#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from nano_harness.baseline import load_cases, load_manifest, sha256_file


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/harness/qwen35_adapter_regression_canary_v1.yaml"
RAW = (
    ROOT
    / "results/harness/qwen35-v11-adapter-regression-canary-v1/"
    "candidate/cases.jsonl"
)
NAMESPACE = ROOT / "results/serving/qwen35-v11-vllm-adapter.receipt.json"
PARITY = ROOT / "results/serving/qwen35-v11-serving-parity.json"
LOCAL = (
    ROOT.parent
    / "nano-train/docs/results/targeted_preservation_sft_smoke_v11.public.json"
)
BASE = (
    ROOT
    / "results/harness/qwen35-three-task-replication-v1/4b/cases.jsonl"
)
V6 = (
    ROOT
    / "results/harness/qwen35-v6-matched-adapter-v1/candidate/cases.jsonl"
)


def latest(path: Path) -> dict[str, dict]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[str(row["case_id"])] = row
    return rows


def sha256_json(value: object) -> str:
    rendered = json.dumps(value, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def main() -> None:
    manifest = load_manifest(MANIFEST)
    cases = load_cases(manifest, ROOT / "../../datasets")
    candidate = latest(RAW)
    base = latest(BASE)
    v6 = latest(V6)
    namespace = json.loads(NAMESPACE.read_text(encoding="utf-8"))
    parity = json.loads(PARITY.read_text(encoding="utf-8"))
    local = json.loads(LOCAL.read_text(encoding="utf-8"))
    case_ids = [case.case_id for case in cases]
    if set(candidate) != set(case_ids):
        raise SystemExit("candidate raw case set differs from frozen canary")
    if not set(case_ids) <= set(base) or not set(case_ids) <= set(v6):
        raise SystemExit("calibration raw results lack canary cases")

    thresholds = {
        "gsm8k": 14,
        "mmlu": 13,
        "gpqa_diamond": 3,
    }
    expected_cases = {
        "gsm8k": 16,
        "mmlu": 16,
        "gpqa_diamond": 8,
    }
    by_benchmark = {}
    for benchmark in thresholds:
        benchmark_ids = [
            case.case_id for case in cases if case.benchmark == benchmark
        ]
        rows = [candidate[case_id] for case_id in benchmark_ids]
        base_rows = [base[case_id] for case_id in benchmark_ids]
        by_benchmark[benchmark] = {
            "cases": len(rows),
            "correct": int(sum(float(row["score"]) for row in rows)),
            "threshold": thresholds[benchmark],
            "threshold_passed": (
                sum(float(row["score"]) for row in rows)
                >= thresholds[benchmark]
            ),
            "parse_failures": sum(row.get("prediction") is None for row in rows),
            "base_parse_failures": sum(
                row.get("prediction") is None for row in base_rows
            ),
            "parse_non_regression": (
                sum(row.get("prediction") is None for row in rows)
                <= sum(row.get("prediction") is None for row in base_rows)
            ),
            "api_errors": sum(row.get("status") == "error" for row in rows),
            "length_truncations": sum(
                row.get("finish_reason") == "length" for row in rows
            ),
        }
        if by_benchmark[benchmark]["cases"] != expected_cases[benchmark]:
            raise SystemExit(f"unexpected {benchmark} case count")

    total = sum(row["correct"] for row in by_benchmark.values())
    base_total = sum(float(base[case_id]["score"]) for case_id in case_ids)
    v6_total = sum(float(v6[case_id]["score"]) for case_id in case_ids)
    passed = (
        total >= 30
        and all(row["threshold_passed"] for row in by_benchmark.values())
        and all(row["parse_non_regression"] for row in by_benchmark.values())
        and not any(row["api_errors"] for row in by_benchmark.values())
        and not any(row["length_truncations"] for row in by_benchmark.values())
        and namespace["tensor_count"] == 224
        and namespace["tensor_content_hashes_match"] is True
        and parity["adapter_parent_matches"] is True
        and parity["logits_differ"] is True
        and local["decision"]["sealed_canary_allowed"] is True
    )
    if not passed:
        raise SystemExit("v11 adapter fails the frozen canary gate")

    report = {
        "schema_version": "nano_harness_v11_adapter_canary_v1",
        "suite_id": manifest.suite_id,
        "passed": passed,
        "policy": {
            "source_split": "sealed_eval_canary",
            "training_eligible": False,
            "quality_claim_allowed": False,
            "purpose": "adapter_regression_gate_only",
            "post_v6_calibrated": True,
            "case_level_publication_allowed": False,
        },
        "identity": {
            "model": "qwen3.5-4b-targeted-v11",
            "canary_manifest_sha256": sha256_file(MANIFEST),
            "candidate_raw_sha256": sha256_file(RAW),
            "candidate_case_id_set_sha256": sha256_json(sorted(case_ids)),
            "namespace_receipt_sha256": sha256_file(NAMESPACE),
            "serving_parity_sha256": sha256_file(PARITY),
            "local_gate_report_sha256": sha256_file(LOCAL),
        },
        "serving": {
            "namespace_tensor_count": namespace["tensor_count"],
            "namespace_tensor_content_hashes_match": namespace[
                "tensor_content_hashes_match"
            ],
            "adapter_parent_matches": parity["adapter_parent_matches"],
            "base_adapter_logits_differ": parity["logits_differ"],
        },
        "calibration": {
            "base_four_b_total": int(base_total),
            "rejected_v6_total": int(v6_total),
        },
        "candidate": {
            "cases": len(case_ids),
            "correct": total,
            "required_total": 30,
            "by_benchmark": by_benchmark,
            "api_errors": sum(
                row["api_errors"] for row in by_benchmark.values()
            ),
            "parse_failures": sum(
                row["parse_failures"] for row in by_benchmark.values()
            ),
            "length_truncations": sum(
                row["length_truncations"] for row in by_benchmark.values()
            ),
        },
        "decision": {
            "canary_passed": True,
            "independent_quality_claim_allowed": False,
            "full_benchmark_allowed": True,
            "merge_allowed": False,
            "scale_allowed": False,
            "rl_allowed": False,
            "next_action": (
                "Pre-register and run the unchanged adapter on the frozen "
                "211-case matched suite. Preserve all raw case evidence "
                "locally and report task-level paired comparisons."
            ),
        },
    }
    markdown = f"""# V11 Adapter Regression Canary Result

## Result

The unchanged v11 adapter passes the sealed regression canary:

- GSM8K: {by_benchmark['gsm8k']['correct']}/16, threshold 14/16;
- MMLU: {by_benchmark['mmlu']['correct']}/16, threshold 13/16;
- GPQA-Diamond: {by_benchmark['gpqa_diamond']['correct']}/8, threshold 3/8;
- total: {total}/40, threshold 30/40;
- API errors: {report['candidate']['api_errors']};
- parse failures: {report['candidate']['parse_failures']};
- length truncations: {report['candidate']['length_truncations']}.

The calibration remains base 4B {int(base_total)}/40 and rejected v6
{int(v6_total)}/40. Namespace conversion preserves all 224 tensor contents,
the adapter parent is correct, and base/adapter logits differ.

## Boundary

This post-v6-calibrated canary is a regression gate only. It cannot establish
quality uplift and its case-level outputs or IDs must not enter training.

Passing permits only the exact adapter to run the frozen 211-case matched
suite. Merge, scale-up, and RL remain forbidden.

## Identity

- manifest SHA256: `{report['identity']['canary_manifest_sha256']}`;
- candidate raw SHA256: `{report['identity']['candidate_raw_sha256']}`;
- namespace receipt SHA256:
  `{report['identity']['namespace_receipt_sha256']}`;
- serving parity SHA256: `{report['identity']['serving_parity_sha256']}`;
- local gate report SHA256:
  `{report['identity']['local_gate_report_sha256']}`.
"""
    output = ROOT / "docs/results"
    (output / "v11_adapter_regression_canary_v1.public.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "v11_adapter_regression_canary_v1.md").write_text(
        markdown,
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "passed": passed,
                "correct": total,
                "by_benchmark": {
                    key: value["correct"] for key, value in by_benchmark.items()
                },
                "full_benchmark_allowed": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
