#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from nano_harness.baseline import compare_baselines, load_cases, load_manifest


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = (
    ROOT
    / "results/harness/qwen35-v11-full-matched-adapter-v1/"
    "candidate/cases.jsonl"
)
FOUR_B = (
    ROOT
    / "results/harness/qwen35-three-task-replication-v1/4b/cases.jsonl"
)
NINE_B = (
    ROOT
    / "results/harness/qwen35-three-task-replication-v1/9b/cases.jsonl"
)
MANIFEST = ROOT / "configs/harness/qwen35_v6_matched_adapter_v1.yaml"
NAMESPACE = ROOT / "results/serving/qwen35-v11-vllm-adapter.receipt.json"
PARITY = ROOT / "results/serving/qwen35-v11-serving-parity.json"
LOCAL = (
    ROOT.parent
    / "nano-train/docs/results/targeted_preservation_sft_smoke_v11.public.json"
)
CANARY = ROOT / "docs/results/v11_adapter_regression_canary_v1.public.json"
PRE_REGISTRATION_REVISION = "6ccc260"
EXPECTED_MODELS = {
    "candidate": "qwen3.5-4b-targeted-v11",
    "four_b": "qwen3.5-4b",
    "nine_b": "qwen3.5-9b",
}
LOOSE_NUMERIC_FINAL_PATTERN = re.compile(
    r"FINAL\s*:?\s*([-+]?(?:\d[\d,]*\.?\d*|\.\d+))",
    re.IGNORECASE,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows(path: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            result[str(row["case_id"])] = row
    return result


def cost(path: Path) -> dict[str, Any]:
    records = list(rows(path).values())
    by_benchmark = {}
    for benchmark in sorted({row["benchmark"] for row in records}):
        subset = [row for row in records if row["benchmark"] == benchmark]
        by_benchmark[benchmark] = {
            "cases": len(subset),
            "correct": int(sum(float(row["score"]) for row in subset)),
            "parse_failures": sum(
                row.get("prediction") is None for row in subset
            ),
            "length_truncations": sum(
                row.get("finish_reason") == "length" for row in subset
            ),
            "api_errors": sum(row.get("status") == "error" for row in subset),
            "total_tokens": sum(
                int(row.get("usage", {}).get("total_tokens", 0))
                for row in subset
            ),
            "wall_seconds": sum(
                float(row["latency_seconds"]) for row in subset
            ),
        }
    return {
        "cases": len(records),
        "correct": int(sum(float(row["score"]) for row in records)),
        "parse_failures": sum(
            row.get("prediction") is None for row in records
        ),
        "length_truncations": sum(
            row.get("finish_reason") == "length" for row in records
        ),
        "api_errors": sum(row.get("status") == "error" for row in records),
        "total_tokens": sum(
            int(row.get("usage", {}).get("total_tokens", 0))
            for row in records
        ),
        "wall_seconds": sum(
            float(row["latency_seconds"]) for row in records
        ),
        "by_benchmark": by_benchmark,
    }


def audit_direct(path: Path, expected_model: str) -> dict[str, Any]:
    manifest = load_manifest(MANIFEST)
    cases = {
        case.case_id: case
        for case in load_cases(manifest, ROOT / "../../datasets")
    }
    results = rows(path)
    failures = []
    if set(cases) != set(results):
        failures.append("case identities")
    if {str(row["model"]) for row in results.values()} != {expected_model}:
        failures.append("model identity")
    for case_id, case in cases.items():
        record = results.get(case_id, {})
        expected = hashlib.sha256(case.prompt.encode()).hexdigest()
        direct = record.get("stages", {}).get("direct", {})
        if (
            record.get("selected_strategy") != "direct"
            or record.get("prompt_sha256") != expected
            or direct.get("input_sha256") != expected
            or record.get("max_tokens") != case.max_tokens
        ):
            failures.append(case_id)
    if failures:
        raise SystemExit(f"direct contract audit failed: {failures[:5]}")
    return {
        "passed": True,
        "cases": len(cases),
        "case_id_set_matches": True,
        "model": expected_model,
        "selected_strategy": "direct",
        "prompt_hashes_match": True,
        "stage_input_hashes_match": True,
        "output_budgets_match": True,
    }


def task_non_regression(comparison: dict[str, Any]) -> bool:
    return all(
        metrics["delta"] >= 0
        for metrics in comparison["benchmarks"].values()
    )


def parse_non_regression(comparison: dict[str, Any]) -> bool:
    return all(
        len(metrics["candidate_parse_failures"])
        <= len(metrics["baseline_parse_failures"])
        for metrics in comparison["benchmarks"].values()
    )


def gsm8k_diagnostic(
    candidate: dict[str, dict[str, Any]],
    four_b: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    failures = [
        row
        for row in candidate.values()
        if row["benchmark"] == "gsm8k" and float(row["score"]) == 0.0
    ]
    parse_failures = [
        row for row in failures if row.get("prediction") is None
    ]
    loose_correct = []
    loose_wrong = []
    no_loose_final = []
    for row in parse_failures:
        match = LOOSE_NUMERIC_FINAL_PATTERN.search(str(row.get("output", "")))
        if match is None:
            no_loose_final.append(row["case_id"])
            continue
        prediction = match.group(1).replace(",", "")
        if prediction == str(row["expected"]):
            loose_correct.append(row["case_id"])
        else:
            loose_wrong.append(row["case_id"])
    parsed_wrong = [
        row["case_id"]
        for row in failures
        if row.get("prediction") is not None
    ]
    base_only = [
        case_id
        for case_id, row in candidate.items()
        if row["benchmark"] == "gsm8k"
        and float(row["score"]) == 0.0
        and float(four_b[case_id]["score"]) == 1.0
    ]
    return {
        "official_score_changed": False,
        "official_failures": len(failures),
        "official_parse_failures": len(parse_failures),
        "length_truncations": sum(
            row.get("finish_reason") == "length" for row in failures
        ),
        "loose_final_matches_reference": len(loose_correct),
        "loose_final_wrong": len(loose_wrong),
        "no_loose_final": len(no_loose_final),
        "parsed_numeric_wrong": len(parsed_wrong),
        "base_four_b_only_failures": len(base_only),
        "case_ids": {
            "loose_final_matches_reference": sorted(loose_correct),
            "loose_final_wrong": sorted(loose_wrong),
            "no_loose_final": sorted(no_loose_final),
            "parsed_numeric_wrong": sorted(parsed_wrong),
            "base_four_b_only_failures": sorted(base_only),
        },
    }


def main() -> None:
    required = (
        CANDIDATE,
        FOUR_B,
        NINE_B,
        NAMESPACE,
        PARITY,
        LOCAL,
        CANARY,
    )
    for path in required:
        if not path.is_file():
            raise SystemExit(f"missing result artifact: {path}")
    candidate_rows = rows(CANDIDATE)
    four_b_rows = rows(FOUR_B)
    if len(candidate_rows) != 211:
        raise SystemExit(f"candidate run incomplete: {len(candidate_rows)}/211")

    versus_four = compare_baselines(CANDIDATE, FOUR_B)
    versus_nine = compare_baselines(CANDIDATE, NINE_B)
    costs = {
        "candidate": cost(CANDIDATE),
        "four_b": cost(FOUR_B),
        "nine_b": cost(NINE_B),
    }
    no_api_errors = all(not item["api_errors"] for item in costs.values())
    four_task = task_non_regression(versus_four)
    four_parse = parse_non_regression(versus_four)
    four_macro = (
        versus_four["candidate_macro_accuracy"]
        >= versus_four["baseline_macro_accuracy"]
    )
    four_micro = versus_four["overall_micro"]["delta"] >= 0
    passed_four = (
        four_task and four_parse and four_macro and four_micro and no_api_errors
    )
    nine_overall = versus_nine["overall_micro"]
    nine_task = task_non_regression(versus_nine)
    significant_nine = (
        versus_nine["candidate_macro_accuracy"]
        > versus_nine["baseline_macro_accuracy"]
        and nine_overall["paired_bootstrap_95_ci"][0] > 0
        and nine_overall["mcnemar_exact_p"] < 0.05
        and nine_task
        and no_api_errors
    )

    namespace = json.loads(NAMESPACE.read_text(encoding="utf-8"))
    parity = json.loads(PARITY.read_text(encoding="utf-8"))
    local = json.loads(LOCAL.read_text(encoding="utf-8"))
    canary = json.loads(CANARY.read_text(encoding="utf-8"))
    if (
        namespace["tensor_count"] != 224
        or namespace["tensor_content_hashes_match"] is not True
        or parity["adapter_parent_matches"] is not True
        or parity["logits_differ"] is not True
        or local["decision"]["sealed_canary_allowed"] is not True
        or canary["decision"]["full_benchmark_allowed"] is not True
    ):
        raise SystemExit("serving or gate receipt does not authorize reporting")

    diagnostic = gsm8k_diagnostic(candidate_rows, four_b_rows)
    report = {
        "schema_version": "nano_harness_public_v11_full_adapter_v1",
        "experiment_id": "qwen35-v11-full-matched-adapter-v1",
        "pre_registration_revision": PRE_REGISTRATION_REVISION,
        "comparisons": {
            "candidate_vs_four_b": versus_four,
            "candidate_vs_nine_b": versus_nine,
        },
        "costs": costs,
        "failure_diagnostic": {"gsm8k": diagnostic},
        "contract_audits": {
            "candidate": audit_direct(
                CANDIDATE, EXPECTED_MODELS["candidate"]
            ),
            "four_b": audit_direct(FOUR_B, EXPECTED_MODELS["four_b"]),
            "nine_b": audit_direct(NINE_B, EXPECTED_MODELS["nine_b"]),
            "serving": {
                "namespace_tensor_count": namespace["tensor_count"],
                "namespace_tensor_content_matches": namespace[
                    "tensor_content_hashes_match"
                ],
                "adapter_parent_matches": parity["adapter_parent_matches"],
                "base_adapter_logits_differ": parity["logits_differ"],
                "local_gate_passed": local["passed"],
                "canary_passed": canary["passed"],
            },
        },
        "artifacts": {
            "candidate_raw_sha256": sha256_file(CANDIDATE),
            "four_b_raw_sha256": sha256_file(FOUR_B),
            "nine_b_raw_sha256": sha256_file(NINE_B),
            "namespace_receipt_sha256": sha256_file(NAMESPACE),
            "serving_parity_sha256": sha256_file(PARITY),
            "local_gate_report_sha256": sha256_file(LOCAL),
            "canary_report_sha256": sha256_file(CANARY),
            "adapter_tree_sha256": local["artifacts"]["adapter_sha256"],
        },
        "decision": {
            "passed_four_b_non_regression": passed_four,
            "four_b_task_non_regression": four_task,
            "four_b_parse_non_regression": four_parse,
            "four_b_macro_non_regression": four_macro,
            "four_b_micro_non_regression": four_micro,
            "significantly_exceeds_nine_b": significant_nine,
            "nine_b_task_non_regression": nine_task,
            "nine_b_macro_above": (
                versus_nine["candidate_macro_accuracy"]
                > versus_nine["baseline_macro_accuracy"]
            ),
            "nine_b_micro_ci_lower_above_zero": (
                nine_overall["paired_bootstrap_95_ci"][0] > 0
            ),
            "nine_b_mcnemar_below_005": (
                nine_overall["mcnemar_exact_p"] < 0.05
            ),
            "no_api_errors": no_api_errors,
            "merge_allowed": False,
            "scale_allowed": False,
            "rl_allowed": False,
            "next_action": (
                "Reject v11 for promotion. Abstract the four base-4B-only "
                "failures into non-evaluation failure families, preserve the "
                "GPQA gain and format stability, and pre-register a new data "
                "ablation without training on benchmark or canary rows."
            ),
        },
    }
    table_rows = []
    for benchmark in ("gsm8k", "mmlu", "gpqa_diamond"):
        candidate = costs["candidate"]["by_benchmark"][benchmark]
        four = costs["four_b"]["by_benchmark"][benchmark]
        nine = costs["nine_b"]["by_benchmark"][benchmark]
        table_rows.append(
            f"| {benchmark} | {candidate['correct']}/{candidate['cases']} | "
            f"{four['correct']}/{four['cases']} | "
            f"{nine['correct']}/{nine['cases']} |"
        )
    four_overall = versus_four["overall_micro"]
    markdown = f"""# V11 Full Matched Adapter Result

## Task Results

| Benchmark | V11 adapter | Base 4B | 9B |
| --- | ---: | ---: | ---: |
{chr(10).join(table_rows)}

V11 scores {costs['candidate']['correct']}/211, versus base 4B
{costs['four_b']['correct']}/211 and 9B {costs['nine_b']['correct']}/211.

## Candidate Versus Base 4B

- candidate macro: {versus_four['candidate_macro_accuracy']:.4f};
- base 4B macro: {versus_four['baseline_macro_accuracy']:.4f};
- micro delta: {four_overall['delta']:+.4f};
- paired 95% CI:
  [{four_overall['paired_bootstrap_95_ci'][0]:+.4f},
  {four_overall['paired_bootstrap_95_ci'][1]:+.4f}];
- exact McNemar p: {four_overall['mcnemar_exact_p']:.6f};
- task non-regression: {four_task};
- parse non-regression: {four_parse}.

## Candidate Versus 9B

- candidate macro: {versus_nine['candidate_macro_accuracy']:.4f};
- 9B macro: {versus_nine['baseline_macro_accuracy']:.4f};
- micro delta: {nine_overall['delta']:+.4f};
- paired 95% CI:
  [{nine_overall['paired_bootstrap_95_ci'][0]:+.4f},
  {nine_overall['paired_bootstrap_95_ci'][1]:+.4f}];
- exact McNemar p: {nine_overall['mcnemar_exact_p']:.6f};
- task non-regression: {nine_task}.

The 11-case point improvement over 9B is not statistically supported because
the confidence interval crosses zero and McNemar p is above 0.05.

## Failure Diagnostic

Candidate GSM8K has {diagnostic['official_failures']}/96 official failures,
including {diagnostic['official_parse_failures']} parse failures and
{diagnostic['length_truncations']} length truncations. The official score is
unchanged by diagnostics.

Against base 4B, there are
{four_overall['paired_counts']['candidate_only']} candidate-only wins and
{four_overall['paired_counts']['baseline_only']} base-only wins. The latter
are the bounded source for abstract failure-family analysis; their benchmark
rows remain ineligible for training.

## Decision

V11 fails base-4B task non-regression because GSM8K and MMLU are each one case
lower. It also fails the pre-registered statistical superiority gate versus
9B. Reject promotion, merge, scale-up, and RL.

Preserve the GPQA gain, local family improvement, and format stability. The
next data ablation may use only abstract failure families, never benchmark or
canary prompts, outputs, references, or case payloads.

## Reproduction Identity

- pre-registration revision: `{PRE_REGISTRATION_REVISION}`;
- candidate raw SHA256: `{report['artifacts']['candidate_raw_sha256']}`;
- base 4B raw SHA256: `{report['artifacts']['four_b_raw_sha256']}`;
- 9B raw SHA256: `{report['artifacts']['nine_b_raw_sha256']}`;
- adapter tree SHA256: `{report['artifacts']['adapter_tree_sha256']}`;
- namespace receipt SHA256:
  `{report['artifacts']['namespace_receipt_sha256']}`;
- serving parity SHA256: `{report['artifacts']['serving_parity_sha256']}`;
- canary report SHA256: `{report['artifacts']['canary_report_sha256']}`.
"""
    output = ROOT / "docs/results"
    (output / "v11_full_matched_adapter_v1.public.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "v11_full_matched_adapter_v1.md").write_text(
        markdown,
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "candidate_correct": costs["candidate"]["correct"],
                "four_b_correct": costs["four_b"]["correct"],
                "nine_b_correct": costs["nine_b"]["correct"],
                "passed_four_b_non_regression": passed_four,
                "significantly_exceeds_nine_b": significant_nine,
                "rl_allowed": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
