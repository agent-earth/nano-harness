#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from nano_harness.baseline import (
    compare_baselines,
    load_cases,
    load_manifest,
)


CANDIDATE = Path(
    "results/harness/qwen35-v6-matched-adapter-v1/candidate/cases.jsonl"
)
FOUR_B = Path(
    "results/harness/qwen35-three-task-replication-v1/4b/cases.jsonl"
)
NINE_B = Path(
    "results/harness/qwen35-three-task-replication-v1/9b/cases.jsonl"
)
MANIFEST = Path("configs/harness/qwen35_v6_matched_adapter_v1.yaml")
SERVING_RECEIPT = Path(
    "results/serving/qwen35-v6-vllm-adapter.receipt.json"
)
PARITY_RECEIPT = Path(
    "results/serving/qwen35-v6-serving-parity.json"
)
PRE_REGISTRATION_REVISION = "2250ed0"
SERVING_PARITY_REVISION = "f7875fe"
EXPECTED_MODELS = {
    "candidate": "qwen3.5-4b-process-v6",
    "four_b": "qwen3.5-4b",
    "nine_b": "qwen3.5-9b",
}
LOOSE_NUMERIC_FINAL_PATTERN = re.compile(
    r"FINAL\s*:\s*([-+]?(?:\d[\d,]*\.?\d*|\.\d+))",
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
            "api_errors": sum(
                row.get("status") == "error" for row in subset
            ),
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
        "api_errors": sum(
            row.get("status") == "error" for row in records
        ),
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
        for case in load_cases(manifest, Path("../../datasets"))
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
        actual = record.get("stages", {}).get("direct", {}).get(
            "input_sha256"
        )
        if (
            record.get("selected_strategy") != "direct"
            or record.get("prompt_sha256") != expected
            or actual != expected
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


def task_non_regression(
    comparison: dict[str, Any],
) -> bool:
    return all(
        metrics["delta"] >= 0
        for metrics in comparison["benchmarks"].values()
    )


def parse_non_regression(
    comparison: dict[str, Any],
) -> bool:
    return all(
        len(metrics["candidate_parse_failures"])
        <= len(metrics["baseline_parse_failures"])
        for metrics in comparison["benchmarks"].values()
    )


def gsm8k_failure_diagnostic(
    candidate: dict[str, dict[str, Any]],
    four_b: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    official_failures = [
        row
        for row in candidate.values()
        if row["benchmark"] == "gsm8k" and float(row["score"]) == 0.0
    ]
    parse_failures = [
        row for row in official_failures if row.get("prediction") is None
    ]
    loose_correct_ids = []
    loose_wrong_ids = []
    no_loose_final_ids = []
    for row in parse_failures:
        match = LOOSE_NUMERIC_FINAL_PATTERN.search(str(row.get("output", "")))
        if match is None:
            no_loose_final_ids.append(row["case_id"])
            continue
        loose_prediction = match.group(1).replace(",", "")
        if loose_prediction == str(row["expected"]):
            loose_correct_ids.append(row["case_id"])
        else:
            loose_wrong_ids.append(row["case_id"])
    parsed_wrong_ids = [
        row["case_id"]
        for row in official_failures
        if row.get("prediction") is not None
    ]
    base_only_ids = [
        case_id
        for case_id, row in candidate.items()
        if row["benchmark"] == "gsm8k"
        and float(row["score"]) == 0.0
        and float(four_b[case_id]["score"]) == 1.0
    ]
    return {
        "official_score_changed": False,
        "official_failures": len(official_failures),
        "official_parse_failures": len(parse_failures),
        "loose_inline_final_matches_reference": len(loose_correct_ids),
        "loose_inline_final_wrong": len(loose_wrong_ids),
        "no_loose_final": len(no_loose_final_ids),
        "parsed_numeric_wrong": len(parsed_wrong_ids),
        "base_four_b_only_failures": len(base_only_ids),
        "case_ids": {
            "loose_inline_final_matches_reference": sorted(loose_correct_ids),
            "loose_inline_final_wrong": sorted(loose_wrong_ids),
            "no_loose_final": sorted(no_loose_final_ids),
            "parsed_numeric_wrong": sorted(parsed_wrong_ids),
            "base_four_b_only_failures": sorted(base_only_ids),
        },
    }


def main() -> None:
    for path in (
        CANDIDATE,
        FOUR_B,
        NINE_B,
        SERVING_RECEIPT,
        PARITY_RECEIPT,
    ):
        if not path.is_file():
            raise SystemExit(f"missing result artifact: {path}")

    candidate_rows = rows(CANDIDATE)
    four_b_rows = rows(FOUR_B)
    if len(candidate_rows) != 211:
        raise SystemExit(
            f"candidate run is incomplete: {len(candidate_rows)}/211"
        )
    candidate_vs_four = compare_baselines(CANDIDATE, FOUR_B)
    candidate_vs_nine = compare_baselines(CANDIDATE, NINE_B)
    costs = {
        "candidate": cost(CANDIDATE),
        "four_b": cost(FOUR_B),
        "nine_b": cost(NINE_B),
    }
    no_api_errors = all(not item["api_errors"] for item in costs.values())
    four_task_non_regression = task_non_regression(candidate_vs_four)
    four_parse_non_regression = parse_non_regression(candidate_vs_four)
    four_macro_non_regression = (
        candidate_vs_four["candidate_macro_accuracy"]
        >= candidate_vs_four["baseline_macro_accuracy"]
    )
    four_micro_non_regression = (
        candidate_vs_four["overall_micro"]["delta"] >= 0
    )
    passed_four_b_non_regression = (
        four_task_non_regression
        and four_parse_non_regression
        and four_macro_non_regression
        and four_micro_non_regression
        and no_api_errors
    )

    nine_overall = candidate_vs_nine["overall_micro"]
    nine_task_non_regression = task_non_regression(candidate_vs_nine)
    significantly_exceeds_nine = (
        candidate_vs_nine["candidate_macro_accuracy"]
        > candidate_vs_nine["baseline_macro_accuracy"]
        and nine_overall["paired_bootstrap_95_ci"][0] > 0
        and nine_overall["mcnemar_exact_p"] < 0.05
        and nine_task_non_regression
        and no_api_errors
    )
    failure_diagnostic = {
        "gsm8k": gsm8k_failure_diagnostic(candidate_rows, four_b_rows)
    }
    serving_receipt = json.loads(
        SERVING_RECEIPT.read_text(encoding="utf-8")
    )
    parity_receipt = json.loads(PARITY_RECEIPT.read_text(encoding="utf-8"))
    if (
        serving_receipt["tensor_content_hashes_match"] is not True
        or serving_receipt["tensor_count"] != 224
        or parity_receipt["full_benchmark_allowed"] is not True
        or parity_receipt["logits_differ"] is not True
    ):
        raise SystemExit("serving parity receipt does not authorize reporting")

    report = {
        "schema_version": "nano_harness_public_v6_matched_adapter_v1",
        "experiment_id": "qwen35-v6-matched-adapter-v1",
        "pre_registration_revision": PRE_REGISTRATION_REVISION,
        "serving_parity_revision": SERVING_PARITY_REVISION,
        "comparisons": {
            "candidate_vs_four_b": candidate_vs_four,
            "candidate_vs_nine_b": candidate_vs_nine,
        },
        "costs": costs,
        "failure_diagnostic": failure_diagnostic,
        "contract_audits": {
            "candidate": audit_direct(
                CANDIDATE,
                EXPECTED_MODELS["candidate"],
            ),
            "four_b": audit_direct(
                FOUR_B,
                EXPECTED_MODELS["four_b"],
            ),
            "nine_b": audit_direct(
                NINE_B,
                EXPECTED_MODELS["nine_b"],
            ),
            "serving": {
                "namespace_tensor_content_matches": True,
                "tensor_count": serving_receipt["tensor_count"],
                "logits_differ": parity_receipt["logits_differ"],
                "known_case_adapter_exact": parity_receipt["known_case"][
                    EXPECTED_MODELS["candidate"]
                ]["exact"],
                "known_case_adapter_semantic": parity_receipt["known_case"][
                    EXPECTED_MODELS["candidate"]
                ]["semantic_valid"],
            },
        },
        "artifacts": {
            "candidate_raw_sha256": sha256_file(CANDIDATE),
            "four_b_raw_sha256": sha256_file(FOUR_B),
            "nine_b_raw_sha256": sha256_file(NINE_B),
            "serving_namespace_receipt_sha256": sha256_file(
                SERVING_RECEIPT
            ),
            "serving_parity_receipt_sha256": sha256_file(
                PARITY_RECEIPT
            ),
            "source_adapter_weights_sha256": serving_receipt[
                "source_adapter_weights_sha256"
            ],
            "serving_adapter_weights_sha256": serving_receipt[
                "serving_adapter_weights_sha256"
            ],
        },
        "decision": {
            "passed_four_b_non_regression": passed_four_b_non_regression,
            "four_b_task_non_regression": four_task_non_regression,
            "four_b_parse_non_regression": four_parse_non_regression,
            "four_b_macro_non_regression": four_macro_non_regression,
            "four_b_micro_non_regression": four_micro_non_regression,
            "significantly_exceeds_nine_b": significantly_exceeds_nine,
            "nine_b_task_non_regression": nine_task_non_regression,
            "nine_b_macro_above": (
                candidate_vs_nine["candidate_macro_accuracy"]
                > candidate_vs_nine["baseline_macro_accuracy"]
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
                "Preserve matched benchmark evidence and choose the next "
                "ablation from task-level candidate-vs-base failures; do not "
                "merge, scale, or start RL without a separate decision."
            ),
        },
    }
    rows_markdown = []
    for benchmark in ("gsm8k", "mmlu", "gpqa_diamond"):
        candidate = costs["candidate"]["by_benchmark"][benchmark]
        four = costs["four_b"]["by_benchmark"][benchmark]
        nine = costs["nine_b"]["by_benchmark"][benchmark]
        rows_markdown.append(
            f"| {benchmark} | {candidate['correct']}/{candidate['cases']} | "
            f"{four['correct']}/{four['cases']} | "
            f"{nine['correct']}/{nine['cases']} |"
        )
    four_overall = candidate_vs_four["overall_micro"]
    gsm8k_diagnostic = failure_diagnostic["gsm8k"]
    markdown = f"""# V6 Matched Adapter Evaluation Result

## Task Results

| Benchmark | V6 adapter | Base 4B | 9B |
| --- | ---: | ---: | ---: |
{chr(10).join(rows_markdown)}

## Candidate Versus Base 4B

- candidate macro: {candidate_vs_four['candidate_macro_accuracy']:.4f};
- base 4B macro: {candidate_vs_four['baseline_macro_accuracy']:.4f};
- micro delta: {four_overall['delta']:+.4f};
- paired 95% CI:
  [{four_overall['paired_bootstrap_95_ci'][0]:+.4f},
  {four_overall['paired_bootstrap_95_ci'][1]:+.4f}];
- exact McNemar p: {four_overall['mcnemar_exact_p']:.6f};
- task non-regression: {four_task_non_regression};
- parse non-regression: {four_parse_non_regression}.

## Candidate Versus 9B

- candidate macro: {candidate_vs_nine['candidate_macro_accuracy']:.4f};
- 9B macro: {candidate_vs_nine['baseline_macro_accuracy']:.4f};
- micro delta: {nine_overall['delta']:+.4f};
- paired 95% CI:
  [{nine_overall['paired_bootstrap_95_ci'][0]:+.4f},
  {nine_overall['paired_bootstrap_95_ci'][1]:+.4f}];
- exact McNemar p: {nine_overall['mcnemar_exact_p']:.6f};
- task non-regression: {nine_task_non_regression}.

## GSM8K Failure Diagnostic

Official candidate GSM8K failures:
{gsm8k_diagnostic['official_failures']}/96.

- official parse failures:
  {gsm8k_diagnostic['official_parse_failures']};
- non-scoring inline `FINAL:` values matching the reference:
  {gsm8k_diagnostic['loose_inline_final_matches_reference']};
- parseable but numerically wrong outputs:
  {gsm8k_diagnostic['parsed_numeric_wrong']};
- base-4B-only correct cases:
  {gsm8k_diagnostic['base_four_b_only_failures']}.

The inline diagnostic does not change the official score. It separates output
contract regressions from genuine numeric or modeling regressions.

## Decision

- passes base 4B non-regression:
  {passed_four_b_non_regression};
- significantly exceeds matched 9B:
  {significantly_exceeds_nine};
- API errors across all arms: {sum(x['api_errors'] for x in costs.values())}.

Regardless of outcome, this evaluation does not directly authorize merge,
scale-up, or RL. Preserve task-level discordances and serving parity evidence
for the next separately pre-registered ablation.

## Reproduction Identity

- pre-registration revision: `{PRE_REGISTRATION_REVISION}`;
- serving parity revision: `{SERVING_PARITY_REVISION}`;
- candidate raw SHA256:
  `{report['artifacts']['candidate_raw_sha256']}`;
- base 4B raw SHA256:
  `{report['artifacts']['four_b_raw_sha256']}`;
- 9B raw SHA256:
  `{report['artifacts']['nine_b_raw_sha256']}`;
- source adapter weights SHA256:
  `{report['artifacts']['source_adapter_weights_sha256']}`;
- serving adapter weights SHA256:
  `{report['artifacts']['serving_adapter_weights_sha256']}`.
"""
    Path(
        "docs/results/v6_matched_adapter_v1.public.json"
    ).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    Path("docs/results/v6_matched_adapter_v1.md").write_text(
        markdown,
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "experiment_id": report["experiment_id"],
                "passed_four_b_non_regression": passed_four_b_non_regression,
                "significantly_exceeds_nine_b": significantly_exceeds_nine,
                "candidate_correct": costs["candidate"]["correct"],
                "four_b_correct": costs["four_b"]["correct"],
                "nine_b_correct": costs["nine_b"]["correct"],
                "candidate_vs_four_b_micro_delta": four_overall["delta"],
                "candidate_vs_nine_b_micro_delta": nine_overall["delta"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
