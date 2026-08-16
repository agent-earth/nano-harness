#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from nano_harness.baseline import compare_baselines


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = (
    ROOT
    / "results/harness/qwen35-anchored-v1-full-development/"
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
LOCAL = (
    ROOT.parent
    / "nano-train/docs/results/"
    "v11_schedule_b_only_anchor_continuation_v1.public.json"
)
CANARY = ROOT / "docs/results/anchored_v1_regression_canary.public.json"
NAMESPACE = (
    ROOT / "results/serving/qwen35-anchored-v1-vllm-adapter.receipt.json"
)
PARITY = ROOT / "results/serving/qwen35-anchored-v1-serving-parity.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    candidate = rows(CANDIDATE)
    if len(candidate) != 211 or any(row["status"] != "completed" for row in candidate):
        raise SystemExit("anchored development run is incomplete")
    versus_four = compare_baselines(CANDIDATE, FOUR_B)
    versus_nine = compare_baselines(CANDIDATE, NINE_B)
    by_benchmark = {}
    for benchmark in ("gsm8k", "mmlu", "gpqa_diamond"):
        subset = [row for row in candidate if row["benchmark"] == benchmark]
        by_benchmark[benchmark] = {
            "cases": len(subset),
            "correct": int(sum(float(row["score"]) for row in subset)),
            "api_errors": sum(row["status"] == "error" for row in subset),
            "parse_failures": sum(row.get("prediction") is None for row in subset),
            "length_truncations": sum(
                row.get("finish_reason") == "length" for row in subset
            ),
            "total_tokens": sum(
                int(row.get("usage", {}).get("total_tokens", 0))
                for row in subset
            ),
            "wall_seconds": sum(float(row["latency_seconds"]) for row in subset),
        }
    local = json.loads(LOCAL.read_text(encoding="utf-8"))
    canary = json.loads(CANARY.read_text(encoding="utf-8"))
    namespace = json.loads(NAMESPACE.read_text(encoding="utf-8"))
    parity = json.loads(PARITY.read_text(encoding="utf-8"))
    if (
        local["decision"]["sealed_canary_allowed"] is not True
        or canary["decision"]["full_development_suite_allowed"] is not True
        or namespace["tensor_content_hashes_match"] is not True
        or parity["adapter_parent_matches"] is not True
        or parity["logits_differ"] is not True
    ):
        raise SystemExit("staged serving receipts do not authorize reporting")

    four_task_non_regression = all(
        metric["delta"] >= 0
        for metric in versus_four["benchmarks"].values()
    )
    four_micro_non_regression = versus_four["overall_micro"]["delta"] >= 0
    four_macro_non_regression = (
        versus_four["candidate_macro_accuracy"]
        >= versus_four["baseline_macro_accuracy"]
    )
    nine = versus_nine["overall_micro"]
    significant_over_nine = (
        versus_nine["candidate_macro_accuracy"]
        > versus_nine["baseline_macro_accuracy"]
        and nine["paired_bootstrap_95_ci"][0] > 0
        and nine["mcnemar_exact_p"] < 0.05
        and all(
            metric["delta"] >= 0
            for metric in versus_nine["benchmarks"].values()
        )
    )
    report = {
        "schema_version": "nano_harness_anchored_v1_full_development_public",
        "experiment_id": "qwen35-anchored-v1-full-development",
        "pre_registration_revision": "4396a51",
        "comparisons": {
            "candidate_vs_four_b": versus_four,
            "candidate_vs_nine_b": versus_nine,
        },
        "candidate": {
            "cases": len(candidate),
            "correct": sum(row["correct"] for row in by_benchmark.values()),
            "by_benchmark": by_benchmark,
        },
        "identity": {
            "candidate_raw_sha256": sha256(CANDIDATE),
            "four_b_raw_sha256": sha256(FOUR_B),
            "nine_b_raw_sha256": sha256(NINE_B),
            "local_report_sha256": sha256(LOCAL),
            "canary_report_sha256": sha256(CANARY),
            "namespace_receipt_sha256": sha256(NAMESPACE),
            "serving_parity_sha256": sha256(PARITY),
            "adapter_tree_sha256": local["identity"]["adapter_tree_sha256"],
        },
        "decision": {
            "aggregate_above_four_b": four_micro_non_regression,
            "macro_above_four_b": four_macro_non_regression,
            "four_b_task_non_regression": four_task_non_regression,
            "passed_four_b_non_regression": (
                four_task_non_regression
                and four_micro_non_regression
                and four_macro_non_regression
            ),
            "nine_b_ci_lower_above_zero": (
                nine["paired_bootstrap_95_ci"][0] > 0
            ),
            "nine_b_mcnemar_below_005": nine["mcnemar_exact_p"] < 0.05,
            "significantly_exceeds_nine_b": significant_over_nine,
            "independent_holdout_allowed": False,
            "merge_allowed": False,
            "scale_allowed": False,
            "rl_allowed": False,
            "next_action": (
                "Preserve anchored v1 as the strongest aggregate candidate. "
                "Do not open the independent holdout because MMLU is one "
                "point below base 4B and McNemar versus 9B is 0.066. "
                "Design a public-safe generic choice-preservation objective."
            ),
        },
    }
    markdown = f"""# Anchored v1 Full Development Result

| Benchmark | Anchored v1 | Base 4B | 9B |
| --- | ---: | ---: | ---: |
| GSM8K | {by_benchmark['gsm8k']['correct']}/96 | 90/96 | 89/96 |
| MMLU | {by_benchmark['mmlu']['correct']}/96 | 67/96 | 58/96 |
| GPQA-Diamond | {by_benchmark['gpqa_diamond']['correct']}/19 | 6/19 | 4/19 |

Anchored v1 scores 164/211, versus base 4B at 163/211 and 9B at
151/211.

Versus base 4B, micro delta is
{versus_four['overall_micro']['delta']:+.4f} with 95% CI
[{versus_four['overall_micro']['paired_bootstrap_95_ci'][0]:+.4f},
{versus_four['overall_micro']['paired_bootstrap_95_ci'][1]:+.4f}] and
McNemar p={versus_four['overall_micro']['mcnemar_exact_p']:.3f}. MMLU is one
case lower, so the frozen per-task non-regression gate fails.

Versus 9B, micro delta is {nine['delta']:+.4f}, 95% CI
[{nine['paired_bootstrap_95_ci'][0]:+.4f},
{nine['paired_bootstrap_95_ci'][1]:+.4f}], and McNemar
p={nine['mcnemar_exact_p']:.3f}. The CI is above zero, but p does not pass the
pre-registered 0.05 gate.

Do not open the independent holdout. Merge, scale, and RL remain forbidden.
"""
    output = ROOT / "docs/results"
    (output / "anchored_v1_full_development.public.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    (output / "anchored_v1_full_development.md").write_text(markdown)
    print(
        json.dumps(
            {
                "correct": report["candidate"]["correct"],
                "passed_four_b_non_regression": report["decision"][
                    "passed_four_b_non_regression"
                ],
                "significantly_exceeds_nine_b": significant_over_nine,
                "independent_holdout_allowed": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
