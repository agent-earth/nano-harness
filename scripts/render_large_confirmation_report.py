#!/usr/bin/env python3

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_three_task_replication_report as base  # noqa: E402


PATHS = {
    "four_b": Path(
        "results/harness/qwen35-large-confirmation-v1/4b/cases.jsonl"
    ),
    "nine_b": Path(
        "results/harness/qwen35-large-confirmation-v1/9b/cases.jsonl"
    ),
}
MANIFEST = Path("configs/harness/qwen35_large_confirmation_v1.yaml")


def format_diagnostic() -> dict:
    records = base.rows(PATHS["nine_b"])
    failures = [
        row
        for row in records.values()
        if row["benchmark"] == "mmlu" and row.get("prediction") is None
    ]
    matching = []
    mismatching = []
    other = []
    for row in failures:
        output = str(row.get("output", "")).strip()
        match = re.fullmatch(r"FINAL\s+([A-D])", output, re.IGNORECASE)
        if match is None:
            other.append(row["case_id"])
            continue
        record = {
            "case_id": row["case_id"],
            "letter": match.group(1).upper(),
            "expected": row["expected"],
        }
        if record["letter"] == record["expected"]:
            matching.append(record)
        else:
            mismatching.append(record)
    official_correct = sum(
        float(row["score"])
        for row in records.values()
        if row["benchmark"] == "mmlu"
    )
    return {
        "scoring_effect": "none",
        "warning": (
            "Diagnostic only: official outputs, predictions, scores, paired "
            "statistics, and decisions are unchanged."
        ),
        "official_parse_failures": len(failures),
        "final_without_colon": len(matching) + len(mismatching),
        "other_shapes": other,
        "letters_matching_reference": len(matching),
        "letters_mismatching_reference": len(mismatching),
        "matching_cases": matching,
        "mismatching_cases": mismatching,
        "official_nine_b_mmlu_correct": int(official_correct),
        "hypothetical_correct_if_colon_normalized": int(
            official_correct + len(matching)
        ),
        "hypothetical_accuracy_if_colon_normalized": (
            official_correct + len(matching)
        )
        / 256,
    }


def main() -> None:
    base.PATHS = PATHS
    base.MANIFEST = MANIFEST
    for path in PATHS.values():
        if not path.is_file():
            raise SystemExit(f"missing result: {path}")

    comparison = base.compare_baselines(PATHS["four_b"], PATHS["nine_b"])
    overall = comparison["overall_micro"]
    costs = {label: base.cost(path) for label, path in PATHS.items()}
    diagnostic = format_diagnostic()
    per_benchmark_non_regression = all(
        result["delta"] >= 0 for result in comparison["benchmarks"].values()
    )
    no_api_errors = all(not value["api_errors"] for value in costs.values())
    accepted = (
        comparison["candidate_macro_accuracy"]
        > comparison["baseline_macro_accuracy"]
        and overall["paired_bootstrap_95_ci"][0] > 0
        and overall["mcnemar_exact_p"] < 0.05
        and per_benchmark_non_regression
        and no_api_errors
    )
    report = {
        "schema_version": "nano_harness_public_large_confirmation_v1",
        "experiment_id": "qwen35-large-confirmation-v1",
        "code_revision": base.git_revision(),
        "comparison": comparison,
        "costs": costs,
        "contract_audits": {
            "four_b": base.audit_direct(PATHS["four_b"]),
            "nine_b": base.audit_direct(PATHS["nine_b"]),
        },
        "artifacts": {
            f"{label}_raw_sha256": base.sha256_file(path)
            for label, path in PATHS.items()
        },
        "non_scoring_format_diagnostic": diagnostic,
        "decision": {
            "accepted": accepted,
            "macro_above_9b": (
                comparison["candidate_macro_accuracy"]
                > comparison["baseline_macro_accuracy"]
            ),
            "paired_micro_lower_bound_above_zero": (
                overall["paired_bootstrap_95_ci"][0] > 0
            ),
            "mcnemar_below_005": overall["mcnemar_exact_p"] < 0.05,
            "per_benchmark_non_regression": per_benchmark_non_regression,
            "no_api_errors": no_api_errors,
            "mmlu_official_advantage_significant": (
                comparison["benchmarks"]["mmlu"]["paired_bootstrap_95_ci"][0] > 0
                and comparison["benchmarks"]["mmlu"]["mcnemar_exact_p"] < 0.05
            ),
            "gsm8k_non_regression": (
                comparison["benchmarks"]["gsm8k"]["delta"] >= 0
            ),
            "next_experiment": (
                "Stop direct-only confirmation. Separate format compliance "
                "from semantic quality, and use versioned discordances to "
                "design format-aware harness and training data ablations."
            ),
        },
    }

    rows = []
    for benchmark in ("gsm8k", "mmlu"):
        result = comparison["benchmarks"][benchmark]
        rows.append(
            f"| {benchmark} | {result['candidate_correct']}/{result['cases']} "
            f"({result['candidate_accuracy']:.4f}) | "
            f"{result['baseline_correct']}/{result['cases']} "
            f"({result['baseline_accuracy']:.4f}) | "
            f"{result['delta']:+.4f} | "
            f"[{result['paired_bootstrap_95_ci'][0]:+.4f}, "
            f"{result['paired_bootstrap_95_ci'][1]:+.4f}] | "
            f"{result['mcnemar_exact_p']:.4g} |"
        )

    markdown = f"""# Large Independent 4B/9B Confirmation Result

## Official Result

| Benchmark | Qwen3.5-4B | Qwen3.5-9B | Delta | Paired 95% CI | McNemar p |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

- 4B macro accuracy: {comparison['candidate_macro_accuracy']:.4f};
- 9B macro accuracy: {comparison['baseline_macro_accuracy']:.4f};
- macro delta: {comparison['macro_delta']:+.4f}.

Across all 512 matched cases, 4B scores
{overall['candidate_correct']}/512 ({overall['candidate_accuracy']:.4f}) and
9B scores {overall['baseline_correct']}/512
({overall['baseline_accuracy']:.4f}). The paired micro delta is
{overall['delta']:+.4f}, with 95% bootstrap CI
[{overall['paired_bootstrap_95_ci'][0]:+.4f},
{overall['paired_bootstrap_95_ci'][1]:+.4f}] and exact McNemar
`p={overall['mcnemar_exact_p']:.6f}`.

There are {overall['paired_counts']['candidate_only']} 4B-only wins and
{overall['paired_counts']['baseline_only']} 9B-only wins. MMLU significantly
favors 4B under the official strict answer contract, while GSM8K favors 9B.
The aggregate interval crosses zero and the per-task non-regression rule
fails.

## Non-Scoring Format Diagnostic

The official results above are unchanged. All
{diagnostic['official_parse_failures']} 9B MMLU parse failures have the form
`FINAL <letter>` without the required colon. Of those letters,
{diagnostic['letters_matching_reference']} match the reference and
{diagnostic['letters_mismatching_reference']} do not.

If a colon-only normalization were applied hypothetically, 9B MMLU would be
{diagnostic['hypothetical_correct_if_colon_normalized']}/256
({diagnostic['hypothetical_accuracy_if_colon_normalized']:.4f}), compared with
the official 4B result of 180/256. This value is diagnostic only and is not
used in any score, confidence interval, p-value, or decision. It shows that
the official MMLU advantage primarily measures answer-contract compliance,
not stable semantic superiority.

## Parse And Cost

- 4B: {costs['four_b']['parse_failures']} parse failures,
  {costs['four_b']['length_truncations']} length truncations,
  {costs['four_b']['total_tokens']} tokens,
  {costs['four_b']['wall_seconds']:.1f}s summed request latency;
- 9B: {costs['nine_b']['parse_failures']} parse failures,
  {costs['nine_b']['length_truncations']} length truncations,
  {costs['nine_b']['total_tokens']} tokens,
  {costs['nine_b']['wall_seconds']:.1f}s summed request latency;
- both arms: zero API errors.

## Contract Audit

Both arms contain exactly the committed 512 unique case IDs. Prompt hashes,
direct-stage input hashes, strategies, dataset versions, scorers, and budgets
match the pre-registration. Raw outputs remain local and ignored.

## Decision

The confirmation fails the pre-registered superiority rule: aggregate
significance and GSM8K non-regression both fail. Stop direct-only confirmation.
Separate format-compliance examples from semantic discordances before any
harness or training ablation.

## Reproduction Identity

- Pre-registration/code revision: `{report['code_revision']}`
- 4B raw SHA256: `{report['artifacts']['four_b_raw_sha256']}`
- 9B raw SHA256: `{report['artifacts']['nine_b_raw_sha256']}`
"""
    Path("docs/results/large_confirmation_v1.public.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    Path("docs/results/large_confirmation_v1.md").write_text(
        markdown,
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "experiment_id": report["experiment_id"],
                "accepted": accepted,
                "macro_delta": comparison["macro_delta"],
                "micro_delta": overall["delta"],
                "micro_ci": overall["paired_bootstrap_95_ci"],
                "mcnemar_exact_p": overall["mcnemar_exact_p"],
                "per_benchmark_non_regression": per_benchmark_non_regression,
                "nine_b_format_matches": diagnostic["letters_matching_reference"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
