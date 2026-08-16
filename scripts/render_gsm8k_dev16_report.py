#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_gsm8k_dev15_report as audit  # noqa: E402


PATHS = {
    "four_b_direct": Path(
        "results/harness/qwen35-gsm8k-dev16-direct-v1/4b/cases.jsonl"
    ),
    "four_b_treatment": Path(
        "results/harness/qwen35-gsm8k-dev16-constrained-v1/4b/cases.jsonl"
    ),
    "nine_b_direct": Path(
        "results/harness/qwen35-gsm8k-dev16-direct-v1/9b/cases.jsonl"
    ),
}
DIRECT_MANIFEST = Path("configs/harness/qwen35_gsm8k_dev16_direct_v1.yaml")
TREATMENT_MANIFEST = Path(
    "configs/harness/qwen35_gsm8k_dev16_constrained_v1.yaml"
)


def _next_experiment(accepted: bool, recovery_count: int) -> str:
    if accepted:
        return (
            "Integrate the frozen constrained recovery guard into the next "
            "pre-registered sealed multi-task holdout."
        )
    if recovery_count == 0:
        return (
            "Stop enlarging GSM8K development windows; retain the validated "
            "mechanism as an optional no-op parse guard and return to "
            "higher-leverage harness hypotheses."
        )
    return (
        "Reject promotion of constrained recovery because observed triggers "
        "did not satisfy every pre-registered benefit and safety rule."
    )


def main() -> None:
    audit.PATHS = PATHS
    audit.DIRECT_MANIFEST = DIRECT_MANIFEST
    audit.TREATMENT_MANIFEST = TREATMENT_MANIFEST

    for path in PATHS.values():
        if not path.is_file():
            raise SystemExit(f"missing result: {path}")

    versus_4b = audit.compact(
        audit.compare_baselines(PATHS["four_b_treatment"], PATHS["four_b_direct"])
    )
    versus_9b = audit.compact(
        audit.compare_baselines(PATHS["four_b_treatment"], PATHS["nine_b_direct"])
    )
    costs = {label: audit.cost(path) for label, path in PATHS.items()}
    analysis = audit.recovery_analysis()
    token_ratio = (
        costs["four_b_treatment"]["total_tokens"]
        / costs["four_b_direct"]["total_tokens"]
    )
    accepted = (
        analysis["recovery_count"] >= 1
        and len(analysis["recovery_wins"]) >= 1
        and versus_4b["paired_counts"]["baseline_only"] == 0
        and costs["four_b_treatment"]["parse_failures"]
        < costs["four_b_direct"]["parse_failures"]
        and costs["four_b_treatment"]["regex_matches"]
        == costs["four_b_treatment"]["recovery_calls"]
        and not costs["four_b_treatment"]["api_errors"]
        and token_ratio < 1.2
    )
    report = {
        "schema_version": "nano_harness_public_gsm8k_dev_v1",
        "experiment_id": "qwen35-gsm8k-dev16-constrained-recovery-v1",
        "code_revision": audit.git_revision(),
        "versus_4b_direct": versus_4b,
        "versus_9b_direct": versus_9b,
        "costs": costs,
        "token_ratio_vs_4b_direct": token_ratio,
        "recovery_analysis": analysis,
        "contract_audits": {
            "four_b_direct": audit.audit_direct(PATHS["four_b_direct"]),
            "four_b_treatment": audit.audit_treatment(),
            "nine_b_direct": audit.audit_direct(PATHS["nine_b_direct"]),
        },
        "artifacts": {
            f"{label}_raw_sha256": audit.sha256_file(path)
            for label, path in PATHS.items()
        },
        "decision": {
            "accepted": accepted,
            "recovery_triggered": analysis["recovery_count"] >= 1,
            "at_least_one_recovery_win": len(analysis["recovery_wins"]) >= 1,
            "zero_direct_only_losses": (
                versus_4b["paired_counts"]["baseline_only"] == 0
            ),
            "parse_failures_reduced": (
                costs["four_b_treatment"]["parse_failures"]
                < costs["four_b_direct"]["parse_failures"]
            ),
            "all_recoveries_match_regex": (
                costs["four_b_treatment"]["regex_matches"]
                == costs["four_b_treatment"]["recovery_calls"]
            ),
            "no_api_errors": not costs["four_b_treatment"]["api_errors"],
            "token_ratio_below_1_2": token_ratio < 1.2,
            "next_experiment": _next_experiment(
                accepted,
                analysis["recovery_count"],
            ),
        },
    }
    recovery_case_text = (
        ", ".join(f"`{case_id}`" for case_id in analysis["recovery_cases"])
        if analysis["recovery_cases"]
        else "none"
    )
    treatment_win_text = (
        ", ".join(
            f"`{case_id}`" for case_id in versus_4b["candidate_only_cases"]
        )
        if versus_4b["candidate_only_cases"]
        else "none"
    )
    direct_win_text = (
        ", ".join(
            f"`{case_id}`" for case_id in versus_4b["baseline_only_cases"]
        )
        if versus_4b["baseline_only_cases"]
        else "none"
    )
    trigger_interpretation = (
        "Recovery remains unobserved despite doubling the fresh slice, so "
        "dev16 cannot establish mechanism benefit."
        if not analysis["recovery_count"]
        else (
            "Observed recovery calls satisfy every pre-registered promotion rule."
            if accepted
            else "Observed recovery calls fail at least one pre-registered rule."
        )
    )
    markdown = f"""# GSM8K Dev16 Constrained Recovery Confirmation

## Result

- 4B direct: {versus_4b['baseline_accuracy']:.4f}
  ({costs['four_b_direct']['correct']}/{costs['four_b_direct']['cases']});
- 4B constrained recovery: {versus_4b['candidate_accuracy']:.4f}
  ({costs['four_b_treatment']['correct']}/{costs['four_b_treatment']['cases']});
- 9B direct: {versus_9b['baseline_accuracy']:.4f}
  ({costs['nine_b_direct']['correct']}/{costs['nine_b_direct']['cases']}).

Treatment versus 4B direct is {versus_4b['delta']:+.4f}, with paired bootstrap
95% CI [{versus_4b['paired_bootstrap_95_ci'][0]:+.4f},
{versus_4b['paired_bootstrap_95_ci'][1]:+.4f}] and exact McNemar
`p={versus_4b['mcnemar_exact_p']:.4g}`.

Treatment versus 9B direct is {versus_9b['delta']:+.4f}, with paired bootstrap
95% CI [{versus_9b['paired_bootstrap_95_ci'][0]:+.4f},
{versus_9b['paired_bootstrap_95_ci'][1]:+.4f}] and exact McNemar
`p={versus_9b['mcnemar_exact_p']:.4g}`. There are
{versus_9b['paired_counts']['candidate_only']} treatment-only wins and
{versus_9b['paired_counts']['baseline_only']} 9B-only wins.

Recovery triggers {analysis['recovery_count']} times, produces
{len(analysis['recovery_wins'])} correct recoveries, and leaves
{len(analysis['unresolved_recoveries'])} unparseable. All
{costs['four_b_treatment']['regex_matches']} recovery outputs full-match the
committed regex. Treatment/direct token ratio is {token_ratio:.3f}x.

{trigger_interpretation}

## Case-Level Evidence

- Recovery cases: {recovery_case_text}
- Treatment-only wins versus 4B direct: {treatment_win_text}
- Direct-only wins versus treatment: {direct_win_text}
- 4B direct parse failures:
  {json.dumps(versus_4b['baseline_parse_failures'])}
- 9B direct parse failures:
  {json.dumps(versus_9b['baseline_parse_failures'])}

## Contract Audit

All arms contain the committed 96 case IDs. Direct stage input hashes match
the committed prompts. Treatment invokes recovery exactly for direct parse
failures, preserves every parseable direct prediction, carries the committed
`structured_outputs.regex`, full-matches each recovery output, and selects the
recorded result deterministically. Raw outputs remain local and ignored.

## Cost

- 4B direct: {costs['four_b_direct']['total_tokens']} tokens,
  {costs['four_b_direct']['wall_seconds']:.1f}s summed request latency,
  {costs['four_b_direct']['api_errors']} API errors;
- 4B treatment: {costs['four_b_treatment']['total_tokens']} tokens,
  {costs['four_b_treatment']['wall_seconds']:.1f}s summed request latency,
  {costs['four_b_treatment']['api_errors']} API errors;
- 9B direct: {costs['nine_b_direct']['total_tokens']} tokens,
  {costs['nine_b_direct']['wall_seconds']:.1f}s summed request latency,
  {costs['nine_b_direct']['api_errors']} API errors.

## Decision

{('Dev16 satisfies every pre-registered directional promotion rule.'
   if accepted
   else 'Dev16 fails at least one pre-registered directional promotion rule.')}

{report['decision']['next_experiment']}

## Reproduction Identity

- Pre-registration/code revision: `{report['code_revision']}`
- 4B direct raw SHA256: `{report['artifacts']['four_b_direct_raw_sha256']}`
- 4B treatment raw SHA256: `{report['artifacts']['four_b_treatment_raw_sha256']}`
- 9B direct raw SHA256: `{report['artifacts']['nine_b_direct_raw_sha256']}`
"""
    Path(
        "docs/results/gsm8k_dev16_constrained_recovery_v1.public.json"
    ).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    Path("docs/results/gsm8k_dev16_constrained_recovery_v1.md").write_text(
        markdown,
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "experiment_id": report["experiment_id"],
                "accepted": accepted,
                "treatment_vs_4b_delta": versus_4b["delta"],
                "treatment_vs_9b_delta": versus_9b["delta"],
                "recoveries": analysis["recovery_count"],
                "recovery_wins": len(analysis["recovery_wins"]),
                "regex_matches": costs["four_b_treatment"]["regex_matches"],
                "token_ratio": token_ratio,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
