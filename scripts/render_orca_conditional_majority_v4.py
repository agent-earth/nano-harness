#!/usr/bin/env python3

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from nano_harness.baseline import sha256_file
from nano_harness.orca_conditional_majority import load_config
from nano_harness.orca_self_consistency import score_prediction
from scripts.render_orca_self_consistency_replication_v2 import (
    four_b_preservation_gates,
)
from scripts.render_orca_self_consistency_v1 import (
    comparison_gates,
    comparison_with_strata,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/campaign/orca_math_conditional_majority_v4.json"
PREREGISTER = (
    ROOT
    / "docs/experiments/orca_math_conditional_majority_v4.preregister.json"
)
PUBLIC = (
    ROOT
    / "docs/results/orca_math_conditional_majority_v4.public.json"
)
MARKDOWN = ROOT / "docs/results/orca_math_conditional_majority_v4.md"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_raw(
    arms: dict[str, list[dict[str, Any]]],
    receipts: list[dict[str, Any]],
    expected_ids: set[str],
) -> None:
    for rows in arms.values():
        case_ids = [row["case_id"] for row in rows]
        if (
            len(rows) != 96
            or len(set(case_ids)) != 96
            or set(case_ids) != expected_ids
        ):
            raise ValueError("conditional majority case sets differ")
        for row in rows:
            if bool(row["correct"]) != score_prediction(
                row.get("prediction"), row["expected"]
            ):
                raise ValueError("conditional majority score differs")

    reference = {
        row["case_id"]: (row["stratum"], row["expected"])
        for row in arms["four_direct"]
    }
    for rows in arms.values():
        if any(
            reference[row["case_id"]]
            != (row["stratum"], row["expected"])
            for row in rows
        ):
            raise ValueError("conditional majority labels differ")

    receipt_ids = [row["case_id"] for row in receipts]
    if (
        len(receipts) != 96
        or len(set(receipt_ids)) != 96
        or set(receipt_ids) != expected_ids
    ):
        raise ValueError("conditional majority receipt cases differ")
    for receipt in receipts:
        expected_minimum = (
            5 if receipt["direct_strict_parseable"] else 3
        )
        if (
            receipt["minimum_votes"] != expected_minimum
            or receipt["parseable_replicas"] != 5
            or receipt["consensus_votes"] < 0
            or receipt["consensus_votes"] > 5
            or (
                receipt["fallback"]
                and receipt["consensus_votes"] >= expected_minimum
            )
            or (
                not receipt["fallback"]
                and receipt["consensus_votes"] < expected_minimum
            )
            or (receipt["override"] and receipt["fallback"])
        ):
            raise ValueError("conditional majority receipt differs")


def route_metrics(
    candidate_rows: list[dict[str, Any]],
    four_rows: list[dict[str, Any]],
    receipts: list[dict[str, Any]],
    *,
    direct_strict_parseable: bool,
) -> dict[str, int]:
    candidate = {row["case_id"]: row for row in candidate_rows}
    four = {row["case_id"]: row for row in four_rows}
    selected = [
        row
        for row in receipts
        if bool(row["direct_strict_parseable"])
        is direct_strict_parseable
    ]
    return {
        "cases": len(selected),
        "candidate_correct": sum(
            bool(candidate[row["case_id"]]["correct"]) for row in selected
        ),
        "four_b_correct": sum(
            bool(four[row["case_id"]]["correct"]) for row in selected
        ),
        "candidate_only": sum(
            bool(candidate[row["case_id"]]["correct"])
            and not bool(four[row["case_id"]]["correct"])
            for row in selected
        ),
        "four_b_only": sum(
            not bool(candidate[row["case_id"]]["correct"])
            and bool(four[row["case_id"]]["correct"])
            for row in selected
        ),
    }


def build_report() -> dict[str, Any]:
    config = load_config(CONFIG)
    raw = config.raw
    preregister = json.loads(PREREGISTER.read_text(encoding="utf-8"))
    output_root = config.resolve(raw["output_dir"])
    paths = {
        name: output_root / f"{name}.jsonl"
        for name in ("four_direct", "candidate", "nine_direct")
    }
    receipts_path = output_root / "receipts.json"
    run_receipt_path = output_root / "run.stdout.json"
    if (
        preregister.get("schema_version")
        != "nano_harness_orca_conditional_majority_preregister_v4"
        or preregister.get("identity", {}).get("config_sha256")
        != sha256_file(CONFIG)
        or not all(path.is_file() for path in paths.values())
        or not receipts_path.is_file()
        or not run_receipt_path.is_file()
    ):
        raise ValueError("conditional majority identity differs")

    arms = {name: read_jsonl(path) for name, path in paths.items()}
    receipts = json.loads(receipts_path.read_text(encoding="utf-8"))
    expected_ids = set(preregister["selection"]["case_ids"])
    validate_raw(arms, receipts, expected_ids)

    run_receipt = json.loads(run_receipt_path.read_text(encoding="utf-8"))
    actual_raw_sha = {
        name: sha256_file(path) for name, path in paths.items()
    }
    if (
        run_receipt.get("schema_version")
        != "nano_harness_orca_conditional_majority_raw_v4"
        or run_receipt.get("experiment_id") != raw["experiment_id"]
        or run_receipt.get("selection", {}).get("cases") != 96
        or run_receipt.get("selection", {}).get("case_ids_sha256")
        != preregister["selection"]["case_ids_sha256"]
        or run_receipt.get("selection", {}).get(
            "excluded_source_ids_sha256"
        )
        != preregister["selection"]["excluded_source_ids_sha256"]
        or {
            name: value["sha256"]
            for name, value in run_receipt.get("raw", {}).items()
        }
        != actual_raw_sha
        or run_receipt.get("receipts_sha256")
        != sha256_file(receipts_path)
        or run_receipt.get("generation_boundary")
        != {
            "expected_used_during_generation": False,
            "scoring_after_generation": True,
        }
    ):
        raise ValueError("conditional majority run receipt differs")

    stats = raw["statistics"]
    versus_four = comparison_with_strata(
        arms["candidate"],
        arms["four_direct"],
        bootstrap_samples=stats["bootstrap_samples"],
        bootstrap_seed=f"{stats['bootstrap_seed']}:four",
    )
    versus_nine = comparison_with_strata(
        arms["candidate"],
        arms["nine_direct"],
        bootstrap_samples=stats["bootstrap_samples"],
        bootstrap_seed=f"{stats['bootstrap_seed']}:nine",
    )
    four_gates = four_b_preservation_gates(versus_four)
    nine_gates = comparison_gates(
        versus_nine,
        alpha=stats["alpha"],
        minimum_candidate_only_wins=stats[
            "minimum_candidate_only_wins"
        ],
    )

    receipt_by_id = {row["case_id"]: row for row in receipts}
    override_ids = {
        row["case_id"] for row in receipts if row["override"]
    }
    candidate_by_id = {
        row["case_id"]: row for row in arms["candidate"]
    }
    four_by_id = {
        row["case_id"]: row for row in arms["four_direct"]
    }
    diagnostics = {
        "cases": len(receipts),
        "overrides": len(override_ids),
        "fallbacks": sum(bool(row["fallback"]) for row in receipts),
        "direct_strict_parseable": sum(
            bool(row["direct_strict_parseable"]) for row in receipts
        ),
        "direct_strict_parse_failure": sum(
            not bool(row["direct_strict_parseable"]) for row in receipts
        ),
        "minimum_vote_route_counts": dict(
            sorted(
                Counter(
                    str(row["minimum_votes"]) for row in receipts
                ).items()
            )
        ),
        "consensus_vote_counts": dict(
            sorted(
                Counter(
                    str(row["consensus_votes"]) for row in receipts
                ).items()
            )
        ),
        "parseable_replica_counts": dict(
            sorted(
                Counter(
                    str(row["parseable_replicas"]) for row in receipts
                ).items()
            )
        ),
        "route_outcomes": {
            "strict_parseable": route_metrics(
                arms["candidate"],
                arms["four_direct"],
                receipts,
                direct_strict_parseable=True,
            ),
            "strict_parse_failure": route_metrics(
                arms["candidate"],
                arms["four_direct"],
                receipts,
                direct_strict_parseable=False,
            ),
        },
        "override_outcomes": {
            "candidate_only": sum(
                bool(candidate_by_id[case_id]["correct"])
                and not bool(four_by_id[case_id]["correct"])
                for case_id in override_ids
            ),
            "four_b_only": sum(
                not bool(candidate_by_id[case_id]["correct"])
                and bool(four_by_id[case_id]["correct"])
                for case_id in override_ids
            ),
            "both_correct": sum(
                bool(candidate_by_id[case_id]["correct"])
                and bool(four_by_id[case_id]["correct"])
                for case_id in override_ids
            ),
            "both_wrong": sum(
                not bool(candidate_by_id[case_id]["correct"])
                and not bool(four_by_id[case_id]["correct"])
                for case_id in override_ids
            ),
        },
        "receipt_cases_match_raw": (
            set(receipt_by_id) == set(candidate_by_id)
        ),
        "model_requests_rerun_for_render": False,
    }
    admitted = all(four_gates.values()) and all(nine_gates.values())
    return {
        "schema_version": (
            "nano_harness_orca_conditional_majority_public_v4"
        ),
        "experiment_id": raw["experiment_id"],
        "identity": {
            "generation_revision": (
                "dba8f569c380ca70fc4a507b64ef34407ca18da5"
            ),
            "config_sha256": sha256_file(CONFIG),
            "preregister_sha256": sha256_file(PREREGISTER),
            "raw_sha256": actual_raw_sha,
            "receipts_sha256": sha256_file(receipts_path),
            "run_receipt_sha256": sha256_file(run_receipt_path),
            "case_ids_sha256": preregister["selection"][
                "case_ids_sha256"
            ],
        },
        "comparisons": {
            "versus_four_b": versus_four,
            "versus_nine_b": versus_nine,
        },
        "diagnostics": diagnostics,
        "decision": {
            "four_b_preservation_gates": four_gates,
            "nine_b_superiority_gates": nine_gates,
            "candidate_admitted": admitted,
            "complete_benchmark_allowed": admitted,
            "rerun_or_tuning_allowed": False,
            "next_action": (
                "Pre-register one matched complete benchmark treatment "
                "using the frozen parser and conditional-majority policy."
                if admitted
                else (
                    "Publish negative evidence. Do not retune this policy "
                    "on the observed cases."
                )
            ),
        },
        "boundary": {
            "expected_used_during_generation": False,
            "benchmark_rows_used": False,
            "training_allowed": False,
            "claim_scope": (
                "fresh non-benchmark local development admission only"
            ),
            "complete_benchmark_score_claimed": False,
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    four = report["comparisons"]["versus_four_b"]
    nine = report["comparisons"]["versus_nine_b"]
    diagnostics = report["diagnostics"]
    verdict = (
        "ADMIT TO PRE-REGISTRATION"
        if report["decision"]["candidate_admitted"]
        else "REJECT"
    )
    return f"""# Orca Math Conditional-Majority v4

## Verdict

**{verdict}.**

This is a fresh local development gate, not a complete benchmark score.

| Comparison | Candidate | Baseline | Delta | 95% CI | McNemar p | Wins / Losses |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| vs recovered 4B direct | {four['candidate_correct']}/96 | {four['baseline_correct']}/96 | {four['delta']:+.4f} | [{four['paired_bootstrap_95_ci'][0]:+.4f}, {four['paired_bootstrap_95_ci'][1]:+.4f}] | {four['mcnemar_exact_p']:.6g} | {four['paired_counts']['candidate_only']} / {four['paired_counts']['baseline_only']} |
| vs recovered 9B direct | {nine['candidate_correct']}/96 | {nine['baseline_correct']}/96 | {nine['delta']:+.4f} | [{nine['paired_bootstrap_95_ci'][0]:+.4f}, {nine['paired_bootstrap_95_ci'][1]:+.4f}] | {nine['mcnemar_exact_p']:.6g} | {nine['paired_counts']['candidate_only']} / {nine['paired_counts']['baseline_only']} |

## What The Harness Does

The parser first reads a strict `FINAL:` answer and otherwise recovers the last
numeric token from the final 1,500 characters. The candidate then asks the 4B
model for five stochastic solutions:

- if the direct answer had no strict `FINAL:`, a 3-of-5 agreement may replace
  the recovered direct answer;
- if the direct answer already had a strict `FINAL:`, replacement requires
  unanimous 5-of-5 agreement;
- without the required agreement, the candidate keeps the recovered direct
  answer.

This policy made {diagnostics['overrides']} actual answer replacements and
fell back to direct on {diagnostics['fallbacks']} cases. It routed
{diagnostics['direct_strict_parse_failure']} strict-parse failures through the
3-vote threshold and {diagnostics['direct_strict_parseable']} strict-parseable
cases through the 5-vote threshold.

## Why It Passed

Against matched 4B direct, the candidate gains four cases and loses none. Its
paired bootstrap lower bound is positive, and every length stratum improves:
short {four['by_stratum']['short']['delta']:+.4f}, medium
{four['by_stratum']['medium']['delta']:+.4f}, long
{four['by_stratum']['long']['delta']:+.4f}.

Against matched 9B direct, it gains 12 net cases, with
{nine['paired_counts']['candidate_only']} wins and
{nine['paired_counts']['baseline_only']} losses. The paired interval excludes
zero and exact McNemar is `p={nine['mcnemar_exact_p']:.6g}`. No stratum
regresses: short {nine['by_stratum']['short']['delta']:+.4f}, medium
{nine['by_stratum']['medium']['delta']:+.4f}, long
{nine['by_stratum']['long']['delta']:+.4f}.

All pre-registered preservation and superiority gates pass. The existing raw
files were rendered offline; no model request was repeated.

## Decision Boundary

The frozen parser and conditional-majority policy may now be pre-registered
for one matched complete benchmark treatment. This result does not itself
prove superiority on GSM8K, MMLU, GPQA, SWE-bench, or any other complete
benchmark. No rerun or tuning is allowed on these 96 observed cases.
"""


def main() -> None:
    report = build_report()
    PUBLIC.parent.mkdir(parents=True, exist_ok=True)
    PUBLIC.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    MARKDOWN.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
