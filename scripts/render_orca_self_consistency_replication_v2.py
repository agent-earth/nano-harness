#!/usr/bin/env python3

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from nano_harness.baseline import sha256_file
from nano_harness.orca_self_consistency_replication import load_config
from scripts.render_orca_self_consistency_v1 import (
    comparison_gates,
    comparison_with_strata,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/campaign/orca_math_self_consistency_replication_v2.json"
)
PREREGISTER = (
    ROOT
    / "docs/experiments/"
    "orca_math_self_consistency_replication_v2.preregister.json"
)
PRIOR_PUBLIC = (
    ROOT / "docs/results/orca_math_self_consistency_v1.public.json"
)
PUBLIC = (
    ROOT
    / "docs/results/"
    "orca_math_self_consistency_replication_v2.public.json"
)
MARKDOWN = (
    ROOT / "docs/results/orca_math_self_consistency_replication_v2.md"
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def four_b_preservation_gates(comparison: dict) -> dict[str, bool]:
    return {
        "point_delta_nonnegative": comparison["delta"] >= 0,
        "bootstrap_ci_lower_nonnegative": (
            comparison["paired_bootstrap_95_ci"][0] >= 0
        ),
        "no_significant_regression": not (
            comparison["delta"] < 0
            and comparison["mcnemar_exact_p"] < 0.05
        ),
        "every_stratum_non_regression": all(
            row["delta"] >= 0
            for row in comparison["by_stratum"].values()
        ),
    }


def build_report() -> dict:
    config = load_config(CONFIG)
    raw = config.raw
    preregister = json.loads(PREREGISTER.read_text(encoding="utf-8"))
    prior = json.loads(PRIOR_PUBLIC.read_text(encoding="utf-8"))
    output_root = config.resolve(raw["output_dir"])
    paths = {
        name: output_root / f"{name}.jsonl"
        for name in ("four_direct", "candidate", "nine_direct")
    }
    receipts_path = output_root / "receipts.json"
    if (
        preregister.get("schema_version")
        != (
            "nano_harness_orca_self_consistency_"
            "replication_preregister_v2"
        )
        or preregister.get("identity", {}).get("config_sha256")
        != sha256_file(CONFIG)
        or prior.get("schema_version")
        != "nano_harness_orca_self_consistency_public_v1"
        or sha256_file(PRIOR_PUBLIC)
        != raw["prior_self_consistency_result_sha256"]
        or not all(path.is_file() for path in paths.values())
        or not receipts_path.is_file()
    ):
        raise ValueError("self-consistency replication identity differs")
    arms = {name: read_jsonl(path) for name, path in paths.items()}
    expected_ids = set(preregister["selection"]["case_ids"])
    if any(
        {row["case_id"] for row in rows} != expected_ids
        for rows in arms.values()
    ):
        raise ValueError("self-consistency replication case sets differ")
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
    receipts = json.loads(receipts_path.read_text(encoding="utf-8"))
    diagnostics = {
        "cases": len(receipts),
        "overrides": sum(bool(row["override"]) for row in receipts),
        "fallbacks": sum(bool(row["fallback"]) for row in receipts),
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
    }

    prior_root = ROOT / "results/harness/orca-math-self-consistency-v1"
    pooled = {}
    for baseline in ("four_direct", "nine_direct"):
        pooled[baseline] = comparison_with_strata(
            read_jsonl(prior_root / "candidate.jsonl")
            + arms["candidate"],
            read_jsonl(prior_root / f"{baseline}.jsonl")
            + arms[baseline],
            bootstrap_samples=stats["bootstrap_samples"],
            bootstrap_seed=f"{stats['bootstrap_seed']}:pooled:{baseline}",
        )
    admitted = all(four_gates.values()) and all(nine_gates.values())
    return {
        "schema_version": (
            "nano_harness_orca_self_consistency_replication_public_v2"
        ),
        "experiment_id": raw["experiment_id"],
        "identity": {
            "result_revision": (
                "e5de3e1cff5836a8104a2c68e392cd5c6b8a5b81"
            ),
            "config_sha256": sha256_file(CONFIG),
            "preregister_sha256": sha256_file(PREREGISTER),
            "prior_public_sha256": sha256_file(PRIOR_PUBLIC),
            "raw_sha256": {
                name: sha256_file(path) for name, path in paths.items()
            },
            "receipts_sha256": sha256_file(receipts_path),
            "case_ids_sha256": preregister["selection"][
                "case_ids_sha256"
            ],
        },
        "comparisons": {
            "versus_four_b": versus_four,
            "versus_nine_b": versus_nine,
        },
        "diagnostics": diagnostics,
        "pooled_descriptive_only": {
            "cases": 256,
            "not_a_preregistered_gate": True,
            "versus_four_b": pooled["four_direct"],
            "versus_nine_b": pooled["nine_direct"],
        },
        "decision": {
            "four_b_preservation_gates": four_gates,
            "nine_b_superiority_gates": nine_gates,
            "replication_admitted": admitted,
            "complete_benchmark_allowed": admitted,
            "rerun_or_tuning_allowed": False,
            "next_action": (
                "Pre-register one matched complete benchmark treatment."
                if admitted
                else (
                    "Publish the failed long-stratum gate; freeze this exact "
                    "policy and test a new harness only on fresh data."
                )
            ),
        },
        "boundary": {
            "expected_used_during_generation": False,
            "benchmark_rows_used": False,
            "training_allowed": False,
            "pooled_result_overrides_replication_gate": False,
        },
    }


def render_markdown(report: dict) -> str:
    four = report["comparisons"]["versus_four_b"]
    nine = report["comparisons"]["versus_nine_b"]
    pooled = report["pooled_descriptive_only"]["versus_nine_b"]
    return f"""# Orca Math Self-Consistency Replication v2

## Verdict

**REJECT.**

| Comparison | Candidate | Baseline | Delta | 95% CI | McNemar p | Wins / Losses |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| vs 4B direct | {four['candidate_correct']}/160 | {four['baseline_correct']}/160 | {four['delta']:+.4f} | [{four['paired_bootstrap_95_ci'][0]:+.4f}, {four['paired_bootstrap_95_ci'][1]:+.4f}] | {four['mcnemar_exact_p']:.6g} | {four['paired_counts']['candidate_only']} / {four['paired_counts']['baseline_only']} |
| vs 9B direct | {nine['candidate_correct']}/160 | {nine['baseline_correct']}/160 | {nine['delta']:+.4f} | [{nine['paired_bootstrap_95_ci'][0]:+.4f}, {nine['paired_bootstrap_95_ci'][1]:+.4f}] | {nine['mcnemar_exact_p']:.6g} | {nine['paired_counts']['candidate_only']} / {nine['paired_counts']['baseline_only']} |

## Gate Failure

The exact frozen policy preserves 4B: 3 wins, 0 losses, non-negative bootstrap
lower bound, and no 4B stratum regression. It also beats 9B overall
significantly. However, the long stratum scores 6/40 versus 9B 8/40, so the
pre-registered per-stratum non-regression gate fails. Complete benchmark access
remains closed.

## Descriptive Pool

Across v1 plus replication, candidate scores
{pooled['candidate_correct']}/256 versus 9B
{pooled['baseline_correct']}/256, delta
{pooled['delta']:+.4f}, 95% CI
[{pooled['paired_bootstrap_95_ci'][0]:+.4f},
{pooled['paired_bootstrap_95_ci'][1]:+.4f}], McNemar
`p={pooled['mcnemar_exact_p']:.6g}`. This pooled view was not a
pre-registered replication gate and cannot override the formal rejection.

No rerun or tuning is allowed on either observed surface.
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
