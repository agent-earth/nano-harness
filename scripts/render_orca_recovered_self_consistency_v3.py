#!/usr/bin/env python3

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from nano_harness.baseline import sha256_file
from nano_harness.orca_recovered_self_consistency import load_config
from scripts.render_orca_self_consistency_replication_v2 import (
    four_b_preservation_gates,
)
from scripts.render_orca_self_consistency_v1 import (
    comparison_gates,
    comparison_with_strata,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/campaign/orca_math_recovered_self_consistency_v3.json"
)
PREREGISTER = (
    ROOT
    / "docs/experiments/"
    "orca_math_recovered_self_consistency_v3.preregister.json"
)
PUBLIC = (
    ROOT
    / "docs/results/orca_math_recovered_self_consistency_v3.public.json"
)
MARKDOWN = (
    ROOT / "docs/results/orca_math_recovered_self_consistency_v3.md"
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_report() -> dict:
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
        != "nano_harness_orca_recovered_self_consistency_preregister_v3"
        or preregister.get("identity", {}).get("config_sha256")
        != sha256_file(CONFIG)
        or not all(path.is_file() for path in paths.values())
        or not receipts_path.is_file()
        or not run_receipt_path.is_file()
    ):
        raise ValueError("recovered self-consistency identity differs")
    arms = {name: read_jsonl(path) for name, path in paths.items()}
    expected_ids = set(preregister["selection"]["case_ids"])
    if any(
        len(rows) != 96
        or {row["case_id"] for row in rows} != expected_ids
        for rows in arms.values()
    ):
        raise ValueError("recovered self-consistency case sets differ")
    run_receipt = json.loads(run_receipt_path.read_text(encoding="utf-8"))
    if (
        run_receipt.get("selection", {}).get("cases") != 96
        or run_receipt.get("selection", {}).get("case_ids_sha256")
        != preregister["selection"]["case_ids_sha256"]
        or run_receipt.get("generation_boundary", {}).get(
            "expected_used_during_generation"
        )
        is not False
        or run_receipt.get("generation_boundary", {}).get(
            "scoring_after_generation"
        )
        is not True
    ):
        raise ValueError("recovered self-consistency run receipt differs")
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
        "metadata_finalize_required": True,
        "model_requests_rerun_for_finalize": False,
    }
    admitted = all(four_gates.values()) and all(nine_gates.values())
    return {
        "schema_version": (
            "nano_harness_orca_recovered_self_consistency_public_v3"
        ),
        "experiment_id": raw["experiment_id"],
        "identity": {
            "generation_revision": (
                "10b1449782e6bb849a08a48a41107a9a451ab3cd"
            ),
            "finalize_revision": (
                "d181198327a48ef520fe1a60fb03da759e1b37fc"
            ),
            "config_sha256": sha256_file(CONFIG),
            "preregister_sha256": sha256_file(PREREGISTER),
            "raw_sha256": {
                name: sha256_file(path) for name, path in paths.items()
            },
            "receipts_sha256": sha256_file(receipts_path),
            "run_receipt_sha256": sha256_file(run_receipt_path),
            "case_ids_sha256": preregister["selection"]["case_ids_sha256"],
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
                "Pre-register one matched complete benchmark treatment."
                if admitted
                else (
                    "Publish negative evidence. Do not retune this parser "
                    "or consensus on observed cases."
                )
            ),
        },
        "boundary": {
            "expected_used_during_generation": False,
            "benchmark_rows_used": False,
            "training_allowed": False,
            "historical_parser_audit_not_used_as_score": True,
        },
    }


def render_markdown(report: dict) -> str:
    four = report["comparisons"]["versus_four_b"]
    nine = report["comparisons"]["versus_nine_b"]
    return f"""# Orca Math Recovered Self-Consistency v3

## Verdict

**REJECT.**

| Comparison | Candidate | Baseline | Delta | 95% CI | McNemar p | Wins / Losses |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| vs recovered 4B direct | {four['candidate_correct']}/96 | {four['baseline_correct']}/96 | {four['delta']:+.4f} | [{four['paired_bootstrap_95_ci'][0]:+.4f}, {four['paired_bootstrap_95_ci'][1]:+.4f}] | {four['mcnemar_exact_p']:.6g} | {four['paired_counts']['candidate_only']} / {four['paired_counts']['baseline_only']} |
| vs recovered 9B direct | {nine['candidate_correct']}/96 | {nine['baseline_correct']}/96 | {nine['delta']:+.4f} | [{nine['paired_bootstrap_95_ci'][0]:+.4f}, {nine['paired_bootstrap_95_ci'][1]:+.4f}] | {nine['mcnemar_exact_p']:.6g} | {nine['paired_counts']['candidate_only']} / {nine['paired_counts']['baseline_only']} |

## What Changed

All arms use the same target-blind parser: strict `FINAL:` first, otherwise the
last numeric token from the final 1,500 characters. This removed parse failures
without using references. The candidate stayed level with 4B direct, but its
paired bootstrap interval versus 4B still crosses below zero.

Against 9B, candidate leads by 9 cases and has a positive bootstrap interval,
but exact McNemar is `p={nine['mcnemar_exact_p']:.6g}` and the short stratum
regresses by one case. The strict gate therefore fails.

The 672 model requests completed before a metadata-only return-field error.
All three raw arms and receipts contain the exact 96 pre-registered IDs.
Finalization read those existing files and made no additional model request.

No rerun or tuning is allowed on this surface.
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
