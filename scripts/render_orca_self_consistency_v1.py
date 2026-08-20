#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

from nano_harness.baseline import sha256_file
from nano_harness.orca_self_consistency import load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/campaign/orca_math_self_consistency_v1.json"
PREREGISTER = (
    ROOT / "docs/experiments/orca_math_self_consistency_v1.preregister.json"
)
PUBLIC = (
    ROOT / "docs/results/orca_math_self_consistency_v1.public.json"
)
MARKDOWN = ROOT / "docs/results/orca_math_self_consistency_v1.md"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mcnemar_exact(candidate_only: int, baseline_only: int) -> float:
    discordant = candidate_only + baseline_only
    if discordant == 0:
        return 1.0
    tail = min(candidate_only, baseline_only)
    probability = sum(
        math.comb(discordant, index)
        for index in range(tail + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * probability)


def paired_metrics(
    candidate_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: str,
) -> dict[str, Any]:
    candidate = {row["case_id"]: row for row in candidate_rows}
    baseline = {row["case_id"]: row for row in baseline_rows}
    if set(candidate) != set(baseline) or not candidate:
        raise ValueError("self-consistency paired case sets differ")
    case_ids = sorted(candidate)
    deltas = [
        int(bool(candidate[case_id]["correct"]))
        - int(bool(baseline[case_id]["correct"]))
        for case_id in case_ids
    ]
    randomizer = random.Random(bootstrap_seed)
    estimates = sorted(
        sum(
            deltas[randomizer.randrange(len(deltas))]
            for _ in deltas
        )
        / len(deltas)
        for _ in range(bootstrap_samples)
    )
    candidate_only = sum(delta == 1 for delta in deltas)
    baseline_only = sum(delta == -1 for delta in deltas)
    return {
        "cases": len(case_ids),
        "candidate_correct": sum(
            bool(candidate[case_id]["correct"]) for case_id in case_ids
        ),
        "baseline_correct": sum(
            bool(baseline[case_id]["correct"]) for case_id in case_ids
        ),
        "candidate_accuracy": sum(
            bool(candidate[case_id]["correct"]) for case_id in case_ids
        )
        / len(case_ids),
        "baseline_accuracy": sum(
            bool(baseline[case_id]["correct"]) for case_id in case_ids
        )
        / len(case_ids),
        "delta": sum(deltas) / len(deltas),
        "paired_bootstrap_95_ci": [
            estimates[int(bootstrap_samples * 0.025)],
            estimates[
                min(
                    bootstrap_samples - 1,
                    int(bootstrap_samples * 0.975),
                )
            ],
        ],
        "mcnemar_exact_p": mcnemar_exact(
            candidate_only,
            baseline_only,
        ),
        "paired_counts": {
            "candidate_only": candidate_only,
            "baseline_only": baseline_only,
            "both_correct": sum(
                bool(candidate[case_id]["correct"])
                and bool(baseline[case_id]["correct"])
                for case_id in case_ids
            ),
            "both_wrong": sum(
                not bool(candidate[case_id]["correct"])
                and not bool(baseline[case_id]["correct"])
                for case_id in case_ids
            ),
        },
        "candidate_parse_failures": sum(
            candidate[case_id].get("prediction") is None
            for case_id in case_ids
        ),
        "baseline_parse_failures": sum(
            baseline[case_id].get("prediction") is None
            for case_id in case_ids
        ),
    }


def comparison_with_strata(
    candidate: list[dict[str, Any]],
    baseline: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: str,
) -> dict[str, Any]:
    result = paired_metrics(
        candidate,
        baseline,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed + ":overall",
    )
    result["by_stratum"] = {}
    for stratum in ("short", "medium", "long"):
        candidate_rows = [
            row for row in candidate if row["stratum"] == stratum
        ]
        baseline_rows = [
            row for row in baseline if row["stratum"] == stratum
        ]
        result["by_stratum"][stratum] = paired_metrics(
            candidate_rows,
            baseline_rows,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=f"{bootstrap_seed}:{stratum}",
        )
    return result


def comparison_gates(
    comparison: dict[str, Any],
    *,
    alpha: float,
    minimum_candidate_only_wins: int,
) -> dict[str, bool]:
    return {
        "point_delta_positive": comparison["delta"] > 0,
        "bootstrap_ci_lower_positive": (
            comparison["paired_bootstrap_95_ci"][0] > 0
        ),
        "mcnemar_below_alpha": comparison["mcnemar_exact_p"] < alpha,
        "minimum_candidate_only_wins": (
            comparison["paired_counts"]["candidate_only"]
            >= minimum_candidate_only_wins
        ),
        "candidate_only_exceeds_baseline_only": (
            comparison["paired_counts"]["candidate_only"]
            > comparison["paired_counts"]["baseline_only"]
        ),
        "every_stratum_non_regression": all(
            row["delta"] >= 0
            for row in comparison["by_stratum"].values()
        ),
    }


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


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
    if (
        preregister.get("schema_version")
        != "nano_harness_orca_self_consistency_preregister_v1"
        or preregister.get("identity", {}).get("config_sha256")
        != sha256_file(CONFIG)
        or not all(path.is_file() for path in paths.values())
        or not receipts_path.is_file()
    ):
        raise ValueError("self-consistency result identity differs")
    arms = {name: read_jsonl(path) for name, path in paths.items()}
    expected_ids = set(preregister["selection"]["case_ids"])
    if any(
        {row["case_id"] for row in rows} != expected_ids
        for rows in arms.values()
    ):
        raise ValueError("self-consistency raw case identities differ")
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
    four_gates = comparison_gates(
        versus_four,
        alpha=stats["alpha"],
        minimum_candidate_only_wins=stats[
            "minimum_candidate_only_wins"
        ],
    )
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
    admitted = all(four_gates.values()) and all(nine_gates.values())
    return {
        "schema_version": "nano_harness_orca_self_consistency_public_v1",
        "experiment_id": raw["experiment_id"],
        "identity": {
            "result_revision": (
                "b578aabc1b286f5797d59f2635a6c4dc419b93ce"
            ),
            "config_sha256": sha256_file(CONFIG),
            "preregister_sha256": sha256_file(PREREGISTER),
            "raw_sha256": {
                name: sha256_file(path) for name, path in paths.items()
            },
            "receipts_sha256": sha256_file(receipts_path),
            "case_ids_sha256": preregister["selection"][
                "case_ids_sha256"
            ],
            "raw_bundle_sha256": canonical_sha256(
                {
                    name: sha256_file(path)
                    for name, path in paths.items()
                }
            ),
        },
        "comparisons": {
            "versus_four_b": versus_four,
            "versus_nine_b": versus_nine,
        },
        "diagnostics": diagnostics,
        "decision": {
            "versus_four_b_gates": four_gates,
            "versus_nine_b_gates": nine_gates,
            "candidate_admitted": admitted,
            "complete_benchmark_allowed": admitted,
            "rerun_or_tuning_allowed": False,
            "next_action": (
                "Pre-register one matched complete benchmark treatment."
                if admitted
                else (
                    "Publish negative evidence and freeze this harness; "
                    "do not rerun or tune on these cases."
                )
            ),
        },
        "boundary": {
            "expected_used_during_generation": False,
            "benchmark_rows_used": False,
            "training_allowed": False,
            "claim_scope": "fresh non-benchmark local development only",
        },
    }


def render_markdown(report: dict) -> str:
    four = report["comparisons"]["versus_four_b"]
    nine = report["comparisons"]["versus_nine_b"]
    verdict = "ADMIT" if report["decision"]["candidate_admitted"] else "REJECT"
    return f"""# Orca Math Self-Consistency v1 Result

## Verdict

**{verdict}.**

| Comparison | Candidate | Baseline | Delta | 95% CI | McNemar p | Wins / Losses |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| vs 4B direct | {four['candidate_correct']}/96 | {four['baseline_correct']}/96 | {four['delta']:+.4f} | [{four['paired_bootstrap_95_ci'][0]:+.4f}, {four['paired_bootstrap_95_ci'][1]:+.4f}] | {four['mcnemar_exact_p']:.6g} | {four['paired_counts']['candidate_only']} / {four['paired_counts']['baseline_only']} |
| vs 9B direct | {nine['candidate_correct']}/96 | {nine['baseline_correct']}/96 | {nine['delta']:+.4f} | [{nine['paired_bootstrap_95_ci'][0]:+.4f}, {nine['paired_bootstrap_95_ci'][1]:+.4f}] | {nine['mcnemar_exact_p']:.6g} | {nine['paired_counts']['candidate_only']} / {nine['paired_counts']['baseline_only']} |

## Harness Behavior

- 5 full-solve replicas per case;
- require 4 agreeing numeric finals before override;
- overrides: {report['diagnostics']['overrides']}/96;
- fallbacks to frozen 4B direct:
  {report['diagnostics']['fallbacks']}/96.

No rerun, prompt, replica-count, threshold, temperature, seed, parser, scorer,
or generation-budget change is allowed after this result.

## Boundary

Fresh non-benchmark local development only. This is not GSM8K, MMLU, GPQA,
9B-complete, 27B, or agent-benchmark evidence.
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
