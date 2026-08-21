#!/usr/bin/env python3

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from nano_harness.baseline import sha256_file
from nano_harness.mbpp_iterative_repair import load_config
from scripts.render_orca_self_consistency_replication_v2 import (
    four_b_preservation_gates,
)
from scripts.render_orca_self_consistency_v1 import paired_metrics


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/campaign/mbpp_sanitized_iterative_repair_train_v2.json"
)
PREREGISTER = (
    ROOT
    / "docs/experiments/"
    "mbpp_sanitized_iterative_repair_train_v2.preregister.json"
)
RAW_RESULT = (
    ROOT
    / "results/harness/mbpp-sanitized-iterative-repair-train-v2/result.json"
)
RAW_CASES = (
    ROOT
    / "results/harness/mbpp-sanitized-iterative-repair-train-v2/cases.jsonl"
)
PUBLIC = (
    ROOT
    / "docs/results/mbpp_sanitized_iterative_repair_train_v2.public.json"
)
MARKDOWN = ROOT / "docs/results/mbpp_sanitized_iterative_repair_train_v2.md"


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def arm_rows(
    rows: list[dict[str, Any]],
    arm: str,
) -> list[dict[str, Any]]:
    return [
        {
            "case_id": row["case_id"],
            "prediction": (
                "pass" if row[arm]["test_result"]["full_pass"] else "fail"
            ),
            "correct": bool(row[arm]["test_result"]["full_pass"]),
        }
        for row in rows
    ]


def build_report() -> dict[str, Any]:
    config = load_config(CONFIG)
    preregister = json.loads(PREREGISTER.read_text(encoding="utf-8"))
    raw = json.loads(RAW_RESULT.read_text(encoding="utf-8"))
    if (
        preregister.get("schema_version")
        != "nano_harness_mbpp_iterative_repair_preregister_v2"
        or preregister.get("identity", {}).get("config_sha256")
        != sha256_file(CONFIG)
        or raw.get("schema_version")
        != "nano_harness_mbpp_iterative_repair_merged_v2"
        or raw.get("identity", {}).get("config_sha256")
        != sha256_file(CONFIG)
        or raw.get("identity", {}).get("raw_sha256")
        != sha256_file(RAW_CASES)
        or len(raw.get("identity", {}).get("shards", []))
        != config["execution"]["num_shards"]
        or any(
            row.get("sha256")
            != sha256_file(
                ROOT
                / config["output_dir"]
                / f"shard-{row.get('shard_id')}.jsonl"
            )
            for row in raw.get("identity", {}).get("shards", [])
        )
        or raw.get("surface")
        != {
            "split": "train",
            "cases": 120,
            "validation_v1_rerun": False,
            "validation_rows_loaded_by_v2": False,
            "test_generation_started": False,
        }
    ):
        raise ValueError("MBPP v2 result identity differs")
    rows = read_jsonl(RAW_CASES)
    case_ids = [row["case_id"] for row in rows]
    if len(rows) != 120 or len(set(case_ids)) != 120:
        raise ValueError("MBPP v2 train rows differ")
    candidate = arm_rows(rows, "candidate")
    four = arm_rows(rows, "four_b_direct")
    nine = arm_rows(rows, "nine_b_direct")
    stats = config["statistics"]
    versus_four = paired_metrics(
        candidate,
        four,
        bootstrap_samples=stats["bootstrap_samples"],
        bootstrap_seed=f"{stats['bootstrap_seed']}:four",
    )
    versus_nine = paired_metrics(
        candidate,
        nine,
        bootstrap_samples=stats["bootstrap_samples"],
        bootstrap_seed=f"{stats['bootstrap_seed']}:nine",
    )
    four_gates = four_b_preservation_gates(
        {**versus_four, "by_stratum": {"all": versus_four}}
    )
    nine_gates = {
        "point_delta_positive": versus_nine["delta"] > 0,
        "bootstrap_ci_lower_positive": (
            versus_nine["paired_bootstrap_95_ci"][0] > 0
        ),
        "mcnemar_p_below_alpha": (
            versus_nine["mcnemar_exact_p"] < stats["alpha"]
        ),
        "candidate_only_exceeds_baseline_only": (
            versus_nine["paired_counts"]["candidate_only"]
            > versus_nine["paired_counts"]["baseline_only"]
        ),
        "minimum_candidate_only_wins": (
            versus_nine["paired_counts"]["candidate_only"]
            >= stats["minimum_candidate_only_wins"]
        ),
    }
    supported = all(four_gates.values()) and all(nine_gates.values())
    return {
        "schema_version": "nano_harness_mbpp_iterative_repair_public_v2",
        "experiment_id": config["experiment_id"],
        "identity": {
            "result_revision": git_revision(),
            "config_sha256": sha256_file(CONFIG),
            "preregister_sha256": sha256_file(PREREGISTER),
            "raw_result_sha256": sha256_file(RAW_RESULT),
            "raw_cases_sha256": sha256_file(RAW_CASES),
            "train_case_ids_sha256": preregister["identity"][
                "train_case_ids_sha256"
            ],
        },
        "comparisons": {
            "versus_four_b": versus_four,
            "versus_nine_b": versus_nine,
        },
        "diagnostics": raw["diagnostics"],
        "decision": {
            "four_b_preservation_gates": four_gates,
            "nine_b_directional_gates": nine_gates,
            "method_supported": supported,
            "fresh_validation_preregistration_allowed": supported,
            "validation_v1_rerun_allowed": False,
            "test_generation_allowed": False,
            "rerun_or_tuning_allowed": False,
            "next_action": (
                "Pre-register one frozen run on the 47 full-validation rows "
                "not present in sanitized validation."
                if supported
                else (
                    "Publish negative evidence. Do not tune or rerun this "
                    "train surface."
                )
            ),
        },
        "boundary": {
            "split": "train",
            "validation_v1_rerun": False,
            "validation_rows_loaded_by_v2": False,
            "test_generation_started": False,
            "reference_solution_used_for_evaluation": False,
            "claim_scope": "method development only, not benchmark score",
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    four = report["comparisons"]["versus_four_b"]
    nine = report["comparisons"]["versus_nine_b"]
    diagnostics = report["diagnostics"]
    verdict = (
        "SUPPORTED"
        if report["decision"]["method_supported"]
        else "REJECT"
    )
    return f"""# MBPP Sanitized Iterative Repair Train v2

## Verdict

**{verdict}.**

This is method-development evidence on the 120-case train split, not an MBPP
validation or test score.

| Comparison | Candidate | Baseline | Delta | 95% CI | McNemar p | Wins / Losses |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| vs 4B direct | {four['candidate_correct']}/120 | {four['baseline_correct']}/120 | {four['delta']:+.4f} | [{four['paired_bootstrap_95_ci'][0]:+.4f}, {four['paired_bootstrap_95_ci'][1]:+.4f}] | {four['mcnemar_exact_p']:.6g} | {four['paired_counts']['candidate_only']} / {four['paired_counts']['baseline_only']} |
| vs 9B direct | {nine['candidate_correct']}/120 | {nine['baseline_correct']}/120 | {nine['delta']:+.4f} | [{nine['paired_bootstrap_95_ci'][0]:+.4f}, {nine['paired_bootstrap_95_ci'][1]:+.4f}] | {nine['mcnemar_exact_p']:.6g} | {nine['paired_counts']['candidate_only']} / {nine['paired_counts']['baseline_only']} |

The harness generated {diagnostics['replicas_generated']} replicas and
{diagnostics['repair_rounds_generated']} repairs, making
{diagnostics['overrides']} strictly test-improving overrides. Reference
solutions remained hidden; only official public assertions and deterministic
sandbox outcomes were used.

No rerun or tuning is allowed on this train surface.
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
