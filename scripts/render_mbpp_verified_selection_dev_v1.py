#!/usr/bin/env python3

from __future__ import annotations

import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from nano_harness.baseline import sha256_file
from nano_harness.mbpp_verified_selection import load_config
from scripts.render_orca_self_consistency_replication_v2 import (
    four_b_preservation_gates,
)
from scripts.render_orca_self_consistency_v1 import paired_metrics


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/campaign/mbpp_sanitized_verified_selection_dev_v1.json"
)
PREREGISTER = (
    ROOT
    / "docs/experiments/"
    "mbpp_sanitized_verified_selection_dev_v1.preregister.json"
)
RAW_RESULT = (
    ROOT
    / "results/harness/mbpp-sanitized-verified-selection-dev-v1/result.json"
)
RAW_CASES = (
    ROOT
    / "results/harness/mbpp-sanitized-verified-selection-dev-v1/cases.jsonl"
)
PUBLIC = (
    ROOT
    / "docs/results/mbpp_sanitized_verified_selection_dev_v1.public.json"
)
MARKDOWN = (
    ROOT / "docs/results/mbpp_sanitized_verified_selection_dev_v1.md"
)


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
                "pass" if row[arm]["test_result"]["full_pass"] else None
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
        != "nano_harness_mbpp_verified_selection_preregister_v1"
        or preregister.get("identity", {}).get("config_sha256")
        != sha256_file(CONFIG)
        or raw.get("schema_version")
        != "nano_harness_mbpp_verified_selection_raw_v1"
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
            "split": "validation",
            "cases": 43,
            "test_feasibility_probe_rows": 1,
            "test_content_used_for_policy_design": False,
        }
    ):
        raise ValueError("MBPP verified-selection result identity differs")
    rows = read_jsonl(RAW_CASES)
    case_ids = [row["case_id"] for row in rows]
    if (
        len(rows) != 43
        or len(set(case_ids)) != 43
        or any(row["task_id"] < 554 or row["task_id"] > 600 for row in rows)
    ):
        raise ValueError("MBPP validation rows differ")
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
        "candidate_only_exceeds_baseline_only": (
            versus_nine["paired_counts"]["candidate_only"]
            > versus_nine["paired_counts"]["baseline_only"]
        ),
        "minimum_candidate_only_wins": (
            versus_nine["paired_counts"]["candidate_only"]
            >= stats["minimum_candidate_only_wins"]
        ),
    }
    admitted = all(four_gates.values()) and all(nine_gates.values())
    direct_failures = [
        row for row in rows if not row["four_b_direct"]["test_result"]["full_pass"]
    ]
    return {
        "schema_version": (
            "nano_harness_mbpp_verified_selection_public_v1"
        ),
        "experiment_id": config["experiment_id"],
        "identity": {
            "result_revision": git_revision(),
            "config_sha256": sha256_file(CONFIG),
            "preregister_sha256": sha256_file(PREREGISTER),
            "raw_result_sha256": sha256_file(RAW_RESULT),
            "raw_cases_sha256": sha256_file(RAW_CASES),
            "validation_case_ids_sha256": preregister["identity"][
                "validation_case_ids_sha256"
            ],
        },
        "comparisons": {
            "versus_four_b": versus_four,
            "versus_nine_b": versus_nine,
        },
        "diagnostics": {
            "cases": len(rows),
            "four_b_direct_full_pass": sum(
                row["four_b_direct"]["test_result"]["full_pass"]
                for row in rows
            ),
            "nine_b_direct_full_pass": sum(
                row["nine_b_direct"]["test_result"]["full_pass"]
                for row in rows
            ),
            "candidate_full_pass": sum(
                row["candidate"]["test_result"]["full_pass"]
                for row in rows
            ),
            "direct_failures_entering_selection": len(direct_failures),
            "candidate_overrides": sum(
                row["receipt"]["override"] for row in rows
            ),
            "replicas_generated": sum(
                row["receipt"]["replicas_generated"] for row in rows
            ),
            "repairs_generated": sum(
                row["receipt"]["repair_generated"] for row in rows
            ),
            "selected_source_counts": dict(
                sorted(
                    Counter(
                        row["receipt"]["selected_source"] for row in rows
                    ).items()
                )
            ),
            "candidate_failure_classes": dict(
                sorted(
                    Counter(
                        failure
                        for row in rows
                        for failure, count in row["candidate"]["test_result"][
                            "failure_classes"
                        ].items()
                        for _ in range(count)
                    ).items()
                )
            ),
        },
        "decision": {
            "four_b_preservation_gates": four_gates,
            "nine_b_directional_gates": nine_gates,
            "validation_admitted": admitted,
            "complete_test_preregistration_allowed": admitted,
            "rerun_or_tuning_allowed": False,
            "next_action": (
                "Pre-register the unchanged harness on all 257 sanitized test cases."
                if admitted
                else (
                    "Publish negative evidence. Do not tune or rerun this "
                    "validation surface."
                )
            ),
        },
        "evaluation_boundary": raw["evaluation_boundary"],
        "claim_boundary": (
            "This is the complete MBPP sanitized validation result. It is "
            "not the 257-case sanitized test score."
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    four = report["comparisons"]["versus_four_b"]
    nine = report["comparisons"]["versus_nine_b"]
    diagnostics = report["diagnostics"]
    verdict = (
        "ADMIT TO TEST PRE-REGISTRATION"
        if report["decision"]["validation_admitted"]
        else "REJECT"
    )
    return f"""# MBPP Sanitized Verified Selection Dev v1

## Verdict

**{verdict}.**

| Comparison | Candidate | Baseline | Delta | 95% CI | McNemar p | Wins / Losses |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| vs 4B direct | {four['candidate_correct']}/43 | {four['baseline_correct']}/43 | {four['delta']:+.4f} | [{four['paired_bootstrap_95_ci'][0]:+.4f}, {four['paired_bootstrap_95_ci'][1]:+.4f}] | {four['mcnemar_exact_p']:.6g} | {four['paired_counts']['candidate_only']} / {four['paired_counts']['baseline_only']} |
| vs 9B direct | {nine['candidate_correct']}/43 | {nine['baseline_correct']}/43 | {nine['delta']:+.4f} | [{nine['paired_bootstrap_95_ci'][0]:+.4f}, {nine['paired_bootstrap_95_ci'][1]:+.4f}] | {nine['mcnemar_exact_p']:.6g} | {nine['paired_counts']['candidate_only']} / {nine['paired_counts']['baseline_only']} |

## What Ran

Direct 4B passed {diagnostics['four_b_direct_full_pass']}/43 and direct 9B
passed {diagnostics['nine_b_direct_full_pass']}/43. Only the
{diagnostics['direct_failures_entering_selection']} direct-4B failures entered
the verifier-selection route. It generated
{diagnostics['replicas_generated']} replicas and
{diagnostics['repairs_generated']} aggregate-feedback repairs, then made
{diagnostics['candidate_overrides']} improvements over direct 4B.

The model saw the public MBPP assertions, matching the benchmark protocol, but
never saw reference solutions. Every assertion ran in a no-network bubblewrap
sandbox with a read-only root filesystem, isolated Python mode, per-test
timeout, CPU, address-space, file-size, and open-file limits. Repair received
the same public assertions plus only aggregate pass count and failure classes.

## Decision Boundary

This result covers all 43 sanitized validation tasks. The 257-case sanitized
test split remains untouched unless every pre-registered validation gate
passes. No validation rerun or post-observation tuning is allowed.
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
