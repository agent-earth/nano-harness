#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nano_harness.skill_evolution import (
    build_candidate_request,
    cluster_failures,
    evaluate_registry,
    load_contract_suite,
    select_candidate,
)
from nano_harness.skill_system import SkillRegistry


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parent-registry",
        default=str(ROOT / "skills/registry_parent_v1.json"),
    )
    parser.add_argument(
        "--candidate-registry",
        default=str(ROOT / "skills/registry_candidate_v2.json"),
    )
    parser.add_argument(
        "--suite",
        default=str(ROOT / "skills/synthetic_contract_suite_v1.json"),
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    suite = load_contract_suite(args.suite)
    parent = SkillRegistry.from_manifest(args.parent_registry)
    candidate = SkillRegistry.from_manifest(args.candidate_registry)
    parent_scorecard = evaluate_registry(parent, suite)
    candidate_request = build_candidate_request(parent, parent_scorecard, suite)
    candidate_scorecard = evaluate_registry(candidate, suite)
    promotion = select_candidate(parent_scorecard, candidate_scorecard, suite)
    receipt = {
        "schema_version": "nano_harness_skill_evolution_smoke_v1",
        "suite_id": suite["suite_id"],
        "parent_scorecard": parent_scorecard,
        "candidate_request": candidate_request,
        "candidate_scorecard": candidate_scorecard,
        "parent_failure_clusters": cluster_failures(parent_scorecard),
        "promotion": promotion,
        "assertions": {
            "parent_has_failures": parent_scorecard["passed"]
            < parent_scorecard["total"],
            "candidate_passes_all": candidate_scorecard["passed"]
            == candidate_scorecard["total"],
            "candidate_is_promoted": promotion["promoted"] is True,
            "protected_families_non_regressing": not any(
                reason.startswith("protected_family_")
                for reason in promotion["reasons"]
            ),
            "claim_is_synthetic_only": "no model or benchmark quality uplift"
            in promotion["claim_boundary"],
        },
    }
    if not all(receipt["assertions"].values()):
        raise SystemExit(json.dumps(receipt, indent=2, sort_keys=True))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "schema_version": receipt["schema_version"],
                "suite_id": receipt["suite_id"],
                "parent": {
                    "passed": parent_scorecard["passed"],
                    "total": parent_scorecard["total"],
                    "aggregate_score": parent_scorecard["aggregate_score"],
                },
                "candidate": {
                    "passed": candidate_scorecard["passed"],
                    "total": candidate_scorecard["total"],
                    "aggregate_score": candidate_scorecard["aggregate_score"],
                },
                "failure_clusters": receipt["parent_failure_clusters"],
                "promotion": promotion,
                "assertions": receipt["assertions"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
