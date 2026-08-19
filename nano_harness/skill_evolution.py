from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from nano_harness.skill_system import (
    SkillRegistry,
    canonical_json,
    sha256_text,
)
from nano_harness.types import Task


SUITE_SCHEMA = "nano_harness_skill_contract_suite_v1"


def load_contract_suite(path: str | Path) -> dict[str, Any]:
    suite = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_contract_suite(suite)
    return suite


def validate_contract_suite(suite: dict[str, Any]) -> None:
    if suite.get("schema_version") != SUITE_SCHEMA:
        raise ValueError("unsupported skill contract suite schema")
    cases = suite.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("skill contract suite needs cases")
    case_ids = []
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("skill contract cases must be objects")
        case_id = str(case.get("case_id", ""))
        if not case_id:
            raise ValueError("skill contract case needs a case_id")
        case_ids.append(case_id)
        if not isinstance(case.get("tags"), list):
            raise ValueError("skill contract case needs tags")
        if not isinstance(case.get("expected_skills"), list):
            raise ValueError("skill contract case needs expected_skills")
        if not isinstance(case.get("required_fragments", []), list):
            raise ValueError("required_fragments must be a list")
        if not isinstance(case.get("forbidden_fragments", []), list):
            raise ValueError("forbidden_fragments must be a list")
        if not case.get("family"):
            raise ValueError("skill contract case needs a family")
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("skill contract case IDs must be unique")
    protected = suite.get("protected_families")
    if not isinstance(protected, list) or not protected:
        raise ValueError("skill contract suite needs protected families")
    allowed_mutations = suite.get("allowed_mutated_skill_ids")
    if not isinstance(allowed_mutations, list) or any(
        not isinstance(skill_id, str) or not skill_id
        for skill_id in allowed_mutations
    ):
        raise ValueError("allowed_mutated_skill_ids must be a string list")
    minimum_delta = suite.get("minimum_aggregate_delta")
    if (
        not isinstance(minimum_delta, (int, float))
        or isinstance(minimum_delta, bool)
        or not 0 < float(minimum_delta) <= 1
    ):
        raise ValueError("minimum aggregate delta must be in (0, 1]")


def evaluate_registry(
    registry: SkillRegistry,
    suite: dict[str, Any],
) -> dict[str, Any]:
    validate_contract_suite(suite)
    rows = []
    for case in suite["cases"]:
        task = Task(
            task_id=case["case_id"],
            benchmark="synthetic",
            messages=[{"role": "user", "content": "synthetic contract case"}],
            metadata={"skill_tags": case["tags"]},
        )
        selected, route = registry.route(task)
        selected_ids = [skill.skill_id for skill in selected]
        instructions = "\n".join(
            skill.instructions.lower() for skill in selected
        )
        failures = []
        if selected_ids != case["expected_skills"]:
            failures.append("route_mismatch")
        for fragment in case.get("required_fragments", []):
            if fragment.lower() not in instructions:
                failures.append(f"missing_instruction:{fragment}")
        for fragment in case.get("forbidden_fragments", []):
            if fragment.lower() in instructions:
                failures.append(f"forbidden_instruction:{fragment}")
        rows.append(
            {
                "case_id": case["case_id"],
                "family": case["family"],
                "protected": case["family"] in suite["protected_families"],
                "passed": not failures,
                "failures": failures,
                "route": route,
            }
        )
    by_family: dict[str, dict[str, Any]] = {}
    families = sorted({row["family"] for row in rows})
    for family in families:
        matched = [row for row in rows if row["family"] == family]
        passed = sum(row["passed"] for row in matched)
        by_family[family] = {
            "passed": passed,
            "total": len(matched),
            "score": passed / len(matched),
            "failed_case_ids": [
                row["case_id"] for row in matched if not row["passed"]
            ],
        }
    passed = sum(row["passed"] for row in rows)
    suite_sha256 = sha256_text(canonical_json(suite))
    case_ids_sha256 = sha256_text(
        canonical_json(sorted(row["case_id"] for row in rows))
    )
    return {
        "schema_version": "nano_harness_skill_scorecard_v1",
        "suite_id": suite["suite_id"],
        "suite_sha256": suite_sha256,
        "case_ids_sha256": case_ids_sha256,
        "registry_id": registry.registry_id,
        "registry_sha256": registry.sha256,
        "skill_sha256": {
            skill.skill_id: skill.sha256 for skill in registry.skills
        },
        "passed": passed,
        "total": len(rows),
        "aggregate_score": passed / len(rows),
        "by_family": by_family,
        "rows": rows,
    }


def cluster_failures(scorecard: dict[str, Any]) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str]] = Counter()
    case_ids: dict[tuple[str, str], list[str]] = {}
    for row in scorecard["rows"]:
        for failure in row["failures"]:
            key = (row["family"], failure)
            counts[key] += 1
            case_ids.setdefault(key, []).append(row["case_id"])
    return [
        {
            "family": family,
            "failure": failure,
            "count": count,
            "case_ids": sorted(case_ids[(family, failure)]),
        }
        for (family, failure), count in sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]


def build_candidate_request(
    parent: SkillRegistry,
    parent_scorecard: dict[str, Any],
    suite: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "nano_harness_skill_candidate_request_v1",
        "parent_registry_id": parent.registry_id,
        "parent_registry_sha256": parent.sha256,
        "suite_id": suite["suite_id"],
        "suite_sha256": parent_scorecard["suite_sha256"],
        "case_ids_sha256": parent_scorecard["case_ids_sha256"],
        "failure_clusters": cluster_failures(parent_scorecard),
        "allowed_mutation_scope": [
            "tool-recovery skill instructions",
        ],
        "forbidden_inputs": [
            "benchmark prompts",
            "benchmark references",
            "canary rows",
            "independent holdout rows",
            "model hidden reasoning",
        ],
        "acceptance": {
            "minimum_aggregate_delta": suite["minimum_aggregate_delta"],
            "allowed_mutated_skill_ids": suite["allowed_mutated_skill_ids"],
            "protected_families": suite["protected_families"],
            "same_case_ids_required": True,
        },
    }


def select_candidate(
    parent: dict[str, Any],
    candidate: dict[str, Any],
    suite: dict[str, Any],
) -> dict[str, Any]:
    validate_contract_suite(suite)
    reasons = []
    if candidate["suite_sha256"] != parent["suite_sha256"]:
        reasons.append("suite_identity_mismatch")
    if candidate["case_ids_sha256"] != parent["case_ids_sha256"]:
        reasons.append("case_identity_mismatch")
    parent_skills = parent.get("skill_sha256", {})
    candidate_skills = candidate.get("skill_sha256", {})
    if set(parent_skills) != set(candidate_skills):
        reasons.append("skill_set_mismatch")
    allowed_mutations = set(suite["allowed_mutated_skill_ids"])
    for skill_id in sorted(set(parent_skills) & set(candidate_skills)):
        if (
            parent_skills[skill_id] != candidate_skills[skill_id]
            and skill_id not in allowed_mutations
        ):
            reasons.append(f"disallowed_skill_mutation:{skill_id}")
    delta = candidate["aggregate_score"] - parent["aggregate_score"]
    if delta < suite["minimum_aggregate_delta"]:
        reasons.append("aggregate_delta_below_threshold")
    for family in suite["protected_families"]:
        parent_score = parent["by_family"].get(family, {}).get("score")
        candidate_score = candidate["by_family"].get(family, {}).get("score")
        if parent_score is None or candidate_score is None:
            reasons.append(f"protected_family_missing:{family}")
        elif candidate_score < parent_score:
            reasons.append(f"protected_family_regression:{family}")
    return {
        "schema_version": "nano_harness_skill_promotion_v1",
        "suite_id": suite["suite_id"],
        "parent_registry_id": parent["registry_id"],
        "parent_registry_sha256": parent["registry_sha256"],
        "candidate_registry_id": candidate["registry_id"],
        "candidate_registry_sha256": candidate["registry_sha256"],
        "aggregate_delta": delta,
        "allowed_mutated_skill_ids": suite["allowed_mutated_skill_ids"],
        "protected_families": suite["protected_families"],
        "promoted": not reasons,
        "reasons": reasons,
        "claim_boundary": (
            "synthetic skill-contract promotion only; no model or benchmark "
            "quality uplift is established"
        ),
    }
