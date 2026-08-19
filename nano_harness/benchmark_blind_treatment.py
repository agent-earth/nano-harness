from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

from nano_harness.baseline import SuiteManifest, load_manifest


CONFIG_SCHEMA = "nano_harness_benchmark_blind_treatment_v1"
RECEIPT_SCHEMA = "nano_harness_benchmark_blind_treatment_receipt_v1"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_exact_keys(
    value: dict[str, Any],
    expected: set[str],
    label: str,
) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{label} fields differ: expected={sorted(expected)} "
            f"actual={sorted(value)}"
        )


def _require_sha256(value: Any, label: str) -> None:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA256")


def _json_pointer(document: Any, pointer: str) -> Any:
    if not pointer.startswith("/"):
        raise ValueError("JSON pointer must start with /")
    current = document
    for token in pointer[1:].split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, list):
            current = current[int(token)]
        else:
            current = current[token]
    return current


def _resolve(root: Path, relative: str) -> Path:
    return (root / relative).resolve()


def load_treatment(path: str | Path) -> dict[str, Any]:
    treatment = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_treatment(treatment)
    return treatment


def validate_treatment(treatment: dict[str, Any]) -> None:
    _require_exact_keys(
        treatment,
        {
            "schema_version",
            "treatment_id",
            "claim_boundary",
            "base_revision",
            "dependencies",
            "prior_evidence",
            "evaluation_surfaces",
            "arms",
            "admission_gates",
            "decision_policy",
            "execution_boundary",
        },
        "treatment",
    )
    if treatment["schema_version"] != CONFIG_SCHEMA:
        raise ValueError("unsupported benchmark-blind treatment schema")
    if not re.fullmatch(r"[0-9a-f]{40}", treatment["base_revision"]):
        raise ValueError("base_revision must be a full Git revision")

    dependencies = treatment["dependencies"]
    if len(dependencies) != 1:
        raise ValueError("exactly one peer consistency dependency is required")
    dependency = dependencies[0]
    _require_exact_keys(
        dependency,
        {
            "dependency_id",
            "owner_lane",
            "required_commit",
            "artifacts",
            "expected_public_result_path",
            "expected_public_result_schema",
            "state_at_preregistration",
        },
        "peer dependency",
    )
    if (
        dependency["dependency_id"]
        != "peer_paired_consistency_replication_v1"
        or dependency["owner_lane"] != "ultimate-distill-skillgen-traex-02"
        or dependency["state_at_preregistration"] != "not_started"
        or not re.fullmatch(r"[0-9a-f]{40}", dependency["required_commit"])
    ):
        raise ValueError("peer consistency dependency differs")
    if len(dependency["artifacts"]) != 3:
        raise ValueError("peer consistency needs config, prereg, and release")
    for artifact in dependency["artifacts"]:
        _require_exact_keys(artifact, {"path", "sha256"}, "peer artifact")
        _require_sha256(artifact["sha256"], "peer artifact sha256")

    if [arm["arm_id"] for arm in treatment["arms"]] != [
        "adapter_only",
        "arbiter_only",
        "adapter_plus_arbiter",
    ]:
        raise ValueError("treatment arm order differs")
    for arm in treatment["arms"]:
        _require_exact_keys(
            arm,
            {
                "arm_id",
                "base_model",
                "harness_manifest",
                "intervention",
                "serving_model_name_policy",
                "variable_changed",
            },
            "treatment arm",
        )
        if arm["base_model"] != "qwen3.5-4b":
            raise ValueError("all treatment arms must use Qwen3.5-4B")

    surfaces = treatment["evaluation_surfaces"]
    _require_exact_keys(surfaces, {"canary", "complete"}, "surfaces")
    _require_exact_keys(
        surfaces["canary"],
        {
            "direct_manifest_path",
            "direct_manifest_sha256",
            "gpqa_arbiter_manifest_path",
            "gpqa_arbiter_manifest_sha256",
            "baseline_case_manifest_path",
            "baseline_case_manifest_sha256",
            "baseline_report_path",
            "baseline_report_sha256",
        },
        "canary surface",
    )
    _require_exact_keys(
        surfaces["complete"],
        {
            "direct_manifest_path",
            "direct_manifest_sha256",
            "gpqa_arbiter_manifest_path",
            "gpqa_arbiter_manifest_sha256",
            "baseline_case_manifest_path",
            "baseline_case_manifest_sha256",
            "baseline_report_path",
            "baseline_report_sha256",
            "preregister_path",
            "preregister_sha256",
        },
        "complete surface",
    )
    for surface in surfaces.values():
        for key, value in surface.items():
            if key.endswith("_sha256"):
                _require_sha256(value, key)

    gates = treatment["admission_gates"]
    _require_exact_keys(
        gates,
        {"peer_consistency", "canary", "complete"},
        "admission gates",
    )
    peer = gates["peer_consistency"]
    if peer != {
        "all_adapter_tensors_finite": True,
        "independent_reload_exact": True,
        "aggregate_bootstrap_ci_lower_gt_zero": True,
        "aggregate_exact_mcnemar_p_lt": 0.05,
        "final_bootstrap_ci_lower_gt_zero": True,
        "final_exact_mcnemar_p_lt": 0.05,
        "pair_bootstrap_ci_lower_gt_zero": True,
        "pair_exact_mcnemar_p_lt": 0.05,
        "minimum_final_only_wins": 6,
        "maximum_final_only_losses": 0,
        "every_json_family_non_regression": True,
    }:
        raise ValueError("peer consistency admission gates differ")
    canary = gates["canary"]
    if (
        canary["arm_order"]
        != ["adapter_only", "arbiter_only", "adapter_plus_arbiter"]
        or canary["cases"] != 211
        or canary["minimum_overall_correct"] != 164
        or canary["benchmark_minimum_correct"]
        != {"gsm8k": 90, "mmlu": 67, "gpqa_diamond": 6}
        or canary["maximum_parse_failures"] != 2
        or canary["maximum_api_errors"] != 0
        or canary["require_exact_case_identity"] is not True
        or canary[
            "paired_candidate_only_wins_gt_base_only_wins"
        ]
        is not True
        or canary[
            "each_arm_evaluated_even_if_an_earlier_arm_fails"
        ]
        is not True
    ):
        raise ValueError("canary admission gates differ")
    complete = gates["complete"]
    if (
        complete["bootstrap_samples"] != 10_000
        or complete["alpha"] != 0.05
        or complete["minimum_benchmarks_significantly_won"] != 3
        or complete["per_benchmark_non_regression_vs_direct_four_b"]
        is not True
        or complete[
            "each_admitted_arm_evaluated_even_if_an_earlier_arm_fails"
        ]
        is not True
        or complete["strict_score_is_official"] is not True
        or complete[
            "track_loose_format_diagnostic_without_rescoring"
        ]
        is not True
    ):
        raise ValueError("complete admission gates differ")

    if treatment["execution_boundary"] != {
        "this_commit_only_preregisters": True,
        "adapter_result_exists": False,
        "treatment_generation_started": False,
        "canary_generation_allowed": False,
        "complete_treatment_generation_allowed": False,
        "independent_holdout_allowed": False,
    }:
        raise ValueError("execution boundary differs")

    evidence_ids = []
    for evidence in treatment["prior_evidence"]:
        _require_exact_keys(
            evidence,
            {"evidence_id", "path", "sha256", "assertions"},
            "prior evidence",
        )
        evidence_ids.append(evidence["evidence_id"])
        _require_sha256(evidence["sha256"], "prior evidence sha256")
        for assertion in evidence["assertions"]:
            _require_exact_keys(
                assertion,
                {"pointer", "equals"},
                "prior evidence assertion",
            )
    if evidence_ids != [
        "complete_direct_baseline",
        "gpqa_arbiter_dev8_direction",
        "gpqa_arbiter_holdout5_rejection",
    ]:
        raise ValueError("prior evidence order differs")

    forbidden = set(
        treatment["decision_policy"][
            "forbidden_after_any_treatment_observation"
        ]
    )
    required_forbidden = {
        "training_data_change",
        "objective_change",
        "learning_rate_change",
        "seed_change",
        "step_change",
        "adapter_checkpoint_choice",
        "adapter_weight_change",
        "prompt_change",
        "parser_change",
        "scorer_change",
        "token_budget_change",
        "benchmark_route_change",
        "case_selection_change",
    }
    if not required_forbidden <= forbidden:
        raise ValueError("no-post-hoc-search policy is incomplete")


def _verify_file(root: Path, path: str, expected_sha256: str) -> Path:
    resolved = _resolve(root, path)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    if sha256_file(resolved) != expected_sha256:
        raise ValueError(f"artifact identity mismatch: {path}")
    return resolved


def _manifest_contract(manifest: SuiteManifest) -> dict[str, Any]:
    value = asdict(manifest)
    value.pop("suite_id")
    value.pop("strategy")
    value.pop("benchmark_routing")
    value.pop("option_evidence_max_tokens")
    value.pop("verifier_max_tokens")
    value.pop("normalize_bare_choice")
    return value


def _verify_manifest_pair(
    direct_path: Path,
    arbiter_path: Path,
) -> dict[str, Any]:
    direct = load_manifest(direct_path)
    arbiter = load_manifest(arbiter_path)
    if direct.strategy != "direct":
        raise ValueError("control manifest must be direct")
    if (
        arbiter.strategy != "benchmark_routing"
        or arbiter.benchmark_routing
        != {
            "gsm8k": "direct",
            "mmlu": "direct",
            "gpqa_diamond": "option_evidence_arbiter",
        }
        or arbiter.option_evidence_max_tokens != 96
        or arbiter.verifier_max_tokens != 64
        or arbiter.normalize_bare_choice is not True
    ):
        raise ValueError("GPQA-only arbiter route differs")
    if _manifest_contract(direct) != _manifest_contract(arbiter):
        raise ValueError("arbiter manifest changes more than the GPQA route")
    return {
        "case_selection_equal": True,
        "dataset_identity_equal": True,
        "gsm8k_direct_unchanged": True,
        "mmlu_direct_unchanged": True,
        "gpqa_only_arbiter": True,
        "option_evidence_max_tokens": 96,
        "arbiter_max_tokens": 64,
    }


def _git_revision(repository: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def _git_is_ancestor(
    repository: Path,
    ancestor: str,
    descendant: str,
) -> bool:
    completed = subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "merge-base",
            "--is-ancestor",
            ancestor,
            descendant,
        ],
        check=False,
    )
    return completed.returncode == 0


def _evaluate_peer_result(
    result: dict[str, Any],
    dependency: dict[str, Any],
) -> dict[str, Any]:
    _require_exact_keys(
        result,
        {
            "schema_version",
            "experiment_id",
            "identity",
            "stability",
            "comparisons",
            "json_families",
            "decision",
            "claim_boundary",
        },
        "peer result",
    )
    if (
        result["schema_version"]
        != dependency["expected_public_result_schema"]
        or result["experiment_id"] != "paired-consistency-replication-v1"
    ):
        raise ValueError("peer result identity differs")
    identity = result["identity"]
    artifact_hashes = {
        Path(row["path"]).name: row["sha256"]
        for row in dependency["artifacts"]
    }
    required_identity = {
        "config_sha256": artifact_hashes[
            "consistency_replication_v1.json"
        ],
        "preregister_sha256": artifact_hashes[
            "consistency_replication_v1.preregister.json"
        ],
        "release_sha256": artifact_hashes[
            "paired_consistency_replication_v1.release.json"
        ],
        "preregister_revision": dependency["required_commit"],
    }
    for key, expected in required_identity.items():
        if identity.get(key) != expected:
            raise ValueError(f"peer result {key} differs")
    _require_sha256(identity.get("adapter_sha256"), "peer adapter sha256")

    stability = result["stability"]
    stability_passed = (
        stability.get("all_adapter_tensors_finite") is True
        and stability.get("independent_reload_exact") is True
        and stability.get("failure_receipt_exists") is False
    )
    comparison_checks = {}
    for name in ("aggregate", "final", "pair"):
        comparison = result["comparisons"].get(name, {})
        comparison_checks[name] = (
            comparison.get("paired_bootstrap_95_ci", [None])[0] is not None
            and comparison["paired_bootstrap_95_ci"][0] > 0
            and comparison.get("mcnemar_exact_p", 1.0) < 0.05
        )
    final = result["comparisons"].get("final", {})
    final_count_passed = (
        final.get("candidate_only_wins", 0) >= 6
        and final.get("baseline_only_wins", 1) == 0
    )
    json_passed = (
        bool(result["json_families"])
        and all(
            row.get("post_correct", -1) >= row.get("baseline_correct", 0)
            for row in result["json_families"].values()
        )
    )
    admitted = (
        stability_passed
        and all(comparison_checks.values())
        and final_count_passed
        and json_passed
        and result["decision"].get("accepted") is True
    )
    return {
        "exists": True,
        "admitted": admitted,
        "adapter_sha256": identity["adapter_sha256"],
        "checks": {
            "stability": stability_passed,
            "aggregate_significance": comparison_checks["aggregate"],
            "final_significance": comparison_checks["final"],
            "pair_significance": comparison_checks["pair"],
            "final_wins_at_least_six_and_losses_zero": final_count_passed,
            "every_json_family_non_regression": json_passed,
            "peer_decision_accepted": (
                result["decision"].get("accepted") is True
            ),
        },
    }


def build_treatment_receipt(
    config_path: str | Path,
    *,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else config_path.parents[2]
    )
    treatment = load_treatment(config_path)

    verified_dependencies = []
    dependency = treatment["dependencies"][0]
    for artifact in dependency["artifacts"]:
        _verify_file(root, artifact["path"], artifact["sha256"])
        verified_dependencies.append({**artifact, "verified": True})
    peer_repository = _resolve(root, dependency["artifacts"][0]["path"]).parents[
        2
    ]
    peer_revision = _git_revision(peer_repository)
    if not _git_is_ancestor(
        peer_repository,
        dependency["required_commit"],
        peer_revision,
    ):
        raise ValueError("peer result does not descend from preregistration")

    evidence_receipts = []
    for evidence in treatment["prior_evidence"]:
        path = _verify_file(root, evidence["path"], evidence["sha256"])
        document = json.loads(path.read_text(encoding="utf-8"))
        assertion_receipts = []
        for assertion in evidence["assertions"]:
            actual = _json_pointer(document, assertion["pointer"])
            if actual != assertion["equals"]:
                raise ValueError(
                    f"prior evidence assertion differs: "
                    f"{evidence['evidence_id']} {assertion['pointer']}"
                )
            assertion_receipts.append(
                {**assertion, "actual": actual, "passed": True}
            )
        evidence_receipts.append(
            {
                **evidence,
                "assertions": assertion_receipts,
                "verified": True,
            }
        )

    surface_receipts = {}
    for surface_name, surface in treatment["evaluation_surfaces"].items():
        verified_paths = {}
        for key, value in surface.items():
            if not key.endswith("_path"):
                continue
            sha_key = key.removesuffix("_path") + "_sha256"
            _verify_file(root, value, surface[sha_key])
            verified_paths[key] = {
                "path": value,
                "sha256": surface[sha_key],
                "verified": True,
            }
        surface_receipts[surface_name] = {
            "artifacts": verified_paths,
            "manifest_invariance": _verify_manifest_pair(
                _resolve(root, surface["direct_manifest_path"]),
                _resolve(root, surface["gpqa_arbiter_manifest_path"]),
            ),
        }

    result_path = _resolve(root, dependency["expected_public_result_path"])
    if result_path.exists():
        peer_result = _evaluate_peer_result(
            json.loads(result_path.read_text(encoding="utf-8")),
            dependency,
        )
        peer_result["sha256"] = sha256_file(result_path)
        peer_result["path"] = dependency["expected_public_result_path"]
    else:
        peer_result = {
            "exists": False,
            "admitted": False,
            "adapter_sha256": None,
            "checks": {
                "result_missing": False,
            },
            "path": dependency["expected_public_result_path"],
            "sha256": None,
        }

    canary_allowed = peer_result["admitted"]
    return {
        "schema_version": RECEIPT_SCHEMA,
        "treatment_id": treatment["treatment_id"],
        "claim_boundary": treatment["claim_boundary"],
        "identity": {
            "base_revision": treatment["base_revision"],
            "config_sha256": sha256_file(config_path),
            "peer_revision": peer_revision,
            "dependencies": verified_dependencies,
        },
        "prior_evidence": evidence_receipts,
        "evaluation_surfaces": surface_receipts,
        "arms": treatment["arms"],
        "admission_gates": treatment["admission_gates"],
        "decision_policy": treatment["decision_policy"],
        "peer_result": peer_result,
        "readiness": {
            "canary_generation_allowed": canary_allowed,
            "complete_treatment_generation_allowed": False,
            "independent_holdout_allowed": False,
            "reason": (
                "Peer consistency replication passed every frozen gate; "
                "run the three canary arms in pre-registered order."
                if canary_allowed
                else "Peer consistency replication result is absent or "
                "fails at least one frozen admission gate."
            ),
        },
        "checks": {
            "dependency_identities_verified": True,
            "peer_owner_and_revision_verified": True,
            "prior_evidence_verified": True,
            "canary_surface_frozen": True,
            "complete_surface_frozen": True,
            "only_gpqa_harness_changes": True,
            "strict_score_remains_official": True,
            "no_post_hoc_search_policy_frozen": True,
            "independent_holdout_sealed": True,
        },
        "execution_boundary": treatment["execution_boundary"],
        "next_action": (
            "Run adapter-only, arbiter-only, and adapter-plus-arbiter on the "
            "frozen 211-case canary without changing any identity."
            if canary_allowed
            else "Wait for and consume the peer-owned unique paired-"
            "consistency replication result; do not run treatment canary or "
            "complete benchmarks."
        ),
    }
