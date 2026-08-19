from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from nano_harness.skill_evolution import (
    evaluate_registry,
    load_contract_suite,
    select_candidate,
)
from nano_harness.skill_system import SkillRegistry


CAMPAIGN_SCHEMA = "nano_harness_fullstack_campaign_v1"
RECEIPT_SCHEMA = "nano_harness_fullstack_campaign_receipt_v1"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_campaign(path: str | Path) -> dict[str, Any]:
    campaign = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_campaign(campaign)
    return campaign


def _require_sha256(value: Any, label: str) -> None:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA256")


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


def validate_campaign(campaign: dict[str, Any]) -> None:
    _require_exact_keys(
        campaign,
        {
            "schema_version",
            "campaign_id",
            "claim_boundary",
            "base_revision",
            "artifacts",
            "models",
            "complete_benchmarks",
            "formal_agent_benchmarks",
            "capabilities",
            "peer_dependencies",
            "skill_evolution",
            "prior_evidence",
            "candidate_ladder",
            "acceptance",
            "resource_contract",
            "execution_boundary",
        },
        "campaign",
    )
    if campaign["schema_version"] != CAMPAIGN_SCHEMA:
        raise ValueError("unsupported full-stack campaign schema")
    if not str(campaign["campaign_id"]):
        raise ValueError("campaign_id is required")
    if not re.fullmatch(r"[0-9a-f]{7,40}", str(campaign["base_revision"])):
        raise ValueError("base_revision must be a Git revision")

    artifact_ids = []
    for artifact in campaign["artifacts"]:
        _require_exact_keys(
            artifact,
            {"artifact_id", "path", "sha256", "kind"},
            "artifact",
        )
        artifact_ids.append(artifact["artifact_id"])
        _require_sha256(
            artifact["sha256"],
            f"artifact {artifact['artifact_id']} sha256",
        )
    if len(artifact_ids) != len(set(artifact_ids)):
        raise ValueError("artifact IDs must be unique")
    artifact_id_set = set(artifact_ids)

    model_ids = []
    for model in campaign["models"]:
        model_ids.append(model["model_id"])
        status = model.get("status")
        if status == "ready":
            _require_exact_keys(
                model,
                {
                    "model_id",
                    "status",
                    "path",
                    "config_sha256",
                    "index_sha256",
                    "weight_bytes",
                    "shards",
                    "required_phase",
                },
                "ready model",
            )
            _require_sha256(
                model["config_sha256"],
                f"model {model['model_id']} config_sha256",
            )
            _require_sha256(
                model["index_sha256"],
                f"model {model['model_id']} index_sha256",
            )
            if not model["shards"] or model["weight_bytes"] <= 0:
                raise ValueError("ready model needs shards and weight bytes")
            for shard in model["shards"]:
                _require_exact_keys(
                    shard,
                    {"name", "bytes", "sha256"},
                    "model shard",
                )
                _require_sha256(
                    shard["sha256"],
                    f"model shard {shard['name']} sha256",
                )
        elif status == "missing":
            _require_exact_keys(
                model,
                {
                    "model_id",
                    "status",
                    "path",
                    "oniond_name",
                    "required_phase",
                    "minimum_free_disk_gib",
                    "block_reason",
                },
                "missing model",
            )
        else:
            raise ValueError("model status must be ready or missing")
    if len(model_ids) != len(set(model_ids)):
        raise ValueError("model IDs must be unique")
    if set(model_ids) != {"qwen3.5-4b", "qwen3.5-9b", "qwen3.5-27b"}:
        raise ValueError("campaign must declare 4B, 9B, and 27B identities")

    benchmark_names = []
    for benchmark in campaign["complete_benchmarks"]:
        _require_exact_keys(
            benchmark,
            {
                "name",
                "path",
                "sha256",
                "rows",
                "scorer",
                "complete_definition",
            },
            "complete benchmark",
        )
        benchmark_names.append(benchmark["name"])
        _require_sha256(
            benchmark["sha256"],
            f"benchmark {benchmark['name']} sha256",
        )
        if benchmark["rows"] <= 0:
            raise ValueError("complete benchmark rows must be positive")
    if benchmark_names != ["gsm8k", "mmlu", "gpqa_diamond"]:
        raise ValueError("complete benchmark order must be GSM8K, MMLU, GPQA")

    formal_names = []
    for benchmark in campaign["formal_agent_benchmarks"]:
        _require_exact_keys(
            benchmark,
            {
                "name",
                "feasibility_key",
                "status",
                "quality_score_claimed",
                "block_reason",
            },
            "formal agent benchmark",
        )
        formal_names.append(benchmark["name"])
        if (
            benchmark["status"] != "blocked"
            or benchmark["quality_score_claimed"] is not False
        ):
            raise ValueError("formal agent benchmark scans are not scores")
    if len(formal_names) < 3 or len(formal_names) != len(set(formal_names)):
        raise ValueError("at least three formal agent benchmarks are required")

    capability_names = []
    for capability in campaign["capabilities"]:
        _require_exact_keys(
            capability,
            {"name", "status", "evidence_paths", "admission_gate"},
            "capability",
        )
        capability_names.append(capability["name"])
        if capability["status"] not in {"implemented", "missing"}:
            raise ValueError("capability status differs")
        if (
            capability["status"] == "implemented"
            and not capability["evidence_paths"]
        ):
            raise ValueError("implemented capability needs evidence")
        if (
            capability["status"] == "missing"
            and capability["evidence_paths"]
        ):
            raise ValueError("missing capability cannot claim evidence")
    if capability_names != [
        "skill_harness",
        "sft",
        "paired_consistency",
        "rl",
        "opd",
    ]:
        raise ValueError("full-stack capability order differs")

    for dependency in campaign["peer_dependencies"]:
        _require_exact_keys(
            dependency,
            {"dependency_id", "artifact_ids", "state", "owner_lane"},
            "peer dependency",
        )
        if not set(dependency["artifact_ids"]) <= artifact_id_set:
            raise ValueError("peer dependency references unknown artifacts")

    skill = campaign["skill_evolution"]
    _require_exact_keys(
        skill,
        {
            "parent_registry_path",
            "candidate_registry_path",
            "suite_path",
            "expected_parent_passed",
            "expected_candidate_passed",
            "expected_total",
            "expected_promoted",
        },
        "skill evolution",
    )

    for evidence in campaign["prior_evidence"]:
        _require_exact_keys(
            evidence,
            {"evidence_id", "artifact_id", "assertions", "claim_boundary"},
            "prior evidence",
        )
        if evidence["artifact_id"] not in artifact_id_set:
            raise ValueError("prior evidence references unknown artifact")
        for assertion in evidence["assertions"]:
            _require_exact_keys(
                assertion,
                {"pointer", "equals"},
                "prior evidence assertion",
            )

    ladder_ids = [stage["stage_id"] for stage in campaign["candidate_ladder"]]
    if len(ladder_ids) != len(set(ladder_ids)) or len(ladder_ids) < 5:
        raise ValueError("candidate ladder needs unique full-stack stages")
    prior_stage_ids: set[str] = set()
    for stage in campaign["candidate_ladder"]:
        _require_exact_keys(
            stage,
            {
                "stage_id",
                "treatment",
                "depends_on",
                "admission",
                "stop_rule",
            },
            "candidate ladder stage",
        )
        dependencies = set(stage["depends_on"])
        if len(dependencies) != len(stage["depends_on"]):
            raise ValueError("candidate ladder dependencies must be unique")
        if not dependencies <= prior_stage_ids:
            raise ValueError(
                "candidate ladder may depend only on prior stages"
            )
        prior_stage_ids.add(stage["stage_id"])

    acceptance = campaign["acceptance"]
    _require_exact_keys(
        acceptance,
        {
            "bootstrap_samples",
            "alpha",
            "complete_benchmark_superiority",
            "minimum_complete_benchmarks_won",
            "twenty_seven_b_parity",
        },
        "acceptance",
    )
    if (
        acceptance["bootstrap_samples"] != 10_000
        or acceptance["alpha"] != 0.05
        or acceptance["minimum_complete_benchmarks_won"] < 3
    ):
        raise ValueError("campaign significance contract differs")
    parity = acceptance["twenty_seven_b_parity"]
    if (
        parity.get("benchmarks") != ["gsm8k", "mmlu"]
        or parity.get("noninferiority_margin") != 0.02
        or parity.get("minimum_benchmarks_at_parity") != 2
    ):
        raise ValueError("27B parity contract differs")

    boundary = campaign["execution_boundary"]
    if boundary != {
        "this_commit_only_audits_and_preregisters": True,
        "model_generation_started": False,
        "training_started": False,
        "benchmark_scoring_started": False,
        "rl_started": False,
        "opd_started": False,
    }:
        raise ValueError("execution boundary differs")


def _resolve(root: Path, relative: str) -> Path:
    return (root / relative).resolve()


def _json_pointer(document: Any, pointer: str) -> Any:
    if pointer == "":
        return document
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


def _parquet_rows(path: Path) -> int:
    import pyarrow.parquet as parquet

    return parquet.ParquetFile(path).metadata.num_rows


def build_campaign_receipt(
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
    campaign = load_campaign(config_path)

    artifacts = {}
    for artifact in campaign["artifacts"]:
        path = _resolve(root, artifact["path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != artifact["sha256"]:
            raise ValueError(
                f"artifact identity mismatch: {artifact['artifact_id']}"
            )
        artifacts[artifact["artifact_id"]] = {
            **artifact,
            "verified": True,
        }

    models = []
    for model in campaign["models"]:
        path = _resolve(root, model["path"])
        if model["status"] == "missing":
            if path.exists():
                raise ValueError(
                    f"missing-model contract is stale: {model['model_id']}"
                )
            models.append(
                {
                    "model_id": model["model_id"],
                    "status": "blocked_missing",
                    "required_phase": model["required_phase"],
                    "oniond_name": model["oniond_name"],
                    "minimum_free_disk_gib": model[
                        "minimum_free_disk_gib"
                    ],
                    "block_reason": model["block_reason"],
                }
            )
            continue
        config_actual = sha256_file(path / "config.json")
        index_actual = sha256_file(path / "model.safetensors.index.json")
        if (
            config_actual != model["config_sha256"]
            or index_actual != model["index_sha256"]
        ):
            raise ValueError(f"model identity mismatch: {model['model_id']}")
        observed_bytes = 0
        shard_receipts = []
        for shard in model["shards"]:
            shard_path = path / shard["name"]
            observed_bytes += shard_path.stat().st_size
            actual = sha256_file(shard_path)
            if (
                shard_path.stat().st_size != shard["bytes"]
                or actual != shard["sha256"]
            ):
                raise ValueError(
                    f"model shard identity mismatch: {model['model_id']} "
                    f"{shard['name']}"
                )
            shard_receipts.append({**shard, "verified": True})
        if observed_bytes != model["weight_bytes"]:
            raise ValueError(f"model weight bytes differ: {model['model_id']}")
        models.append(
            {
                "model_id": model["model_id"],
                "status": "ready",
                "required_phase": model["required_phase"],
                "config_sha256": config_actual,
                "index_sha256": index_actual,
                "weight_bytes": observed_bytes,
                "shards": shard_receipts,
            }
        )

    complete_benchmarks = []
    for benchmark in campaign["complete_benchmarks"]:
        path = _resolve(root, benchmark["path"])
        actual = sha256_file(path)
        rows = _parquet_rows(path)
        if actual != benchmark["sha256"] or rows != benchmark["rows"]:
            raise ValueError(
                f"complete benchmark identity mismatch: {benchmark['name']}"
            )
        complete_benchmarks.append(
            {
                **benchmark,
                "rows_verified": True,
                "full_dataset_required": True,
            }
        )

    feasibility = json.loads(
        _resolve(
            root,
            artifacts["benchmark-feasibility-v1"]["path"],
        ).read_text(encoding="utf-8")
    )
    if (
        feasibility["quality_score_claimed"] is not False
        or feasibility["summary"]["formal_container_benchmarks_runnable"]
        is not False
    ):
        raise ValueError("benchmark feasibility boundary differs")
    formal_agent_benchmarks = []
    for benchmark in campaign["formal_agent_benchmarks"]:
        scan = feasibility["scans"][benchmark["feasibility_key"]]
        if scan["status"] != "passed":
            raise ValueError("formal benchmark feasibility scan regressed")
        formal_agent_benchmarks.append(
            {
                **benchmark,
                "scan_status": scan["status"],
                "evidence_kind": scan["evidence_kind"],
                "formal_score_available": False,
            }
        )

    capabilities = []
    for capability in campaign["capabilities"]:
        for relative in capability["evidence_paths"]:
            if not _resolve(root, relative).is_file():
                raise FileNotFoundError(relative)
        capabilities.append(
            {
                **capability,
                "admitted_now": capability["status"] == "implemented",
            }
        )

    peer_dependencies = []
    for dependency in campaign["peer_dependencies"]:
        peer_dependencies.append(
            {
                **dependency,
                "identities_verified": all(
                    artifacts[artifact_id]["verified"]
                    for artifact_id in dependency["artifact_ids"]
                ),
            }
        )

    skill = campaign["skill_evolution"]
    suite = load_contract_suite(_resolve(root, skill["suite_path"]))
    parent = evaluate_registry(
        SkillRegistry.from_manifest(
            _resolve(root, skill["parent_registry_path"])
        ),
        suite,
    )
    candidate = evaluate_registry(
        SkillRegistry.from_manifest(
            _resolve(root, skill["candidate_registry_path"])
        ),
        suite,
    )
    promotion = select_candidate(parent, candidate, suite)
    if (
        parent["passed"] != skill["expected_parent_passed"]
        or candidate["passed"] != skill["expected_candidate_passed"]
        or parent["total"] != skill["expected_total"]
        or candidate["total"] != skill["expected_total"]
        or promotion["promoted"] is not skill["expected_promoted"]
    ):
        raise ValueError("skill evolution evidence differs")

    prior_evidence = []
    for evidence in campaign["prior_evidence"]:
        artifact = artifacts[evidence["artifact_id"]]
        document = json.loads(
            _resolve(root, artifact["path"]).read_text(encoding="utf-8")
        )
        checks = []
        for assertion in evidence["assertions"]:
            actual = _json_pointer(document, assertion["pointer"])
            passed = actual == assertion["equals"]
            if not passed:
                raise ValueError(
                    f"prior evidence assertion differs: "
                    f"{evidence['evidence_id']} {assertion['pointer']}"
                )
            checks.append({**assertion, "actual": actual, "passed": True})
        prior_evidence.append(
            {
                **evidence,
                "assertions": checks,
                "identity_verified": True,
            }
        )

    ready_models = {
        row["model_id"] for row in models if row["status"] == "ready"
    }
    implemented = {
        row["name"] for row in capabilities if row["status"] == "implemented"
    }
    readiness = {
        "matched_4b_9b_complete_benchmarks": {
            "ready": {"qwen3.5-4b", "qwen3.5-9b"} <= ready_models,
            "benchmarks": [
                row["name"] for row in complete_benchmarks
            ],
        },
        "twenty_seven_b_parity": {
            "ready": "qwen3.5-27b" in ready_models,
            "blocked_by": [
                "qwen3.5-27b model is not installed",
                "safe download headroom is not available in the frozen audit",
            ],
        },
        "rl": {
            "ready": "rl" in implemented,
            "blocked_by": ["no versioned RL implementation or smoke receipt"],
        },
        "opd": {
            "ready": "opd" in implemented,
            "blocked_by": [
                "no versioned on-policy distillation implementation or "
                "smoke receipt"
            ],
        },
        "formal_agent_benchmarks": {
            "ready": False,
            "blocked_by": [
                "container mount namespaces are unavailable on this devbox"
            ],
        },
    }

    checks = {
        "all_artifact_identities_verified": all(
            row["verified"] for row in artifacts.values()
        ),
        "four_b_and_nine_b_ready": (
            {"qwen3.5-4b", "qwen3.5-9b"} <= ready_models
        ),
        "twenty_seven_b_missing_is_explicit": (
            "qwen3.5-27b" not in ready_models
        ),
        "three_complete_benchmarks_verified": (
            len(complete_benchmarks) == 3
            and all(row["rows_verified"] for row in complete_benchmarks)
        ),
        "formal_scans_not_reported_as_scores": all(
            row["formal_score_available"] is False
            for row in formal_agent_benchmarks
        ),
        "skill_evolution_reproduced": promotion["promoted"] is True,
        "peer_dependencies_verified": all(
            row["identities_verified"] for row in peer_dependencies
        ),
        "rl_and_opd_fail_closed": (
            readiness["rl"]["ready"] is False
            and readiness["opd"]["ready"] is False
        ),
        "no_model_or_training_execution": all(
            value is False
            for key, value in campaign["execution_boundary"].items()
            if key != "this_commit_only_audits_and_preregisters"
        ),
    }
    if not all(checks.values()):
        raise ValueError("full-stack campaign checks differ")

    return {
        "schema_version": RECEIPT_SCHEMA,
        "campaign_id": campaign["campaign_id"],
        "identity": {
            "config_sha256": sha256_file(config_path),
            "base_revision": campaign["base_revision"],
            "artifacts": {
                artifact_id: row["sha256"]
                for artifact_id, row in sorted(artifacts.items())
            },
        },
        "inventory": {
            "models": models,
            "complete_benchmarks": complete_benchmarks,
            "formal_agent_benchmarks": formal_agent_benchmarks,
            "capabilities": capabilities,
            "peer_dependencies": peer_dependencies,
        },
        "skill_evolution": {
            "parent": {
                "passed": parent["passed"],
                "total": parent["total"],
                "registry_sha256": parent["registry_sha256"],
            },
            "candidate": {
                "passed": candidate["passed"],
                "total": candidate["total"],
                "registry_sha256": candidate["registry_sha256"],
            },
            "promotion": promotion,
        },
        "prior_evidence": prior_evidence,
        "candidate_ladder": campaign["candidate_ladder"],
        "acceptance": campaign["acceptance"],
        "resource_contract": campaign["resource_contract"],
        "readiness": readiness,
        "checks": checks,
        "execution_boundary": campaign["execution_boundary"],
        "claim_boundary": campaign["claim_boundary"],
        "next_executable_slice": {
            "stage_id": "matched-direct-complete-baselines",
            "action": (
                "Generate the frozen all-row case manifest and execute matched "
                "4B/9B direct baselines on GSM8K, MMLU, and GPQA-Diamond "
                "without consuming benchmark outputs as training data."
            ),
            "training_allowed": False,
            "rl_allowed": False,
            "opd_allowed": False,
            "twenty_seven_b_allowed": False,
        },
    }
