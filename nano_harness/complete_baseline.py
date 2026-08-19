from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from nano_harness.baseline import (
    load_cases,
    load_manifest,
    public_case_contract,
    select_case_shard,
    sha256_file,
)


CONFIG_SCHEMA = "nano_harness_complete_baseline_preregister_v1"
RECEIPT_SCHEMA = "nano_harness_complete_baseline_receipt_v1"


def _sha256_lines(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def load_config(path: str | Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = {
        "schema_version",
        "experiment_id",
        "suite_manifest_path",
        "suite_manifest_sha256",
        "dataset_root",
        "model_contracts",
        "serving",
        "execution",
        "historical_cost_sources",
        "uncertainty",
        "policy",
        "execution_boundary",
    }
    if set(config) != expected:
        raise ValueError("complete baseline config fields differ")
    if config["schema_version"] != CONFIG_SCHEMA:
        raise ValueError("unsupported complete baseline config schema")
    if config["execution"]["num_shards"] != 16:
        raise ValueError("complete baseline freezes 16 shards")
    if config["execution"]["merge_requires_exact_case_set"] is not True:
        raise ValueError("complete baseline merge must require exact case set")
    if config["uncertainty"] != {
        "bootstrap_samples": 10000,
        "bootstrap_seed": 20260820,
        "exact_mcnemar": True,
        "alpha": 0.05,
    }:
        raise ValueError("complete baseline uncertainty contract differs")
    boundary = {
        "this_commit_only_preregisters": True,
        "model_generation_started": False,
        "benchmark_scoring_started": False,
        "training_started": False,
        "rl_started": False,
        "opd_started": False,
    }
    if config["execution_boundary"] != boundary:
        raise ValueError("complete baseline execution boundary differs")
    if config["policy"] != {
        "benchmark_rows_training_eligible": False,
        "raw_outputs_committed": False,
        "benchmark_outputs_may_enter_sft_dpo_rl_reward_verifier": False,
        "post_observation_prompt_budget_parser_scorer_search": False,
    }:
        raise ValueError("complete baseline policy differs")
    model_ids = [row["model_id"] for row in config["model_contracts"]]
    if model_ids != ["qwen3.5-4b", "qwen3.5-9b"]:
        raise ValueError("complete baseline model order differs")
    if any(row["status"] != "ready" for row in config["model_contracts"]):
        raise ValueError("complete baseline models must be ready")
    if config["serving"] != {
        "vllm_version": "0.19.1",
        "dtype": "float16",
        "max_model_len": 4096,
        "gpu_memory_utilization": 0.85,
        "enforce_eager": True,
        "max_num_batched_tokens": 4096,
        "max_num_seqs": 1,
        "health_path": "/v1/models",
        "startup_timeout_seconds": 900,
    }:
        raise ValueError("complete baseline serving contract differs")
    return config


def _model_identity(root: Path, model: dict[str, Any]) -> dict[str, Any]:
    path = (root / model["path"]).resolve()
    config_sha = sha256_file(path / "config.json")
    index_sha = sha256_file(path / "model.safetensors.index.json")
    if (
        config_sha != model["config_sha256"]
        or index_sha != model["index_sha256"]
    ):
        raise ValueError(f"model identity mismatch: {model['model_id']}")
    return {
        "model_id": model["model_id"],
        "served_model_name": model["served_model_name"],
        "path": model["path"],
        "config_sha256": config_sha,
        "index_sha256": index_sha,
        "status": "ready",
    }


def _historical_costs(
    root: Path,
    sources: list[dict[str, Any]],
    target_counts: dict[str, int],
) -> dict[str, Any]:
    samples: dict[str, dict[str, list[float]]] = {}
    identities = {}
    for source in sources:
        path = (root / source["path"]).resolve()
        actual_sha = sha256_file(path)
        if actual_sha != source["sha256"]:
            raise ValueError(f"historical cost identity mismatch: {source['id']}")
        identities[source["id"]] = actual_sha
        document = json.loads(path.read_text(encoding="utf-8"))
        for model_key, model_cost in document["costs"].items():
            for benchmark, metrics in model_cost["by_benchmark"].items():
                cases = int(metrics["cases"])
                samples.setdefault(model_key, {}).setdefault(
                    benchmark, []
                ).append(float(metrics["wall_seconds"]) / cases)
    projections = {}
    for model_key, by_benchmark in samples.items():
        projections[model_key] = {}
        for benchmark, seconds_per_case in by_benchmark.items():
            low = min(seconds_per_case)
            high = max(seconds_per_case)
            rows = target_counts[benchmark]
            projections[model_key][benchmark] = {
                "historical_seconds_per_case_min": low,
                "historical_seconds_per_case_max": high,
                "projected_wall_seconds_min": low * rows,
                "projected_wall_seconds_max": high * rows,
            }
    total_per_model = {}
    for model_key, by_benchmark in projections.items():
        total_per_model[model_key] = {
            "projected_wall_seconds_min": sum(
                row["projected_wall_seconds_min"]
                for row in by_benchmark.values()
            ),
            "projected_wall_seconds_max": sum(
                row["projected_wall_seconds_max"]
                for row in by_benchmark.values()
            ),
        }
    return {
        "source_identities": identities,
        "by_model_and_benchmark": projections,
        "total_per_model": total_per_model,
        "scope": (
            "Historical single-request wall times are planning estimates only; "
            "actual complete-run times remain measured evidence."
        ),
    }


def build_receipt(
    config_path: str | Path,
    *,
    repository_root: str | Path | None = None,
    tokenizer: Any | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from transformers import AutoTokenizer

    config_path = Path(config_path).resolve()
    root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else config_path.parents[2]
    )
    config = load_config(config_path)
    suite_path = (root / config["suite_manifest_path"]).resolve()
    if sha256_file(suite_path) != config["suite_manifest_sha256"]:
        raise ValueError("complete suite manifest identity mismatch")
    manifest = load_manifest(suite_path)
    if (
        manifest.case_id_policy != "row_stable_v2"
        or manifest.strategy != "direct"
        or manifest.temperature != 0
        or manifest.chat_template_kwargs != {"enable_thinking": False}
    ):
        raise ValueError("complete suite inference contract differs")
    cases = load_cases(manifest, (root / config["dataset_root"]).resolve())
    public_cases = public_case_contract(cases)
    expected_total = sum(spec.limit for spec in manifest.datasets)
    if len(cases) != expected_total or len({case.case_id for case in cases}) != len(
        cases
    ):
        raise ValueError("complete suite row identity differs")
    benchmark_counts = Counter(case.benchmark for case in cases)
    expected_counts = {
        spec.name: spec.limit for spec in manifest.datasets
    }
    if dict(benchmark_counts) != expected_counts:
        raise ValueError("complete suite benchmark counts differ")

    shards = [
        select_case_shard(
            cases,
            num_shards=config["execution"]["num_shards"],
            shard_id=shard_id,
        )
        for shard_id in range(config["execution"]["num_shards"])
    ]
    shard_case_ids = [case.case_id for shard in shards for case in shard]
    if (
        len(shard_case_ids) != len(cases)
        or len(set(shard_case_ids)) != len(cases)
        or set(shard_case_ids) != {case.case_id for case in cases}
    ):
        raise ValueError("complete suite shard coverage differs")

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            (root / config["model_contracts"][0]["path"]).resolve(),
            local_files_only=True,
        )
    input_lengths = {}
    total_lengths = {}
    for case in cases:
        text = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": case.system_prompt},
                {"role": "user", "content": case.prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
            **manifest.chat_template_kwargs,
        )
        input_length = len(tokenizer.encode(text))
        input_lengths.setdefault(case.benchmark, []).append(input_length)
        total_lengths.setdefault(case.benchmark, []).append(
            input_length + case.max_tokens
        )
    context = {}
    for benchmark in sorted(input_lengths):
        sorted_inputs = sorted(input_lengths[benchmark])
        context[benchmark] = {
            "input_max": max(sorted_inputs),
            "input_p99": sorted_inputs[
                min(
                    len(sorted_inputs) - 1,
                    math.floor(len(sorted_inputs) * 0.99),
                )
            ],
            "input_plus_budget_max": max(total_lengths[benchmark]),
        }
        if (
            context[benchmark]["input_plus_budget_max"]
            > config["serving"]["max_model_len"]
        ):
            raise ValueError(f"{benchmark} exceeds serving context")

    models = [
        _model_identity(root, model)
        for model in config["model_contracts"]
    ]
    costs = _historical_costs(
        root,
        config["historical_cost_sources"],
        expected_counts,
    )
    output_paths = {
        model["model_id"]: {
            "shard_pattern": config["execution"]["raw_output_patterns"][
                model["model_id"]
            ],
            "merged": config["execution"]["merged_outputs"][
                model["model_id"]
            ],
        }
        for model in config["model_contracts"]
    }
    serving_commands = {
        model["model_id"]: [
            "env",
            f"CUDA_VISIBLE_DEVICES={model['gpu_index']}",
            config["execution"]["vllm_binary"],
            "serve",
            model["path"],
            "--host",
            model["host"],
            "--port",
            str(model["port"]),
            "--served-model-name",
            model["served_model_name"],
            "--dtype",
            config["serving"]["dtype"],
            "--max-model-len",
            str(config["serving"]["max_model_len"]),
            "--gpu-memory-utilization",
            str(config["serving"]["gpu_memory_utilization"]),
            "--enforce-eager",
            "--max-num-batched-tokens",
            str(config["serving"]["max_num_batched_tokens"]),
            "--max-num-seqs",
            str(config["serving"]["max_num_seqs"]),
        ]
        for model in config["model_contracts"]
    }
    run_commands = {
        model["model_id"]: [
            config["execution"]["python_binary"],
            "-m",
            "nano_harness.cli",
            "baseline",
            "--manifest",
            config["suite_manifest_path"],
            "--dataset-root",
            config["dataset_root"],
            "--model",
            model["served_model_name"],
            "--base-url",
            f"http://{model['host']}:{model['port']}/v1",
            "--output",
            config["execution"]["raw_output_template"][
                model["model_id"]
            ],
            "--num-shards",
            str(config["execution"]["num_shards"]),
            "--shard-id",
            "<shard_id>",
        ]
        for model in config["model_contracts"]
    }
    checks = {
        "suite_identity_pass": True,
        "all_dataset_rows_included": len(cases) == 15559,
        "case_ids_unique": len({case.case_id for case in cases}) == len(cases),
        "row_stable_identity_enabled": manifest.case_id_policy == "row_stable_v2",
        "all_shards_disjoint_and_complete": len(set(shard_case_ids)) == len(cases),
        "context_4096_pass": all(
            row["input_plus_budget_max"]
            <= config["serving"]["max_model_len"]
            for row in context.values()
        ),
        "models_ready": all(model["status"] == "ready" for model in models),
        "policy_fail_closed": all(
            value is False
            for value in config["policy"].values()
        ),
        "no_execution_started": all(
            value is False
            for key, value in config["execution_boundary"].items()
            if key != "this_commit_only_preregisters"
        ),
    }
    if not all(checks.values()):
        raise ValueError("complete baseline preregistration checks differ")
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "experiment_id": config["experiment_id"],
        "identity": {
            "config_sha256": sha256_file(config_path),
            "suite_manifest_sha256": config["suite_manifest_sha256"],
            "case_contract_sha256": hashlib.sha256(
                json.dumps(
                    public_cases,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "case_ids_sha256": _sha256_lines(
                sorted(case.case_id for case in cases)
            ),
            "models": models,
        },
        "cases": {
            "total": len(cases),
            "by_benchmark": dict(benchmark_counts),
            "case_id_policy": manifest.case_id_policy,
            "public_contract_excludes": [
                "task prompts",
                "reference answers",
                "model outputs",
            ],
        },
        "sharding": {
            "num_shards": config["execution"]["num_shards"],
            "algorithm": "sha256(case_id) mod num_shards",
            "counts": [len(shard) for shard in shards],
            "minimum": min(len(shard) for shard in shards),
            "maximum": max(len(shard) for shard in shards),
            "merge_requires_exact_case_set": True,
        },
        "context": {
            "max_model_len": config["serving"]["max_model_len"],
            "by_benchmark": context,
        },
        "commands": {
            "serving": serving_commands,
            "health": {
                model["model_id"]: [
                    "curl",
                    "-fsS",
                    f"http://{model['host']}:{model['port']}"
                    + config["serving"]["health_path"],
                ]
                for model in config["model_contracts"]
            },
            "run_shard": run_commands,
            "output_paths": output_paths,
        },
        "resume": {
            "unit": "stable case_id",
            "existing_completed_case_ids_skipped": True,
            "api_error_rows_retried": True,
            "merge_requires_all_expected_case_ids": True,
        },
        "historical_cost_projection": costs,
        "uncertainty": config["uncertainty"],
        "policy": config["policy"],
        "checks": checks,
        "execution_boundary": config["execution_boundary"],
        "claim_boundary": (
            "This receipt freezes the complete matched direct baseline. It "
            "does not start serving, generation, scoring, training, RL, or "
            "OPD and it does not establish a new model-quality result."
        ),
    }
    return receipt, public_cases
