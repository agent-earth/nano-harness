from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from datasets import Dataset
from openai import OpenAI

from nano_harness.baseline import sha256_file
from nano_harness.mbpp_iterative_repair import (
    generate_case,
    load_few_shots,
    read_jsonl,
    select_shard,
    verify_services,
)
from nano_harness.mbpp_verified_selection import MbppCase


CONFIG_SHA256 = (
    "43675163a1d4fb576404f70a13f62e74e90b7dea275f56f06fe5dc6c118db19b"
)


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if sha256_file(config_path) != CONFIG_SHA256:
        raise ValueError("MBPP confirmation config SHA differs")
    if (
        config.get("schema_version")
        != "nano_harness_mbpp_confirmation_v2"
        or config.get("experiment_id")
        != "mbpp-full-validation-confirmation-v2"
        or config.get("execution_boundary")
        != {
            "confirmation_generation_started": False,
            "sanitized_validation_v1_rerun": False,
            "train_v2_rerun": False,
            "test_generation_started": False,
            "this_commit_only_preregisters": True,
            "training_started": False,
            "rl_or_opd_started": False,
        }
        or config.get("policy")
        != {
            "reference_solution_used": False,
            "public_test_source_visible_to_model": True,
            "test_outcome_used_by_verifier": True,
            "confirmation_rows_training_eligible": False,
            "test_rows_training_eligible": False,
            "outputs_may_enter_training_reward_or_verifier": False,
            "raw_outputs_committed": False,
            "post_observation_tuning": False,
        }
    ):
        raise ValueError("MBPP confirmation contract differs")
    return config


def load_confirmation_cases(
    config: dict[str, Any],
    root: Path,
) -> list[MbppCase]:
    dataset = config["dataset"]
    full_path = (root / dataset["full_validation_path"]).resolve()
    excluded_path = (
        root / dataset["excluded_sanitized_validation_path"]
    ).resolve()
    if (
        sha256_file(full_path) != dataset["full_validation_sha256"]
        or sha256_file(excluded_path)
        != dataset["excluded_sanitized_validation_sha256"]
    ):
        raise ValueError("MBPP confirmation dataset differs")
    excluded = {
        int(row["task_id"])
        for row in Dataset.from_parquet(str(excluded_path))
    }
    rows = [
        row
        for row in Dataset.from_parquet(str(full_path))
        if int(row["task_id"]) not in excluded
    ]
    cases = [
        MbppCase(
            case_id=f"mbpp-full-validation-{int(row['task_id'])}",
            task_id=int(row["task_id"]),
            prompt=str(row["text"]),
            test_imports=(
                (str(row["test_setup_code"]).strip(),)
                if str(row["test_setup_code"]).strip()
                else ()
            ),
            test_list=tuple(str(value) for value in row["test_list"]),
        )
        for row in rows
    ]
    if (
        len(excluded) != dataset["excluded_rows"]
        or len(cases) != dataset["confirmation_rows"]
        or len({case.case_id for case in cases}) != len(cases)
    ):
        raise ValueError("MBPP confirmation case identity differs")
    return sorted(cases, key=lambda case: case.case_id)


def run(
    config_path: str | Path,
    *,
    num_shards: int,
    shard_id: int,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    root = config_path.parents[2]
    config = load_config(config_path)
    predecessor_path = root / config["predecessor"]["v2_result_path"]
    if (
        sha256_file(predecessor_path)
        != config["predecessor"]["v2_result_sha256"]
        or json.loads(predecessor_path.read_text(encoding="utf-8"))
        .get("decision", {})
        .get("method_supported")
        is not config["predecessor"]["method_supported"]
    ):
        raise ValueError("MBPP confirmation predecessor differs")
    test_path = (root / config["dataset"]["test_path"]).resolve()
    if sha256_file(test_path) != config["dataset"]["test_sha256"]:
        raise ValueError("MBPP confirmation test identity differs")
    cases = load_confirmation_cases(config, root)
    few_shots = load_few_shots(config, root)
    if (
        num_shards != config["execution"]["num_shards"]
        or config["execution"]["assignment"]
        != "sorted_case_index_mod_num_shards"
        or config["execution"]["merge_requires_exact_case_set"] is not True
    ):
        raise ValueError("MBPP confirmation execution differs")
    selected = select_shard(
        cases,
        num_shards=num_shards,
        shard_id=shard_id,
    )
    service_sha = verify_services(config)
    output_root = root / config["output_dir"]
    output_path = output_root / f"shard-{shard_id}.jsonl"
    completed = {row["case_id"] for row in read_jsonl(output_path)}
    expected_ids = {case.case_id for _, case in selected}
    if not completed.issubset(expected_ids):
        raise ValueError("MBPP confirmation output IDs differ")
    output_root.mkdir(parents=True, exist_ok=True)
    clients = {
        name: OpenAI(
            api_key="local-vllm",
            base_url=model["base_url"],
            timeout=240,
            max_retries=0,
        )
        for name, model in config["models"].items()
    }
    started = time.time()
    for case_index, case in selected:
        if case.case_id in completed:
            continue
        row = generate_case(
            config,
            case,
            few_shots,
            four_client=clients["four_b"],
            nine_client=clients["nine_b"],
            case_index=case_index,
        )
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    rows = read_jsonl(output_path)
    if (
        len(rows) != len(selected)
        or {row["case_id"] for row in rows} != expected_ids
    ):
        raise ValueError("MBPP confirmation generation incomplete")
    result = {
        "schema_version": "nano_harness_mbpp_confirmation_raw_v2",
        "experiment_id": config["experiment_id"],
        "identity": {
            "config_sha256": sha256_file(config_path),
            "raw_sha256": sha256_file(output_path),
            "service_sha256": service_sha,
        },
        "surface": {
            "split": "full_validation_minus_sanitized_validation",
            "cases": len(rows),
            "num_shards": num_shards,
            "shard_id": shard_id,
            "sanitized_validation_v1_rerun": False,
            "train_v2_rerun": False,
            "test_generation_started": False,
        },
        "wall_seconds": time.time() - started,
    }
    (output_root / f"shard-{shard_id}.result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def case_ids_sha256(cases: list[MbppCase]) -> str:
    return hashlib.sha256(
        "\n".join(case.case_id for case in cases).encode()
    ).hexdigest()
