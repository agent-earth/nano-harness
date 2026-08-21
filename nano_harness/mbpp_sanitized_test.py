from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from datasets import Dataset
from openai import OpenAI

from nano_harness.baseline import sha256_file
from nano_harness.mbpp_full_train_replication import POLICY_KEYS
from nano_harness.mbpp_iterative_repair import (
    generate_case,
    load_few_shots,
    read_jsonl,
    select_shard,
    verify_services,
)
from nano_harness.mbpp_verified_selection import MbppCase


CONFIG_SHA256 = (
    "f37ab18661ce84a8a7adec664ae8a6114ca745266108099a77916b37528bdc5b"
)


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if sha256_file(config_path) != CONFIG_SHA256:
        raise ValueError("MBPP sanitized-test config SHA differs")
    if (
        config.get("schema_version") != "nano_harness_mbpp_sanitized_test_v2"
        or config.get("experiment_id") != "mbpp-sanitized-test-v2"
        or config.get("execution_boundary")
        != {
            "test_generation_started": False,
            "sanitized_train_v2_rerun": False,
            "sanitized_validation_v1_rerun": False,
            "full_validation_confirmation_rerun": False,
            "full_train_replication_rerun": False,
            "this_commit_only_preregisters": True,
            "training_started": False,
            "rl_or_opd_started": False,
        }
        or config.get("policy")
        != {
            "reference_solution_used": False,
            "public_test_source_visible_to_model": True,
            "test_outcome_used_by_verifier": True,
            "train_rows_training_eligible": False,
            "validation_rows_training_eligible": False,
            "test_rows_training_eligible": False,
            "outputs_may_enter_training_reward_or_verifier": False,
            "raw_outputs_committed": False,
            "post_observation_tuning": False,
        }
    ):
        raise ValueError("MBPP sanitized-test contract differs")
    return config


def verify_unchanged_policy(config: dict[str, Any], root: Path) -> None:
    predecessor_path = root / config["predecessor"]["v2_config_path"]
    if (
        not predecessor_path.is_file()
        or sha256_file(predecessor_path)
        != config["predecessor"]["v2_config_sha256"]
    ):
        raise ValueError("MBPP v2 predecessor config identity differs")
    predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
    if any(config[key] != predecessor[key] for key in POLICY_KEYS):
        raise ValueError("MBPP sanitized-test policy differs from v2")


def load_test_cases(
    config: dict[str, Any],
    root: Path,
) -> list[MbppCase]:
    dataset = config["dataset"]
    path = (root / dataset["test_path"]).resolve()
    if sha256_file(path) != dataset["test_sha256"]:
        raise ValueError("MBPP sanitized-test dataset differs")
    rows = list(Dataset.from_parquet(str(path)))
    cases = [
        MbppCase(
            case_id=f"mbpp-sanitized-test-{int(row['task_id'])}",
            task_id=int(row["task_id"]),
            prompt=str(row["prompt"]),
            test_imports=tuple(str(value) for value in row["test_imports"]),
            test_list=tuple(str(value) for value in row["test_list"]),
        )
        for row in rows
    ]
    if (
        len(cases) != dataset["test_rows"]
        or len({case.case_id for case in cases}) != len(cases)
    ):
        raise ValueError("MBPP sanitized-test case identity differs")
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
    verify_unchanged_policy(config, root)
    predecessor_path = root / config["predecessor"]["replication_result_path"]
    if (
        not predecessor_path.is_file()
        or sha256_file(predecessor_path)
        != config["predecessor"]["replication_result_sha256"]
    ):
        raise ValueError("MBPP sanitized-test predecessor result differs")
    predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
    if (
        predecessor.get("decision", {}).get("replication_admitted")
        is not config["predecessor"]["replication_admitted"]
        or predecessor.get("decision", {}).get(
            "sanitized_test_preregistration_allowed"
        )
        is not True
    ):
        raise ValueError("MBPP sanitized-test predecessor gate differs")
    cases = load_test_cases(config, root)
    few_shots = load_few_shots(config, root)
    if (
        num_shards != config["execution"]["num_shards"]
        or config["execution"]["assignment"]
        != "sorted_case_index_mod_num_shards"
        or config["execution"]["merge_requires_exact_case_set"] is not True
    ):
        raise ValueError("MBPP sanitized-test execution differs")
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
        raise ValueError("MBPP sanitized-test output IDs differ")
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
        raise ValueError("MBPP sanitized-test generation incomplete")
    result = {
        "schema_version": "nano_harness_mbpp_sanitized_test_raw_v2",
        "experiment_id": config["experiment_id"],
        "identity": {
            "config_sha256": sha256_file(config_path),
            "raw_sha256": sha256_file(output_path),
            "service_sha256": service_sha,
        },
        "surface": {
            "split": "sanitized_test",
            "cases": len(rows),
            "num_shards": num_shards,
            "shard_id": shard_id,
            "sanitized_train_v2_rerun": False,
            "full_validation_confirmation_rerun": False,
            "full_train_replication_rerun": False,
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
