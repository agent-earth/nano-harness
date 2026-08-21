from __future__ import annotations

import hashlib
import json
import time
import urllib.request
from pathlib import Path
from typing import Any

from openai import OpenAI

from nano_harness.baseline import sha256_file
from nano_harness.mbpp_iterative_repair import (
    load_few_shots,
    read_jsonl,
    request_code,
    run_public_tests,
    select_shard,
    task_messages,
)
from nano_harness.mbpp_sanitized_test import load_test_cases


CONFIG_SHA256 = (
    "dbeccb9afc6acc4100c853a4e8bc52385e9fbc38d52204486e15296c286549e1"
)


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if sha256_file(config_path) != CONFIG_SHA256:
        raise ValueError("MBPP 27B parity config SHA differs")
    if (
        config.get("schema_version") != "nano_harness_mbpp_27b_parity_v1"
        or config.get("experiment_id") != "mbpp-27b-parity-v1"
        or config.get("execution_boundary")
        != {
            "parity_generation_started": False,
            "four_b_candidate_reused": True,
            "four_b_generation_repeated": False,
            "nine_b_generation_repeated": False,
            "test_policy_changed": False,
            "this_commit_only_preregisters": True,
            "training_started": False,
            "rl_or_opd_started": False,
        }
        or config.get("policy")
        != {
            "reference_solution_used": False,
            "public_test_source_visible_to_model": True,
            "test_outcome_used_only_for_final_scoring": True,
            "benchmark_rows_training_eligible": False,
            "benchmark_outputs_may_enter_training_reward_or_verifier": False,
            "raw_outputs_committed": False,
            "post_observation_tuning_or_rerun": False,
        }
        or config.get("statistics", {}).get("noninferiority_margin") != 0.02
    ):
        raise ValueError("MBPP 27B parity contract differs")
    return config


def load_policy_config(
    config: dict[str, Any],
    root: Path,
) -> dict[str, Any]:
    path = root / config["policy_source"]["config_path"]
    if (
        not path.is_file()
        or sha256_file(path) != config["policy_source"]["config_sha256"]
    ):
        raise ValueError("MBPP 27B parity policy identity differs")
    policy = json.loads(path.read_text(encoding="utf-8"))
    if policy.get("direct") != config["direct"]:
        raise ValueError("MBPP 27B parity direct policy differs")
    return policy


def verify_service(config: dict[str, Any], root: Path) -> str:
    model = config["twenty_seven_b"]
    model_path = Path(model["path"]).resolve()
    serving_path = root / model["serving_report_path"]
    if (
        sha256_file(model_path / "config.json")
        != model["model_config_sha256"]
        or sha256_file(model_path / "model.safetensors.index.json")
        != model["model_index_sha256"]
        or sha256_file(serving_path) != model["serving_report_sha256"]
    ):
        raise ValueError("MBPP 27B parity service identity differs")
    serving = json.loads(serving_path.read_text(encoding="utf-8"))
    if (
        serving.get("decision", {}).get("bf16_tp2_service_ready") is not True
        or serving.get("decision", {}).get("parity_preregistration_allowed")
        is not True
        or serving.get("service", {}).get("max_model_len")
        != model["max_model_len"]
    ):
        raise ValueError("MBPP 27B parity service gate differs")
    with urllib.request.urlopen(model["base_url"] + "/models", timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    rows = payload.get("data", [])
    if (
        len(rows) != 1
        or rows[0].get("id") != model["model"]
        or rows[0].get("max_model_len") != model["max_model_len"]
        or rows[0].get("owned_by") != "vllm"
    ):
        raise ValueError("MBPP 27B parity live service differs")
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_candidate_rows(
    config: dict[str, Any],
    root: Path,
) -> list[dict[str, Any]]:
    candidate = config["candidate"]
    raw_path = root / candidate["raw_path"]
    report_path = root / candidate["report_path"]
    policy_path = root / config["policy_source"]["config_path"]
    if (
        sha256_file(raw_path) != candidate["raw_sha256"]
        or sha256_file(report_path) != candidate["report_sha256"]
        or sha256_file(policy_path)
        != config["policy_source"]["config_sha256"]
    ):
        raise ValueError("MBPP 4B candidate identity differs")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("decision", {}).get("complete_benchmark_superiority")
        is not candidate["complete_benchmark_superiority"]
        or report.get("decision", {}).get("rerun_or_tuning_allowed")
        is not False
    ):
        raise ValueError("MBPP 4B candidate gate differs")
    rows = read_jsonl(raw_path)
    if len(rows) != config["dataset"]["test_rows"]:
        raise ValueError("MBPP 4B candidate row count differs")
    return rows


def run(
    config_path: str | Path,
    *,
    num_shards: int,
    shard_id: int,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    root = config_path.parents[2]
    config = load_config(config_path)
    policy_config = load_policy_config(config, root)
    cases = load_test_cases(config, root)
    candidate_rows = load_candidate_rows(config, root)
    candidate_by_id = {row["case_id"]: row for row in candidate_rows}
    if set(candidate_by_id) != {case.case_id for case in cases}:
        raise ValueError("MBPP 27B parity candidate case set differs")
    few_shots = load_few_shots(policy_config, root)
    if (
        num_shards != config["execution"]["num_shards"]
        or config["execution"]["assignment"]
        != "sorted_case_index_mod_num_shards"
        or config["execution"]["merge_requires_exact_case_set"] is not True
    ):
        raise ValueError("MBPP 27B parity execution differs")
    selected = select_shard(
        cases,
        num_shards=num_shards,
        shard_id=shard_id,
    )
    service_sha256 = verify_service(config, root)
    output_root = root / config["output_dir"]
    output_path = output_root / f"shard-{shard_id}.jsonl"
    completed = {row["case_id"] for row in read_jsonl(output_path)}
    expected_ids = {case.case_id for _, case in selected}
    if not completed.issubset(expected_ids):
        raise ValueError("MBPP 27B parity output IDs differ")
    output_root.mkdir(parents=True, exist_ok=True)
    client = OpenAI(
        api_key="local-vllm",
        base_url=config["twenty_seven_b"]["base_url"],
        timeout=300,
        max_retries=0,
    )
    started = time.time()
    for case_index, case in selected:
        if case.case_id in completed:
            continue
        reply = request_code(
            client,
            model=config["twenty_seven_b"]["model"],
            messages=task_messages(policy_config, case, few_shots),
            temperature=config["direct"]["temperature"],
            top_p=config["direct"]["top_p"],
            max_tokens=config["direct"]["max_tokens"],
            seed=config["direct"]["seed_base"] + case_index * 10,
        )
        test_result = run_public_tests(
            reply["code"], case, policy_config["sandbox"]
        )
        candidate = candidate_by_id[case.case_id]["candidate"]
        row = {
            "schema_version": "nano_harness_mbpp_27b_parity_case_v1",
            "case_id": case.case_id,
            "task_id": case.task_id,
            "candidate": {
                "correct": bool(candidate["test_result"]["full_pass"]),
                "test_result_sha256": hashlib.sha256(
                    json.dumps(
                        candidate["test_result"],
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest(),
            },
            "twenty_seven_b": {
                **reply,
                "test_result": test_result,
            },
        }
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    rows = read_jsonl(output_path)
    if (
        len(rows) != len(selected)
        or {row["case_id"] for row in rows} != expected_ids
    ):
        raise ValueError("MBPP 27B parity generation incomplete")
    result = {
        "schema_version": "nano_harness_mbpp_27b_parity_raw_v1",
        "experiment_id": config["experiment_id"],
        "identity": {
            "config_sha256": sha256_file(config_path),
            "raw_sha256": sha256_file(output_path),
            "service_sha256": service_sha256,
            "candidate_raw_sha256": config["candidate"]["raw_sha256"],
        },
        "surface": {
            "split": "sanitized_test",
            "cases": len(rows),
            "num_shards": num_shards,
            "shard_id": shard_id,
            "four_b_generation_repeated": False,
            "nine_b_generation_repeated": False,
            "test_policy_changed": False,
        },
        "wall_seconds": time.time() - started,
    }
    (output_root / f"shard-{shard_id}.result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def case_ids_sha256(case_ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(case_ids)).encode()).hexdigest()
