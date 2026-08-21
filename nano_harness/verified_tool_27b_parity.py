from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.request
from pathlib import Path
from typing import Any

from nano_harness.baseline import sha256_file
from nano_harness.client import OpenRouterClient
from nano_harness.config import ModelConfig
from nano_harness.verified_tool_execution import (
    build_cases,
    public_case_contract,
)
from nano_harness.verified_tool_execution_v2 import (
    load_config as load_source_config,
    parent_config,
)


CONFIG_SHA256 = (
    "71d0b98a6f7d857fba48588ce81ec0aed0d97f5862930869bcc283193c7f57f2"
)


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if sha256_file(config_path) != CONFIG_SHA256:
        raise ValueError("verified-tool 27B parity config SHA differs")
    if (
        config.get("schema_version")
        != "nano_harness_verified_tool_27b_parity_v1"
        or config.get("experiment_id")
        != "verified-tool-27b-parity-v1"
        or config.get("execution_boundary")
        != {
            "parity_generation_started": False,
            "four_b_harness_reused": True,
            "four_b_generation_repeated": False,
            "nine_b_generation_repeated": False,
            "suite_changed": False,
            "this_commit_only_preregisters": True,
            "training_started": False,
            "rl_or_opd_started": False,
        }
        or config.get("policy")
        != {
            "synthetic_evaluation_only": True,
            "training_eligible": False,
            "uses_benchmark_rows_or_outputs": False,
            "expected_answer_used_during_generation": False,
            "case_correctness_used_during_generation": False,
            "raw_outputs_committed": False,
            "post_observation_tuning_or_rerun": False,
        }
        or config.get("statistics", {}).get("noninferiority_margin") != 0.02
        or config.get("statistics", {}).get(
            "require_overall_and_every_family"
        )
        is not True
    ):
        raise ValueError("verified-tool 27B parity contract differs")
    return config


def load_source(config: dict[str, Any], root: Path) -> tuple[Any, dict]:
    source = config["source"]
    source_config_path = root / source["config_path"]
    raw_path = root / source["raw_path"]
    report_path = root / source["report_path"]
    if (
        sha256_file(source_config_path) != source["config_sha256"]
        or sha256_file(raw_path) != source["raw_sha256"]
        or sha256_file(report_path) != source["report_sha256"]
    ):
        raise ValueError("verified-tool 27B source identity differs")
    source_config = load_source_config(source_config_path)
    parent = parent_config(source_config)
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    cases = build_cases(parent)
    if (
        public_case_contract(cases)["case_contract_sha256"]
        != source["case_contract_sha256"]
        or raw.get("identity", {}).get("case_contract", {}).get(
            "case_contract_sha256"
        )
        != source["case_contract_sha256"]
        or len(raw.get("harness_rows", [])) != source["cases"]
        or sum(row["correct"] for row in raw["harness_rows"])
        != source["harness_correct"]
        or report.get("decision", {}).get("local_harness_admitted") is not True
        or report.get("decision", {}).get("benchmark_allowed") is not False
    ):
        raise ValueError("verified-tool 27B source contract differs")
    return parent, raw


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
        raise ValueError("verified-tool 27B service identity differs")
    serving = json.loads(serving_path.read_text(encoding="utf-8"))
    if (
        serving.get("decision", {}).get("bf16_tp2_service_ready") is not True
        or serving.get("decision", {}).get("parity_preregistration_allowed")
        is not True
        or serving.get("service", {}).get("max_model_len")
        != model["max_model_len"]
    ):
        raise ValueError("verified-tool 27B service gate differs")
    with urllib.request.urlopen(model["base_url"] + "/models", timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    rows = payload.get("data", [])
    if (
        len(rows) != 1
        or rows[0].get("id") != model["model"]
        or rows[0].get("max_model_len") != model["max_model_len"]
        or rows[0].get("owned_by") != "vllm"
    ):
        raise ValueError("verified-tool 27B live service differs")
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def select_shard(
    cases: list[dict[str, Any]],
    *,
    num_shards: int,
    shard_id: int,
) -> list[tuple[int, dict[str, Any]]]:
    if num_shards <= 0 or shard_id not in range(num_shards):
        raise ValueError("verified-tool 27B shard differs")
    return [
        (index, case)
        for index, case in enumerate(cases)
        if index % num_shards == shard_id
    ]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run(
    config_path: str | Path,
    *,
    num_shards: int,
    shard_id: int,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    root = config_path.parents[2]
    config = load_config(config_path)
    parent, source_raw = load_source(config, root)
    cases = build_cases(parent)
    harness_by_id = {
        row["case_id"]: row for row in source_raw["harness_rows"]
    }
    if set(harness_by_id) != {case["case_id"] for case in cases}:
        raise ValueError("verified-tool 4B harness case set differs")
    if (
        num_shards != config["execution"]["num_shards"]
        or config["execution"]["assignment"]
        != "sorted_case_index_mod_num_shards"
        or config["execution"]["merge_requires_exact_case_set"] is not True
    ):
        raise ValueError("verified-tool 27B execution differs")
    selected = select_shard(
        cases,
        num_shards=num_shards,
        shard_id=shard_id,
    )
    service_sha256 = verify_service(config, root)
    output_root = root / config["output_dir"]
    output_path = output_root / f"shard-{shard_id}.jsonl"
    completed = {row["case_id"] for row in read_jsonl(output_path)}
    expected_ids = {case["case_id"] for _, case in selected}
    if not completed.issubset(expected_ids):
        raise ValueError("verified-tool 27B output IDs differ")
    output_root.mkdir(parents=True, exist_ok=True)
    direct = config["direct"]
    client = OpenRouterClient(
        ModelConfig(
            name=config["twenty_seven_b"]["model"],
            base_url=config["twenty_seven_b"]["base_url"],
            api_key_env="NANO_HARNESS_API_KEY",
            temperature=direct["temperature"],
            max_tokens=direct["max_tokens"],
            timeout_seconds=300,
            max_retries=1,
            chat_template_kwargs=direct["chat_template_kwargs"],
        )
    )
    parser = re.compile(direct["parser_regex"])
    started = time.time()
    for case_index, case in selected:
        if case["case_id"] in completed:
            continue
        reply = client.complete(
            [
                {"role": "system", "content": direct["system"]},
                {"role": "user", "content": case["prompt"]},
            ],
            extra_body={
                "structured_outputs": {
                    "regex": direct["structured_output_regex"]
                },
                "seed": direct["seed_base"] + case_index,
            },
        )
        output = reply.content.strip()
        match = parser.fullmatch(output)
        prediction = int(match.group(1)) if match else None
        harness = harness_by_id[case["case_id"]]
        row = {
            "schema_version": "nano_harness_verified_tool_27b_case_v1",
            "case_id": case["case_id"],
            "family": case["family"],
            "four_b_harness": {
                "correct": bool(harness["correct"]),
                "prediction": harness["prediction"],
            },
            "twenty_seven_b_direct": {
                "model": config["twenty_seven_b"]["model"],
                "output": output,
                "prediction": prediction,
                "parseable": prediction is not None,
                "correct": prediction == case["expected"],
                "finish_reason": reply.raw["choices"][0]["finish_reason"],
                "usage": reply.usage,
            },
        }
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    rows = read_jsonl(output_path)
    if (
        len(rows) != len(selected)
        or {row["case_id"] for row in rows} != expected_ids
    ):
        raise ValueError("verified-tool 27B generation incomplete")
    result = {
        "schema_version": "nano_harness_verified_tool_27b_raw_v1",
        "experiment_id": config["experiment_id"],
        "identity": {
            "config_sha256": sha256_file(config_path),
            "raw_sha256": sha256_file(output_path),
            "service_sha256": service_sha256,
            "source_raw_sha256": config["source"]["raw_sha256"],
        },
        "surface": {
            "cases": len(rows),
            "num_shards": num_shards,
            "shard_id": shard_id,
            "four_b_generation_repeated": False,
            "nine_b_generation_repeated": False,
            "suite_changed": False,
        },
        "wall_seconds": time.time() - started,
    }
    (output_root / f"shard-{shard_id}.result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result
