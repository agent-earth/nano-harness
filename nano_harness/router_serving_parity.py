from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nano_harness.baseline import sha256_file
from nano_harness.client import OpenRouterClient
from nano_harness.config import ModelConfig
from nano_harness.router_adapter_integration import _sha256_tree


CONFIG_SCHEMA = "nano_harness_router_serving_parity_v1"
RESULT_SCHEMA = "nano_harness_router_serving_parity_result_v1"


@dataclass(frozen=True)
class RouterServingParityConfig:
    schema_version: str
    experiment_id: str
    dataset_path: str
    dataset_sha256: str
    sft_report_path: str
    sft_report_sha256: str
    integration_report_path: str
    integration_report_sha256: str
    hf_generations_path: str
    hf_generations_sha256: str
    hf_reload_path: str
    hf_reload_sha256: str
    original_adapter_path: str
    original_adapter_tree_sha256: str
    original_adapter_weights_sha256: str
    remapped_adapter_path: str
    remapped_adapter_tree_sha256: str
    remapped_adapter_weights_sha256: str
    remap_receipt_path: str
    remap_receipt_sha256: str
    remap_converter_path: str
    remap_converter_sha256: str
    base_tokenizer_path: str
    base_tokenizer_json_sha256: str
    adapter_tokenizer_path: str
    adapter_tokenizer_json_sha256: str
    namespace_audit: dict[str, str]
    served_models: dict[str, str]
    service_launch: dict[str, Any]
    service_receipt_path: str
    output_path: str
    validation_rows: int
    system_prompt: str
    route_structured_output_regex: str
    generation_max_tokens: int
    temperature: float
    chat_template_kwargs: dict[str, bool]
    policy: dict[str, bool]
    execution_boundary: dict[str, bool]
    vllm_source_files: dict[str, str]


def load_config(path: str | Path) -> RouterServingParityConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(raw) != set(RouterServingParityConfig.__dataclass_fields__):
        raise ValueError("router serving parity config fields differ")
    config = RouterServingParityConfig(**raw)
    if config.schema_version != CONFIG_SCHEMA:
        raise ValueError("unsupported router serving parity schema")
    frozen = {
        "experiment_id": "qwen35-router-serving-parity-v1",
        "dataset_path": (
            "../nano-data-pipeline-fullstack-traex-03/datasets/"
            "qwen35_router_classification_v1.json"
        ),
        "dataset_sha256": (
            "dacd3663639fe9ddc054865b87afdd0c918f0fddb12c8c9355819d4bbce95d65"
        ),
        "sft_report_path": (
            "../nano-train-fullstack-traex-03/docs/results/"
            "qwen35_router_classification_sft_v1.public.json"
        ),
        "sft_report_sha256": (
            "c8af17cfa2fb77b594a9b34deaeccf27273491da6c350c3c5deb1435a9336c69"
        ),
        "integration_report_path": (
            "docs/results/qwen35_router_adapter_integration_v1.public.json"
        ),
        "integration_report_sha256": (
            "9b01a9b6d6011f657696b0cebf9de8853b16fd2406802b14ca203d3500288f70"
        ),
        "hf_generations_path": (
            "../nano-train-fullstack-traex-03/artifacts/"
            "qwen35-router-classification-sft-smoke-v1/generations.json"
        ),
        "hf_generations_sha256": (
            "226d8ed88197ebb847d216364f4e84b6654e69150c4fdca75867d4297d9dcfff"
        ),
        "hf_reload_path": (
            "../nano-train-fullstack-traex-03/artifacts/"
            "qwen35-router-classification-sft-smoke-v1/reload_validation.json"
        ),
        "hf_reload_sha256": (
            "645b61a679b213e6bdb86d9c884e47c25049ad671051bd026050833ad97fd316"
        ),
        "original_adapter_path": (
            "../nano-train-fullstack-traex-03/artifacts/"
            "qwen35-router-classification-sft-smoke-v1/adapter"
        ),
        "original_adapter_tree_sha256": (
            "48d72666cf269f64825791801c71aad5ae1d9e97cbc70574c216b12974e46e63"
        ),
        "original_adapter_weights_sha256": (
            "8dfe3a89bf20325a50b9a4c168c4e9039c5f7e84721dd8d5f8b21e3ad829b9ec"
        ),
        "remapped_adapter_path": "results/serving/qwen35-router-v1-remapped",
        "remapped_adapter_tree_sha256": (
            "fbaa39dcb3fcf34e9aab280308cb5a5416094c1968e4ac3a69cd739a806ecc49"
        ),
        "remapped_adapter_weights_sha256": (
            "9475d69207fa1db9b0106e420637c6f764d907baa2048c4b73f19773d6e2042b"
        ),
        "remap_receipt_path": (
            "results/serving/qwen35-router-v1-remapped.receipt.json"
        ),
        "remap_receipt_sha256": (
            "04ae7daa74eefbd891f4fb1ea9b4d29cc3cbcd59dbb22fcb6565488e5363cefb"
        ),
        "remap_converter_path": "scripts/build_qwen35_vllm_adapter.py",
        "remap_converter_sha256": (
            "355c25fd07a38aae456a01982351c6836fdceff2f12cacb146138d810d260702"
        ),
        "base_tokenizer_path": "../../../models/Qwen3.5-4B",
        "base_tokenizer_json_sha256": (
            "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
        ),
        "adapter_tokenizer_path": (
            "../nano-train-fullstack-traex-03/artifacts/"
            "qwen35-router-classification-sft-smoke-v1/adapter"
        ),
        "adapter_tokenizer_json_sha256": (
            "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"
        ),
        "namespace_audit": {
            "source_prefix": "base_model.model.model.layers.",
            "target_prefix": (
                "base_model.model.language_model.model.layers."
            ),
            "original_parsed_module": "model.layers.0.mlp.down_proj",
            "remapped_parsed_module": (
                "language_model.model.layers.0.mlp.down_proj"
            ),
            "vllm_text_module_prefix": "language_model.model.layers.",
        },
        "served_models": {
            "base": "qwen3.5-4b",
            "original": "qwen3.5-router-original-v1",
            "remapped": "qwen3.5-router-remapped-v1",
        },
        "service_launch": {
            "gpu_index": 0,
            "host": "127.0.0.1",
            "port": 8000,
            "base_url": "http://127.0.0.1:8000/v1",
            "vllm_version": "0.19.1",
            "dtype": "float16",
            "max_model_len": 4096,
            "gpu_memory_utilization": 0.85,
            "enforce_eager": True,
            "max_num_batched_tokens": 4096,
            "max_num_seqs": 1,
            "enable_lora": True,
            "max_lora_rank": 8,
            "max_loras": 2,
            "triton_libcuda_path": "/usr/lib/x86_64-linux-gnu",
        },
        "service_receipt_path": (
            "docs/experiments/qwen35_router_serving_parity_service_v1.public.json"
        ),
        "output_path": (
            "results/harness/qwen35-router-serving-parity-v1/result.json"
        ),
        "validation_rows": 192,
        "system_prompt": (
            "Classify the task for a semantic tool router. Return exactly one "
            "line: FINAL: A for implicit rectangular scale totals, FINAL: B "
            "for first strictly profitable whole periods, or FINAL: C for "
            "every unsupported task."
        ),
        "route_structured_output_regex": r"FINAL: [A-C]",
        "generation_max_tokens": 8,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
        "policy": {
            "diagnostic_only": True,
            "training_eligible": False,
            "contains_benchmark_rows": False,
            "contains_benchmark_outputs": False,
            "contains_canary_rows": False,
            "contains_holdout_rows": False,
            "contains_fresh_integration_rows": False,
            "post_observation_prompt_parser_budget_search": False,
        },
        "execution_boundary": {
            "parity_service_started": False,
            "model_generation_started": False,
            "evaluation_started": False,
            "benchmark_accessed": False,
            "canary_accessed": False,
            "holdout_accessed": False,
            "training_started": False,
            "this_commit_only_preregisters": True,
        },
        "vllm_source_files": {
            "../../.venv/lib/python3.11/site-packages/vllm/"
            "model_executor/models/qwen3_5.py": (
                "5ce2151170c38c2a2718e3a9386598c0660d30db5ab8844fd284e08633ebc904"
            ),
            "../../.venv/lib/python3.11/site-packages/vllm/"
            "model_executor/models/qwen3_vl.py": (
                "78ef9783baa036b471fe3773d709ecfb631523e17c6013a5b31ba659df86438a"
            ),
            "../../.venv/lib/python3.11/site-packages/vllm/"
            "lora/lora_model.py": (
                "5b8da6cd6dc5c63d2974ff2eec7157db6c8657d0aa21a93755c8345e03674a0d"
            ),
            "../../.venv/lib/python3.11/site-packages/vllm/lora/utils.py": (
                "f0fd4df62c028e2d408edf7020dae7374ff0c2f3d3f878315f96f869c26e65b4"
            ),
            "../../.venv/lib/python3.11/site-packages/vllm/"
            "lora/worker_manager.py": (
                "8d457a062d6e02b6630a6bfc721d70269afa56d1a97511d0e6b4b4946519fe85"
            ),
        },
    }
    for field, expected in frozen.items():
        if getattr(config, field) != expected:
            raise ValueError(f"router serving parity freezes {field}={expected}")
    for source, digest in (
        (config.dataset_path, config.dataset_sha256),
        (config.sft_report_path, config.sft_report_sha256),
        (config.integration_report_path, config.integration_report_sha256),
        (config.hf_generations_path, config.hf_generations_sha256),
        (config.hf_reload_path, config.hf_reload_sha256),
        (config.remap_receipt_path, config.remap_receipt_sha256),
        (config.remap_converter_path, config.remap_converter_sha256),
        (
            Path(config.base_tokenizer_path) / "tokenizer.json",
            config.base_tokenizer_json_sha256,
        ),
        (
            Path(config.adapter_tokenizer_path) / "tokenizer.json",
            config.adapter_tokenizer_json_sha256,
        ),
    ):
        if sha256_file(Path(source)) != digest:
            raise ValueError("router serving parity evidence identity differs")
    if any(
        sha256_file(Path(source)) != digest
        for source, digest in config.vllm_source_files.items()
    ):
        raise ValueError("router serving parity vLLM source identity differs")
    if (
        _sha256_tree(Path(config.original_adapter_path))
        != config.original_adapter_tree_sha256
        or _sha256_tree(Path(config.remapped_adapter_path))
        != config.remapped_adapter_tree_sha256
        or sha256_file(
            Path(config.original_adapter_path) / "adapter_model.safetensors"
        )
        != config.original_adapter_weights_sha256
        or sha256_file(
            Path(config.remapped_adapter_path) / "adapter_model.safetensors"
        )
        != config.remapped_adapter_weights_sha256
    ):
        raise ValueError("router serving parity adapter identity differs")
    receipt = json.loads(
        Path(config.remap_receipt_path).read_text(encoding="utf-8")
    )
    if (
        receipt.get("source_prefix") != config.namespace_audit["source_prefix"]
        or receipt.get("target_prefix")
        != config.namespace_audit["target_prefix"]
        or receipt.get("source_adapter_weights_sha256")
        != config.original_adapter_weights_sha256
        or receipt.get("serving_adapter_weights_sha256")
        != config.remapped_adapter_weights_sha256
        or receipt.get("tensor_count") != 224
        or receipt.get("remapped_key_count") != 224
        or receipt.get("tensor_content_hashes_match") is not True
    ):
        raise ValueError("router serving parity remap receipt differs")
    sft = json.loads(Path(config.sft_report_path).read_text(encoding="utf-8"))
    integration = json.loads(
        Path(config.integration_report_path).read_text(encoding="utf-8")
    )
    if (
        sft.get("validation", {}).get("post_exact") != 192
        or sft.get("identity", {}).get("adapter_sha256")
        != config.original_adapter_tree_sha256
        or integration.get("decision", {}).get(
            "adapter_integration_admitted"
        )
        is not False
        or integration.get("decision", {}).get(
            "question_only_scan_preregistration_allowed"
        )
        is not False
    ):
        raise ValueError("router serving parity predecessor decision differs")
    return config


def validation_cases(
    config: RouterServingParityConfig,
) -> list[dict[str, str]]:
    dataset = json.loads(Path(config.dataset_path).read_text(encoding="utf-8"))
    rows = [row for row in dataset["samples"] if row["split"] == "validation"]
    if len(rows) != config.validation_rows:
        raise ValueError("router serving parity validation count differs")
    cases = []
    for row in rows:
        messages = row["messages"]
        if (
            len(messages) != 3
            or messages[0]
            != {"role": "system", "content": config.system_prompt}
            or messages[1]["role"] != "user"
            or messages[2]["role"] != "assistant"
            or messages[2]["content"] != f"FINAL: {row['route_label']}"
        ):
            raise ValueError("router serving parity message contract differs")
        cases.append(
            {
                "sample_id": row["sample_id"],
                "task_family": row["task_family"],
                "label": row["route_label"],
                "prompt": messages[1]["content"],
                "target": messages[2]["content"],
            }
        )
    return cases


def case_contract(cases: list[dict[str, str]]) -> dict[str, Any]:
    rows = [
        {
            "sample_id": case["sample_id"],
            "task_family": case["task_family"],
            "label": case["label"],
            "prompt_sha256": hashlib.sha256(
                case["prompt"].encode()
            ).hexdigest(),
            "target": case["target"],
        }
        for case in cases
    ]
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return {
        "schema_version": "nano_harness_router_serving_parity_contract_v1",
        "cases": rows,
        "case_count": len(rows),
        "case_contract_sha256": hashlib.sha256(
            canonical.encode()
        ).hexdigest(),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_family = {}
    for family in ("router_a", "router_b", "router_c"):
        selected = [row for row in rows if row["task_family"] == family]
        by_family[family] = {
            "samples": len(selected),
            "exact": sum(row["exact"] for row in selected),
        }
    exact = sum(row["exact"] for row in rows)
    return {
        "samples": len(rows),
        "exact": exact,
        "accuracy": exact / len(rows),
        "by_family": by_family,
    }


def run(config: RouterServingParityConfig) -> dict[str, Any]:
    service = json.loads(
        Path(config.service_receipt_path).read_text(encoding="utf-8")
    )
    if (
        service.get("schema_version")
        != "nano_harness_router_serving_parity_service_v1"
        or service.get("healthy") is not True
        or service.get("generation_started") is not False
        or service.get("models") != config.served_models
        or service.get("original_adapter_sha256")
        != config.original_adapter_tree_sha256
        or service.get("remapped_adapter_sha256")
        != config.remapped_adapter_tree_sha256
    ):
        raise ValueError("router serving parity service receipt differs")
    cases = validation_cases(config)
    hf = json.loads(
        Path(config.hf_generations_path).read_text(encoding="utf-8")
    )["post_sft"]
    hf_by_id = {row["sample_id"]: row for row in hf}
    if (
        len(hf_by_id) != config.validation_rows
        or set(hf_by_id) != {case["sample_id"] for case in cases}
        or any(not row["exact"] for row in hf_by_id.values())
    ):
        raise ValueError("router serving parity HF generations differ")

    arms: dict[str, list[dict[str, Any]]] = {}
    for arm, model in config.served_models.items():
        client = OpenRouterClient(
            ModelConfig(
                name=model,
                base_url=config.service_launch["base_url"],
                api_key_env="NANO_HARNESS_API_KEY",
                temperature=config.temperature,
                max_tokens=config.generation_max_tokens,
                timeout_seconds=180.0,
                max_retries=3,
                chat_template_kwargs=config.chat_template_kwargs,
            )
        )
        arm_rows = []
        for case in cases:
            reply = client.complete(
                [
                    {"role": "system", "content": config.system_prompt},
                    {"role": "user", "content": case["prompt"]},
                ],
                extra_body={
                    "structured_outputs": {
                        "regex": config.route_structured_output_regex
                    }
                },
            )
            output = reply.content.strip()
            arm_rows.append(
                {
                    "sample_id": case["sample_id"],
                    "task_family": case["task_family"],
                    "target": case["target"],
                    "output": output,
                    "exact": output == case["target"],
                    "hf_output": hf_by_id[case["sample_id"]]["output"],
                    "hf_output_match": (
                        output == hf_by_id[case["sample_id"]]["output"]
                    ),
                    "usage": reply.usage,
                }
            )
        arms[arm] = arm_rows
    result = {
        "schema_version": RESULT_SCHEMA,
        "experiment_id": config.experiment_id,
        "config": config.__dict__,
        "identity": {
            "case_contract": case_contract(cases),
            "original_adapter_sha256": config.original_adapter_tree_sha256,
            "remapped_adapter_sha256": config.remapped_adapter_tree_sha256,
            "remap_receipt_sha256": config.remap_receipt_sha256,
            "hf_generations_sha256": config.hf_generations_sha256,
        },
        "summaries": {arm: summarize(rows) for arm, rows in arms.items()},
        "hf_reference": summarize(
            [
                {
                    "task_family": row["task_family"],
                    "exact": row["exact"],
                }
                for row in hf
            ]
        ),
        "hf_output_matches": {
            arm: sum(row["hf_output_match"] for row in rows)
            for arm, rows in arms.items()
        },
        "arms": arms,
        "service_receipt": service,
        "evaluation_boundary": {
            "training_eligible_cases": 0,
            "benchmark_rows_loaded": False,
            "benchmark_outputs_loaded": False,
            "canary_rows_loaded": False,
            "holdout_rows_loaded": False,
            "fresh_integration_rows_loaded": False,
            "only_observed_sft_validation_rows_loaded": True,
        },
    }
    output = Path(config.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result
