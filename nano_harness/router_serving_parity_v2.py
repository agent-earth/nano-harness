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


CONFIG_SCHEMA = "nano_harness_router_serving_parity_v2"
RESULT_SCHEMA = "nano_harness_router_serving_parity_result_v2"
CONFIG_SHA256 = (
    "10f351e9e7209695a5a66f79b18a7265a7da3c66410be21fef584f0474e28b3b"
)


@dataclass(frozen=True)
class RouterServingParityV2Config:
    adapter_tokenizer_json_sha256: str
    adapter_tokenizer_path: str
    base_tokenizer_json_sha256: str
    base_tokenizer_path: str
    chat_template_kwargs: dict[str, bool]
    dataset_path: str
    dataset_sha256: str
    execution_boundary: dict[str, bool]
    experiment_id: str
    generation_max_tokens: int
    hf_generations_path: str
    hf_generations_sha256: str
    hf_reload_path: str
    hf_reload_sha256: str
    namespace_audit: dict[str, str]
    namespace_root_cause_report_path: str
    namespace_root_cause_report_sha256: str
    original_adapter_path: str
    original_adapter_tree_sha256: str
    original_adapter_weights_sha256: str
    output_path: str
    policy: dict[str, bool]
    remap_converter_path: str
    remap_converter_sha256: str
    remap_receipt_path: str
    remap_receipt_sha256: str
    remapped_adapter_path: str
    remapped_adapter_tree_sha256: str
    remapped_adapter_weights_sha256: str
    route_structured_output_regex: str
    schema_version: str
    served_models: dict[str, str]
    service_launch: dict[str, Any]
    service_receipt_path: str
    sft_report_path: str
    sft_report_sha256: str
    system_prompt: str
    temperature: float
    validation_rows: int
    vllm_source_files: dict[str, str]


def load_config(path: str | Path) -> RouterServingParityV2Config:
    config_path = Path(path)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if set(raw) != set(RouterServingParityV2Config.__dataclass_fields__):
        raise ValueError("router serving parity v2 config fields differ")
    if sha256_file(config_path) != CONFIG_SHA256:
        raise ValueError("router serving parity v2 config SHA differs")
    config = RouterServingParityV2Config(**raw)
    if config.schema_version != CONFIG_SCHEMA:
        raise ValueError("unsupported router serving parity v2 schema")
    for source, digest in (
        (config.dataset_path, config.dataset_sha256),
        (config.sft_report_path, config.sft_report_sha256),
        (
            config.namespace_root_cause_report_path,
            config.namespace_root_cause_report_sha256,
        ),
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
            raise ValueError("router serving parity v2 evidence identity differs")
    if any(
        sha256_file(Path(source)) != digest
        for source, digest in config.vllm_source_files.items()
    ):
        raise ValueError("router serving parity v2 vLLM identity differs")
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
        raise ValueError("router serving parity v2 adapter identity differs")
    remap = json.loads(
        Path(config.remap_receipt_path).read_text(encoding="utf-8")
    )
    if (
        remap.get("source_prefix") != config.namespace_audit["source_prefix"]
        or remap.get("target_prefix")
        != config.namespace_audit["target_prefix"]
        or remap.get("source_adapter_weights_sha256")
        != config.original_adapter_weights_sha256
        or remap.get("serving_adapter_weights_sha256")
        != config.remapped_adapter_weights_sha256
        or remap.get("tensor_count") != 224
        or remap.get("remapped_key_count") != 224
        or remap.get("tensor_content_hashes_match") is not True
    ):
        raise ValueError("router serving parity v2 remap receipt differs")
    sft = json.loads(Path(config.sft_report_path).read_text(encoding="utf-8"))
    root_cause = json.loads(
        Path(config.namespace_root_cause_report_path).read_text(encoding="utf-8")
    )
    reload = json.loads(
        Path(config.hf_reload_path).read_text(encoding="utf-8")
    )
    if (
        sft.get("validation", {}).get("post_exact") != 1536
        or sft.get("identity", {}).get("adapter_sha256")
        != config.original_adapter_tree_sha256
        or sft.get("decision", {}).get(
            "serving_parity_preregistration_allowed"
        )
        is not True
        or root_cause.get("decision", {}).get(
            "serving_namespace_root_cause_supported"
        )
        is not True
        or reload.get("reload_success") is not True
        or reload.get("metrics_exact") is not True
        or reload.get("generations_exact") is not True
        or reload.get("adapter_sha256")
        != config.original_adapter_tree_sha256
    ):
        raise ValueError("router serving parity v2 predecessor decision differs")
    return config


def validation_cases(
    config: RouterServingParityV2Config,
) -> list[dict[str, str | None]]:
    dataset = json.loads(Path(config.dataset_path).read_text(encoding="utf-8"))
    rows = [row for row in dataset["samples"] if row["split"] == "validation"]
    if len(rows) != config.validation_rows:
        raise ValueError("router serving parity v2 validation count differs")
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
            raise ValueError("router serving parity v2 message contract differs")
        cases.append(
            {
                "sample_id": row["sample_id"],
                "task_family": row["task_family"],
                "negative_subtype": row["negative_subtype"],
                "label": row["route_label"],
                "prompt": messages[1]["content"],
                "target": messages[2]["content"],
            }
        )
    return cases


def case_contract(cases: list[dict[str, str | None]]) -> dict[str, Any]:
    rows = [
        {
            "sample_id": case["sample_id"],
            "task_family": case["task_family"],
            "negative_subtype": case["negative_subtype"],
            "label": case["label"],
            "prompt_sha256": hashlib.sha256(
                str(case["prompt"]).encode()
            ).hexdigest(),
            "target": case["target"],
        }
        for case in cases
    ]
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return {
        "schema_version": "nano_harness_router_serving_parity_contract_v2",
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
            "hf_output_matches": sum(
                row["hf_output_match"] for row in selected
            ),
        }
    c_by_subtype = {}
    for subtype in sorted(
        {
            str(row["negative_subtype"])
            for row in rows
            if row["task_family"] == "router_c"
        }
    ):
        selected = [
            row for row in rows if row["negative_subtype"] == subtype
        ]
        c_by_subtype[subtype] = {
            "samples": len(selected),
            "exact": sum(row["exact"] for row in selected),
            "hf_output_matches": sum(
                row["hf_output_match"] for row in selected
            ),
        }
    exact = sum(row["exact"] for row in rows)
    matches = sum(row["hf_output_match"] for row in rows)
    return {
        "samples": len(rows),
        "exact": exact,
        "accuracy": exact / len(rows),
        "hf_output_matches": matches,
        "by_family": by_family,
        "c_by_subtype": c_by_subtype,
    }


def run(config: RouterServingParityV2Config) -> dict[str, Any]:
    service = json.loads(
        Path(config.service_receipt_path).read_text(encoding="utf-8")
    )
    if (
        service.get("schema_version")
        != "nano_harness_router_serving_parity_service_v2"
        or service.get("healthy") is not True
        or service.get("generation_started") is not False
        or service.get("models") != config.served_models
        or service.get("remapped_adapter_sha256")
        != config.remapped_adapter_tree_sha256
    ):
        raise ValueError("router serving parity v2 service receipt differs")
    cases = validation_cases(config)
    hf = json.loads(
        Path(config.hf_generations_path).read_text(encoding="utf-8")
    )["post_sft"]
    hf_by_id = {row["sample_id"]: row for row in hf}
    if (
        len(hf_by_id) != config.validation_rows
        or set(hf_by_id) != {str(case["sample_id"]) for case in cases}
        or any(not row["exact"] for row in hf_by_id.values())
    ):
        raise ValueError("router serving parity v2 HF generations differ")
    client = OpenRouterClient(
        ModelConfig(
            name=config.served_models["remapped"],
            base_url=config.service_launch["base_url"],
            api_key_env="NANO_HARNESS_API_KEY",
            temperature=config.temperature,
            max_tokens=config.generation_max_tokens,
            timeout_seconds=180.0,
            max_retries=3,
            chat_template_kwargs=config.chat_template_kwargs,
        )
    )
    rows = []
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
        hf_output = hf_by_id[str(case["sample_id"])]["output"]
        rows.append(
            {
                "sample_id": case["sample_id"],
                "task_family": case["task_family"],
                "negative_subtype": case["negative_subtype"],
                "target": case["target"],
                "output": output,
                "exact": output == case["target"],
                "hf_output": hf_output,
                "hf_output_match": output == hf_output,
                "usage": reply.usage,
            }
        )
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
        "summary": summarize(rows),
        "rows": rows,
        "service_receipt": service,
        "evaluation_boundary": {
            "training_eligible_cases": 0,
            "benchmark_rows_loaded": False,
            "benchmark_outputs_loaded": False,
            "canary_rows_loaded": False,
            "holdout_rows_loaded": False,
            "fresh_integration_rows_loaded": False,
            "fresh_integration_outputs_loaded": False,
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
