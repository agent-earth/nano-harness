from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from nano_harness.client import OpenRouterClient
from nano_harness.config import ModelConfig


@dataclass(frozen=True)
class AnalogContractConfig:
    schema_version: str
    experiment_id: str
    model: str
    base_url: str
    dataset_path: str
    dataset_sha256: str
    serving_receipt_path: str
    serving_receipt_sha256: str
    serving_adapter_weights_sha256: str
    output_path: str
    temperature: float
    direct_max_tokens: int
    calculation_max_tokens: int
    selector_max_tokens: int
    selector_regex: str
    chat_template_kwargs: dict[str, Any]
    calculation_system_prompt: str
    selector_system_prompt: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: str | Path) -> AnalogContractConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = set(AnalogContractConfig.__dataclass_fields__)
    if set(raw) != expected:
        raise ValueError("analog contract config fields differ")
    config = AnalogContractConfig(**raw)
    frozen = {
        "schema_version": "nano_harness_analog_contract_v1",
        "experiment_id": "anchored-v1-choice-calculation-selector-v1",
        "temperature": 0.0,
        "direct_max_tokens": 128,
        "calculation_max_tokens": 128,
        "selector_max_tokens": 8,
        "selector_regex": r"FINAL: [A-D]",
        "chat_template_kwargs": {"enable_thinking": False},
    }
    for field, expected_value in frozen.items():
        if getattr(config, field) != expected_value:
            raise ValueError(f"analog contract freezes {field}={expected_value}")
    return config


def load_dataset(config: AnalogContractConfig) -> dict[str, Any]:
    path = Path(config.dataset_path)
    if sha256_file(path) != config.dataset_sha256:
        raise ValueError("analog contract dataset identity mismatch")
    dataset = json.loads(path.read_text(encoding="utf-8"))
    if dataset.get("dataset_id") != "generic-choice-replay-v11":
        raise ValueError("analog contract requires generic choice replay v11")
    policy = dataset.get("policy", {})
    if (
        policy.get("contains_benchmark_content") is not False
        or policy.get("contains_model_outputs") is not False
        or policy.get("contains_teacher_outputs") is not False
        or policy.get("sealed_canary_used_for_training") is not False
        or policy.get("independent_holdout_used_for_training") is not False
    ):
        raise ValueError("analog contract dataset boundary differs")
    validation = [
        sample
        for sample in dataset.get("samples", [])
        if sample.get("split") == "validation"
    ]
    if len(validation) != 32:
        raise ValueError("analog contract requires 32 development rows")
    if sum(
        sample.get("format_family") == "final_choice"
        for sample in validation
    ) != 8:
        raise ValueError("analog contract requires 8 choice rows")
    return dataset


def validate_serving_receipt(config: AnalogContractConfig) -> dict[str, Any]:
    path = Path(config.serving_receipt_path)
    if sha256_file(path) != config.serving_receipt_sha256:
        raise ValueError("serving receipt identity mismatch")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if (
        receipt.get("schema_version")
        != "qwen35_vllm_adapter_namespace_receipt_v1"
        or receipt.get("serving_adapter_weights_sha256")
        != config.serving_adapter_weights_sha256
        or receipt.get("tensor_count") != 224
        or receipt.get("remapped_key_count") != 224
        or receipt.get("tensor_content_hashes_match") is not True
    ):
        raise ValueError("serving receipt contract differs")
    return receipt


def _sum_usage(*items: dict[str, Any]) -> dict[str, Any]:
    keys = {
        key
        for item in items
        for key, value in item.items()
        if isinstance(value, (int, float))
    }
    return {
        key: sum(
            float(item.get(key, 0))
            for item in items
            if isinstance(item.get(key, 0), (int, float))
        )
        for key in sorted(keys)
    }


def run_choice_calculation_selector(
    sample: dict[str, Any],
    config: AnalogContractConfig,
    calculation_client: Any,
    selector_client: Any,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    if sample.get("format_family") != "final_choice":
        raise ValueError("calculation selector requires final_choice")
    messages = sample["messages"]
    original_system = str(messages[0]["content"])
    original_task = str(messages[1]["content"])
    calculation = calculation_client.complete(
        [
            {
                "role": "system",
                "content": config.calculation_system_prompt,
            },
            {
                "role": "user",
                "content": (
                    f"<original_system>\n{original_system}\n</original_system>\n\n"
                    f"<original_task>\n{original_task}\n</original_task>"
                ),
            },
        ]
    )
    selector_input = (
        f"<original_task>\n{original_task}\n</original_task>\n\n"
        f"<calculation>\n{calculation.content}\n</calculation>"
    )
    selector = selector_client.complete(
        [
            {
                "role": "system",
                "content": config.selector_system_prompt,
            },
            {"role": "user", "content": selector_input},
        ],
        extra_body={
            "structured_outputs": {"regex": config.selector_regex}
        },
    )
    stages = {
        "calculation": {
            "max_tokens": config.calculation_max_tokens,
            "input_sha256": hashlib.sha256(
                original_task.encode()
            ).hexdigest(),
            "output": calculation.content,
            "output_sha256": hashlib.sha256(
                calculation.content.encode()
            ).hexdigest(),
            "usage": calculation.usage,
        },
        "selector": {
            "max_tokens": config.selector_max_tokens,
            "input_sha256": hashlib.sha256(
                selector_input.encode()
            ).hexdigest(),
            "structured_outputs": {"regex": config.selector_regex},
            "usage": selector.usage,
        },
    }
    return selector.content, _sum_usage(
        calculation.usage,
        selector.usage,
    ), stages


def _semantic_validator() -> Callable[[dict[str, Any], str], bool]:
    try:
        from nano_train.data import TokenizedSample, semantic_output_valid
    except ImportError as error:
        raise RuntimeError(
            "analog contract evaluation requires nano-train on PYTHONPATH"
        ) from error

    def validate(sample: dict[str, Any], output: str) -> bool:
        target = str(sample["messages"][-1]["content"])
        tokenized = TokenizedSample(
            sample_id=str(sample["sample_id"]),
            split=str(sample["split"]),
            input_ids=[],
            labels=[],
            prompt_ids=[],
            target=target,
            format_family=str(sample["format_family"]),
            verifier=sample.get("verifier"),
            task_family=str(sample["task_family"]),
        )
        return semantic_output_valid(tokenized, output)

    return validate


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    families = sorted({str(row["task_family"]) for row in rows})
    return {
        "samples": len(rows),
        "exact": sum(bool(row["exact"]) for row in rows),
        "semantic_exact": sum(bool(row["semantic_valid"]) for row in rows),
        "by_family": {
            family: {
                "samples": len(subset := [
                    row for row in rows if row["task_family"] == family
                ]),
                "exact": sum(bool(row["exact"]) for row in subset),
                "semantic_exact": sum(
                    bool(row["semantic_valid"]) for row in subset
                ),
                "failure_sample_ids": [
                    row["sample_id"]
                    for row in subset
                    if not row["semantic_valid"]
                ],
            }
            for family in families
        },
    }


def run(config: AnalogContractConfig) -> dict[str, Any]:
    dataset = load_dataset(config)
    receipt = validate_serving_receipt(config)
    samples = [
        sample
        for sample in dataset["samples"]
        if sample["split"] == "validation"
    ]
    validate_semantic = _semantic_validator()
    direct_client = OpenRouterClient(
        ModelConfig(
            name=config.model,
            base_url=config.base_url,
            api_key_env="NANO_HARNESS_API_KEY",
            temperature=config.temperature,
            max_tokens=config.direct_max_tokens,
            timeout_seconds=180.0,
            max_retries=3,
            chat_template_kwargs=config.chat_template_kwargs,
        )
    )
    calculation_client = OpenRouterClient(
        ModelConfig(
            name=config.model,
            base_url=config.base_url,
            api_key_env="NANO_HARNESS_API_KEY",
            temperature=config.temperature,
            max_tokens=config.calculation_max_tokens,
            timeout_seconds=180.0,
            max_retries=3,
            chat_template_kwargs=config.chat_template_kwargs,
        )
    )
    selector_client = OpenRouterClient(
        ModelConfig(
            name=config.model,
            base_url=config.base_url,
            api_key_env="NANO_HARNESS_API_KEY",
            temperature=config.temperature,
            max_tokens=config.selector_max_tokens,
            timeout_seconds=180.0,
            max_retries=3,
            chat_template_kwargs=config.chat_template_kwargs,
        )
    )

    baseline_rows = []
    candidate_rows = []
    started = time.time()
    for sample in samples:
        prompt_messages = sample["messages"][:-1]
        direct = direct_client.complete(prompt_messages)
        target = str(sample["messages"][-1]["content"])
        baseline = {
            "sample_id": sample["sample_id"],
            "task_family": sample["task_family"],
            "format_family": sample["format_family"],
            "target": target,
            "output": direct.content,
            "exact": direct.content.strip() == target,
            "semantic_valid": validate_semantic(sample, direct.content),
            "usage": direct.usage,
        }
        baseline_rows.append(baseline)
        if sample["format_family"] == "final_choice":
            output, usage, stages = run_choice_calculation_selector(
                sample,
                config,
                calculation_client,
                selector_client,
            )
            candidate_rows.append(
                {
                    **{key: value for key, value in baseline.items() if key not in {
                        "output",
                        "exact",
                        "semantic_valid",
                        "usage",
                    }},
                    "output": output,
                    "exact": output.strip() == target,
                    "semantic_valid": validate_semantic(sample, output),
                    "usage": usage,
                    "stages": stages,
                    "route": "choice_calculation_selector",
                }
            )
        else:
            candidate_rows.append(
                {
                    **baseline,
                    "route": "reuse_direct_baseline",
                }
            )

    result = {
        "schema_version": "nano_harness_analog_contract_result_v1",
        "experiment_id": config.experiment_id,
        "config": config.__dict__,
        "identity": {
            "dataset_sha256": sha256_file(Path(config.dataset_path)),
            "serving_receipt_sha256": sha256_file(
                Path(config.serving_receipt_path)
            ),
            "serving_adapter_weights_sha256": receipt[
                "serving_adapter_weights_sha256"
            ],
        },
        "baseline": summarize_rows(baseline_rows),
        "candidate": summarize_rows(candidate_rows),
        "baseline_rows": baseline_rows,
        "candidate_rows": candidate_rows,
        "choice_treated_rows": sum(
            row["route"] == "choice_calculation_selector"
            for row in candidate_rows
        ),
        "non_choice_reused_rows": sum(
            row["route"] == "reuse_direct_baseline"
            for row in candidate_rows
        ),
        "wall_seconds": time.time() - started,
        "evaluation_boundary": {
            "benchmark_rows_loaded": False,
            "sealed_canary_run": False,
            "prior_full_suite_run": False,
            "independent_holdout_run": False,
        },
    }
    output = Path(config.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result
