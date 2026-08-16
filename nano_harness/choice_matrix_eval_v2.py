from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nano_harness.choice_matrix_eval import (
    _executor_rows,
    summarize_rows,
)
from nano_harness.client import OpenRouterClient
from nano_harness.config import ModelConfig
from nano_harness.verified_choice import sha256_file


FINAL_CHOICE = re.compile(r"^FINAL: ([A-D])$")


@dataclass(frozen=True)
class ChoiceMatrixEvalV2Config:
    schema_version: str
    experiment_id: str
    matrix_path: str
    matrix_sha256: str
    four_b_model: str
    four_b_base_url: str
    four_b_model_config_path: str
    four_b_model_config_sha256: str
    four_b_serving_receipt_path: str
    four_b_serving_receipt_sha256: str
    four_b_serving_adapter_weights_sha256: str
    nine_b_model: str
    nine_b_base_url: str
    nine_b_model_config_path: str
    nine_b_model_config_sha256: str
    v1_public_report_path: str
    v1_public_report_sha256: str
    system_prompt: str
    max_tokens: int
    temperature: float
    chat_template_kwargs: dict[str, Any]
    structured_output_regex: str
    parser_version: str
    output_path: str


def load_config(path: str | Path) -> ChoiceMatrixEvalV2Config:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = set(ChoiceMatrixEvalV2Config.__dataclass_fields__)
    if set(raw) != expected:
        raise ValueError("choice matrix v2 config fields differ")
    config = ChoiceMatrixEvalV2Config(**raw)
    frozen = {
        "schema_version": "nano_harness_choice_matrix_eval_v2",
        "experiment_id": "generic-choice-capability-matrix-eval-v2",
        "four_b_model": "qwen3.5-4b-anchor-v1",
        "nine_b_model": "qwen3.5-9b",
        "max_tokens": 32,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
        "structured_output_regex": r"FINAL: [A-D]",
        "parser_version": "explicit_two_expression_average_v1",
    }
    for field, expected_value in frozen.items():
        if getattr(config, field) != expected_value:
            raise ValueError(
                f"choice matrix v2 freezes {field}={expected_value}"
            )
    return config


def _validate_inputs(config: ChoiceMatrixEvalV2Config) -> dict[str, Any]:
    paths = {
        "matrix": Path(config.matrix_path),
        "four_b_model_config": Path(config.four_b_model_config_path),
        "four_b_serving_receipt": Path(config.four_b_serving_receipt_path),
        "nine_b_model_config": Path(config.nine_b_model_config_path),
        "v1_public_report": Path(config.v1_public_report_path),
    }
    expected_hashes = {
        "matrix": config.matrix_sha256,
        "four_b_model_config": config.four_b_model_config_sha256,
        "four_b_serving_receipt": config.four_b_serving_receipt_sha256,
        "nine_b_model_config": config.nine_b_model_config_sha256,
        "v1_public_report": config.v1_public_report_sha256,
    }
    for name, path in paths.items():
        if sha256_file(path) != expected_hashes[name]:
            raise ValueError(f"choice matrix v2 {name} identity mismatch")
    matrix = json.loads(paths["matrix"].read_text(encoding="utf-8"))
    receipt = json.loads(
        paths["four_b_serving_receipt"].read_text(encoding="utf-8")
    )
    v1 = json.loads(paths["v1_public_report"].read_text(encoding="utf-8"))
    if (
        matrix.get("schema_version") != "nano_choice_capability_matrix_v1"
        or matrix.get("summary", {}).get("cases") != 48
        or matrix.get("summary", {}).get("training_eligible_cases") != 0
        or matrix.get("policy", {}).get("training_allowed") is not False
    ):
        raise ValueError("choice matrix v2 matrix boundary differs")
    if (
        receipt.get("serving_adapter_weights_sha256")
        != config.four_b_serving_adapter_weights_sha256
        or receipt.get("tensor_count") != 224
        or receipt.get("tensor_content_hashes_match") is not True
    ):
        raise ValueError("choice matrix v2 serving boundary differs")
    if (
        v1.get("passed") is not True
        or v1.get("arms", {})
        .get("nine_b_direct", {})
        .get("valid_for_quality_comparison")
        is not False
        or v1.get("decision", {}).get("four_b_exceeds_nine_b_claim_allowed")
        is not False
    ):
        raise ValueError("choice matrix v1 receipt does not authorize repair")
    return matrix


def _client(
    config: ChoiceMatrixEvalV2Config,
    *,
    four_b: bool,
) -> OpenRouterClient:
    return OpenRouterClient(
        ModelConfig(
            name=config.four_b_model if four_b else config.nine_b_model,
            base_url=(
                config.four_b_base_url
                if four_b
                else config.nine_b_base_url
            ),
            api_key_env="NANO_HARNESS_API_KEY",
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout_seconds=180.0,
            max_retries=3,
            chat_template_kwargs=config.chat_template_kwargs,
        )
    )


def _direct_rows(
    cases: list[dict[str, Any]],
    client: Any,
    config: ChoiceMatrixEvalV2Config,
    *,
    model: str,
) -> list[dict[str, Any]]:
    rows = []
    structured_outputs = {"regex": config.structured_output_regex}
    for case in cases:
        started = time.perf_counter()
        reply = client.complete(
            [
                {"role": "system", "content": config.system_prompt},
                {"role": "user", "content": case["prompt"]},
            ],
            extra_body={"structured_outputs": structured_outputs},
        )
        output = reply.content.strip()
        match = FINAL_CHOICE.fullmatch(output)
        prediction = match.group(1) if match else None
        reference = case["reference"]
        rows.append(
            {
                "case_id": case["case_id"],
                "family": case["family"],
                "expected_route": case["expected_route"],
                "reference": reference,
                "model": model,
                "output": output,
                "prediction": prediction,
                "parseable": prediction is not None,
                "scored": reference is not None,
                "correct": (
                    prediction == reference
                    if reference is not None
                    else None
                ),
                "usage": reply.usage,
                "latency_seconds": time.perf_counter() - started,
                "structured_outputs": structured_outputs,
            }
        )
    return rows


def run(config: ChoiceMatrixEvalV2Config) -> dict[str, Any]:
    matrix = _validate_inputs(config)
    cases = list(matrix["cases"])
    four_b = _direct_rows(
        cases,
        _client(config, four_b=True),
        config,
        model=config.four_b_model,
    )
    nine_b = _direct_rows(
        cases,
        _client(config, four_b=False),
        config,
        model=config.nine_b_model,
    )
    executor, receipts = _executor_rows(cases, four_b)
    result = {
        "schema_version": "nano_harness_choice_matrix_eval_result_v2",
        "experiment_id": config.experiment_id,
        "config": config.__dict__,
        "identity": {
            "matrix_sha256": sha256_file(Path(config.matrix_path)),
            "four_b_model_config_sha256": sha256_file(
                Path(config.four_b_model_config_path)
            ),
            "four_b_serving_receipt_sha256": sha256_file(
                Path(config.four_b_serving_receipt_path)
            ),
            "nine_b_model_config_sha256": sha256_file(
                Path(config.nine_b_model_config_path)
            ),
            "v1_public_report_sha256": sha256_file(
                Path(config.v1_public_report_path)
            ),
        },
        "arms": {
            "four_b_constrained": summarize_rows(four_b),
            "nine_b_constrained": summarize_rows(nine_b),
            "four_b_constrained_verified_executor": summarize_rows(executor),
        },
        "four_b_rows": four_b,
        "nine_b_rows": nine_b,
        "executor_rows": executor,
        "executor_receipts": receipts,
        "routing": {
            "verified_overrides": sum(
                receipt["override"] for receipt in receipts.values()
            ),
            "fallbacks": sum(
                not receipt["override"] for receipt in receipts.values()
            ),
            "expected_route_matches": sum(
                (
                    case["expected_route"] == "verified_override"
                    and receipts[case["case_id"]]["override"]
                )
                or (
                    case["expected_route"] != "verified_override"
                    and not receipts[case["case_id"]]["override"]
                )
                for case in cases
            ),
        },
        "evaluation_boundary": {
            "training_eligible_cases": 0,
            "target_used_by_executor_parser": False,
            "benchmark_rows_loaded": False,
            "canary_rows_loaded": False,
            "independent_holdout_rows_loaded": False,
            "v1_nine_b_outputs_reused": False,
        },
    }
    output = Path(config.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result
