from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nano_harness.client import OpenRouterClient
from nano_harness.config import ModelConfig
from nano_harness.verified_choice import sha256_file, verify_explicit_average_choice


FINAL_CHOICE = re.compile(r"^FINAL: ([A-D])$")


@dataclass(frozen=True)
class ChoiceMatrixEvalConfig:
    schema_version: str
    experiment_id: str
    matrix_path: str
    matrix_sha256: str
    four_b_model: str
    four_b_base_url: str
    four_b_serving_receipt_path: str
    four_b_serving_receipt_sha256: str
    four_b_serving_adapter_weights_sha256: str
    nine_b_model: str
    nine_b_base_url: str
    system_prompt: str
    max_tokens: int
    temperature: float
    chat_template_kwargs: dict[str, Any]
    parser_version: str
    output_path: str


def load_config(path: str | Path) -> ChoiceMatrixEvalConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = set(ChoiceMatrixEvalConfig.__dataclass_fields__)
    if set(raw) != expected:
        raise ValueError("choice matrix eval config fields differ")
    config = ChoiceMatrixEvalConfig(**raw)
    frozen = {
        "schema_version": "nano_harness_choice_matrix_eval_v1",
        "experiment_id": "generic-choice-capability-matrix-eval-v1",
        "four_b_model": "qwen3.5-4b-anchor-v1",
        "nine_b_model": "qwen3.5-9b",
        "max_tokens": 32,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
        "parser_version": "explicit_two_expression_average_v1",
    }
    for field, expected_value in frozen.items():
        if getattr(config, field) != expected_value:
            raise ValueError(f"choice matrix eval freezes {field}={expected_value}")
    return config


def _validate_inputs(config: ChoiceMatrixEvalConfig) -> dict[str, Any]:
    matrix_path = Path(config.matrix_path)
    receipt_path = Path(config.four_b_serving_receipt_path)
    if sha256_file(matrix_path) != config.matrix_sha256:
        raise ValueError("choice matrix identity mismatch")
    if sha256_file(receipt_path) != config.four_b_serving_receipt_sha256:
        raise ValueError("choice matrix serving receipt identity mismatch")
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        matrix.get("schema_version") != "nano_choice_capability_matrix_v1"
        or matrix.get("summary", {}).get("cases") != 48
        or matrix.get("summary", {}).get("training_eligible_cases") != 0
        or matrix.get("policy", {}).get("training_allowed") is not False
    ):
        raise ValueError("choice matrix boundary differs")
    if (
        receipt.get("schema_version")
        != "qwen35_vllm_adapter_namespace_receipt_v1"
        or receipt.get("serving_adapter_weights_sha256")
        != config.four_b_serving_adapter_weights_sha256
        or receipt.get("tensor_count") != 224
        or receipt.get("tensor_content_hashes_match") is not True
    ):
        raise ValueError("choice matrix serving adapter boundary differs")
    return matrix


def _client(config: ChoiceMatrixEvalConfig, *, four_b: bool) -> OpenRouterClient:
    return OpenRouterClient(
        ModelConfig(
            name=config.four_b_model if four_b else config.nine_b_model,
            base_url=(
                config.four_b_base_url if four_b else config.nine_b_base_url
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
    config: ChoiceMatrixEvalConfig,
    *,
    model: str,
) -> list[dict[str, Any]]:
    rows = []
    for case in cases:
        started = time.perf_counter()
        reply = client.complete(
            [
                {"role": "system", "content": config.system_prompt},
                {"role": "user", "content": case["prompt"]},
            ]
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
                    prediction == reference if reference is not None else None
                ),
                "usage": reply.usage,
                "latency_seconds": time.perf_counter() - started,
            }
        )
    return rows


def _executor_rows(
    cases: list[dict[str, Any]],
    direct_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    direct = {row["case_id"]: row for row in direct_rows}
    receipts = {}
    routed = []
    for case in cases:
        baseline = direct[case["case_id"]]
        receipt = verify_explicit_average_choice(case["prompt"])
        receipts[case["case_id"]] = receipt
        output = baseline["output"]
        route = "reuse_direct_output"
        if receipt["override"]:
            output = f"FINAL: {receipt['selected_letter']}"
            route = "verified_choice_override"
        match = FINAL_CHOICE.fullmatch(output)
        prediction = match.group(1) if match else None
        reference = case["reference"]
        routed.append(
            {
                **baseline,
                "model": "qwen3.5-4b-anchor-v1+verified-choice-v1",
                "output": output,
                "prediction": prediction,
                "parseable": prediction is not None,
                "correct": (
                    prediction == reference if reference is not None else None
                ),
                "verified_choice_route": route,
            }
        )
    return routed, receipts


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [row for row in rows if row["scored"]]
    ambiguity = [row for row in rows if not row["scored"]]
    by_family = {}
    for family in sorted({str(row["family"]) for row in rows}):
        subset = [row for row in rows if row["family"] == family]
        family_scored = [row for row in subset if row["scored"]]
        by_family[family] = {
            "cases": len(subset),
            "scored_cases": len(family_scored),
            "correct": sum(bool(row["correct"]) for row in family_scored),
            "parseable": sum(bool(row["parseable"]) for row in subset),
        }
    return {
        "cases": len(rows),
        "scored_cases": len(scored),
        "correct": sum(bool(row["correct"]) for row in scored),
        "accuracy": (
            sum(bool(row["correct"]) for row in scored) / len(scored)
            if scored
            else None
        ),
        "ambiguity_cases": len(ambiguity),
        "parseable": sum(bool(row["parseable"]) for row in rows),
        "by_family": by_family,
    }


def run(config: ChoiceMatrixEvalConfig) -> dict[str, Any]:
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
        "schema_version": "nano_harness_choice_matrix_eval_result_v1",
        "experiment_id": config.experiment_id,
        "config": config.__dict__,
        "identity": {
            "matrix_sha256": sha256_file(Path(config.matrix_path)),
            "four_b_serving_receipt_sha256": sha256_file(
                Path(config.four_b_serving_receipt_path)
            ),
        },
        "arms": {
            "four_b_direct": summarize_rows(four_b),
            "nine_b_direct": summarize_rows(nine_b),
            "four_b_verified_executor": summarize_rows(executor),
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
        },
    }
    output = Path(config.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result
