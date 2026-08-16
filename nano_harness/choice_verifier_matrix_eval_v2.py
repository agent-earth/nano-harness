from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nano_harness.choice_matrix_eval import summarize_rows
from nano_harness.client import OpenRouterClient
from nano_harness.config import ModelConfig
from nano_harness.verified_choice import sha256_file
from nano_harness.verified_choice_v2 import verify_choice_v2


FINAL_CHOICE = re.compile(r"^FINAL: ([A-D])$")


@dataclass(frozen=True)
class ChoiceVerifierMatrixEvalConfig:
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
    prior_matrix_report_path: str
    prior_matrix_report_sha256: str
    system_prompt: str
    max_tokens: int
    temperature: float
    chat_template_kwargs: dict[str, Any]
    structured_output_regex: str
    parser_version: str
    output_path: str


def load_config(path: str | Path) -> ChoiceVerifierMatrixEvalConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = set(ChoiceVerifierMatrixEvalConfig.__dataclass_fields__)
    if set(raw) != expected:
        raise ValueError("choice verifier matrix config fields differ")
    config = ChoiceVerifierMatrixEvalConfig(**raw)
    frozen = {
        "schema_version": "nano_harness_choice_verifier_matrix_eval_v2",
        "experiment_id": "generic-choice-verifier-matrix-eval-v2",
        "four_b_model": "qwen3.5-4b-anchor-v1",
        "nine_b_model": "qwen3.5-9b",
        "max_tokens": 32,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
        "structured_output_regex": r"FINAL: [A-D]",
        "parser_version": "host_count_and_verbal_average_v2",
    }
    for field, expected_value in frozen.items():
        if getattr(config, field) != expected_value:
            raise ValueError(
                f"choice verifier matrix freezes {field}={expected_value}"
            )
    return config


def _validate_inputs(
    config: ChoiceVerifierMatrixEvalConfig,
) -> dict[str, Any]:
    paths = {
        "matrix": Path(config.matrix_path),
        "four_b_model_config": Path(config.four_b_model_config_path),
        "four_b_serving_receipt": Path(config.four_b_serving_receipt_path),
        "nine_b_model_config": Path(config.nine_b_model_config_path),
        "prior_matrix_report": Path(config.prior_matrix_report_path),
    }
    expected_hashes = {
        "matrix": config.matrix_sha256,
        "four_b_model_config": config.four_b_model_config_sha256,
        "four_b_serving_receipt": config.four_b_serving_receipt_sha256,
        "nine_b_model_config": config.nine_b_model_config_sha256,
        "prior_matrix_report": config.prior_matrix_report_sha256,
    }
    for name, path in paths.items():
        if sha256_file(path) != expected_hashes[name]:
            raise ValueError(
                f"choice verifier matrix {name} identity mismatch"
            )
    matrix = json.loads(paths["matrix"].read_text(encoding="utf-8"))
    receipt = json.loads(
        paths["four_b_serving_receipt"].read_text(encoding="utf-8")
    )
    prior = json.loads(
        paths["prior_matrix_report"].read_text(encoding="utf-8")
    )
    if (
        matrix.get("schema_version") != "nano_choice_verifier_matrix_v2"
        or matrix.get("summary", {}).get("cases") != 48
        or matrix.get("summary", {}).get("training_eligible_cases") != 0
        or matrix.get("policy", {}).get("training_allowed") is not False
    ):
        raise ValueError("choice verifier matrix data boundary differs")
    if (
        receipt.get("serving_adapter_weights_sha256")
        != config.four_b_serving_adapter_weights_sha256
        or receipt.get("tensor_count") != 224
        or receipt.get("tensor_content_hashes_match") is not True
    ):
        raise ValueError("choice verifier matrix serving boundary differs")
    if (
        prior.get("passed") is not True
        or prior.get("decision", {}).get(
            "executor_significantly_exceeds_nine_b_on_matrix"
        )
        is not True
        or prior.get("decision", {}).get("benchmark_superiority_claim_allowed")
        is not False
    ):
        raise ValueError("prior matrix receipt does not authorize expansion")
    return matrix


def _client(
    config: ChoiceVerifierMatrixEvalConfig,
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
    config: ChoiceVerifierMatrixEvalConfig,
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


def _executor_rows(
    cases: list[dict[str, Any]],
    four_b_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    direct = {row["case_id"]: row for row in four_b_rows}
    rows = []
    receipts = {}
    for case in cases:
        baseline = direct[case["case_id"]]
        receipt = verify_choice_v2(case["prompt"])
        receipts[case["case_id"]] = receipt
        output = baseline["output"]
        route = "reuse_direct_output"
        if receipt["override"]:
            output = f"FINAL: {receipt['selected_letter']}"
            route = "verified_choice_override"
        match = FINAL_CHOICE.fullmatch(output)
        prediction = match.group(1) if match else None
        reference = case["reference"]
        rows.append(
            {
                **baseline,
                "model": "qwen3.5-4b-anchor-v1+verified-choice-v2",
                "output": output,
                "prediction": prediction,
                "parseable": prediction is not None,
                "correct": (
                    prediction == reference
                    if reference is not None
                    else None
                ),
                "verified_choice_route": route,
            }
        )
    return rows, receipts


def run(config: ChoiceVerifierMatrixEvalConfig) -> dict[str, Any]:
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
        "schema_version": "nano_harness_choice_verifier_matrix_result_v2",
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
            "prior_matrix_report_sha256": sha256_file(
                Path(config.prior_matrix_report_path)
            ),
        },
        "arms": {
            "four_b_constrained": summarize_rows(four_b),
            "nine_b_constrained": summarize_rows(nine_b),
            "four_b_verified_executor_v2": summarize_rows(executor),
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
