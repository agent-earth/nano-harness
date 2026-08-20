from __future__ import annotations

import json
import re
import time
import urllib.request
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

from nano_harness.baseline import (
    BaselineCase,
    compare_baselines,
    load_cases,
    load_manifest,
    score_output,
    sha256_file,
    summarize_baseline,
)
from nano_harness.client import OpenRouterClient
from nano_harness.config import ModelConfig
from nano_harness.grounded_calculator_canary import (
    parse_and_execute_grounded_expression,
)


CONFIG_SHA256 = (
    "b083d320e7103cb0809b5c22e6f8ebbc9330a3eb96d3910fd17d34c8f9a52f10"
)


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if sha256_file(config_path) != CONFIG_SHA256:
        raise ValueError("V5 complete treatment config SHA differs")
    if config.get("schema_version") != "nano_harness_v5_complete_treatment_v1":
        raise ValueError("unsupported V5 complete treatment schema")
    if config.get("execution_boundary") != {
        "benchmark_generation_started": False,
        "benchmark_outputs_loaded_by_preregister": False,
        "canary_rerun": False,
        "holdout_accessed": False,
        "rl_started": False,
        "this_commit_only_preregisters": True,
        "training_started": False,
    }:
        raise ValueError("V5 complete treatment boundary differs")
    if config.get("policy") != {
        "benchmark_rows_training_eligible": False,
        "benchmark_outputs_may_enter_training_reward_or_verifier": False,
        "case_id_allowlist_for_routing": False,
        "expected_answer_used_by_routing_or_execution": False,
        "post_observation_search": False,
        "raw_outputs_committed": False,
    }:
        raise ValueError("V5 complete treatment policy differs")
    if config.get("routes") != {
        "gpqa_diamond": {
            "choice_regex": "FINAL: [A-D]",
            "confirmation_max_tokens": 64,
            "option_review_max_tokens": 96,
            "override_rule": (
                "two_independent_reviews_and_confirmation_agree_on_same_non_direct_choice"
            ),
            "otherwise": "preserve_frozen_4b_direct",
            "strategy": "conservative_choice_consensus",
        },
        "gsm8k": {
            "expression_regex": r"CALC: [0-9+\-*/(). ]+",
            "maximum_absolute_value": 10**15,
            "maximum_ast_nodes": 64,
            "maximum_expression_chars": 160,
            "otherwise": "preserve_frozen_4b_direct",
            "plan_max_tokens": 128,
            "plan_replicas": 3,
            "strategy": "grounded_expression_consensus",
            "verified_result_rule": (
                "at_least_two_executed_grounded_expressions_agree"
            ),
        },
        "mmlu": {"strategy": "preserve_frozen_4b_direct"},
    }:
        raise ValueError("V5 complete treatment routes differ")
    return config


def jsonl_ids(path: Path) -> list[str]:
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        match = re.search(r'"case_id"\s*:\s*"([^"]+)"', line)
        if match is None:
            raise ValueError("V5 complete treatment raw case ID is missing")
        values.append(match.group(1))
    if len(values) != len(set(values)):
        raise ValueError("V5 complete treatment raw IDs are duplicated")
    return values


def jsonl_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sum_usage(*values: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for value in values:
        for key, item in value.items():
            if isinstance(item, int):
                result[key] = int(result.get(key, 0)) + item
    return result


def _client(*, max_tokens: int) -> OpenRouterClient:
    return OpenRouterClient(
        ModelConfig(
            name="qwen3.5-4b",
            base_url="http://127.0.0.1:8000/v1",
            api_key_env="NANO_HARNESS_API_KEY",
            temperature=0.0,
            max_tokens=max_tokens,
            timeout_seconds=180.0,
            max_retries=3,
            chat_template_kwargs={"enable_thinking": False},
        )
    )


def verify_four_b_service() -> dict[str, Any]:
    with urllib.request.urlopen(
        "http://127.0.0.1:8000/v1/models",
        timeout=30,
    ) as response:
        health = json.loads(response.read().decode("utf-8"))
    rows = health.get("data", [])
    if len(rows) != 1:
        raise ValueError("V5 complete treatment expects one 4B service model")
    row = rows[0]
    if (
        row.get("id") != "qwen3.5-4b"
        or row.get("root") != "../../../models/Qwen3.5-4B"
        or row.get("parent") is not None
        or row.get("max_model_len") != 4096
        or row.get("owned_by") != "vllm"
    ):
        raise ValueError("V5 complete treatment 4B service identity differs")
    return health


def generate_candidate(
    case: BaselineCase,
    direct: dict[str, Any],
    config: dict[str, Any],
    *,
    calculator_client: Any,
    choice_review_client: Any,
    choice_confirmation_client: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.perf_counter()
    if case.benchmark == "mmlu":
        return (
            {
                **direct,
                "model": "qwen3.5-4b-v5-complete-treatment",
                "treatment_route": "mmlu_direct_preserve",
            },
            {
                "route": "mmlu_direct_preserve",
                "model_calls": 0,
                "api_errors": 0,
                "override": False,
            },
        )
    if case.benchmark == "gsm8k":
        route = config["routes"]["gsm8k"]
        perspectives = (
            "Translate the original math problem into one exact arithmetic expression.",
            "Independently derive a compact arithmetic expression for the requested quantity.",
            "Re-solve from scratch and encode the full calculation as one arithmetic expression.",
        )
        attempts = []
        usages = []
        api_errors = 0
        for perspective in perspectives:
            try:
                reply = calculator_client.complete(
                    [
                        {
                            "role": "system",
                            "content": (
                                perspective
                                + " Every numeric literal must appear in the original "
                                "task; source values may be reused. Use only +, -, *, /, "
                                "parentheses, and unary signs. Return only CALC: <expression>."
                            ),
                        },
                        {"role": "user", "content": case.prompt},
                    ],
                    extra_body={
                        "structured_outputs": {
                            "regex": route["expression_regex"]
                        }
                    },
                )
            except Exception as exc:
                api_errors += 1
                attempts.append(
                    {
                        "executed": False,
                        "reason": "model_api_error",
                        "error_type": type(exc).__name__,
                    }
                )
                continue
            usages.append(reply.usage)
            receipt = parse_and_execute_grounded_expression(
                reply.content,
                prompt=case.prompt,
                maximum_expression_chars=route["maximum_expression_chars"],
                maximum_ast_nodes=route["maximum_ast_nodes"],
                maximum_absolute_value=route["maximum_absolute_value"],
            )
            attempts.append(
                {
                    "executed": receipt["executed"],
                    "reason": receipt["reason"],
                    "expression_sha256": receipt.get("expression_sha256"),
                    "result": receipt.get("result"),
                    "output_sha256": __import__("hashlib")
                    .sha256(reply.content.encode())
                    .hexdigest(),
                }
            )
        counts = Counter(
            row["result"] for row in attempts if row.get("executed")
        )
        consensus = [
            result for result, count in counts.items() if count >= 2
        ]
        output = direct["output"]
        override = len(consensus) == 1
        if override:
            output = f"FINAL: {consensus[0]}"
        return (
            {
                **direct,
                "model": "qwen3.5-4b-v5-complete-treatment",
                "output": output,
                "treatment_route": (
                    "gsm8k_grounded_consensus"
                    if override
                    else "gsm8k_direct_preserve"
                ),
                "usage": sum_usage(direct.get("usage", {}), *usages),
                "latency_seconds": time.perf_counter() - started,
            },
            {
                "route": "gsm8k_grounded_expression_consensus",
                "model_calls": len(perspectives),
                "api_errors": api_errors,
                "attempts": attempts,
                "consensus_result": consensus[0] if override else None,
                "override": override,
            },
        )
    if case.benchmark != "gpqa_diamond":
        raise ValueError(f"unsupported treatment benchmark: {case.benchmark}")
    route = config["routes"]["gpqa_diamond"]
    direct_prediction = direct.get("prediction")
    review_prompts = (
        "Independently solve the science question. Check every option and return only FINAL: <letter>.",
        "Re-evaluate the question from first principles with special attention to distractors. Return only FINAL: <letter>.",
    )
    reviews = []
    usages = []
    api_errors = 0
    for instruction in review_prompts:
        try:
            reply = choice_review_client.complete(
                [
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": case.draft_prompt},
                ],
                extra_body={
                    "structured_outputs": {
                        "regex": route["choice_regex"]
                    }
                },
            )
        except Exception as exc:
            api_errors += 1
            reviews.append(
                {
                    "prediction": None,
                    "reason": "model_api_error",
                    "error_type": type(exc).__name__,
                }
            )
            continue
        usages.append(reply.usage)
        prediction = re.fullmatch(
            r"FINAL: ([A-D])", reply.content.strip()
        )
        reviews.append(
            {
                "prediction": prediction.group(1) if prediction else None,
                "output_sha256": __import__("hashlib")
                .sha256(reply.content.encode())
                .hexdigest(),
            }
        )
    review_choices = [row["prediction"] for row in reviews]
    agreed = (
        len(review_choices) == 2
        and review_choices[0] is not None
        and review_choices[0] == review_choices[1]
        and review_choices[0] != direct_prediction
    )
    confirmation = None
    if agreed:
        try:
            reply = choice_confirmation_client.complete(
                [
                    {
                        "role": "system",
                        "content": (
                            "Independently confirm or reject the proposed answer "
                            "against the original task. Return only FINAL: <letter>."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"<original_task>\n{case.draft_prompt}\n</original_task>\n\n"
                            f"<proposed_answer>\nFINAL: {review_choices[0]}\n"
                            "</proposed_answer>"
                        ),
                    },
                ],
                extra_body={
                    "structured_outputs": {
                        "regex": route["choice_regex"]
                    }
                },
            )
            usages.append(reply.usage)
            match = re.fullmatch(r"FINAL: ([A-D])", reply.content.strip())
            confirmation = match.group(1) if match else None
        except Exception:
            api_errors += 1
    override = agreed and confirmation == review_choices[0]
    output = (
        f"FINAL: {review_choices[0]}" if override else direct["output"]
    )
    return (
        {
            **direct,
            "model": "qwen3.5-4b-v5-complete-treatment",
            "output": output,
            "treatment_route": (
                "gpqa_conservative_consensus"
                if override
                else "gpqa_direct_preserve"
            ),
            "usage": sum_usage(direct.get("usage", {}), *usages),
            "latency_seconds": time.perf_counter() - started,
        },
        {
            "route": "gpqa_conservative_choice_consensus",
            "model_calls": len(reviews) + int(agreed),
            "api_errors": api_errors,
            "reviews": reviews,
            "confirmation": confirmation,
            "override": override,
        },
    )


def run(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    root = config_path.resolve().parents[2]
    config = load_config(config_path)
    manifest = load_manifest(root / config["baseline"]["suite_manifest_path"])
    cases = load_cases(manifest, (root / "../../../datasets").resolve())
    case_by_id = {case.case_id: case for case in cases}
    direct_rows = jsonl_rows(root / config["baseline"]["four_b_raw_path"])
    direct_by_id = {str(row["case_id"]): row for row in direct_rows}
    service_health = verify_four_b_service()
    output_path = root / config["output"]["candidate_path"]
    completed = set(jsonl_ids(output_path)) if output_path.exists() else set()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    calculator = _client(max_tokens=config["routes"]["gsm8k"]["plan_max_tokens"])
    choice_review = _client(
        max_tokens=config["routes"]["gpqa_diamond"]["option_review_max_tokens"]
    )
    choice_confirmation = _client(
        max_tokens=config["routes"]["gpqa_diamond"]["confirmation_max_tokens"]
    )
    receipt_path = output_path.with_suffix(".receipts.jsonl")
    for case in cases:
        if case.case_id in completed:
            continue
        safe_direct = {
            key: value
            for key, value in direct_by_id[case.case_id].items()
            if key not in {"expected", "score"}
        }
        candidate, receipt = generate_candidate(
            replace(case, expected="__SEALED_DURING_GENERATION__"),
            safe_direct,
            config,
            calculator_client=calculator,
            choice_review_client=choice_review,
            choice_confirmation_client=choice_confirmation,
        )
        score, prediction = score_output(
            candidate["output"], case.expected, case.scorer
        )
        candidate.update(
            {
                "suite_id": config["experiment_id"],
                "benchmark": case.benchmark,
                "prediction": prediction,
                "score": score,
                "status": "completed",
                "expected": case.expected,
                "source_index": case.source_index,
                "prompt_sha256": __import__("hashlib")
                .sha256(case.prompt.encode())
                .hexdigest(),
                "system_prompt_sha256": __import__("hashlib")
                .sha256(case.system_prompt.encode())
                .hexdigest(),
            }
        )
        with receipt_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {"case_id": case.case_id, "receipt": receipt},
                    sort_keys=True,
                )
                + "\n"
            )
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(candidate, sort_keys=True) + "\n")
    summary = summarize_baseline(output_path)
    comparison_four = compare_baselines(
        output_path,
        root / config["baseline"]["four_b_raw_path"],
        bootstrap_samples=config["statistics"]["bootstrap_samples"],
        bootstrap_seed=config["statistics"]["bootstrap_seed"] + ":four",
    )
    comparison_nine = compare_baselines(
        output_path,
        root / config["baseline"]["nine_b_raw_path"],
        bootstrap_samples=config["statistics"]["bootstrap_samples"],
        bootstrap_seed=config["statistics"]["bootstrap_seed"] + ":nine",
    )
    result = {
        "schema_version": "nano_harness_v5_complete_treatment_result_v1",
        "experiment_id": config["experiment_id"],
        "identity": {
            "config_sha256": sha256_file(config_path),
            "candidate_sha256": sha256_file(output_path),
            "receipts_sha256": sha256_file(receipt_path),
            "service_health_sha256": __import__("hashlib")
            .sha256(
                json.dumps(
                    service_health,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            )
            .hexdigest(),
        },
        "summary": summary,
        "comparisons": {
            "versus_four_b": comparison_four,
            "versus_nine_b": comparison_nine,
        },
        "evaluation_boundary": {
            "benchmark_rows_training_eligible": False,
            "expected_answer_used_during_candidate_generation": False,
            "case_correctness_used_during_candidate_generation": False,
            "scoring_applied_after_candidate_generation": True,
            "raw_outputs_committed": False,
        },
    }
    result_path = root / config["output"]["result_path"]
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result
