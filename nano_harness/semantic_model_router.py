from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from nano_harness.baseline import sha256_file
from nano_harness.semantic_skill_execution import (
    FAMILIES,
    SKILL_PROMPTS,
    TOOL_REGEX_BY_FAMILY,
    execute_semantic_tool,
    load_config as load_mechanism_config,
    parent_config as load_parent_runtime,
    parse_and_execute_plan,
)
from nano_harness.verified_tool_execution import (
    _client,
    _direct_row,
    _sum_usage,
    public_case_contract,
    verify_inputs,
)


CONFIG_SCHEMA = "nano_harness_semantic_model_router_v1"
RESULT_SCHEMA = "nano_harness_semantic_model_router_result_v1"
ROUTE_REGEX = (
    r"ROUTE: (?:implicit_scale_total|first_strict_profit_period|NONE)"
)
ROUTE_PATTERN = re.compile(
    r"^ROUTE: (implicit_scale_total|first_strict_profit_period|NONE)$"
)
POSITIVE_FAMILIES = FAMILIES
NEGATIVE_FAMILIES = ("box_total", "remaining_stock")
ALL_FAMILIES = (*POSITIVE_FAMILIES, *NEGATIVE_FAMILIES)
ROUTER_PROMPT = (
    "Choose exactly one route. Use implicit_scale_total only when the task "
    "asks for double or triple a rectangular rows-by-columns capacity plus "
    "an extra. Use first_strict_profit_period only when the task asks for "
    "the first whole period with cumulative profit strictly above zero from "
    "setup cost, units, unit price, and recurring cost. Use NONE for every "
    "other task. Return only ROUTE: <name>."
)


@dataclass(frozen=True)
class SemanticModelRouterConfig:
    schema_version: str
    experiment_id: str
    mechanism_config_path: str
    mechanism_config_sha256: str
    applicability_report_path: str
    applicability_report_sha256: str
    output_path: str
    case_seed: int
    positive_cases_per_family: int
    negative_cases_per_family: int
    router_structured_output_regex: str
    router_max_tokens: int
    direct_max_tokens: int
    plan_max_tokens: int
    final_max_tokens: int
    plan_retry_limit: int
    bootstrap_samples: int
    bootstrap_seed: str
    significance_alpha: float
    minimum_harness_wins: int
    maximum_harness_losses: int
    policy: dict[str, bool]
    execution_boundary: dict[str, bool]


def load_config(path: str | Path) -> SemanticModelRouterConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(raw) != set(SemanticModelRouterConfig.__dataclass_fields__):
        raise ValueError("semantic model router config fields differ")
    config = SemanticModelRouterConfig(**raw)
    expected = {
        "schema_version": CONFIG_SCHEMA,
        "experiment_id": "qwen35-semantic-model-router-v1",
        "mechanism_config_path": (
            "configs/harness/qwen35_semantic_skill_execution_v1.json"
        ),
        "mechanism_config_sha256": (
            "4e5785b9b4fc9dd5b3a76a95b438c2f2c3d505a2fae6231b81cfbf53ebbd6ad9"
        ),
        "applicability_report_path": (
            "docs/results/qwen35_semantic_skill_applicability_v1.public.json"
        ),
        "applicability_report_sha256": (
            "26d99e46d9b0523598ca863c9440d1519119d3ef3a1a928e881a35f3f9806d0c"
        ),
        "output_path": (
            "results/harness/qwen35-semantic-model-router-v1/result.json"
        ),
        "case_seed": 20260822,
        "positive_cases_per_family": 64,
        "negative_cases_per_family": 64,
        "router_structured_output_regex": ROUTE_REGEX,
        "router_max_tokens": 16,
        "direct_max_tokens": 32,
        "plan_max_tokens": 96,
        "final_max_tokens": 32,
        "plan_retry_limit": 1,
        "bootstrap_samples": 10_000,
        "bootstrap_seed": "qwen35-semantic-model-router-v1",
        "significance_alpha": 0.05,
        "minimum_harness_wins": 12,
        "maximum_harness_losses": 0,
        "policy": {
            "evaluation_only": True,
            "training_eligible": False,
            "contains_benchmark_rows": False,
            "contains_benchmark_outputs": False,
            "contains_canary_rows": False,
            "contains_holdout_rows": False,
            "router_uses_case_metadata": False,
            "router_uses_expected_answer": False,
            "router_uses_case_correctness": False,
            "executor_uses_expected_answer": False,
            "executor_uses_case_correctness": False,
            "post_observation_prompt_parser_budget_search": False,
        },
        "execution_boundary": {
            "service_reused": True,
            "model_generation_started": False,
            "evaluation_started": False,
            "benchmark_accessed": False,
            "canary_accessed": False,
            "holdout_accessed": False,
            "training_started": False,
            "this_commit_only_preregisters": True,
        },
    }
    for field, expected_value in expected.items():
        if getattr(config, field) != expected_value:
            raise ValueError(
                f"semantic model router freezes {field}={expected_value}"
            )
    for path_value, digest in (
        (config.mechanism_config_path, config.mechanism_config_sha256),
        (config.applicability_report_path, config.applicability_report_sha256),
    ):
        if sha256_file(Path(path_value)) != digest:
            raise ValueError("semantic model router evidence identity differs")
    report = json.loads(
        Path(config.applicability_report_path).read_text(encoding="utf-8")
    )
    if (
        report.get("decision", {}).get("transfer_preregistration_allowed")
        is not False
        or report.get("decision", {}).get(
            "post_scan_route_or_extractor_change_allowed"
        )
        is not False
    ):
        raise ValueError("semantic model router predecessor decision differs")
    return config


def parent_config(config: SemanticModelRouterConfig):
    mechanism = load_mechanism_config(config.mechanism_config_path)
    parent = load_parent_runtime(mechanism)
    return replace(
        parent,
        experiment_id=config.experiment_id,
        output_path=config.output_path,
        case_seed=config.case_seed,
        direct_max_tokens=config.direct_max_tokens,
        plan_max_tokens=config.plan_max_tokens,
        final_max_tokens=config.final_max_tokens,
        plan_retry_limit=config.plan_retry_limit,
        bootstrap_samples=config.bootstrap_samples,
        bootstrap_seed=config.bootstrap_seed,
        significance_alpha=config.significance_alpha,
        minimum_harness_wins=config.minimum_harness_wins,
        maximum_harness_losses=config.maximum_harness_losses,
    )


def build_cases(config: SemanticModelRouterConfig) -> list[dict[str, Any]]:
    rows = []
    for family in POSITIVE_FAMILIES:
        for index in range(config.positive_cases_per_family):
            value = 3000 + index
            if family == "implicit_scale_total":
                scale_word = "double" if index % 2 == 0 else "triple"
                source_facts: dict[str, Any] = {
                    "rows": value * 3 + 11,
                    "columns": value * 2 + 7,
                    "extra": value * 5 + 13,
                    "scale_word": scale_word,
                }
                scale_phrase = "twice" if scale_word == "double" else "three times"
                prompt = (
                    f"A theater has {source_facts['rows']} tiers with "
                    f"{source_facts['columns']} chairs per tier. The organizer "
                    f"orders {source_facts['extra']} spare chairs in addition "
                    f"to {scale_phrase} the existing capacity. Compute the "
                    "exact total chair order."
                )
            else:
                units = value * 2 + 5
                price = value * 3 + 7
                net = value * 4 + 29
                recurring = units * price - net
                threshold = value * 2 + 17
                source_facts = {
                    "setup_cost": net * threshold,
                    "units_per_period": units,
                    "price_per_unit": price,
                    "recurring_cost": recurring,
                }
                prompt = (
                    f"Opening a service costs {source_facts['setup_cost']}. "
                    f"Each month it sells {source_facts['units_per_period']} "
                    f"passes at {source_facts['price_per_unit']} each and pays "
                    f"{source_facts['recurring_cost']} in monthly costs. In "
                    "which first whole month is cumulative profit greater "
                    "than zero?"
                )
            expected = execute_semantic_tool(family, source_facts)
            rows.append(
                _case(family, prompt, source_facts, expected, positive=True)
            )
    for family in NEGATIVE_FAMILIES:
        for index in range(config.negative_cases_per_family):
            value = 5000 + index
            if family == "box_total":
                source_facts = {
                    "boxes": value * 2 + 3,
                    "items_per_box": value * 3 + 5,
                    "loose_items": value * 7 + 11,
                }
                expected = (
                    source_facts["boxes"] * source_facts["items_per_box"]
                    + source_facts["loose_items"]
                )
                prompt = (
                    f"A shipment has boxes={source_facts['boxes']}, "
                    f"items_per_box={source_facts['items_per_box']}, and "
                    f"loose_items={source_facts['loose_items']}. Compute the "
                    "exact total number of items."
                )
            else:
                batches = value * 2 + 7
                units = value * 3 + 13
                remaining = value * 5 + 17
                source_facts = {
                    "starting_units": batches * units + remaining,
                    "batches_used": batches,
                    "units_per_batch": units,
                }
                expected = remaining
                prompt = (
                    f"A stock record has starting_units="
                    f"{source_facts['starting_units']}, batches_used="
                    f"{source_facts['batches_used']}, and units_per_batch="
                    f"{source_facts['units_per_batch']}. Compute the exact "
                    "units remaining."
                )
            rows.append(
                _case(family, prompt, source_facts, expected, positive=False)
            )
    rows.sort(
        key=lambda row: hashlib.sha256(
            f"{config.case_seed}\0{row['case_id']}".encode()
        ).hexdigest()
    )
    return rows


def _case(
    family: str,
    prompt: str,
    source_facts: dict[str, Any],
    expected: int,
    *,
    positive: bool,
) -> dict[str, Any]:
    digest = hashlib.sha256(
        f"{family}\0{json.dumps(source_facts, sort_keys=True)}".encode()
    ).hexdigest()
    return {
        "case_id": f"semantic-router-{family}-{digest[:16]}",
        "family": family,
        "prompt": prompt,
        "source_facts": source_facts,
        "expected": expected,
        "expected_route": family if positive else "NONE",
        "positive": positive,
    }


def parse_route(text: str) -> str | None:
    match = ROUTE_PATTERN.fullmatch(text.strip())
    return match.group(1) if match else None


def _candidate_row(
    case: dict[str, Any],
    direct: dict[str, Any],
    router_client: Any,
    plan_client: Any,
    final_client: Any,
    config: SemanticModelRouterConfig,
    mechanism: Any,
    parent: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.perf_counter()
    reply = router_client.complete(
        [
            {"role": "system", "content": ROUTER_PROMPT},
            {"role": "user", "content": case["prompt"]},
        ],
        extra_body={
            "structured_outputs": {
                "regex": config.router_structured_output_regex
            }
        },
    )
    selected = parse_route(reply.content)
    router_receipt = {
        "output": reply.content,
        "output_sha256": hashlib.sha256(reply.content.encode()).hexdigest(),
        "selected_route": selected,
        "expected_route": case["expected_route"],
        "correct": selected == case["expected_route"],
        "router_uses_case_metadata": False,
        "router_uses_expected_answer": False,
        "router_uses_case_correctness": False,
    }
    if selected == "NONE" or selected not in POSITIVE_FAMILIES:
        return (
            {
                **direct,
                "model": f"{parent.four_b_model}+semantic-model-router-v1",
                "route": "direct_preserve_after_none",
                "usage": _sum_usage(direct["usage"], reply.usage),
                "latency_seconds": time.perf_counter() - started,
            },
            {
                "router": router_receipt,
                "exposed_tools": [],
                "plan_attempts": [],
                "receipt": None,
                "final_feedback_sent": False,
                "fallback_used": False,
            },
        )
    routed_case = {**case, "family": selected}
    harness, receipt = _model_selected_tool_row(
        routed_case,
        direct,
        selected,
        plan_client,
        final_client,
        mechanism,
        parent,
    )
    harness["usage"] = _sum_usage(reply.usage, harness["usage"])
    harness["latency_seconds"] = time.perf_counter() - started
    return harness, {"router": router_receipt, **receipt}


def _model_selected_tool_row(
    case: dict[str, Any],
    direct: dict[str, Any],
    selected: str,
    plan_client: Any,
    final_client: Any,
    mechanism: Any,
    parent: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    route = {
        "routed": True,
        "reason": "model_selected_route",
        "family": selected,
        "semantic_operator": (
            case["source_facts"].get("scale_word")
            if selected == "implicit_scale_total"
            else None
        ),
        "router_uses_case_metadata": False,
    }
    plan_messages = [
        {"role": "system", "content": SKILL_PROMPTS[selected]},
        {"role": "user", "content": case["prompt"]},
    ]
    attempts = []
    plan_usage = []
    receipt = None
    for attempt in range(mechanism.plan_retry_limit + 1):
        reply = plan_client.complete(
            plan_messages,
            extra_body={
                "structured_outputs": {
                    "regex": TOOL_REGEX_BY_FAMILY[selected]
                }
            },
        )
        plan_usage.append(reply.usage)
        receipt = parse_and_execute_plan(
            reply.content,
            route=route,
            source_facts=case["source_facts"],
        )
        attempts.append(
            {
                "attempt": attempt + 1,
                "output": reply.content,
                "output_sha256": hashlib.sha256(
                    reply.content.encode()
                ).hexdigest(),
                "reason": receipt["reason"],
                "executed": receipt["executed"],
            }
        )
        if receipt["executed"]:
            break
        if attempt < mechanism.plan_retry_limit:
            plan_messages.extend(
                [
                    {"role": "assistant", "content": reply.content},
                    {
                        "role": "user",
                        "content": (
                            "The typed semantic skill rejected the plan with "
                            f"reason={receipt['reason']}. Copy the original "
                            "facts into the same available tool."
                        ),
                    },
                ]
            )
    assert receipt is not None
    if not receipt["executed"]:
        return (
            {
                **direct,
                "model": f"{parent.four_b_model}+semantic-model-router-v1",
                "route": "direct_fallback_after_invalid_plan",
                "usage": _sum_usage(direct["usage"], *plan_usage),
            },
            {
                "route": route,
                "exposed_tools": [selected],
                "plan_attempts": attempts,
                "receipt": receipt,
                "final_feedback_sent": False,
                "fallback_used": True,
            },
        )
    feedback = (
        f"<original_task>\n{case['prompt']}\n</original_task>\n\n"
        f"<verified_semantic_tool>\nname={receipt['tool_name']}\n"
        f"arguments={json.dumps(receipt['arguments'], sort_keys=True)}\n"
        f"result={receipt['result']}\n</verified_semantic_tool>\n\n"
        "Use the verified result as authoritative. Return only "
        "FINAL: <integer>."
    )
    final = final_client.complete(
        [
            {
                "role": "system",
                "content": (
                    "Return the verified semantic-tool result without "
                    "changing it. Return only one FINAL: <integer> line."
                ),
            },
            {"role": "user", "content": feedback},
        ],
        extra_body={
            "structured_outputs": {
                "regex": parent.direct_structured_output_regex
            }
        },
    )
    output = final.content.strip()
    match = re.fullmatch(parent.direct_structured_output_regex, output)
    prediction = int(output.split(":", 1)[1].strip()) if match else None
    if prediction != receipt["result"]:
        return (
            {
                **direct,
                "model": f"{parent.four_b_model}+semantic-model-router-v1",
                "route": "direct_fallback_after_feedback_mismatch",
                "usage": _sum_usage(
                    direct["usage"],
                    *plan_usage,
                    final.usage,
                ),
            },
            {
                "route": route,
                "exposed_tools": [selected],
                "plan_attempts": attempts,
                "receipt": receipt,
                "final_feedback_sent": True,
                "fallback_used": True,
                "feedback_result_match": False,
            },
        )
    return (
        {
            "case_id": case["case_id"],
            "family": case["family"],
            "model": f"{parent.four_b_model}+semantic-model-router-v1",
            "route": "model_routed_verified_semantic_feedback",
            "output": output,
            "prediction": prediction,
            "parseable": True,
            "correct": prediction == case["expected"],
            "usage": _sum_usage(*plan_usage, final.usage),
            "latency_seconds": 0.0,
        },
        {
            "route": route,
            "exposed_tools": [selected],
            "plan_attempts": attempts,
            "receipt": receipt,
            "final_feedback_sent": True,
            "fallback_used": False,
            "feedback_result_match": True,
        },
    )


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_family = {}
    for family in ALL_FAMILIES:
        selected = [row for row in rows if row["family"] == family]
        by_family[family] = {
            "cases": len(selected),
            "correct": sum(bool(row["correct"]) for row in selected),
            "parseable": sum(bool(row["parseable"]) for row in selected),
        }
    count = len(rows)
    correct = sum(bool(row["correct"]) for row in rows)
    return {
        "cases": count,
        "correct": correct,
        "accuracy": correct / count,
        "parseable": sum(bool(row["parseable"]) for row in rows),
        "by_family": by_family,
    }


def run(config: SemanticModelRouterConfig) -> dict[str, Any]:
    mechanism = load_mechanism_config(config.mechanism_config_path)
    parent = parent_config(config)
    service = verify_inputs(parent)
    cases = build_cases(config)
    four_client = _client(
        parent, four_b=True, max_tokens=config.direct_max_tokens
    )
    nine_client = _client(
        parent, four_b=False, max_tokens=config.direct_max_tokens
    )
    router_client = _client(
        parent, four_b=True, max_tokens=config.router_max_tokens
    )
    plan_client = _client(
        parent, four_b=True, max_tokens=config.plan_max_tokens
    )
    final_client = _client(
        parent, four_b=True, max_tokens=config.final_max_tokens
    )
    four_rows = []
    nine_rows = []
    candidate_rows = []
    receipts = {}
    for case in cases:
        four = _direct_row(
            case, four_client, parent, model=parent.four_b_model
        )
        nine = _direct_row(
            case, nine_client, parent, model=parent.nine_b_model
        )
        candidate, receipt = _candidate_row(
            case,
            four,
            router_client,
            plan_client,
            final_client,
            config,
            mechanism,
            parent,
        )
        four_rows.append(four)
        nine_rows.append(nine)
        candidate_rows.append(candidate)
        receipts[case["case_id"]] = receipt
    result = {
        "schema_version": RESULT_SCHEMA,
        "experiment_id": config.experiment_id,
        "config": config.__dict__,
        "identity": {
            "mechanism_config_sha256": config.mechanism_config_sha256,
            "applicability_report_sha256": (
                config.applicability_report_sha256
            ),
            "case_contract": public_case_contract(cases),
        },
        "arms": {
            "four_b_direct": summarize_rows(four_rows),
            "nine_b_direct": summarize_rows(nine_rows),
            "four_b_model_router": summarize_rows(candidate_rows),
        },
        "four_b_rows": four_rows,
        "nine_b_rows": nine_rows,
        "candidate_rows": candidate_rows,
        "receipts": receipts,
        "routing": {
            "cases": len(cases),
            "router_correct": sum(
                row["router"]["correct"] for row in receipts.values()
            ),
            "positive_cases": sum(case["positive"] for case in cases),
            "positive_route_correct": sum(
                case["positive"] and receipts[case["case_id"]]["router"]["correct"]
                for case in cases
            ),
            "negative_cases": sum(not case["positive"] for case in cases),
            "negative_none_correct": sum(
                not case["positive"]
                and receipts[case["case_id"]]["router"]["selected_route"]
                == "NONE"
                for case in cases
            ),
            "negative_false_positive_routes": sum(
                not case["positive"]
                and receipts[case["case_id"]]["router"]["selected_route"]
                != "NONE"
                for case in cases
            ),
            "verified_executions": sum(
                bool(row["receipt"] and row["receipt"]["executed"])
                for row in receipts.values()
            ),
            "fallbacks": sum(
                row["fallback_used"] for row in receipts.values()
            ),
        },
        "service_receipt": service,
        "evaluation_boundary": {
            "training_eligible_cases": 0,
            "router_uses_case_metadata": False,
            "router_uses_expected_answer": False,
            "router_uses_case_correctness": False,
            "executor_uses_expected_answer": False,
            "executor_uses_case_correctness": False,
            "benchmark_rows_loaded": False,
            "canary_rows_loaded": False,
            "holdout_rows_loaded": False,
        },
    }
    output = Path(config.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result
