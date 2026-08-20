from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from nano_harness.baseline import sha256_file
from nano_harness.verified_tool_execution import (
    _client,
    _direct_row,
    _sum_usage,
    load_config as load_parent_config,
    public_case_contract,
    verify_inputs,
)


CONFIG_SCHEMA = "nano_harness_semantic_skill_execution_v1"
RESULT_SCHEMA = "nano_harness_semantic_skill_execution_result_v1"
FAMILIES = (
    "implicit_scale_total",
    "first_strict_profit_period",
)
FINAL_PATTERN = re.compile(r"^FINAL: (-?[0-9]+)$")
PLAN_PATTERN = re.compile(r"^TOOL: ([a-z_]+) (\{.*\})$")
TOOL_REGEX_BY_FAMILY = {
    "implicit_scale_total": (
        r'TOOL: implicit_scale_total \{"rows":[0-9]+,"columns":[0-9]+,'
        r'"extra":[0-9]+,"scale_word":"(?:double|triple)"\}'
    ),
    "first_strict_profit_period": (
        r'TOOL: first_strict_profit_period \{"setup_cost":[0-9]+,'
        r'"units_per_period":[0-9]+,"price_per_unit":[0-9]+,'
        r'"recurring_cost":[0-9]+\}'
    ),
}
ROUTE_MARKERS = {
    "implicit_scale_total": (
        "more than double the number of slots",
        "more than triple the number of slots",
    ),
    "first_strict_profit_period": (
        "first whole period when cumulative profit is strictly positive",
    ),
}
TOOL_FIELDS = {
    "implicit_scale_total": (
        "rows",
        "columns",
        "extra",
        "scale_word",
    ),
    "first_strict_profit_period": (
        "setup_cost",
        "units_per_period",
        "price_per_unit",
        "recurring_cost",
    ),
}
SKILL_PROMPTS = {
    "implicit_scale_total": (
        "The prompt-triggered router selected the implicit-scale-total skill. "
        "Copy rows, columns, extra, and the exact scale word double or triple "
        "from the task into the one available typed tool. Do not calculate. "
        "Return only the TOOL line."
    ),
    "first_strict_profit_period": (
        "The prompt-triggered router selected the first-strict-profit-period "
        "skill. Copy setup_cost, units_per_period, price_per_unit, and "
        "recurring_cost exactly into the one available typed tool. Do not "
        "calculate. Return only the TOOL line."
    ),
}


@dataclass(frozen=True)
class SemanticSkillExecutionConfig:
    schema_version: str
    experiment_id: str
    parent_config_path: str
    parent_config_sha256: str
    v2_report_path: str
    v2_report_sha256: str
    canary_rejection_path: str
    canary_rejection_sha256: str
    service_receipt_sha256: str
    output_path: str
    case_seed: int
    cases_per_family: int
    value_offset: int
    skill_router: str
    route_markers: dict[str, list[str]]
    plan_structured_output_regex_by_family: dict[str, str]
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


def load_config(path: str | Path) -> SemanticSkillExecutionConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(raw) != set(SemanticSkillExecutionConfig.__dataclass_fields__):
        raise ValueError("semantic skill config fields differ")
    config = SemanticSkillExecutionConfig(**raw)
    expected = {
        "schema_version": CONFIG_SCHEMA,
        "experiment_id": "qwen35-semantic-skill-execution-v1",
        "parent_config_path": (
            "configs/harness/qwen35_verified_tool_execution_v1.json"
        ),
        "parent_config_sha256": (
            "538cbcde51ccb8ad43e4f91db4201a2ffd835c1493a5b2bd58177db7dcab3cd3"
        ),
        "v2_report_path": (
            "docs/results/qwen35_verified_tool_execution_v2.public.json"
        ),
        "v2_report_sha256": (
            "cd20bd3f6abccf3e8b70f8ec6504150dc30665fbe477364608ec8b34366ab0cc"
        ),
        "canary_rejection_path": (
            "docs/results/qwen35_grounded_calculator_canary_v1.public.json"
        ),
        "canary_rejection_sha256": (
            "6f0fcebabd0bfb8099ec34e6465362c1c884524605484aec47251068e9f5b056"
        ),
        "service_receipt_sha256": (
            "895bd40d886d870f99230441baecdf1feb926e2992b6a151e54d8165145f1c0d"
        ),
        "output_path": (
            "results/harness/qwen35-semantic-skill-execution-v1/result.json"
        ),
        "case_seed": 20260820,
        "cases_per_family": 128,
        "value_offset": 120000,
        "skill_router": "prompt_marker_to_single_semantic_skill_v1",
        "route_markers": {
            family: list(markers)
            for family, markers in ROUTE_MARKERS.items()
        },
        "plan_structured_output_regex_by_family": TOOL_REGEX_BY_FAMILY,
        "direct_max_tokens": 32,
        "plan_max_tokens": 96,
        "final_max_tokens": 32,
        "plan_retry_limit": 1,
        "bootstrap_samples": 10_000,
        "bootstrap_seed": "qwen35-semantic-skill-execution-v1",
        "significance_alpha": 0.05,
        "minimum_harness_wins": 12,
        "maximum_harness_losses": 0,
        "policy": {
            "evaluation_only": True,
            "training_eligible": False,
            "contains_benchmark_rows": False,
            "contains_benchmark_outputs": False,
            "contains_canary_rows": False,
            "contains_canary_outputs": False,
            "contains_holdout_rows": False,
            "uses_observed_quality_outputs": False,
            "router_uses_case_metadata": False,
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
                f"semantic skill execution freezes {field}={expected_value}"
            )
    for path_value, digest in (
        (config.parent_config_path, config.parent_config_sha256),
        (config.v2_report_path, config.v2_report_sha256),
        (config.canary_rejection_path, config.canary_rejection_sha256),
    ):
        if sha256_file(Path(path_value)) != digest:
            raise ValueError("semantic skill evidence identity differs")
    return config


def parent_config(config: SemanticSkillExecutionConfig):
    parent = load_parent_config(config.parent_config_path)
    if (
        sha256_file(Path(parent.service_receipt_path))
        != config.service_receipt_sha256
    ):
        raise ValueError("semantic skill service receipt identity differs")
    return replace(
        parent,
        experiment_id=config.experiment_id,
        output_path=config.output_path,
        case_seed=config.case_seed,
        cases_per_family=config.cases_per_family,
        value_offset=config.value_offset,
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


def execute_semantic_tool(name: str, arguments: dict[str, Any]) -> int:
    if name == "implicit_scale_total":
        scale = {"double": 2, "triple": 3}[arguments["scale_word"]]
        return (
            scale * arguments["rows"] * arguments["columns"]
            + arguments["extra"]
        )
    if name == "first_strict_profit_period":
        net = (
            arguments["units_per_period"] * arguments["price_per_unit"]
            - arguments["recurring_cost"]
        )
        if net <= 0:
            raise ValueError("period profit must be positive")
        return arguments["setup_cost"] // net + 1
    raise ValueError(f"unsupported semantic tool: {name}")


def build_cases(
    config: SemanticSkillExecutionConfig,
) -> list[dict[str, Any]]:
    rows = []
    for family in FAMILIES:
        for index in range(config.cases_per_family):
            value = config.value_offset + index
            if family == "implicit_scale_total":
                scale_word = "double" if index % 2 == 0 else "triple"
                source_facts: dict[str, Any] = {
                    "rows": value * 3 + 17,
                    "columns": value * 2 + 13,
                    "extra": value * 5 + 19,
                    "scale_word": scale_word,
                }
                prompt = (
                    "A capacity ledger states: "
                    f"rows={source_facts['rows']}, "
                    f"columns={source_facts['columns']}, and "
                    f"extra={source_facts['extra']}. A planner requests extra "
                    f"more than {scale_word} the number of slots in the "
                    "rows-by-columns layout. Compute the exact requested total."
                )
            else:
                units = value * 2 + 11
                price = value * 3 + 7
                net = value * 5 + 97
                recurring = units * price - net
                threshold_period = value * 4 + 33
                source_facts = {
                    "setup_cost": net * threshold_period,
                    "units_per_period": units,
                    "price_per_unit": price,
                    "recurring_cost": recurring,
                }
                prompt = (
                    "A project ledger states: "
                    f"setup_cost={source_facts['setup_cost']}, "
                    f"units_per_period={source_facts['units_per_period']}, "
                    f"price_per_unit={source_facts['price_per_unit']}, and "
                    f"recurring_cost={source_facts['recurring_cost']}. Revenue "
                    "is units_per_period times price_per_unit each period. "
                    "Find the first whole period when cumulative profit is "
                    "strictly positive."
                )
            expected = execute_semantic_tool(family, source_facts)
            digest = hashlib.sha256(
                f"{family}\0{json.dumps(source_facts, sort_keys=True)}".encode()
            ).hexdigest()
            rows.append(
                {
                    "case_id": f"semantic-skill-{family}-{digest[:16]}",
                    "family": family,
                    "prompt": prompt,
                    "source_facts": source_facts,
                    "expected": expected,
                }
            )
    rows.sort(
        key=lambda row: hashlib.sha256(
            f"{config.case_seed}\0{row['case_id']}".encode()
        ).hexdigest()
    )
    return rows


def route_prompt(prompt: str) -> dict[str, Any]:
    matches = []
    for family, markers in ROUTE_MARKERS.items():
        if any(marker in prompt for marker in markers):
            matches.append(family)
    if len(matches) != 1:
        return {
            "routed": False,
            "reason": "route_ambiguous" if matches else "route_missing",
            "router_uses_case_metadata": False,
        }
    family = matches[0]
    semantic = None
    if family == "implicit_scale_total":
        words = [
            word
            for word in ("double", "triple")
            if f"more than {word} the number of slots" in prompt
        ]
        if len(words) != 1:
            return {
                "routed": False,
                "reason": "scale_word_ambiguous",
                "router_uses_case_metadata": False,
            }
        semantic = words[0]
    return {
        "routed": True,
        "reason": "prompt_marker_route",
        "family": family,
        "semantic_operator": semantic,
        "router_uses_case_metadata": False,
    }


def parse_and_execute_plan(
    text: str,
    *,
    route: dict[str, Any],
    source_facts: dict[str, Any],
) -> dict[str, Any]:
    base = {
        "schema_version": "nano_harness_semantic_skill_receipt_v1",
        "eligible": False,
        "executed": False,
        "reason": "",
        "router_uses_case_metadata": False,
        "executor_uses_expected_answer": False,
        "executor_uses_case_correctness": False,
    }
    if not route.get("routed"):
        return {**base, "reason": route["reason"]}
    match = PLAN_PATTERN.fullmatch(text.strip())
    if not match:
        return {**base, "reason": "plan_parse_failure"}
    tool_name = match.group(1)
    try:
        arguments = json.loads(match.group(2))
    except json.JSONDecodeError:
        return {**base, "reason": "arguments_json_failure"}
    if tool_name != route["family"]:
        return {
            **base,
            "reason": "tool_name_mismatch",
            "tool_name": tool_name,
        }
    if set(arguments) != set(TOOL_FIELDS[tool_name]):
        return {
            **base,
            "reason": "argument_fields_mismatch",
            "tool_name": tool_name,
        }
    if arguments != source_facts:
        return {
            **base,
            "reason": "source_facts_mismatch",
            "tool_name": tool_name,
            "arguments_sha256": hashlib.sha256(
                json.dumps(
                    arguments,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
        }
    if (
        tool_name == "implicit_scale_total"
        and arguments["scale_word"] != route["semantic_operator"]
    ):
        return {
            **base,
            "reason": "semantic_operator_mismatch",
            "tool_name": tool_name,
        }
    result = execute_semantic_tool(tool_name, arguments)
    return {
        **base,
        "eligible": True,
        "executed": True,
        "reason": "verified_semantic_execution",
        "tool_name": tool_name,
        "arguments": arguments,
        "result": result,
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_family = {}
    for family in FAMILIES:
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
        "accuracy": correct / count if count else None,
        "parseable": sum(bool(row["parseable"]) for row in rows),
        "by_family": by_family,
    }


def _harness_row(
    case: dict[str, Any],
    direct: dict[str, Any],
    plan_client: Any,
    final_client: Any,
    config: SemanticSkillExecutionConfig,
    parent: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.perf_counter()
    route = route_prompt(case["prompt"])
    if not route["routed"]:
        return (
            {
                **direct,
                "model": f"{parent.four_b_model}+semantic-skills-v1",
                "route": "direct_fallback_after_route_failure",
            },
            {
                "route": route,
                "exposed_tools": [],
                "plan_attempts": [],
                "receipt": None,
                "final_feedback_sent": False,
                "fallback_used": True,
            },
        )
    family = route["family"]
    plan_messages = [
        {"role": "system", "content": SKILL_PROMPTS[family]},
        {"role": "user", "content": case["prompt"]},
    ]
    attempts = []
    plan_usage = []
    receipt = None
    for attempt in range(config.plan_retry_limit + 1):
        reply = plan_client.complete(
            plan_messages,
            extra_body={
                "structured_outputs": {
                    "regex": config.plan_structured_output_regex_by_family[
                        family
                    ]
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
        if attempt < config.plan_retry_limit:
            plan_messages.extend(
                [
                    {"role": "assistant", "content": reply.content},
                    {
                        "role": "user",
                        "content": (
                            "The typed semantic skill rejected the plan with "
                            f"reason={receipt['reason']}. Copy the original "
                            "labeled facts and semantic word exactly into the "
                            "same available tool."
                        ),
                    },
                ]
            )
    assert receipt is not None
    if not receipt["executed"]:
        return (
            {
                **direct,
                "model": f"{parent.four_b_model}+semantic-skills-v1",
                "route": "direct_fallback_after_invalid_plan",
                "usage": _sum_usage(direct["usage"], *plan_usage),
                "latency_seconds": time.perf_counter() - started,
            },
            {
                "route": route,
                "exposed_tools": [family],
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
                    "Return the verified semantic-tool result without changing "
                    "it. Return only one FINAL: <integer> line."
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
    match = FINAL_PATTERN.fullmatch(output)
    prediction = int(match.group(1)) if match else None
    if prediction != receipt["result"]:
        return (
            {
                **direct,
                "model": f"{parent.four_b_model}+semantic-skills-v1",
                "route": "direct_fallback_after_feedback_mismatch",
                "usage": _sum_usage(
                    direct["usage"],
                    *plan_usage,
                    final.usage,
                ),
                "latency_seconds": time.perf_counter() - started,
            },
            {
                "route": route,
                "exposed_tools": [family],
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
            "model": f"{parent.four_b_model}+semantic-skills-v1",
            "route": "prompt_routed_verified_semantic_feedback",
            "output": output,
            "prediction": prediction,
            "parseable": True,
            "correct": prediction == case["expected"],
            "usage": _sum_usage(*plan_usage, final.usage),
            "latency_seconds": time.perf_counter() - started,
        },
        {
            "route": route,
            "exposed_tools": [family],
            "plan_attempts": attempts,
            "receipt": receipt,
            "final_feedback_sent": True,
            "fallback_used": False,
            "feedback_result_match": True,
        },
    )


def run(config: SemanticSkillExecutionConfig) -> dict[str, Any]:
    parent = parent_config(config)
    service = verify_inputs(parent)
    cases = build_cases(config)
    four_client = _client(
        parent, four_b=True, max_tokens=config.direct_max_tokens
    )
    nine_client = _client(
        parent, four_b=False, max_tokens=config.direct_max_tokens
    )
    plan_client = _client(
        parent, four_b=True, max_tokens=config.plan_max_tokens
    )
    final_client = _client(
        parent, four_b=True, max_tokens=config.final_max_tokens
    )
    four_rows = []
    nine_rows = []
    harness_rows = []
    receipts = {}
    for case in cases:
        four = _direct_row(
            case, four_client, parent, model=parent.four_b_model
        )
        nine = _direct_row(
            case, nine_client, parent, model=parent.nine_b_model
        )
        harness, receipt = _harness_row(
            case,
            four,
            plan_client,
            final_client,
            config,
            parent,
        )
        four_rows.append(four)
        nine_rows.append(nine)
        harness_rows.append(harness)
        receipts[case["case_id"]] = receipt
    result = {
        "schema_version": RESULT_SCHEMA,
        "experiment_id": config.experiment_id,
        "config": config.__dict__,
        "identity": {
            "parent_config_sha256": config.parent_config_sha256,
            "v2_report_sha256": config.v2_report_sha256,
            "canary_rejection_sha256": config.canary_rejection_sha256,
            "service_receipt_sha256": config.service_receipt_sha256,
            "case_contract": public_case_contract(cases),
        },
        "arms": {
            "four_b_direct": summarize_rows(four_rows),
            "nine_b_direct": summarize_rows(nine_rows),
            "four_b_semantic_skills": summarize_rows(harness_rows),
        },
        "four_b_rows": four_rows,
        "nine_b_rows": nine_rows,
        "harness_rows": harness_rows,
        "harness_receipts": receipts,
        "routing": {
            "prompt_routes": sum(
                row["route"]["routed"] for row in receipts.values()
            ),
            "single_tool_exposures": sum(
                len(row["exposed_tools"]) == 1 for row in receipts.values()
            ),
            "verified_executions": sum(
                bool(row["receipt"] and row["receipt"]["executed"])
                for row in receipts.values()
            ),
            "plan_retries": sum(
                len(row["plan_attempts"]) - 1 for row in receipts.values()
            ),
            "fallbacks": sum(
                row["fallback_used"] for row in receipts.values()
            ),
            "final_feedback_calls": sum(
                row["final_feedback_sent"] for row in receipts.values()
            ),
            "feedback_result_matches": sum(
                row.get("feedback_result_match", False)
                for row in receipts.values()
            ),
        },
        "service_receipt": service,
        "evaluation_boundary": {
            "training_eligible_cases": 0,
            "router_uses_case_metadata": False,
            "executor_uses_expected_answer": False,
            "executor_uses_case_correctness": False,
            "benchmark_rows_loaded": False,
            "benchmark_outputs_loaded": False,
            "canary_rows_loaded": False,
            "canary_outputs_loaded": False,
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
