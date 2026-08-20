from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from nano_harness.baseline import sha256_file
from nano_harness.router_adapter_integration import (
    LABEL_TO_ROUTE,
    ROUTER_SYSTEM,
    _sha256_tree,
    parse_route,
)
from nano_harness.semantic_model_router import _model_selected_tool_row
from nano_harness.semantic_skill_execution import (
    load_config as load_mechanism_config,
    parent_config as load_parent_runtime,
)
from nano_harness.verified_tool_execution import (
    _client,
    _direct_row,
    _sum_usage,
    public_case_contract,
    verify_inputs,
)


CONFIG_SCHEMA = "nano_harness_router_skill_fallback_v4"
RESULT_SCHEMA = "nano_harness_router_skill_fallback_result_v4"
CONFIG_SHA256 = (
    "240b6e5273c5c0bd111e5c76fa3ab37697f19f6d7666d64129ecdb6d00563f89"
)
POSITIVE_FAMILIES = (
    "implicit_scale_total",
    "first_strict_profit_period",
)
C_FAMILIES = (
    "box_total",
    "remaining_stock",
    "paired_average",
    "single_operation",
    "weighted_total",
    "quotient_remainder",
    "time_conversion",
    "percentage_change",
)
PLAN_PATTERN = re.compile(r"^TOOL: ([a-z_]+) (\{.*\})$")
TOOL_FIELDS = {
    "box_total": ("boxes", "items_per_box", "loose_items"),
    "remaining_stock": (
        "starting_units",
        "batches_used",
        "units_per_batch",
    ),
    "paired_average": ("first_total", "second_total"),
    "single_operation": ("left", "right", "operation"),
    "weighted_total": (
        "first_count",
        "first_weight",
        "second_count",
        "second_weight",
    ),
    "quotient": ("dividend", "divisor"),
    "time_to_minutes": ("days", "hours", "minutes"),
    "percentage_change": ("original", "percent", "direction"),
}
FAMILY_TO_TOOL = {
    "box_total": "box_total",
    "remaining_stock": "remaining_stock",
    "paired_average": "paired_average",
    "single_operation": "single_operation",
    "weighted_total": "weighted_total",
    "quotient_remainder": "quotient",
    "time_conversion": "time_to_minutes",
    "percentage_change": "percentage_change",
}
C_SKILL_PROMPT = (
    "Select exactly one typed arithmetic skill and copy the task facts into "
    "its JSON without calculating. Available schemas: "
    'box_total {"boxes":N,"items_per_box":N,"loose_items":N}; '
    'remaining_stock {"starting_units":N,"batches_used":N,'
    '"units_per_batch":N}; '
    'paired_average {"first_total":N,"second_total":N}; '
    'single_operation {"left":N,"right":N,"operation":"sum|difference|'
    'product|quotient"}; '
    'weighted_total {"first_count":N,"first_weight":N,"second_count":N,'
    '"second_weight":N}; '
    'quotient {"dividend":N,"divisor":N}; '
    'time_to_minutes {"days":N,"hours":N,"minutes":N}; '
    'percentage_change {"original":N,"percent":N,"direction":"increase|'
    'decrease"}. Return only TOOL: <name> <json>.'
)


@dataclass(frozen=True)
class RouterSkillFallbackV4Config:
    adapter_path: str
    adapter_tree_sha256: str
    adapter_weights_sha256: str
    bootstrap_samples: int
    bootstrap_seed: str
    case_seed: int
    cases_per_family: int
    direct_max_tokens: int
    execution_boundary: dict[str, bool]
    experiment_id: str
    final_max_tokens: int
    integration_v3_report_path: str
    integration_v3_report_sha256: str
    maximum_harness_losses: int
    mechanism_config_path: str
    mechanism_config_sha256: str
    minimum_harness_wins: int
    output_path: str
    parent_config_path: str
    parent_config_sha256: str
    plan_max_tokens: int
    plan_retry_limit: int
    policy: dict[str, bool]
    route_max_tokens: int
    route_structured_output_regex: str
    router_training_data_path: str
    router_training_data_sha256: str
    schema_version: str
    served_adapter_name: str
    service_models: dict[str, str]
    service_receipt_path: str
    significance_alpha: float
    skill_plan_structured_output_regex: str
    value_offset: int


def load_config(path: str | Path) -> RouterSkillFallbackV4Config:
    config_path = Path(path)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if set(raw) != set(RouterSkillFallbackV4Config.__dataclass_fields__):
        raise ValueError("router skill fallback v4 config fields differ")
    if sha256_file(config_path) != CONFIG_SHA256:
        raise ValueError("router skill fallback v4 config SHA differs")
    config = RouterSkillFallbackV4Config(**raw)
    if config.schema_version != CONFIG_SCHEMA:
        raise ValueError("unsupported router skill fallback v4 schema")
    for source, digest in (
        (config.mechanism_config_path, config.mechanism_config_sha256),
        (config.parent_config_path, config.parent_config_sha256),
        (config.router_training_data_path, config.router_training_data_sha256),
        (config.integration_v3_report_path, config.integration_v3_report_sha256),
    ):
        if sha256_file(Path(source)) != digest:
            raise ValueError("router skill fallback v4 evidence identity differs")
    adapter = Path(config.adapter_path)
    if (
        _sha256_tree(adapter) != config.adapter_tree_sha256
        or sha256_file(adapter / "adapter_model.safetensors")
        != config.adapter_weights_sha256
    ):
        raise ValueError("router skill fallback v4 adapter identity differs")
    v3 = json.loads(
        Path(config.integration_v3_report_path).read_text(encoding="utf-8")
    )
    if (
        v3.get("decision", {}).get("adapter_integration_v3_admitted") is not False
        or v3.get("decision", {}).get("integration_v3_rerun_allowed")
        is not False
        or v3.get("mechanism_conclusion", {}).get("router_transfer_succeeded")
        is not True
        or v3.get("mechanism_conclusion", {}).get(
            "verified_positive_execution_succeeded"
        )
        is not True
        or v3.get("mechanism_conclusion", {}).get(
            "direct_preserve_policy_rejected"
        )
        is not True
        or v3.get("mechanism_conclusion", {}).get(
            "post_observation_policy_tuning_allowed"
        )
        is not False
        or v3.get("identity", {}).get("remapped_adapter_sha256")
        != config.adapter_tree_sha256
    ):
        raise ValueError("router skill fallback v4 predecessor decision differs")
    return config


def parent_config(config: RouterSkillFallbackV4Config):
    mechanism = load_mechanism_config(config.mechanism_config_path)
    parent = load_parent_runtime(mechanism)
    return mechanism, replace(
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


def execute_c_skill(name: str, arguments: dict[str, Any]) -> int:
    if name == "box_total":
        return (
            arguments["boxes"] * arguments["items_per_box"]
            + arguments["loose_items"]
        )
    if name == "remaining_stock":
        return (
            arguments["starting_units"]
            - arguments["batches_used"] * arguments["units_per_batch"]
        )
    if name == "paired_average":
        total = arguments["first_total"] + arguments["second_total"]
        if total % 2:
            raise ValueError("paired average must be integral")
        return total // 2
    if name == "single_operation":
        operation = arguments["operation"]
        if operation == "sum":
            return arguments["left"] + arguments["right"]
        if operation == "difference":
            return arguments["left"] - arguments["right"]
        if operation == "product":
            return arguments["left"] * arguments["right"]
        if arguments["right"] == 0 or arguments["left"] % arguments["right"]:
            raise ValueError("quotient must be exact")
        return arguments["left"] // arguments["right"]
    if name == "weighted_total":
        return (
            arguments["first_count"] * arguments["first_weight"]
            + arguments["second_count"] * arguments["second_weight"]
        )
    if name == "quotient":
        if arguments["divisor"] == 0:
            raise ValueError("divisor must be nonzero")
        return arguments["dividend"] // arguments["divisor"]
    if name == "time_to_minutes":
        return (
            arguments["days"] * 24 * 60
            + arguments["hours"] * 60
            + arguments["minutes"]
        )
    if name == "percentage_change":
        delta = arguments["original"] * arguments["percent"]
        if delta % 100:
            raise ValueError("percentage result must be integral")
        delta //= 100
        if arguments["direction"] == "increase":
            return arguments["original"] + delta
        return arguments["original"] - delta
    raise ValueError(f"unsupported C skill: {name}")


def parse_and_execute_c_plan(
    text: str,
    *,
    source_facts: dict[str, Any],
) -> dict[str, Any]:
    base = {
        "schema_version": "nano_harness_router_c_skill_receipt_v4",
        "eligible": False,
        "executed": False,
        "reason": "",
        "selector_uses_case_metadata": False,
        "executor_uses_expected_answer": False,
        "executor_uses_case_correctness": False,
    }
    match = PLAN_PATTERN.fullmatch(text.strip())
    if not match:
        return {**base, "reason": "plan_parse_failure"}
    tool_name = match.group(1)
    if tool_name not in TOOL_FIELDS:
        return {**base, "reason": "unsupported_tool", "tool_name": tool_name}
    try:
        arguments = json.loads(match.group(2))
    except json.JSONDecodeError:
        return {**base, "reason": "arguments_json_failure"}
    if tuple(arguments) != TOOL_FIELDS[tool_name]:
        return {
            **base,
            "reason": "argument_fields_or_order_mismatch",
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
    try:
        result = execute_c_skill(tool_name, arguments)
    except (KeyError, TypeError, ValueError) as exc:
        return {
            **base,
            "reason": "deterministic_execution_rejected",
            "tool_name": tool_name,
            "error_type": type(exc).__name__,
        }
    return {
        **base,
        "eligible": True,
        "executed": True,
        "reason": "verified_c_skill_execution",
        "tool_name": tool_name,
        "arguments": arguments,
        "result": result,
    }


def _case_row(
    *,
    family: str,
    prompt: str,
    facts: dict[str, Any],
    expected: int,
    label: str,
) -> dict[str, Any]:
    digest = hashlib.sha256(
        f"{family}\0{json.dumps(facts, sort_keys=True)}".encode()
    ).hexdigest()
    return {
        "case_id": f"router-skill-v4-{family}-{digest[:16]}",
        "family": family,
        "prompt": prompt,
        "source_facts": facts,
        "expected": expected,
        "expected_label": label,
        "expected_route": LABEL_TO_ROUTE[label],
        "positive": family in POSITIVE_FAMILIES,
    }


def build_cases(config: RouterSkillFallbackV4Config) -> list[dict[str, Any]]:
    rows = []
    for family_index, family in enumerate(POSITIVE_FAMILIES):
        for index in range(config.cases_per_family):
            value = config.value_offset + family_index * 100_000 + index
            if family == "implicit_scale_total":
                scale_word = "double" if index % 2 == 0 else "triple"
                facts: dict[str, Any] = {
                    "rows": value * 3 + 401,
                    "columns": value * 2 + 409,
                    "extra": value * 5 + 419,
                    "scale_word": scale_word,
                }
                multiplier = "twice" if scale_word == "double" else "three times"
                prompt = (
                    f"A lunar greenhouse has {facts['rows']} planting rows and "
                    f"{facts['columns']} cells in every row. The expansion "
                    f"contains {multiplier} the rectangular cell count plus "
                    f"{facts['extra']} reserve cells. Compute its exact size."
                )
                expected = (
                    (2 if scale_word == "double" else 3)
                    * facts["rows"]
                    * facts["columns"]
                    + facts["extra"]
                )
                label = "A"
            else:
                units = value * 2 + 421
                price = value * 3 + 431
                net = value * 4 + 433
                recurring = units * price - net
                threshold = value * 2 + 439
                facts = {
                    "setup_cost": net * threshold,
                    "units_per_period": units,
                    "price_per_unit": price,
                    "recurring_cost": recurring,
                }
                prompt = (
                    f"A satellite network requires {facts['setup_cost']} to "
                    f"deploy. Each full quarter it sells "
                    f"{facts['units_per_period']} connections at "
                    f"{facts['price_per_unit']} each and spends "
                    f"{facts['recurring_cost']}. Find the first full quarter "
                    "after which cumulative profit exceeds zero."
                )
                expected = facts["setup_cost"] // net + 1
                label = "B"
            rows.append(
                _case_row(
                    family=family,
                    prompt=prompt,
                    facts=facts,
                    expected=expected,
                    label=label,
                )
            )
    for family_index, family in enumerate(C_FAMILIES):
        for index in range(config.cases_per_family):
            value = (
                config.value_offset
                + 1_000_000
                + family_index * 100_000
                + index
            )
            prompt, facts = _build_c_prompt(family, value=value, index=index)
            expected = execute_c_skill(FAMILY_TO_TOOL[family], facts)
            rows.append(
                _case_row(
                    family=family,
                    prompt=prompt,
                    facts=facts,
                    expected=expected,
                    label="C",
                )
            )
    rows.sort(
        key=lambda row: hashlib.sha256(
            f"{config.case_seed}\0{row['case_id']}".encode()
        ).hexdigest()
    )
    return rows


def _build_c_prompt(
    family: str,
    *,
    value: int,
    index: int,
) -> tuple[str, dict[str, Any]]:
    if family == "box_total":
        facts = {
            "boxes": value * 2 + 443,
            "items_per_box": value * 3 + 449,
            "loose_items": value * 7 + 457,
        }
        return (
            f"A seed vault stores {facts['boxes']} cases with "
            f"{facts['items_per_box']} packets per case and "
            f"{facts['loose_items']} unpacked packets. Find the exact total.",
            facts,
        )
    if family == "remaining_stock":
        batches = value * 2 + 461
        units = value * 3 + 463
        remaining = value * 5 + 467
        facts = {
            "starting_units": batches * units + remaining,
            "batches_used": batches,
            "units_per_batch": units,
        }
        return (
            f"A medical reserve starts with {facts['starting_units']} doses "
            f"and ships {facts['batches_used']} batches of "
            f"{facts['units_per_batch']} doses. How many doses remain?",
            facts,
        )
    if family == "paired_average":
        facts = {
            "first_total": value * 10 + 470,
            "second_total": value * 14 + 474,
        }
        return (
            f"Two telescope counters read {facts['first_total']} and "
            f"{facts['second_total']}. Compute their exact arithmetic mean.",
            facts,
        )
    if family == "single_operation":
        operations = ("sum", "difference", "product", "quotient")
        operation = operations[index % 4]
        right = value * 2 + 479
        left = value * 5 + 487
        if operation == "quotient":
            quotient = value % 97 + 23
            left = right * quotient
        facts = {"left": left, "right": right, "operation": operation}
        return (
            f"A signal processor receives {left} and {right}. Return their "
            f"exact {operation}.",
            facts,
        )
    if family == "weighted_total":
        facts = {
            "first_count": value * 2 + 491,
            "first_weight": value % 19 + 17,
            "second_count": value * 3 + 499,
            "second_weight": value % 23 + 19,
        }
        return (
            f"A rover carries {facts['first_count']} light cells of "
            f"{facts['first_weight']} units each and "
            f"{facts['second_count']} heavy cells of "
            f"{facts['second_weight']} units each. Find total mass.",
            facts,
        )
    if family == "quotient_remainder":
        divisor = value % 83 + 31
        quotient = value * 2 + 503
        remainder = value % divisor
        facts = {
            "dividend": divisor * quotient + remainder,
            "divisor": divisor,
        }
        return (
            f"A telemetry archive divides {facts['dividend']} records into "
            f"groups of {facts['divisor']}. How many complete groups are "
            "formed?",
            facts,
        )
    if family == "time_conversion":
        facts = {
            "days": value % 31 + 7,
            "hours": value % 23,
            "minutes": value % 59,
        }
        return (
            f"An orbital test lasts {facts['days']} days, {facts['hours']} "
            f"hours, and {facts['minutes']} minutes. Convert all of it to "
            "minutes.",
            facts,
        )
    percent = (5, 10, 20, 25)[index % 4]
    direction = "increase" if index % 2 == 0 else "decrease"
    facts = {
        "original": (value * 4 + 509) * 20,
        "percent": percent,
        "direction": direction,
    }
    return (
        f"A habitat score begins at {facts['original']} and has a "
        f"{percent} percent {direction}. Find the updated score.",
        facts,
    )


def _c_skill_candidate(
    case: dict[str, Any],
    direct: dict[str, Any],
    planner: Any,
    router_usage: dict[str, Any],
    config: RouterSkillFallbackV4Config,
    parent: Any,
    router_receipt: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.perf_counter()
    messages = [
        {"role": "system", "content": C_SKILL_PROMPT},
        {"role": "user", "content": case["prompt"]},
    ]
    attempts = []
    usages = [router_usage]
    receipt = None
    for attempt in range(config.plan_retry_limit + 1):
        reply = planner.complete(
            messages,
            extra_body={
                "structured_outputs": {
                    "regex": config.skill_plan_structured_output_regex
                }
            },
        )
        usages.append(reply.usage)
        receipt = parse_and_execute_c_plan(
            reply.content,
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
            messages.extend(
                [
                    {"role": "assistant", "content": reply.content},
                    {
                        "role": "user",
                        "content": (
                            "The typed verifier rejected that plan with "
                            f"reason={receipt['reason']}. Select the matching "
                            "schema and copy every original number exactly."
                        ),
                    },
                ]
            )
    assert receipt is not None
    if not receipt["executed"]:
        return (
            {
                **direct,
                "model": f"{parent.four_b_model}+router-c-skills-v4",
                "route": "direct_fallback_after_invalid_c_skill",
                "usage": _sum_usage(direct["usage"], *usages),
                "latency_seconds": time.perf_counter() - started,
            },
            {
                "router": router_receipt,
                "c_skill_attempts": attempts,
                "c_skill_receipt": receipt,
                "fallback_used": True,
            },
        )
    output = f"FINAL: {receipt['result']}"
    return (
        {
            "case_id": case["case_id"],
            "family": case["family"],
            "model": f"{parent.four_b_model}+router-c-skills-v4",
            "route": "router_c_typed_skill_verified",
            "output": output,
            "prediction": receipt["result"],
            "parseable": True,
            "correct": receipt["result"] == case["expected"],
            "usage": _sum_usage(*usages),
            "latency_seconds": time.perf_counter() - started,
        },
        {
            "router": router_receipt,
            "c_skill_attempts": attempts,
            "c_skill_receipt": receipt,
            "fallback_used": False,
        },
    )


def run(config: RouterSkillFallbackV4Config) -> dict[str, Any]:
    mechanism, parent = parent_config(config)
    base_service = verify_inputs(parent)
    service = json.loads(
        Path(config.service_receipt_path).read_text(encoding="utf-8")
    )
    if (
        service.get("schema_version")
        != "nano_harness_router_skill_fallback_v4_service"
        or service.get("healthy") is not True
        or service.get("v4_generation_started") is not False
        or service.get("integration_v1_v2_v3_rerun") is not False
        or service.get("models") != config.service_models
        or service.get("remapped_adapter_sha256")
        != config.adapter_tree_sha256
    ):
        raise ValueError("router skill fallback v4 service receipt differs")
    cases = build_cases(config)
    four = _client(parent, four_b=True, max_tokens=config.direct_max_tokens)
    nine = _client(parent, four_b=False, max_tokens=config.direct_max_tokens)
    router = _client(parent, four_b=True, max_tokens=config.route_max_tokens)
    router.config = replace(
        router.config,
        name=config.served_adapter_name,
        max_tokens=config.route_max_tokens,
    )
    planner = _client(parent, four_b=True, max_tokens=config.plan_max_tokens)
    final = _client(parent, four_b=True, max_tokens=config.final_max_tokens)
    four_rows = []
    nine_rows = []
    candidate_rows = []
    receipts = {}
    for case in cases:
        direct = _direct_row(case, four, parent, model=parent.four_b_model)
        baseline_nine = _direct_row(
            case,
            nine,
            parent,
            model=parent.nine_b_model,
        )
        route_reply = router.complete(
            [
                {"role": "system", "content": ROUTER_SYSTEM},
                {"role": "user", "content": case["prompt"]},
            ],
            extra_body={
                "structured_outputs": {
                    "regex": config.route_structured_output_regex
                }
            },
        )
        label = parse_route(route_reply.content)
        selected = LABEL_TO_ROUTE.get(label) if label is not None else None
        router_receipt = {
            "output": route_reply.content,
            "output_sha256": hashlib.sha256(
                route_reply.content.encode()
            ).hexdigest(),
            "label": label,
            "selected_route": selected,
            "expected_label": case["expected_label"],
            "expected_route": case["expected_route"],
            "correct": selected == case["expected_route"],
            "model": config.served_adapter_name,
            "adapter_sha256": config.adapter_tree_sha256,
            "uses_case_metadata": False,
            "uses_expected_answer": False,
            "uses_case_correctness": False,
        }
        if selected in POSITIVE_FAMILIES:
            routed_case = {**case, "family": selected}
            candidate, tool_receipt = _model_selected_tool_row(
                routed_case,
                direct,
                selected,
                planner,
                final,
                mechanism,
                parent,
            )
            candidate["family"] = case["family"]
            candidate["model"] = f"{parent.four_b_model}+router-c-skills-v4"
            candidate["usage"] = _sum_usage(
                route_reply.usage,
                candidate["usage"],
            )
            receipt = {"router": router_receipt, **tool_receipt}
        elif selected == "NONE":
            candidate, receipt = _c_skill_candidate(
                case,
                direct,
                planner,
                route_reply.usage,
                config,
                parent,
                router_receipt,
            )
        else:
            candidate = {
                **direct,
                "model": f"{parent.four_b_model}+router-c-skills-v4",
                "route": "direct_fallback_after_router_parse_failure",
                "usage": _sum_usage(direct["usage"], route_reply.usage),
            }
            receipt = {
                "router": router_receipt,
                "fallback_used": True,
            }
        four_rows.append(direct)
        nine_rows.append(baseline_nine)
        candidate_rows.append(candidate)
        receipts[case["case_id"]] = receipt
    result = {
        "schema_version": RESULT_SCHEMA,
        "experiment_id": config.experiment_id,
        "config": config.__dict__,
        "identity": {
            "integration_v3_report_sha256": (
                config.integration_v3_report_sha256
            ),
            "router_training_data_sha256": (
                config.router_training_data_sha256
            ),
            "adapter_sha256": config.adapter_tree_sha256,
            "case_contract": public_case_contract(cases),
        },
        "four_b_rows": four_rows,
        "nine_b_rows": nine_rows,
        "candidate_rows": candidate_rows,
        "receipts": receipts,
        "routing": {
            "cases": len(cases),
            "correct": sum(
                receipt["router"]["correct"] for receipt in receipts.values()
            ),
            "positive_cases": sum(case["positive"] for case in cases),
            "negative_cases": sum(not case["positive"] for case in cases),
            "c_skill_executions": sum(
                bool(receipt.get("c_skill_receipt", {}).get("executed"))
                for receipt in receipts.values()
            ),
            "ab_verified_executions": sum(
                bool(receipt.get("receipt", {}).get("executed"))
                for receipt in receipts.values()
            ),
            "fallbacks": sum(
                bool(receipt.get("fallback_used"))
                for receipt in receipts.values()
            ),
        },
        "base_service_receipt": base_service,
        "v4_service_receipt": service,
        "evaluation_boundary": {
            "training_eligible_cases": 0,
            "router_uses_case_metadata": False,
            "router_uses_expected_answer": False,
            "router_uses_case_correctness": False,
            "skill_selector_uses_case_metadata": False,
            "executor_uses_expected_answer": False,
            "executor_uses_case_correctness": False,
            "integration_v1_v2_v3_rows_or_outputs_loaded": False,
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
