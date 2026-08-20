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
from nano_harness.router_skill_fallback_v4 import (
    C_FAMILIES,
    FAMILY_TO_TOOL,
    POSITIVE_FAMILIES,
    TOOL_FIELDS,
    execute_c_skill,
    parse_and_execute_c_plan,
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


CONFIG_SCHEMA = "nano_harness_router_skill_registry_v5"
RESULT_SCHEMA = "nano_harness_router_skill_registry_result_v5"
CONFIG_SHA256 = (
    "f8544c6337948be87e8a34721bc10906fc724712d8b0ffed24381ef13a59ee91"
)
SKILL_REGEX = {
    "box_total": (
        r'TOOL: box_total \{"boxes":[0-9]+,"items_per_box":[0-9]+,'
        r'"loose_items":[0-9]+\}'
    ),
    "remaining_stock": (
        r'TOOL: remaining_stock \{"starting_units":[0-9]+,'
        r'"batches_used":[0-9]+,"units_per_batch":[0-9]+\}'
    ),
    "paired_average": (
        r'TOOL: paired_average \{"first_total":[0-9]+,'
        r'"second_total":[0-9]+\}'
    ),
    "single_operation": (
        r'TOOL: single_operation \{"left":[0-9]+,"right":[0-9]+,'
        r'"operation":"(?:sum|difference|product|quotient)"\}'
    ),
    "weighted_total": (
        r'TOOL: weighted_total \{"first_count":[0-9]+,'
        r'"first_weight":[0-9]+,"second_count":[0-9]+,'
        r'"second_weight":[0-9]+\}'
    ),
    "quotient_remainder": (
        r'TOOL: quotient \{"dividend":[0-9]+,"divisor":[0-9]+\}'
    ),
    "time_conversion": (
        r'TOOL: time_to_minutes \{"days":[0-9]+,"hours":[0-9]+,'
        r'"minutes":[0-9]+\}'
    ),
    "percentage_change": (
        r'TOOL: percentage_change \{"original":[0-9]+,"percent":[0-9]+,'
        r'"direction":"(?:increase|decrease)"\}'
    ),
}
SKILL_PROMPTS = {
    "box_total": (
        'Copy the case count, pieces per case, and loose pieces into exactly: '
        'TOOL: box_total {"boxes":N,"items_per_box":N,"loose_items":N}. '
        "Do not calculate. Return only that TOOL line."
    ),
    "remaining_stock": (
        "Copy starting quantity, number of batches used, and units per batch "
        'into exactly: TOOL: remaining_stock {"starting_units":N,'
        '"batches_used":N,"units_per_batch":N}. Do not calculate. Return only '
        "that TOOL line."
    ),
    "paired_average": (
        'Copy the first and second readings into exactly: TOOL: paired_average '
        '{"first_total":N,"second_total":N}. Do not calculate. Return only '
        "that TOOL line."
    ),
    "single_operation": (
        "Copy the two displayed integers and requested operation into exactly: "
        'TOOL: single_operation {"left":N,"right":N,"operation":'
        '"sum|difference|product|quotient"}. Do not calculate. Return only '
        "that TOOL line."
    ),
    "weighted_total": (
        "Copy both counts and both per-item masses into exactly: TOOL: "
        'weighted_total {"first_count":N,"first_weight":N,"second_count":N,'
        '"second_weight":N}. Do not calculate. Return only that TOOL line.'
    ),
    "quotient_remainder": (
        'Copy the record total and group size into exactly: TOOL: quotient '
        '{"dividend":N,"divisor":N}. Do not calculate. Return only that '
        "TOOL line."
    ),
    "time_conversion": (
        "Copy days, hours, and minutes into exactly: TOOL: time_to_minutes "
        '{"days":N,"hours":N,"minutes":N}. Do not calculate. Return only '
        "that TOOL line."
    ),
    "percentage_change": (
        "Copy original score, percent, and whether it is an increase or "
        'decrease into exactly: TOOL: percentage_change {"original":N,'
        '"percent":N,"direction":"increase|decrease"}. Do not calculate. '
        "Return only that TOOL line."
    ),
}


@dataclass(frozen=True)
class RouterSkillRegistryV5Config:
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
    skill_registry_policy: str
    v4_report_path: str
    v4_report_sha256: str
    value_offset: int


def load_config(path: str | Path) -> RouterSkillRegistryV5Config:
    config_path = Path(path)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if set(raw) != set(RouterSkillRegistryV5Config.__dataclass_fields__):
        raise ValueError("router skill registry v5 config fields differ")
    if sha256_file(config_path) != CONFIG_SHA256:
        raise ValueError("router skill registry v5 config SHA differs")
    config = RouterSkillRegistryV5Config(**raw)
    if config.schema_version != CONFIG_SCHEMA:
        raise ValueError("unsupported router skill registry v5 schema")
    for source, digest in (
        (config.mechanism_config_path, config.mechanism_config_sha256),
        (config.parent_config_path, config.parent_config_sha256),
        (config.router_training_data_path, config.router_training_data_sha256),
        (config.v4_report_path, config.v4_report_sha256),
    ):
        if sha256_file(Path(source)) != digest:
            raise ValueError("router skill registry v5 evidence identity differs")
    adapter = Path(config.adapter_path)
    if (
        _sha256_tree(adapter) != config.adapter_tree_sha256
        or sha256_file(adapter / "adapter_model.safetensors")
        != config.adapter_weights_sha256
    ):
        raise ValueError("router skill registry v5 adapter identity differs")
    v4 = json.loads(Path(config.v4_report_path).read_text(encoding="utf-8"))
    if (
        v4.get("decision", {}).get("router_skill_fallback_v4_admitted")
        is not False
        or v4.get("decision", {}).get("v1_v2_v3_v4_rerun_allowed")
        is not False
        or v4.get("mechanism_conclusion", {}).get(
            "router_transfer_succeeded"
        )
        is not True
        or v4.get("mechanism_conclusion", {}).get(
            "ab_verified_execution_succeeded"
        )
        is not True
        or v4.get("mechanism_conclusion", {}).get(
            "shared_c_skill_selector_admitted"
        )
        is not False
        or v4.get("mechanism_conclusion", {}).get(
            "post_observation_skill_prompt_or_schema_tuning_allowed"
        )
        is not False
        or v4.get("identity", {}).get("remapped_adapter_sha256")
        != config.adapter_tree_sha256
    ):
        raise ValueError("router skill registry v5 predecessor decision differs")
    return config


def parent_config(config: RouterSkillRegistryV5Config):
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


def applicable_c_skills(prompt: str) -> list[str]:
    normalized = " ".join(prompt.casefold().split())
    predicates = {
        "box_total": (
            "packed cases" in normalized
            and "pieces in each case" in normalized
            and "loose pieces" in normalized
        ),
        "remaining_stock": (
            "starts with" in normalized
            and "batches of" in normalized
            and "remain" in normalized
        ),
        "paired_average": (
            "arithmetic average" in normalized
            and "two calibrated readings" in normalized
        ),
        "single_operation": (
            "two integers" in normalized
            and any(
                f"exact {operation}" in normalized
                for operation in ("sum", "difference", "product", "quotient")
            )
        ),
        "weighted_total": (
            "units each" in normalized
            and "combined payload mass" in normalized
        ),
        "quotient_remainder": (
            "records into groups of" in normalized
            and "complete groups" in normalized
        ),
        "time_conversion": (
            "days" in normalized
            and "hours" in normalized
            and "minutes" in normalized
            and "total minutes" in normalized
        ),
        "percentage_change": (
            "percent" in normalized
            and "updated index" in normalized
            and any(
                marker in normalized for marker in ("increase", "decrease")
            )
        ),
    }
    return [name for name, matched in predicates.items() if matched]


def build_cases(config: RouterSkillRegistryV5Config) -> list[dict[str, Any]]:
    rows = []
    for family_index, family in enumerate(POSITIVE_FAMILIES):
        for index in range(config.cases_per_family):
            value = config.value_offset + family_index * 100_000 + index
            if family == "implicit_scale_total":
                scale_word = "double" if index % 2 == 0 else "triple"
                facts: dict[str, Any] = {
                    "rows": value * 3 + 601,
                    "columns": value * 2 + 607,
                    "extra": value * 5 + 613,
                    "scale_word": scale_word,
                }
                multiplier = (
                    "two times" if scale_word == "double" else "three times"
                )
                prompt = (
                    f"A Martian archive has {facts['rows']} shelves with "
                    f"{facts['columns']} slots per shelf. Its expansion uses "
                    f"{multiplier} the rectangular slot count plus "
                    f"{facts['extra']} overflow slots. Find the exact total."
                )
                expected = (
                    (2 if scale_word == "double" else 3)
                    * facts["rows"]
                    * facts["columns"]
                    + facts["extra"]
                )
                label = "A"
            else:
                units = value * 2 + 617
                price = value * 3 + 619
                net = value * 4 + 631
                recurring = units * price - net
                threshold = value * 2 + 641
                facts = {
                    "setup_cost": net * threshold,
                    "units_per_period": units,
                    "price_per_unit": price,
                    "recurring_cost": recurring,
                }
                prompt = (
                    f"Launching a quantum relay costs {facts['setup_cost']}. "
                    f"Every full cycle it sells {facts['units_per_period']} "
                    f"links at {facts['price_per_unit']} each and spends "
                    f"{facts['recurring_cost']}. Find the first full cycle "
                    "after which cumulative profit is above zero."
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
        "case_id": f"router-skill-v5-{family}-{digest[:16]}",
        "family": family,
        "prompt": prompt,
        "source_facts": facts,
        "expected": expected,
        "expected_label": label,
        "expected_route": LABEL_TO_ROUTE[label],
        "positive": family in POSITIVE_FAMILIES,
    }


def _build_c_prompt(
    family: str,
    *,
    value: int,
    index: int,
) -> tuple[str, dict[str, Any]]:
    if family == "box_total":
        facts = {
            "boxes": value * 2 + 643,
            "items_per_box": value * 3 + 647,
            "loose_items": value * 7 + 653,
        }
        return (
            f"A preservation lab receives {facts['boxes']} packed cases with "
            f"{facts['items_per_box']} pieces in each case and "
            f"{facts['loose_items']} loose pieces. Find the exact total.",
            facts,
        )
    if family == "remaining_stock":
        batches = value * 2 + 659
        units = value * 3 + 661
        remaining = value * 5 + 673
        facts = {
            "starting_units": batches * units + remaining,
            "batches_used": batches,
            "units_per_batch": units,
        }
        return (
            f"A polar supply starts with {facts['starting_units']} units and "
            f"uses {facts['batches_used']} batches of "
            f"{facts['units_per_batch']} units. How many units remain?",
            facts,
        )
    if family == "paired_average":
        facts = {
            "first_total": value * 10 + 680,
            "second_total": value * 14 + 684,
        }
        return (
            f"Two calibrated readings are {facts['first_total']} and "
            f"{facts['second_total']}. Find their exact arithmetic average.",
            facts,
        )
    if family == "single_operation":
        operations = ("sum", "difference", "product", "quotient")
        operation = operations[index % 4]
        right = value * 2 + 691
        left = value * 5 + 701
        if operation == "quotient":
            left = right * (value % 97 + 29)
        facts = {"left": left, "right": right, "operation": operation}
        return (
            f"A control panel shows two integers, {left} and {right}. "
            f"Calculate their exact {operation}.",
            facts,
        )
    if family == "weighted_total":
        facts = {
            "first_count": value * 2 + 709,
            "first_weight": value % 19 + 23,
            "second_count": value * 3 + 719,
            "second_weight": value % 23 + 29,
        }
        return (
            f"A probe carries {facts['first_count']} modules of "
            f"{facts['first_weight']} units each and "
            f"{facts['second_count']} modules of "
            f"{facts['second_weight']} units each. Find the combined payload "
            "mass.",
            facts,
        )
    if family == "quotient_remainder":
        divisor = value % 83 + 37
        quotient = value * 2 + 727
        remainder = value % divisor
        facts = {
            "dividend": divisor * quotient + remainder,
            "divisor": divisor,
        }
        return (
            f"A data hub divides {facts['dividend']} records into groups of "
            f"{facts['divisor']}. How many complete groups are formed?",
            facts,
        )
    if family == "time_conversion":
        facts = {
            "days": value % 31 + 11,
            "hours": value % 23,
            "minutes": value % 59,
        }
        return (
            f"A cryogenic trial lasts {facts['days']} days, {facts['hours']} "
            f"hours, and {facts['minutes']} minutes. Convert it to total "
            "minutes.",
            facts,
        )
    percent = (5, 10, 20, 25)[index % 4]
    direction = "increase" if index % 2 == 0 else "decrease"
    facts = {
        "original": (value * 4 + 733) * 20,
        "percent": percent,
        "direction": direction,
    }
    return (
        f"An atmospheric index begins at {facts['original']} and has a "
        f"{percent} percent {direction}. Find the updated index.",
        facts,
    )


def _registry_c_candidate(
    case: dict[str, Any],
    direct: dict[str, Any],
    planner: Any,
    router_usage: dict[str, Any],
    config: RouterSkillRegistryV5Config,
    parent: Any,
    router_receipt: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.perf_counter()
    applicable = applicable_c_skills(case["prompt"])
    registry_receipt = {
        "schema_version": "nano_harness_skill_registry_receipt_v5",
        "applicable_skills": applicable,
        "unique_match": len(applicable) == 1,
        "policy": config.skill_registry_policy,
        "uses_case_metadata": False,
        "uses_expected_answer": False,
        "uses_case_correctness": False,
    }
    if len(applicable) != 1:
        return (
            {
                **direct,
                "model": f"{parent.four_b_model}+skill-registry-v5",
                "route": "direct_fallback_after_registry_ambiguity",
                "usage": _sum_usage(direct["usage"], router_usage),
            },
            {
                "router": router_receipt,
                "registry": registry_receipt,
                "fallback_used": True,
            },
        )
    family = applicable[0]
    tool_name = FAMILY_TO_TOOL[family]
    messages = [
        {"role": "system", "content": SKILL_PROMPTS[family]},
        {"role": "user", "content": case["prompt"]},
    ]
    attempts = []
    usages = [router_usage]
    receipt = None
    for attempt in range(config.plan_retry_limit + 1):
        reply = planner.complete(
            messages,
            extra_body={
                "structured_outputs": {"regex": SKILL_REGEX[family]}
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
                            "The single typed skill rejected the copied facts "
                            f"with reason={receipt['reason']}. Copy the original "
                            "numbers exactly into the same schema."
                        ),
                    },
                ]
            )
    assert receipt is not None
    if not receipt["executed"] or receipt.get("tool_name") != tool_name:
        return (
            {
                **direct,
                "model": f"{parent.four_b_model}+skill-registry-v5",
                "route": "direct_fallback_after_single_skill_failure",
                "usage": _sum_usage(direct["usage"], *usages),
                "latency_seconds": time.perf_counter() - started,
            },
            {
                "router": router_receipt,
                "registry": registry_receipt,
                "skill_attempts": attempts,
                "skill_receipt": receipt,
                "fallback_used": True,
            },
        )
    return (
        {
            "case_id": case["case_id"],
            "family": case["family"],
            "model": f"{parent.four_b_model}+skill-registry-v5",
            "route": "router_c_registry_single_skill_verified",
            "output": f"FINAL: {receipt['result']}",
            "prediction": receipt["result"],
            "parseable": True,
            "correct": receipt["result"] == case["expected"],
            "usage": _sum_usage(*usages),
            "latency_seconds": time.perf_counter() - started,
        },
        {
            "router": router_receipt,
            "registry": registry_receipt,
            "skill_attempts": attempts,
            "skill_receipt": receipt,
            "fallback_used": False,
        },
    )


def run(config: RouterSkillRegistryV5Config) -> dict[str, Any]:
    mechanism, parent = parent_config(config)
    base_service = verify_inputs(parent)
    service = json.loads(
        Path(config.service_receipt_path).read_text(encoding="utf-8")
    )
    if (
        service.get("schema_version")
        != "nano_harness_router_skill_registry_v5_service"
        or service.get("healthy") is not True
        or service.get("v5_generation_started") is not False
        or service.get("models") != config.service_models
        or service.get("remapped_adapter_sha256")
        != config.adapter_tree_sha256
    ):
        raise ValueError("router skill registry v5 service receipt differs")
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
            case, nine, parent, model=parent.nine_b_model
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
            candidate["model"] = f"{parent.four_b_model}+skill-registry-v5"
            candidate["usage"] = _sum_usage(
                route_reply.usage, candidate["usage"]
            )
            receipt = {"router": router_receipt, **tool_receipt}
        elif selected == "NONE":
            candidate, receipt = _registry_c_candidate(
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
                "model": f"{parent.four_b_model}+skill-registry-v5",
                "route": "direct_fallback_after_router_parse_failure",
                "usage": _sum_usage(direct["usage"], route_reply.usage),
            }
            receipt = {"router": router_receipt, "fallback_used": True}
        four_rows.append(direct)
        nine_rows.append(baseline_nine)
        candidate_rows.append(candidate)
        receipts[case["case_id"]] = receipt
    result = {
        "schema_version": RESULT_SCHEMA,
        "experiment_id": config.experiment_id,
        "config": config.__dict__,
        "identity": {
            "v4_report_sha256": config.v4_report_sha256,
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
            "router_correct": sum(
                receipt["router"]["correct"] for receipt in receipts.values()
            ),
            "registry_unique_matches": sum(
                bool(receipt.get("registry", {}).get("unique_match"))
                for receipt in receipts.values()
            ),
            "c_skill_executions": sum(
                bool(receipt.get("skill_receipt", {}).get("executed"))
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
        "v5_service_receipt": service,
        "evaluation_boundary": {
            "training_eligible_cases": 0,
            "router_uses_case_metadata": False,
            "router_uses_expected_answer": False,
            "router_uses_case_correctness": False,
            "skill_registry_uses_case_metadata": False,
            "skill_registry_uses_expected_answer": False,
            "skill_registry_uses_case_correctness": False,
            "executor_uses_expected_answer": False,
            "executor_uses_case_correctness": False,
            "v1_v2_v3_v4_rows_or_outputs_loaded": False,
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
