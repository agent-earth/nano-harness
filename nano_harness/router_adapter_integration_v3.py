from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from nano_harness.baseline import sha256_file
from nano_harness.router_adapter_integration import (
    LABEL_TO_ROUTE,
    _candidate_row,
    _sha256_tree,
)
from nano_harness.semantic_model_router import summarize_rows
from nano_harness.semantic_skill_execution import (
    execute_semantic_tool,
    load_config as load_mechanism_config,
    parent_config as load_parent_runtime,
)
from nano_harness.verified_tool_execution import (
    _client,
    _direct_row,
    public_case_contract,
    verify_inputs,
)


CONFIG_SCHEMA = "nano_harness_router_adapter_integration_v3"
RESULT_SCHEMA = "nano_harness_router_adapter_integration_result_v3"
CONFIG_SHA256 = (
    "0f7f2aedf12d63a651c79f4b06417de45cc5bd83b48a9be5ab396282baed048a"
)
POSITIVE_FAMILIES = (
    "implicit_scale_total",
    "first_strict_profit_period",
)


@dataclass(frozen=True)
class RouterAdapterIntegrationV3Config:
    adapter_path: str
    adapter_tree_sha256: str
    adapter_weights_sha256: str
    bootstrap_samples: int
    bootstrap_seed: str
    case_seed: int
    direct_max_tokens: int
    execution_boundary: dict[str, bool]
    experiment_id: str
    final_max_tokens: int
    integration_v1_report_path: str
    integration_v1_report_sha256: str
    integration_v2_report_path: str
    integration_v2_report_sha256: str
    maximum_harness_losses: int
    mechanism_config_path: str
    mechanism_config_sha256: str
    minimum_harness_wins: int
    negative_cases_per_subtype: int
    negative_subtypes: list[str]
    output_path: str
    parent_config_path: str
    parent_config_sha256: str
    parity_report_path: str
    parity_report_sha256: str
    plan_max_tokens: int
    plan_retry_limit: int
    policy: dict[str, bool]
    positive_cases_per_family: int
    route_max_tokens: int
    route_structured_output_regex: str
    router_training_data_path: str
    router_training_data_sha256: str
    schema_version: str
    served_adapter_name: str
    service_launch: dict[str, Any]
    service_models: dict[str, str]
    service_receipt_path: str
    significance_alpha: float
    value_offset: int


def load_config(path: str | Path) -> RouterAdapterIntegrationV3Config:
    config_path = Path(path)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if set(raw) != set(RouterAdapterIntegrationV3Config.__dataclass_fields__):
        raise ValueError("router integration v3 config fields differ")
    if sha256_file(config_path) != CONFIG_SHA256:
        raise ValueError("router integration v3 config SHA differs")
    config = RouterAdapterIntegrationV3Config(**raw)
    if config.schema_version != CONFIG_SCHEMA:
        raise ValueError("unsupported router integration v3 schema")
    for source, digest in (
        (config.mechanism_config_path, config.mechanism_config_sha256),
        (config.parent_config_path, config.parent_config_sha256),
        (config.router_training_data_path, config.router_training_data_sha256),
        (config.integration_v1_report_path, config.integration_v1_report_sha256),
        (config.integration_v2_report_path, config.integration_v2_report_sha256),
        (config.parity_report_path, config.parity_report_sha256),
    ):
        if sha256_file(Path(source)) != digest:
            raise ValueError("router integration v3 evidence identity differs")
    adapter = Path(config.adapter_path)
    if (
        _sha256_tree(adapter) != config.adapter_tree_sha256
        or sha256_file(adapter / "adapter_model.safetensors")
        != config.adapter_weights_sha256
    ):
        raise ValueError("router integration v3 adapter identity differs")
    v1 = json.loads(
        Path(config.integration_v1_report_path).read_text(encoding="utf-8")
    )
    v2 = json.loads(
        Path(config.integration_v2_report_path).read_text(encoding="utf-8")
    )
    parity = json.loads(
        Path(config.parity_report_path).read_text(encoding="utf-8")
    )
    if (
        v1.get("decision", {}).get("adapter_integration_admitted") is not False
        or v1.get("decision", {}).get(
            "further_tuning_or_rerun_on_observed_cases_allowed"
        )
        is not False
        or v2.get("decision", {}).get("adapter_integration_v2_admitted")
        is not False
        or v2.get("decision", {}).get("integration_v2_rerun_allowed")
        is not False
        or parity.get("decision", {}).get(
            "remapped_adapter_serving_admitted"
        )
        is not True
        or parity.get("decision", {}).get(
            "fresh_integration_v3_preregistration_allowed"
        )
        is not True
        or parity.get("decision", {}).get(
            "fresh_integration_v3_generation_allowed"
        )
        is not False
        or parity.get("identity", {}).get("remapped_adapter_sha256")
        != config.adapter_tree_sha256
    ):
        raise ValueError("router integration v3 predecessor decision differs")
    return config


def parent_config(config: RouterAdapterIntegrationV3Config):
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


def _positive_case(
    family: str,
    *,
    value: int,
    index: int,
) -> tuple[str, dict[str, Any], int, str]:
    if family == "implicit_scale_total":
        scale_word = "double" if index % 2 == 0 else "triple"
        facts: dict[str, Any] = {
            "rows": value * 3 + 211,
            "columns": value * 2 + 223,
            "extra": value * 5 + 227,
            "scale_word": scale_word,
        }
        multiplier = "two copies of" if scale_word == "double" else "three copies of"
        prompt = (
            f"A polar observatory arranges {facts['rows']} sensor tiers with "
            f"{facts['columns']} sensors on every tier. Its mirrored plan "
            f"needs {multiplier} that rectangular array, together with "
            f"{facts['extra']} calibration sensors. Give the exact planned "
            "sensor count."
        )
        return prompt, facts, execute_semantic_tool(family, facts), "A"
    units = value * 2 + 229
    price = value * 3 + 233
    net = value * 4 + 239
    recurring = units * price - net
    threshold = value * 2 + 241
    facts = {
        "setup_cost": net * threshold,
        "units_per_period": units,
        "price_per_unit": price,
        "recurring_cost": recurring,
    }
    prompt = (
        f"Opening an ocean relay costs {facts['setup_cost']}. In each complete "
        f"month it serves {facts['units_per_period']} links billed at "
        f"{facts['price_per_unit']} each and incurs {facts['recurring_cost']} "
        "in monthly operating cost. Identify the earliest complete month "
        "after which accumulated profit is greater than zero."
    )
    return prompt, facts, execute_semantic_tool(family, facts), "B"


def _negative_case(
    subtype: str,
    *,
    value: int,
    index: int,
) -> tuple[str, dict[str, Any], int, str]:
    if subtype == "box_total":
        facts = {
            "crates": value * 2 + 251,
            "artifacts_per_crate": value * 3 + 257,
            "uncrated_artifacts": value * 7 + 263,
        }
        expected = (
            facts["crates"] * facts["artifacts_per_crate"]
            + facts["uncrated_artifacts"]
        )
        prompt = (
            f"A museum transfer contains {facts['crates']} crates, each with "
            f"{facts['artifacts_per_crate']} catalogued pieces, plus "
            f"{facts['uncrated_artifacts']} pieces shipped separately. How "
            "many pieces arrived altogether?"
        )
    elif subtype == "remaining_stock":
        batches = value * 2 + 269
        per_batch = value * 3 + 271
        expected = value * 5 + 277
        facts = {
            "initial_reserve": batches * per_batch + expected,
            "deployments": batches,
            "units_per_deployment": per_batch,
        }
        prompt = (
            f"A disaster reserve begins with {facts['initial_reserve']} units. "
            f"After {facts['deployments']} deployments using "
            f"{facts['units_per_deployment']} units apiece, what quantity "
            "remains in reserve?"
        )
    elif subtype == "paired_average":
        first = value * 10 + 281
        second = value * 14 + 285
        facts = {"north_reading": first, "south_reading": second}
        expected = (first + second) // 2
        prompt = (
            f"Two climate stations report calibrated readings of {first} and "
            f"{second}. What is their arithmetic average?"
        )
    elif subtype == "single_operation":
        mode = index % 4
        left = value * 5 + 293
        right = value * 2 + 307
        if mode == 0:
            operation, expected = "sum", left + right
        elif mode == 1:
            operation, expected = "difference", left - right
        elif mode == 2:
            operation, expected = "product", left * right
        else:
            expected = value % 97 + 19
            left = right * expected
            operation = "whole-number quotient"
        facts = {"left": left, "right": right, "operation": operation}
        prompt = (
            f"A navigation console displays {left} and {right}. Calculate "
            f"their exact {operation}."
        )
    elif subtype == "weighted_total":
        facts = {
            "light_modules": value * 2 + 311,
            "light_mass": value % 19 + 13,
            "heavy_modules": value * 3 + 313,
            "heavy_mass": value % 23 + 17,
        }
        expected = (
            facts["light_modules"] * facts["light_mass"]
            + facts["heavy_modules"] * facts["heavy_mass"]
        )
        prompt = (
            f"A spacecraft carries {facts['light_modules']} modules of "
            f"{facts['light_mass']} mass units each and "
            f"{facts['heavy_modules']} modules of {facts['heavy_mass']} mass "
            "units each. Find the combined module mass."
        )
    elif subtype == "quotient_remainder":
        divisor = value % 83 + 29
        quotient = value * 2 + 317
        remainder = value % divisor
        total = divisor * quotient + remainder
        facts = {
            "samples": total,
            "group_size": divisor,
            "quotient": quotient,
            "remainder": remainder,
        }
        expected = quotient
        prompt = (
            f"A survey has {total} samples and packs them in groups of "
            f"{divisor}. Determine the number of complete groups and the "
            "number of samples left over."
        )
    elif subtype == "time_conversion":
        facts = {
            "days": value % 31 + 5,
            "hours": value % 23,
            "minutes": value % 59,
        }
        expected = (
            facts["days"] * 24 * 60
            + facts["hours"] * 60
            + facts["minutes"]
        )
        prompt = (
            f"A deep-sea mission lasts {facts['days']} days, "
            f"{facts['hours']} hours, and {facts['minutes']} minutes. Express "
            "the full duration in minutes."
        )
    else:
        percent = (5, 10, 20, 25)[index % 4]
        direction = "rises" if index % 2 == 0 else "falls"
        original = (value * 4 + 331) * 20
        delta = original * percent // 100
        expected = original + delta if direction == "rises" else original - delta
        facts = {
            "original": original,
            "percent": percent,
            "direction": direction,
        }
        prompt = (
            f"An ecological index starts at {original} and then {direction} "
            f"by {percent} percent. What is the updated index value?"
        )
    return prompt, facts, expected, "C"


def build_cases(
    config: RouterAdapterIntegrationV3Config,
) -> list[dict[str, Any]]:
    rows = []
    for family_index, family in enumerate(POSITIVE_FAMILIES):
        for index in range(config.positive_cases_per_family):
            value = config.value_offset + family_index * 100_000 + index
            prompt, facts, expected, label = _positive_case(
                family,
                value=value,
                index=index,
            )
            rows.append(
                _case_row(
                    family=family,
                    prompt=prompt,
                    facts=facts,
                    expected=expected,
                    label=label,
                )
            )
    for subtype_index, subtype in enumerate(config.negative_subtypes):
        for index in range(config.negative_cases_per_subtype):
            value = (
                config.value_offset
                + 1_000_000
                + subtype_index * 100_000
                + index
            )
            prompt, facts, expected, label = _negative_case(
                subtype,
                value=value,
                index=index,
            )
            rows.append(
                _case_row(
                    family=subtype,
                    prompt=prompt,
                    facts=facts,
                    expected=expected,
                    label=label,
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
        "case_id": f"router-adapter-v3-{family}-{digest[:16]}",
        "family": family,
        "prompt": prompt,
        "source_facts": facts,
        "expected": expected,
        "expected_label": label,
        "expected_route": LABEL_TO_ROUTE[label],
        "positive": family in POSITIVE_FAMILIES,
    }


def run(config: RouterAdapterIntegrationV3Config) -> dict[str, Any]:
    mechanism, parent = parent_config(config)
    base_service = verify_inputs(parent)
    service = json.loads(
        Path(config.service_receipt_path).read_text(encoding="utf-8")
    )
    if (
        service.get("schema_version")
        != "nano_harness_router_adapter_integration_v3_service"
        or service.get("healthy") is not True
        or service.get("v3_generation_started") is not False
        or service.get("integration_v1_or_v2_rerun") is not False
        or service.get("models") != config.service_models
        or service.get("remapped_adapter_sha256")
        != config.adapter_tree_sha256
    ):
        raise ValueError("router integration v3 service receipt differs")
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
        candidate, receipt = _candidate_row(
            case,
            direct,
            router,
            planner,
            final,
            config,
            mechanism,
            parent,
        )
        candidate["model"] = f"{parent.four_b_model}+router-adapter-v3"
        four_rows.append(direct)
        nine_rows.append(baseline_nine)
        candidate_rows.append(candidate)
        receipts[case["case_id"]] = receipt
    result = {
        "schema_version": RESULT_SCHEMA,
        "experiment_id": config.experiment_id,
        "config": config.__dict__,
        "identity": {
            "mechanism_config_sha256": config.mechanism_config_sha256,
            "parent_config_sha256": config.parent_config_sha256,
            "router_training_data_sha256": (
                config.router_training_data_sha256
            ),
            "integration_v1_report_sha256": (
                config.integration_v1_report_sha256
            ),
            "integration_v2_report_sha256": (
                config.integration_v2_report_sha256
            ),
            "parity_report_sha256": config.parity_report_sha256,
            "adapter_sha256": config.adapter_tree_sha256,
            "case_contract": public_case_contract(cases),
        },
        "arms": {
            "four_b_direct": summarize_rows(four_rows),
            "nine_b_direct": summarize_rows(nine_rows),
            "four_b_router_adapter_v3": summarize_rows(candidate_rows),
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
            "positive_correct": sum(
                case["positive"]
                and receipts[case["case_id"]]["router"]["correct"]
                for case in cases
            ),
            "negative_cases": sum(not case["positive"] for case in cases),
            "negative_c_correct": sum(
                not case["positive"]
                and receipts[case["case_id"]]["router"]["label"] == "C"
                for case in cases
            ),
            "negative_false_positive_routes": sum(
                not case["positive"]
                and receipts[case["case_id"]]["router"]["label"] != "C"
                for case in cases
            ),
            "verified_executions": sum(
                bool(receipt["receipt"] and receipt["receipt"]["executed"])
                for receipt in receipts.values()
            ),
            "fallbacks": sum(
                receipt["fallback_used"] for receipt in receipts.values()
            ),
        },
        "base_service_receipt": base_service,
        "v3_service_receipt": service,
        "evaluation_boundary": {
            "training_eligible_cases": 0,
            "router_uses_case_metadata": False,
            "router_uses_expected_answer": False,
            "router_uses_case_correctness": False,
            "executor_uses_expected_answer": False,
            "executor_uses_case_correctness": False,
            "integration_v1_or_v2_rows_or_outputs_loaded": False,
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
