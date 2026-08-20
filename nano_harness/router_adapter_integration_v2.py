from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from nano_harness.baseline import sha256_file
from nano_harness.router_adapter_integration import (
    LABEL_TO_ROUTE,
    ROUTER_SYSTEM,
    _candidate_row,
    _sha256_tree,
    parse_route,
)
from nano_harness.semantic_model_router import (
    ALL_FAMILIES,
    POSITIVE_FAMILIES,
    summarize_rows,
)
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


CONFIG_SCHEMA = "nano_harness_router_adapter_integration_v2"
RESULT_SCHEMA = "nano_harness_router_adapter_integration_result_v2"


@dataclass(frozen=True)
class RouterAdapterIntegrationV2Config:
    schema_version: str
    experiment_id: str
    mechanism_config_path: str
    mechanism_config_sha256: str
    router_training_data_path: str
    router_training_data_sha256: str
    prior_router_config_path: str
    prior_router_config_sha256: str
    prior_binary_config_path: str
    prior_binary_config_sha256: str
    integration_v1_config_path: str
    integration_v1_config_sha256: str
    integration_v1_preregister_path: str
    integration_v1_preregister_sha256: str
    integration_v1_report_path: str
    integration_v1_report_sha256: str
    parity_report_path: str
    parity_report_sha256: str
    adapter_path: str
    adapter_tree_sha256: str
    adapter_weights_sha256: str
    served_adapter_name: str
    service_models: dict[str, str]
    service_receipt_path: str
    output_path: str
    case_seed: int
    value_offset: int
    cases_per_family: int
    route_structured_output_regex: str
    route_max_tokens: int
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


def load_config(path: str | Path) -> RouterAdapterIntegrationV2Config:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(raw) != set(RouterAdapterIntegrationV2Config.__dataclass_fields__):
        raise ValueError("router integration v2 config fields differ")
    config = RouterAdapterIntegrationV2Config(**raw)
    expected = {
        "schema_version": CONFIG_SCHEMA,
        "experiment_id": "qwen35-router-adapter-integration-v2",
        "mechanism_config_path": (
            "configs/harness/qwen35_semantic_skill_execution_v1.json"
        ),
        "mechanism_config_sha256": (
            "4e5785b9b4fc9dd5b3a76a95b438c2f2c3d505a2fae6231b81cfbf53ebbd6ad9"
        ),
        "router_training_data_path": (
            "../nano-data-pipeline-fullstack-traex-03/datasets/"
            "qwen35_router_classification_v1.json"
        ),
        "router_training_data_sha256": (
            "dacd3663639fe9ddc054865b87afdd0c918f0fddb12c8c9355819d4bbce95d65"
        ),
        "prior_router_config_path": (
            "configs/harness/qwen35_semantic_model_router_v1.json"
        ),
        "prior_router_config_sha256": (
            "8c6f0215cb2a0f805e04fe6c00a28fc3b5847d0c2a14575a90b3ed2a6586ebce"
        ),
        "prior_binary_config_path": (
            "configs/harness/qwen35_semantic_binary_detectors_v1.json"
        ),
        "prior_binary_config_sha256": (
            "32c1d877a1cecdb9041fc88226fa2e52890390712f449a8552be0a1202d88748"
        ),
        "integration_v1_config_path": (
            "configs/harness/qwen35_router_adapter_integration_v1.json"
        ),
        "integration_v1_config_sha256": (
            "4eb7000201530ecb2ced96f4b1d490d115f4c1e1c6a6cb008cf64d5dc403d4c4"
        ),
        "integration_v1_preregister_path": (
            "docs/experiments/"
            "qwen35_router_adapter_integration_v1.preregister.json"
        ),
        "integration_v1_preregister_sha256": (
            "ed5c4e6800385e7a4bfce0aed027bd1f81a6854bb1ed5b3f6aa0cc6e808491f3"
        ),
        "integration_v1_report_path": (
            "docs/results/qwen35_router_adapter_integration_v1.public.json"
        ),
        "integration_v1_report_sha256": (
            "9b01a9b6d6011f657696b0cebf9de8853b16fd2406802b14ca203d3500288f70"
        ),
        "parity_report_path": (
            "docs/results/qwen35_router_serving_parity_v1.public.json"
        ),
        "parity_report_sha256": (
            "539517c890e53f2a0e4034c724d1324df6cc828186d9621f77c106c08d4a1c01"
        ),
        "adapter_path": "results/serving/qwen35-router-v1-remapped",
        "adapter_tree_sha256": (
            "fbaa39dcb3fcf34e9aab280308cb5a5416094c1968e4ac3a69cd739a806ecc49"
        ),
        "adapter_weights_sha256": (
            "9475d69207fa1db9b0106e420637c6f764d907baa2048c4b73f19773d6e2042b"
        ),
        "served_adapter_name": "qwen3.5-router-remapped-v1",
        "service_models": {
            "base": "qwen3.5-4b",
            "original_unused": "qwen3.5-router-original-v1",
            "remapped_router": "qwen3.5-router-remapped-v1",
        },
        "service_receipt_path": (
            "docs/experiments/"
            "qwen35_router_adapter_integration_v2_service.public.json"
        ),
        "output_path": (
            "results/harness/qwen35-router-adapter-integration-v2/result.json"
        ),
        "case_seed": 20260826,
        "value_offset": 25_000,
        "cases_per_family": 32,
        "route_structured_output_regex": r"FINAL: [A-C]",
        "route_max_tokens": 8,
        "direct_max_tokens": 32,
        "plan_max_tokens": 96,
        "final_max_tokens": 32,
        "plan_retry_limit": 1,
        "bootstrap_samples": 10_000,
        "bootstrap_seed": "qwen35-router-adapter-integration-v2",
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
            "integration_v1_outputs_loaded": False,
            "integration_v1_rerun_allowed": False,
            "router_uses_case_metadata": False,
            "router_uses_expected_answer": False,
            "router_uses_case_correctness": False,
            "executor_uses_expected_answer": False,
            "executor_uses_case_correctness": False,
            "post_observation_prompt_parser_budget_search": False,
        },
        "execution_boundary": {
            "parity_service_reused": True,
            "model_generation_started": False,
            "evaluation_started": False,
            "integration_v1_rerun": False,
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
                f"router integration v2 freezes {field}={expected_value}"
            )
    for source, digest in (
        (config.mechanism_config_path, config.mechanism_config_sha256),
        (config.router_training_data_path, config.router_training_data_sha256),
        (config.prior_router_config_path, config.prior_router_config_sha256),
        (config.prior_binary_config_path, config.prior_binary_config_sha256),
        (config.integration_v1_config_path, config.integration_v1_config_sha256),
        (
            config.integration_v1_preregister_path,
            config.integration_v1_preregister_sha256,
        ),
        (config.integration_v1_report_path, config.integration_v1_report_sha256),
        (config.parity_report_path, config.parity_report_sha256),
    ):
        if sha256_file(Path(source)) != digest:
            raise ValueError("router integration v2 evidence identity differs")
    adapter = Path(config.adapter_path)
    if (
        _sha256_tree(adapter) != config.adapter_tree_sha256
        or sha256_file(adapter / "adapter_model.safetensors")
        != config.adapter_weights_sha256
    ):
        raise ValueError("router integration v2 adapter identity differs")
    v1 = json.loads(
        Path(config.integration_v1_report_path).read_text(encoding="utf-8")
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
        or parity.get("decision", {}).get(
            "serving_namespace_root_cause_supported"
        )
        is not True
        or parity.get("decision", {}).get(
            "fresh_integration_v2_preregistration_allowed"
        )
        is not True
        or parity.get("decision", {}).get(
            "observed_integration_v1_rerun_allowed"
        )
        is not False
        or parity.get("identity", {}).get("remapped_adapter_sha256")
        != config.adapter_tree_sha256
    ):
        raise ValueError("router integration v2 predecessor decision differs")
    return config


def parent_config(config: RouterAdapterIntegrationV2Config):
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


def build_cases(
    config: RouterAdapterIntegrationV2Config,
) -> list[dict[str, Any]]:
    rows = []
    for family in ALL_FAMILIES:
        for index in range(config.cases_per_family):
            value = config.value_offset + index
            if family == "implicit_scale_total":
                scale_word = "double" if index % 2 == 0 else "triple"
                facts: dict[str, Any] = {
                    "rows": value * 3 + 37,
                    "columns": value * 2 + 41,
                    "extra": value * 5 + 43,
                    "scale_word": scale_word,
                }
                scale_phrase = (
                    "twice" if scale_word == "double" else "three times"
                )
                prompt = (
                    f"An auditorium has {facts['rows']} seating rows with "
                    f"{facts['columns']} seats per row. Reserve "
                    f"{scale_phrase} that rectangular capacity and add "
                    f"{facts['extra']} accessibility seats. Compute the exact "
                    "number of seats to reserve."
                )
                expected = execute_semantic_tool(family, facts)
                label = "A"
            elif family == "first_strict_profit_period":
                units = value * 2 + 47
                price = value * 3 + 53
                net = value * 4 + 59
                recurring = units * price - net
                threshold = value * 2 + 61
                facts = {
                    "setup_cost": net * threshold,
                    "units_per_period": units,
                    "price_per_unit": price,
                    "recurring_cost": recurring,
                }
                prompt = (
                    f"Launching a service costs {facts['setup_cost']}. Each "
                    f"week it sells {facts['units_per_period']} subscriptions "
                    f"at {facts['price_per_unit']} each and pays "
                    f"{facts['recurring_cost']} in weekly expenses. What is "
                    "the first whole week when cumulative profit is strictly "
                    "greater than zero?"
                )
                expected = execute_semantic_tool(family, facts)
                label = "B"
            elif family == "box_total":
                facts = {
                    "boxes": value * 2 + 67,
                    "items_per_box": value * 3 + 71,
                    "loose_items": value * 7 + 73,
                }
                expected = (
                    facts["boxes"] * facts["items_per_box"]
                    + facts["loose_items"]
                )
                prompt = (
                    f"A logistics center received {facts['boxes']} sealed "
                    f"cartons with {facts['items_per_box']} components in "
                    f"each carton, plus {facts['loose_items']} loose "
                    "components. Compute the exact inventory total."
                )
                label = "C"
            else:
                batches = value * 2 + 79
                units = value * 3 + 83
                remaining = value * 5 + 89
                facts = {
                    "starting_units": batches * units + remaining,
                    "batches_used": batches,
                    "units_per_batch": units,
                }
                expected = remaining
                prompt = (
                    f"A factory started with {facts['starting_units']} units. "
                    f"It consumed {facts['batches_used']} batches of "
                    f"{facts['units_per_batch']} units each. Compute the exact "
                    "number of units remaining."
                )
                label = "C"
            digest = hashlib.sha256(
                f"{family}\0{json.dumps(facts, sort_keys=True)}".encode()
            ).hexdigest()
            rows.append(
                {
                    "case_id": f"router-adapter-v2-{family}-{digest[:16]}",
                    "family": family,
                    "prompt": prompt,
                    "source_facts": facts,
                    "expected": expected,
                    "expected_label": label,
                    "expected_route": LABEL_TO_ROUTE[label],
                    "positive": family in POSITIVE_FAMILIES,
                }
            )
    rows.sort(
        key=lambda row: hashlib.sha256(
            f"{config.case_seed}\0{row['case_id']}".encode()
        ).hexdigest()
    )
    return rows


def run(config: RouterAdapterIntegrationV2Config) -> dict[str, Any]:
    mechanism, parent = parent_config(config)
    base_service = verify_inputs(parent)
    service = json.loads(
        Path(config.service_receipt_path).read_text(encoding="utf-8")
    )
    if (
        service.get("schema_version")
        != "nano_harness_router_adapter_integration_v2_service"
        or service.get("healthy") is not True
        or service.get("v2_generation_started") is not False
        or service.get("models") != config.service_models
        or service.get("remapped_adapter_sha256")
        != config.adapter_tree_sha256
    ):
        raise ValueError("router integration v2 service receipt differs")
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
        candidate["model"] = f"{parent.four_b_model}+router-adapter-v2"
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
            "router_training_data_sha256": config.router_training_data_sha256,
            "integration_v1_report_sha256": (
                config.integration_v1_report_sha256
            ),
            "parity_report_sha256": config.parity_report_sha256,
            "adapter_sha256": config.adapter_tree_sha256,
            "case_contract": public_case_contract(cases),
        },
        "arms": {
            "four_b_direct": summarize_rows(four_rows),
            "nine_b_direct": summarize_rows(nine_rows),
            "four_b_router_adapter_v2": summarize_rows(candidate_rows),
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
        "v2_service_receipt": service,
        "evaluation_boundary": {
            "training_eligible_cases": 0,
            "router_uses_case_metadata": False,
            "router_uses_expected_answer": False,
            "router_uses_case_correctness": False,
            "executor_uses_expected_answer": False,
            "executor_uses_case_correctness": False,
            "integration_v1_rows_or_outputs_loaded": False,
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
