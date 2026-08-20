from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from nano_harness.baseline import sha256_file
from nano_harness.semantic_model_router import (
    ALL_FAMILIES,
    NEGATIVE_FAMILIES,
    POSITIVE_FAMILIES,
    _model_selected_tool_row,
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
    _sum_usage,
    public_case_contract,
    verify_inputs,
)


CONFIG_SCHEMA = "nano_harness_semantic_binary_detectors_v1"
RESULT_SCHEMA = "nano_harness_semantic_binary_detectors_result_v1"
DETECT_REGEX = r"DETECT: (?:YES|NO)"
DETECT_PATTERN = re.compile(r"^DETECT: (YES|NO)$")
DETECTOR_PROMPTS = {
    "implicit_scale_total": (
        "Answer YES only if the task asks for exactly double or triple a "
        "rectangular rows-by-columns capacity plus an extra quantity. "
        "Otherwise answer NO. Return only DETECT: YES or DETECT: NO."
    ),
    "first_strict_profit_period": (
        "Answer YES only if the task asks for the first whole period when "
        "cumulative profit becomes strictly greater than zero from setup "
        "cost, units sold, unit price, and recurring cost. Otherwise answer "
        "NO. Return only DETECT: YES or DETECT: NO."
    ),
}


@dataclass(frozen=True)
class SemanticBinaryDetectorsConfig:
    schema_version: str
    experiment_id: str
    mechanism_config_path: str
    mechanism_config_sha256: str
    multiclass_report_path: str
    multiclass_report_sha256: str
    output_path: str
    case_seed: int
    cases_per_family: int
    detector_structured_output_regex: str
    detector_max_tokens: int
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


def load_config(path: str | Path) -> SemanticBinaryDetectorsConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(raw) != set(SemanticBinaryDetectorsConfig.__dataclass_fields__):
        raise ValueError("semantic binary detectors config fields differ")
    config = SemanticBinaryDetectorsConfig(**raw)
    expected = {
        "schema_version": CONFIG_SCHEMA,
        "experiment_id": "qwen35-semantic-binary-detectors-v1",
        "mechanism_config_path": (
            "configs/harness/qwen35_semantic_skill_execution_v1.json"
        ),
        "mechanism_config_sha256": (
            "4e5785b9b4fc9dd5b3a76a95b438c2f2c3d505a2fae6231b81cfbf53ebbd6ad9"
        ),
        "multiclass_report_path": (
            "docs/results/qwen35_semantic_model_router_v1.public.json"
        ),
        "multiclass_report_sha256": (
            "c8e4034a27e925025589bc1a8a52abc6720ee0d7fc97e03983ff192cd44c3742"
        ),
        "output_path": (
            "results/harness/qwen35-semantic-binary-detectors-v1/result.json"
        ),
        "case_seed": 20260823,
        "cases_per_family": 32,
        "detector_structured_output_regex": DETECT_REGEX,
        "detector_max_tokens": 8,
        "direct_max_tokens": 32,
        "plan_max_tokens": 96,
        "final_max_tokens": 32,
        "plan_retry_limit": 1,
        "bootstrap_samples": 10_000,
        "bootstrap_seed": "qwen35-semantic-binary-detectors-v1",
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
            "detectors_use_case_metadata": False,
            "detectors_use_expected_answer": False,
            "detectors_use_case_correctness": False,
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
                f"semantic binary detectors freeze {field}={expected_value}"
            )
    for path_value, digest in (
        (config.mechanism_config_path, config.mechanism_config_sha256),
        (config.multiclass_report_path, config.multiclass_report_sha256),
    ):
        if sha256_file(Path(path_value)) != digest:
            raise ValueError("semantic binary detector evidence identity differs")
    report = json.loads(
        Path(config.multiclass_report_path).read_text(encoding="utf-8")
    )
    if (
        report.get("decision", {}).get("router_admitted") is not False
        or report.get("decision", {}).get("router_precision_direction_supported")
        is not True
        or report.get("decision", {}).get("router_recall_supported") is not False
    ):
        raise ValueError("semantic binary detector predecessor differs")
    return config


def parent_config(config: SemanticBinaryDetectorsConfig):
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


def build_cases(config: SemanticBinaryDetectorsConfig) -> list[dict[str, Any]]:
    rows = []
    for family in ALL_FAMILIES:
        for index in range(config.cases_per_family):
            value = 7000 + index
            if family == "implicit_scale_total":
                scale_word = "double" if index % 2 == 0 else "triple"
                facts: dict[str, Any] = {
                    "rows": value * 3 + 11,
                    "columns": value * 2 + 7,
                    "extra": value * 5 + 13,
                    "scale_word": scale_word,
                }
                phrase = "twice" if scale_word == "double" else "three times"
                prompt = (
                    f"A venue records rows={facts['rows']}, "
                    f"columns={facts['columns']}, and extra={facts['extra']}. "
                    f"Its order equals {phrase} the rectangular capacity plus "
                    "the extra. Find the exact order."
                )
                expected = execute_semantic_tool(family, facts)
                expected_detector = family
            elif family == "first_strict_profit_period":
                units = value * 2 + 5
                price = value * 3 + 7
                net = value * 4 + 29
                recurring = units * price - net
                threshold = value * 2 + 17
                facts = {
                    "setup_cost": net * threshold,
                    "units_per_period": units,
                    "price_per_unit": price,
                    "recurring_cost": recurring,
                }
                prompt = (
                    f"A service records setup_cost={facts['setup_cost']}, "
                    f"units_per_period={facts['units_per_period']}, "
                    f"price_per_unit={facts['price_per_unit']}, and "
                    f"recurring_cost={facts['recurring_cost']}. Find the "
                    "earliest whole period where cumulative profit is above zero."
                )
                expected = execute_semantic_tool(family, facts)
                expected_detector = family
            elif family == "box_total":
                facts = {
                    "boxes": value * 2 + 3,
                    "items_per_box": value * 3 + 5,
                    "loose_items": value * 7 + 11,
                }
                expected = (
                    facts["boxes"] * facts["items_per_box"]
                    + facts["loose_items"]
                )
                prompt = (
                    f"A shipment records boxes={facts['boxes']}, "
                    f"items_per_box={facts['items_per_box']}, and "
                    f"loose_items={facts['loose_items']}. Find the exact total."
                )
                expected_detector = "NONE"
            else:
                batches = value * 2 + 7
                units = value * 3 + 13
                remaining = value * 5 + 17
                facts = {
                    "starting_units": batches * units + remaining,
                    "batches_used": batches,
                    "units_per_batch": units,
                }
                expected = remaining
                prompt = (
                    f"A stock record has starting_units={facts['starting_units']}, "
                    f"batches_used={facts['batches_used']}, and "
                    f"units_per_batch={facts['units_per_batch']}. Find the "
                    "exact remaining units."
                )
                expected_detector = "NONE"
            digest = hashlib.sha256(
                f"{family}\0{json.dumps(facts, sort_keys=True)}".encode()
            ).hexdigest()
            rows.append(
                {
                    "case_id": f"binary-detector-{family}-{digest[:16]}",
                    "family": family,
                    "prompt": prompt,
                    "source_facts": facts,
                    "expected": expected,
                    "expected_detector": expected_detector,
                    "positive": family in POSITIVE_FAMILIES,
                }
            )
    rows.sort(
        key=lambda row: hashlib.sha256(
            f"{config.case_seed}\0{row['case_id']}".encode()
        ).hexdigest()
    )
    return rows


def parse_detection(text: str) -> bool | None:
    match = DETECT_PATTERN.fullmatch(text.strip())
    if not match:
        return None
    return match.group(1) == "YES"


def _candidate_row(
    case: dict[str, Any],
    direct: dict[str, Any],
    detector_clients: dict[str, Any],
    plan_client: Any,
    final_client: Any,
    config: SemanticBinaryDetectorsConfig,
    mechanism: Any,
    parent: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.perf_counter()
    detections = {}
    usages = []
    for family in POSITIVE_FAMILIES:
        reply = detector_clients[family].complete(
            [
                {"role": "system", "content": DETECTOR_PROMPTS[family]},
                {"role": "user", "content": case["prompt"]},
            ],
            extra_body={
                "structured_outputs": {
                    "regex": config.detector_structured_output_regex
                }
            },
        )
        usages.append(reply.usage)
        detections[family] = {
            "output": reply.content,
            "output_sha256": hashlib.sha256(reply.content.encode()).hexdigest(),
            "yes": parse_detection(reply.content),
            "uses_case_metadata": False,
            "uses_expected_answer": False,
            "uses_case_correctness": False,
        }
    yes_families = [
        family for family, row in detections.items() if row["yes"] is True
    ]
    selected = yes_families[0] if len(yes_families) == 1 else "NONE"
    detector_correct = selected == case["expected_detector"]
    if selected == "NONE":
        return (
            {
                **direct,
                "model": f"{parent.four_b_model}+semantic-binary-detectors-v1",
                "route": "direct_preserve_after_detector_none",
                "usage": _sum_usage(direct["usage"], *usages),
                "latency_seconds": time.perf_counter() - started,
            },
            {
                "detections": detections,
                "selected_route": selected,
                "expected_route": case["expected_detector"],
                "detector_correct": detector_correct,
                "conflict": len(yes_families) > 1,
                "exposed_tools": [],
                "receipt": None,
                "fallback_used": False,
            },
        )
    routed_case = {**case, "family": selected}
    row, receipt = _model_selected_tool_row(
        routed_case,
        direct,
        selected,
        plan_client,
        final_client,
        mechanism,
        parent,
    )
    row["usage"] = _sum_usage(*usages, row["usage"])
    row["latency_seconds"] = time.perf_counter() - started
    return (
        row,
        {
            "detections": detections,
            "selected_route": selected,
            "expected_route": case["expected_detector"],
            "detector_correct": detector_correct,
            "conflict": False,
            **receipt,
        },
    )


def run(config: SemanticBinaryDetectorsConfig) -> dict[str, Any]:
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
    detector_clients = {
        family: _client(
            parent, four_b=True, max_tokens=config.detector_max_tokens
        )
        for family in POSITIVE_FAMILIES
    }
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
            detector_clients,
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
    routing = {
        "cases": len(cases),
        "detector_correct": sum(
            receipt["detector_correct"] for receipt in receipts.values()
        ),
        "positive_cases": sum(case["positive"] for case in cases),
        "positive_route_correct": sum(
            case["positive"] and receipts[case["case_id"]]["detector_correct"]
            for case in cases
        ),
        "negative_cases": sum(not case["positive"] for case in cases),
        "negative_none_correct": sum(
            not case["positive"]
            and receipts[case["case_id"]]["selected_route"] == "NONE"
            for case in cases
        ),
        "negative_false_positive_routes": sum(
            not case["positive"]
            and receipts[case["case_id"]]["selected_route"] != "NONE"
            for case in cases
        ),
        "conflicts": sum(
            receipt["conflict"] for receipt in receipts.values()
        ),
        "verified_executions": sum(
            bool(receipt["receipt"] and receipt["receipt"]["executed"])
            for receipt in receipts.values()
        ),
        "fallbacks": sum(
            receipt["fallback_used"] for receipt in receipts.values()
        ),
    }
    result = {
        "schema_version": RESULT_SCHEMA,
        "experiment_id": config.experiment_id,
        "config": config.__dict__,
        "identity": {
            "mechanism_config_sha256": config.mechanism_config_sha256,
            "multiclass_report_sha256": config.multiclass_report_sha256,
            "case_contract": public_case_contract(cases),
        },
        "arms": {
            "four_b_direct": summarize_rows(four_rows),
            "nine_b_direct": summarize_rows(nine_rows),
            "four_b_binary_detectors": summarize_rows(candidate_rows),
        },
        "four_b_rows": four_rows,
        "nine_b_rows": nine_rows,
        "candidate_rows": candidate_rows,
        "receipts": receipts,
        "routing": routing,
        "service_receipt": service,
        "evaluation_boundary": {
            "training_eligible_cases": 0,
            "detectors_use_case_metadata": False,
            "detectors_use_expected_answer": False,
            "detectors_use_case_correctness": False,
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
