from __future__ import annotations

import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from nano_harness.baseline import sha256_file
from nano_harness.verified_tool_execution import (
    FAMILIES,
    _client,
    _direct_row,
    _sum_usage,
    build_cases,
    load_config as load_parent_config,
    parse_and_execute_plan,
    public_case_contract,
    summarize_rows,
)


CONFIG_SCHEMA = "nano_harness_verified_tool_execution_v2"
RESULT_SCHEMA = "nano_harness_verified_tool_execution_result_v2"
TOOL_REGEX_BY_FAMILY = {
    "box_total": (
        r'TOOL: box_total \{"boxes":-?[0-9]+,"items_per_box":-?[0-9]+,'
        r'"loose_items":-?[0-9]+\}'
    ),
    "remaining_stock": (
        r'TOOL: remaining_stock \{"starting_units":-?[0-9]+,'
        r'"batches_used":-?[0-9]+,"units_per_batch":-?[0-9]+\}'
    ),
    "paired_average": (
        r'TOOL: paired_average \{"first_total":-?[0-9]+,'
        r'"second_total":-?[0-9]+\}'
    ),
    "labor_total": (
        r'TOOL: labor_total \{"hourly_rate":-?[0-9]+,'
        r'"regular_hours":-?[0-9]+,"bonus":-?[0-9]+\}'
    ),
}
SKILL_PROMPTS = {
    family: (
        f"The capability router selected the {family} skill. Copy every "
        "labeled source fact exactly into the one available typed tool. "
        "Return only the TOOL line required by the structured contract. "
        "Do not calculate and do not choose another tool."
    )
    for family in FAMILIES
}


@dataclass(frozen=True)
class VerifiedToolExecutionV2Config:
    schema_version: str
    experiment_id: str
    parent_config_path: str
    parent_config_sha256: str
    prior_v1_report_path: str
    prior_v1_report_sha256: str
    service_receipt_sha256: str
    output_path: str
    value_offset: int
    skill_router: str
    plan_structured_output_regex_by_family: dict[str, str]
    policy: dict[str, bool]
    execution_boundary: dict[str, bool]


def load_config(path: str | Path) -> VerifiedToolExecutionV2Config:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(raw) != set(VerifiedToolExecutionV2Config.__dataclass_fields__):
        raise ValueError("verified tool v2 config fields differ")
    config = VerifiedToolExecutionV2Config(**raw)
    expected = {
        "schema_version": CONFIG_SCHEMA,
        "experiment_id": "qwen35-verified-tool-execution-v2",
        "parent_config_path": (
            "configs/harness/qwen35_verified_tool_execution_v1.json"
        ),
        "parent_config_sha256": (
            "538cbcde51ccb8ad43e4f91db4201a2ffd835c1493a5b2bd58177db7dcab3cd3"
        ),
        "prior_v1_report_path": (
            "docs/results/qwen35_verified_tool_execution_v1.public.json"
        ),
        "prior_v1_report_sha256": (
            "6baa2f1e5fc30efa1e07f169588847d09f41fca79ead5af235a75f643e0deb07"
        ),
        "service_receipt_sha256": (
            "895bd40d886d870f99230441baecdf1feb926e2992b6a151e54d8165145f1c0d"
        ),
        "output_path": (
            "results/harness/qwen35-verified-tool-execution-v2/result.json"
        ),
        "value_offset": 90000,
        "skill_router": "case_family_to_single_tool_v1",
        "plan_structured_output_regex_by_family": TOOL_REGEX_BY_FAMILY,
        "policy": {
            "evaluation_only": True,
            "training_eligible": False,
            "contains_benchmark_rows": False,
            "contains_benchmark_outputs": False,
            "contains_canary_rows": False,
            "contains_holdout_rows": False,
            "uses_observed_quality_outputs": False,
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
            "training_started": False,
            "this_commit_only_preregisters": True,
        },
    }
    for field, expected_value in expected.items():
        if getattr(config, field) != expected_value:
            raise ValueError(f"verified tool v2 freezes {field}={expected_value}")
    if sha256_file(Path(config.parent_config_path)) != config.parent_config_sha256:
        raise ValueError("verified tool v2 parent config identity differs")
    if (
        sha256_file(Path(config.prior_v1_report_path))
        != config.prior_v1_report_sha256
    ):
        raise ValueError("verified tool v2 prior report identity differs")
    return config


def parent_config(config: VerifiedToolExecutionV2Config):
    parent = load_parent_config(config.parent_config_path)
    if (
        sha256_file(Path(parent.service_receipt_path))
        != config.service_receipt_sha256
    ):
        raise ValueError("verified tool v2 service receipt identity differs")
    return replace(
        parent,
        experiment_id=config.experiment_id,
        output_path=config.output_path,
        value_offset=config.value_offset,
    )


def _harness_row(
    case: dict[str, Any],
    direct: dict[str, Any],
    plan_client: Any,
    final_client: Any,
    config: VerifiedToolExecutionV2Config,
    parent: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.perf_counter()
    family = case["family"]
    plan_messages = [
        {"role": "system", "content": SKILL_PROMPTS[family]},
        {"role": "user", "content": case["prompt"]},
    ]
    attempts = []
    plan_usage = []
    receipt = None
    for attempt in range(parent.plan_retry_limit + 1):
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
            expected_tool=family,
            source_facts=case["source_facts"],
        )
        attempts.append(
            {
                "attempt": attempt + 1,
                "output": reply.content,
                "reason": receipt["reason"],
                "executed": receipt["executed"],
            }
        )
        if receipt["executed"]:
            break
        if attempt < parent.plan_retry_limit:
            plan_messages.extend(
                [
                    {"role": "assistant", "content": reply.content},
                    {
                        "role": "user",
                        "content": (
                            f"The {family} skill rejected this plan: "
                            f"{receipt['reason']}. Copy the original labeled "
                            "facts exactly into the same available tool."
                        ),
                    },
                ]
            )
    assert receipt is not None
    if not receipt["executed"]:
        return (
            {
                **direct,
                "model": f"{parent.four_b_model}+verified-tool-v2",
                "route": "direct_fallback_after_invalid_plan",
                "usage": _sum_usage(direct["usage"], *plan_usage),
                "latency_seconds": time.perf_counter() - started,
            },
            {
                "skill_id": family,
                "exposed_tools": [family],
                "plan_attempts": attempts,
                "receipt": receipt,
                "final_feedback_sent": False,
                "fallback_used": True,
            },
        )
    feedback = (
        f"<original_task>\n{case['prompt']}\n</original_task>\n\n"
        f"<verified_tool>\nname={receipt['tool_name']}\n"
        f"arguments={json.dumps(receipt['arguments'], sort_keys=True)}\n"
        f"result={receipt['result']}\n</verified_tool>\n\n"
        "Use the verified result. Return only FINAL: <integer>."
    )
    final = final_client.complete(
        [
            {
                "role": "system",
                "content": (
                    "Use the verified tool result as authoritative and return "
                    "only one FINAL: <integer> line."
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
    from nano_harness.verified_tool_execution import FINAL_PATTERN

    output = final.content.strip()
    match = FINAL_PATTERN.fullmatch(output)
    prediction = int(match.group(1)) if match else None
    return (
        {
            "case_id": case["case_id"],
            "family": family,
            "model": f"{parent.four_b_model}+verified-tool-v2",
            "route": "skill_routed_verified_tool_feedback",
            "output": output,
            "prediction": prediction,
            "parseable": prediction is not None,
            "correct": prediction == case["expected"],
            "usage": _sum_usage(*plan_usage, final.usage),
            "latency_seconds": time.perf_counter() - started,
        },
        {
            "skill_id": family,
            "exposed_tools": [family],
            "plan_attempts": attempts,
            "receipt": receipt,
            "final_feedback_sent": True,
            "fallback_used": False,
        },
    )


def run(config: VerifiedToolExecutionV2Config) -> dict[str, Any]:
    parent = parent_config(config)
    from nano_harness.verified_tool_execution import (
        _direct_row,
        verify_inputs,
    )

    service = verify_inputs(parent)
    cases = build_cases(parent)
    four_client = _client(
        parent, four_b=True, max_tokens=parent.direct_max_tokens
    )
    nine_client = _client(
        parent, four_b=False, max_tokens=parent.direct_max_tokens
    )
    plan_client = _client(
        parent, four_b=True, max_tokens=parent.plan_max_tokens
    )
    final_client = _client(
        parent, four_b=True, max_tokens=parent.final_max_tokens
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
            case, four, plan_client, final_client, config, parent
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
            "prior_v1_report_sha256": config.prior_v1_report_sha256,
            "service_receipt_sha256": config.service_receipt_sha256,
            "case_contract": public_case_contract(cases),
        },
        "arms": {
            "four_b_direct": summarize_rows(four_rows),
            "nine_b_direct": summarize_rows(nine_rows),
            "four_b_skill_verified_tool": summarize_rows(harness_rows),
        },
        "four_b_rows": four_rows,
        "nine_b_rows": nine_rows,
        "harness_rows": harness_rows,
        "harness_receipts": receipts,
        "routing": {
            "skill_routes": len(receipts),
            "single_tool_exposures": sum(
                len(row["exposed_tools"]) == 1 for row in receipts.values()
            ),
            "verified_executions": sum(
                row["receipt"]["executed"] for row in receipts.values()
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
        },
        "service_receipt": service,
        "evaluation_boundary": {
            "training_eligible_cases": 0,
            "executor_uses_expected_answer": False,
            "executor_uses_case_correctness": False,
            "benchmark_rows_loaded": False,
            "benchmark_outputs_loaded": False,
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
