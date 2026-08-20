from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from nano_harness.baseline import sha256_file
from nano_harness.semantic_skill_execution import (
    FAMILIES,
    _harness_row,
    execute_semantic_tool,
    load_config as load_parent_config,
    parent_config as load_parent_runtime,
    summarize_rows,
)
from nano_harness.verified_tool_execution import (
    _client,
    _direct_row,
    public_case_contract,
    verify_inputs,
)


CONFIG_SCHEMA = "nano_harness_semantic_skill_replication_v1"
RESULT_SCHEMA = "nano_harness_semantic_skill_replication_result_v1"


@dataclass(frozen=True)
class SemanticSkillReplicationConfig:
    schema_version: str
    experiment_id: str
    parent_config_path: str
    parent_config_sha256: str
    parent_preregister_path: str
    parent_preregister_sha256: str
    parent_report_path: str
    parent_report_sha256: str
    output_path: str
    case_seed: int
    cases_per_family: int
    value_regime: str
    prompt_regime: str
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


def load_config(path: str | Path) -> SemanticSkillReplicationConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(raw) != set(SemanticSkillReplicationConfig.__dataclass_fields__):
        raise ValueError("semantic skill replication config fields differ")
    config = SemanticSkillReplicationConfig(**raw)
    expected = {
        "schema_version": CONFIG_SCHEMA,
        "experiment_id": "qwen35-semantic-skill-replication-v1",
        "parent_config_path": (
            "configs/harness/qwen35_semantic_skill_execution_v1.json"
        ),
        "parent_config_sha256": (
            "4e5785b9b4fc9dd5b3a76a95b438c2f2c3d505a2fae6231b81cfbf53ebbd6ad9"
        ),
        "parent_preregister_path": (
            "docs/experiments/qwen35_semantic_skill_execution_v1.preregister.json"
        ),
        "parent_preregister_sha256": (
            "fad4dff56233edc35d10d14f6ff5922c03f054d7d656e779e23c0f1dc102f7fc"
        ),
        "parent_report_path": (
            "docs/results/qwen35_semantic_skill_execution_v1.public.json"
        ),
        "parent_report_sha256": (
            "fe53a512cbf0b6ada65ed3ae27c5f3dc90165e367cfecdb58307dd030d017d5f"
        ),
        "output_path": (
            "results/harness/qwen35-semantic-skill-replication-v1/result.json"
        ),
        "case_seed": 20260821,
        "cases_per_family": 128,
        "value_regime": "small_integer_cross_product_v1",
        "prompt_regime": "unseen_context_paraphrase_v1",
        "direct_max_tokens": 32,
        "plan_max_tokens": 96,
        "final_max_tokens": 32,
        "plan_retry_limit": 1,
        "bootstrap_samples": 10_000,
        "bootstrap_seed": "qwen35-semantic-skill-replication-v1",
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
                f"semantic skill replication freezes {field}={expected_value}"
            )
    for path_value, digest in (
        (config.parent_config_path, config.parent_config_sha256),
        (config.parent_preregister_path, config.parent_preregister_sha256),
        (config.parent_report_path, config.parent_report_sha256),
    ):
        if sha256_file(Path(path_value)) != digest:
            raise ValueError("semantic skill replication parent identity differs")
    parent_report = json.loads(
        Path(config.parent_report_path).read_text(encoding="utf-8")
    )
    if (
        parent_report.get("decision", {}).get(
            "local_semantic_skill_admitted"
        )
        is not True
        or parent_report.get("decision", {}).get(
            "fresh_local_replication_preregistration_allowed"
        )
        is not True
        or parent_report.get("decision", {}).get(
            "fresh_local_replication_generation_allowed"
        )
        is not False
    ):
        raise ValueError("semantic skill replication parent decision differs")
    return config


def parent_config(config: SemanticSkillReplicationConfig):
    parent_experiment = load_parent_config(config.parent_config_path)
    parent = load_parent_runtime(parent_experiment)
    return replace(
        parent,
        experiment_id=config.experiment_id,
        output_path=config.output_path,
        case_seed=config.case_seed,
        cases_per_family=config.cases_per_family,
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
    config: SemanticSkillReplicationConfig,
) -> list[dict[str, Any]]:
    rows = []
    for family in FAMILIES:
        for index in range(config.cases_per_family):
            if family == "implicit_scale_total":
                scale_word = "double" if index % 2 == 0 else "triple"
                source_facts: dict[str, Any] = {
                    "rows": 3 + (index % 17),
                    "columns": 4 + ((index * 3) % 19),
                    "extra": 7 + ((index * 5) % 43),
                    "scale_word": scale_word,
                }
                prompt = (
                    "For a compact display, the audited dimensions are "
                    f"rows={source_facts['rows']} and "
                    f"columns={source_facts['columns']}; the adjustment is "
                    f"extra={source_facts['extra']}. The order calls for extra "
                    f"more than {scale_word} the number of slots in that "
                    "rectangular display. Return the exact order quantity."
                )
            else:
                units = 3 + (index % 19)
                price = 7 + ((index * 2) % 23)
                net = 2 + ((index * 7) % 17)
                recurring = units * price - net
                threshold = 2 + ((index * 5) % 31)
                source_facts = {
                    "setup_cost": net * threshold,
                    "units_per_period": units,
                    "price_per_unit": price,
                    "recurring_cost": recurring,
                }
                prompt = (
                    "A kiosk forecast records "
                    f"setup_cost={source_facts['setup_cost']}, "
                    f"units_per_period={source_facts['units_per_period']}, "
                    f"price_per_unit={source_facts['price_per_unit']}, and "
                    f"recurring_cost={source_facts['recurring_cost']}. Each "
                    "period earns units_per_period times price_per_unit before "
                    "the recurring cost. Report the first whole period when "
                    "cumulative profit is strictly positive."
                )
            expected = execute_semantic_tool(family, source_facts)
            digest = hashlib.sha256(
                f"{family}\0{json.dumps(source_facts, sort_keys=True)}".encode()
            ).hexdigest()
            rows.append(
                {
                    "case_id": (
                        f"semantic-replication-{family}-{digest[:16]}"
                    ),
                    "family": family,
                    "prompt": prompt,
                    "source_facts": source_facts,
                    "expected": expected,
                }
            )
    if len({row["case_id"] for row in rows}) != len(rows):
        raise ValueError("semantic skill replication cases are not unique")
    rows.sort(
        key=lambda row: hashlib.sha256(
            f"{config.case_seed}\0{row['case_id']}".encode()
        ).hexdigest()
    )
    return rows


def run(config: SemanticSkillReplicationConfig) -> dict[str, Any]:
    mechanism = load_parent_config(config.parent_config_path)
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
            mechanism,
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
            "parent_preregister_sha256": config.parent_preregister_sha256,
            "parent_report_sha256": config.parent_report_sha256,
            "case_contract": public_case_contract(cases),
        },
        "mechanism_identity": {
            "skill_router": mechanism.skill_router,
            "route_markers": mechanism.route_markers,
            "plan_structured_output_regex_by_family": (
                mechanism.plan_structured_output_regex_by_family
            ),
            "direct_max_tokens": mechanism.direct_max_tokens,
            "plan_max_tokens": mechanism.plan_max_tokens,
            "final_max_tokens": mechanism.final_max_tokens,
            "plan_retry_limit": mechanism.plan_retry_limit,
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
