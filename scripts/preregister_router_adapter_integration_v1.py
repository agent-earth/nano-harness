#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from transformers import AutoTokenizer

from nano_harness.baseline import sha256_file
from nano_harness.router_adapter_integration import (
    LABEL_TO_ROUTE,
    ROUTER_SYSTEM,
    build_cases,
    load_config,
    parent_config,
)
from nano_harness.semantic_binary_detectors import (
    build_cases as build_binary_cases,
)
from nano_harness.semantic_binary_detectors import (
    load_config as load_binary_config,
)
from nano_harness.semantic_model_router import (
    ALL_FAMILIES,
    POSITIVE_FAMILIES,
    build_cases as build_prior_router_cases,
)
from nano_harness.semantic_model_router import (
    load_config as load_prior_router_config,
)
from nano_harness.semantic_skill_execution import (
    SKILL_PROMPTS,
    TOOL_REGEX_BY_FAMILY,
)
from nano_harness.verified_tool_execution import public_case_contract


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/harness/qwen35_router_adapter_integration_v1.json"
JSON_OUTPUT = (
    ROOT
    / "docs/experiments/qwen35_router_adapter_integration_v1.preregister.json"
)
MARKDOWN_OUTPUT = (
    ROOT / "docs/experiments/qwen35_router_adapter_integration_v1.md"
)


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def _prompt_sha256(prompt: str) -> str:
    normalized = " ".join(prompt.casefold().split())
    return hashlib.sha256(normalized.encode()).hexdigest()


def _training_prompts(path: Path) -> list[str]:
    document = json.loads(path.read_text(encoding="utf-8"))
    rows = document["samples"]
    prompts = []
    for row in rows:
        user_messages = [
            message["content"]
            for message in row["messages"]
            if message["role"] == "user"
        ]
        if len(user_messages) != 1:
            raise ValueError("router training row user-message count differs")
        prompts.append(user_messages[0])
    return prompts


def build_receipt() -> dict:
    config = load_config(CONFIG)
    mechanism, parent = parent_config(config)
    cases = build_cases(config)
    if len(cases) != 128:
        raise ValueError("router adapter integration case count differs")
    counts = {
        family: sum(case["family"] == family for case in cases)
        for family in ALL_FAMILIES
    }
    if counts != {family: 32 for family in ALL_FAMILIES}:
        raise ValueError("router adapter integration family counts differ")
    contract = public_case_contract(cases)

    training_prompts = _training_prompts(Path(config.router_training_data_path))
    prior_router = build_prior_router_cases(
        load_prior_router_config(config.prior_router_config_path)
    )
    prior_binary = build_binary_cases(
        load_binary_config(config.prior_binary_config_path)
    )
    fresh_hashes = {_prompt_sha256(case["prompt"]) for case in cases}
    overlap = {
        "router_training_prompts": len(
            fresh_hashes & {_prompt_sha256(prompt) for prompt in training_prompts}
        ),
        "prior_multiclass_prompts": len(
            fresh_hashes
            & {_prompt_sha256(case["prompt"]) for case in prior_router}
        ),
        "prior_binary_prompts": len(
            fresh_hashes
            & {_prompt_sha256(case["prompt"]) for case in prior_binary}
        ),
    }
    if any(overlap.values()):
        raise ValueError("router adapter integration is not history-disjoint")

    tokenizer = AutoTokenizer.from_pretrained(
        parent.four_b_model_path,
        local_files_only=True,
    )
    route_max_input = max(
        len(
            tokenizer.encode(
                tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": ROUTER_SYSTEM},
                        {"role": "user", "content": case["prompt"]},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                    **parent.chat_template_kwargs,
                )
            )
        )
        for case in cases
    )
    plan_max_input = {
        family: max(
            len(
                tokenizer.encode(
                    tokenizer.apply_chat_template(
                        [
                            {
                                "role": "system",
                                "content": SKILL_PROMPTS[family],
                            },
                            {"role": "user", "content": case["prompt"]},
                        ],
                        tokenize=False,
                        add_generation_prompt=True,
                        **parent.chat_template_kwargs,
                    )
                )
            )
            for case in cases
            if case["family"] == family
        )
        for family in POSITIVE_FAMILIES
    }
    if (
        route_max_input + config.route_max_tokens > parent.max_model_len
        or max(plan_max_input.values()) + config.plan_max_tokens
        > parent.max_model_len
    ):
        raise ValueError("router adapter integration context budget differs")

    return {
        "schema_version": (
            "nano_harness_router_adapter_integration_preregister_v1"
        ),
        "experiment_id": config.experiment_id,
        "identity": {
            "code_revision": git_revision(),
            "config_sha256": sha256_file(CONFIG),
            "mechanism_config_sha256": config.mechanism_config_sha256,
            "sft_report_sha256": config.sft_report_sha256,
            "router_training_data_sha256": (
                config.router_training_data_sha256
            ),
            "adapter_sha256": config.adapter_tree_sha256,
            "adapter_config_sha256": config.adapter_config_sha256,
            "adapter_model_sha256": config.adapter_model_sha256,
            "case_contract_sha256": contract["case_contract_sha256"],
        },
        "case_contract": contract,
        "freshness": {
            "case_seed": config.case_seed,
            "history_disjoint_prompt_overlap": overlap,
            "training_prompts_hashed": len(training_prompts),
            "prior_multiclass_prompts_hashed": len(prior_router),
            "prior_binary_prompts_hashed": len(prior_binary),
            "benchmark_rows_or_outputs_loaded": False,
            "canary_rows_or_outputs_loaded": False,
            "holdout_rows_or_outputs_loaded": False,
        },
        "surface": {
            "cases": len(cases),
            "family_counts": counts,
            "positive_cases": sum(case["positive"] for case in cases),
            "negative_cases": sum(not case["positive"] for case in cases),
            "expected_label_counts": {
                label: sum(case["expected_label"] == label for case in cases)
                for label in LABEL_TO_ROUTE
            },
        },
        "router": {
            "system_prompt_sha256": hashlib.sha256(
                ROUTER_SYSTEM.encode()
            ).hexdigest(),
            "system_prompt_matches_training_contract": True,
            "served_adapter_name": config.served_adapter_name,
            "structured_output_regex": config.route_structured_output_regex,
            "max_tokens": config.route_max_tokens,
            "maximum_input_tokens": route_max_input,
            "uses_case_metadata": False,
            "uses_expected_answer": False,
            "uses_case_correctness": False,
        },
        "executor": {
            "base_model": parent.four_b_model,
            "skill_prompts_sha256": {
                family: hashlib.sha256(
                    SKILL_PROMPTS[family].encode()
                ).hexdigest()
                for family in POSITIVE_FAMILIES
            },
            "tool_regex_by_family": TOOL_REGEX_BY_FAMILY,
            "plan_max_tokens": config.plan_max_tokens,
            "final_max_tokens": config.final_max_tokens,
            "plan_retry_limit": config.plan_retry_limit,
            "maximum_plan_input_tokens": plan_max_input,
            "unsupported_route": "direct_preserve",
            "feedback_result_equality_required": True,
            "direct_fallback_on_invalid_plan_or_feedback": True,
        },
        "service_launch": config.service_launch,
        "acceptance": {
            "all_three_arms_complete_and_parseable_128": True,
            "router_outputs_parseable_128": True,
            "router_a_recall_32": True,
            "router_b_recall_32": True,
            "router_c_precision_64": True,
            "negative_false_positive_routes_zero": True,
            "positive_verified_executions_64": True,
            "positive_feedback_result_matches_64": True,
            "fallbacks_zero": True,
            "negative_candidate_exact_direct_parity": True,
            "candidate_vs_four_b_significant": True,
            "candidate_vs_four_b_minimum_wins": config.minimum_harness_wins,
            "candidate_vs_four_b_maximum_losses": config.maximum_harness_losses,
            "candidate_vs_nine_b_significant": True,
            "candidate_vs_nine_b_minimum_wins": config.minimum_harness_wins,
            "candidate_vs_nine_b_maximum_losses": config.maximum_harness_losses,
            "every_family_non_regression": True,
            "question_only_scan_preregistration_allowed_after_pass": True,
            "benchmark_generation_allowed_after_pass": False,
            "canary_rerun_allowed_after_pass": False,
            "holdout_allowed_after_pass": False,
            "training_or_rl_allowed_after_pass": False,
        },
        "decision_policy": {
            "forbidden_after_observation": [
                "adapter_change",
                "adapter_weight_change",
                "case_change",
                "seed_or_value_range_change",
                "router_prompt_change",
                "router_regex_change",
                "router_budget_change",
                "tool_schema_change",
                "skill_prompt_change",
                "validator_or_feedback_change",
                "fallback_change",
                "gate_change",
                "arm_rerun",
                "benchmark_access",
                "canary_access",
                "holdout_access",
            ],
            "passed": (
                "Publish the local result and separately pre-register one "
                "question-only real-surface adapter scan."
            ),
            "failed": (
                "Publish negative evidence, reject this adapter integration, "
                "and do not tune or rerun on the observed surface."
            ),
        },
        "execution_boundary": config.execution_boundary,
        "claim_boundary": (
            "This pre-registers one fresh synthetic adapter integration. It "
            "starts no service, model generation, evaluation, benchmark, "
            "canary, holdout, training, or RL work."
        ),
    }


def render_markdown(receipt: dict) -> str:
    return f"""# Qwen3.5 Router Adapter Integration v1

## Purpose

The exact router adapter `{receipt['identity']['adapter_sha256']}` is frozen
before serving. It only emits `FINAL: A/B/C`; the unchanged base 4B performs
typed planning and verified feedback. `C` preserves the base direct answer.

## Fresh Surface

- 128 cases: 32 per family;
- positive A/B: 64; unsupported C: 64;
- exact normalized prompt overlap with 960 SFT rows, 256 prior multiclass
  rows, and 128 prior binary rows: all zero;
- benchmark, canary, and holdout rows or outputs loaded: false.

## Gates

- A 32/32, B 32/32, C 64/64, zero unsupported false routes;
- 64 verified executions, 64 feedback matches, zero fallbacks;
- unsupported candidate exactly preserves direct scoring fields;
- candidate significantly beats both direct 4B and direct 9B with at least
  12 wins and zero losses;
- every family is non-regressing.

Passing permits only a separately pre-registered question-only scan. It does
not permit benchmark generation, canary rerun, holdout access, training, or RL.

## Frozen Identity

- config SHA: `{receipt['identity']['config_sha256']}`;
- case contract SHA: `{receipt['identity']['case_contract_sha256']}`;
- SFT report SHA: `{receipt['identity']['sft_report_sha256']}`;
- adapter SHA: `{receipt['identity']['adapter_sha256']}`;
- model generation started: false;
- adapter service started: false.
"""


def main() -> None:
    receipt = build_receipt()
    JSON_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUTPUT.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    MARKDOWN_OUTPUT.write_text(render_markdown(receipt), encoding="utf-8")
    print(
        json.dumps(
            {
                "experiment_id": receipt["experiment_id"],
                "identity": receipt["identity"],
                "freshness": receipt["freshness"],
                "surface": receipt["surface"],
                "acceptance": receipt["acceptance"],
                "execution_boundary": receipt["execution_boundary"],
                "json_output": str(JSON_OUTPUT),
                "markdown_output": str(MARKDOWN_OUTPUT),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
