#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from nano_harness.baseline import sha256_file
from nano_harness.router_adapter_integration import ROUTER_SYSTEM
from nano_harness.router_adapter_integration import (
    build_cases as build_v1_cases,
)
from nano_harness.router_adapter_integration import (
    load_config as load_v1_config,
)
from nano_harness.router_adapter_integration_v2 import (
    build_cases as build_v2_cases,
)
from nano_harness.router_adapter_integration_v2 import (
    load_config as load_v2_config,
)
from nano_harness.router_adapter_integration_v3 import (
    POSITIVE_FAMILIES,
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
    build_cases as build_prior_router_cases,
)
from nano_harness.semantic_model_router import (
    load_config as load_prior_router_config,
)
from nano_harness.semantic_skill_execution import (
    SKILL_PROMPTS,
    TOOL_REGEX_BY_FAMILY,
)
from nano_harness.verified_tool_execution import (
    contamination_audit,
    public_case_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/harness/qwen35_router_adapter_integration_v3.json"
JSON_OUTPUT = (
    ROOT
    / "docs/experiments/qwen35_router_adapter_integration_v3.preregister.json"
)
MARKDOWN_OUTPUT = (
    ROOT / "docs/experiments/qwen35_router_adapter_integration_v3.md"
)
PRIOR_ROUTER_CONFIG = (
    ROOT / "configs/harness/qwen35_semantic_model_router_v1.json"
)
PRIOR_BINARY_CONFIG = (
    ROOT / "configs/harness/qwen35_semantic_binary_detectors_v1.json"
)
V1_CONFIG = ROOT / "configs/harness/qwen35_router_adapter_integration_v1.json"
V2_CONFIG = ROOT / "configs/harness/qwen35_router_adapter_integration_v2.json"


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
    prompts = []
    for row in document["samples"]:
        users = [
            message["content"]
            for message in row["messages"]
            if message["role"] == "user"
        ]
        if len(users) != 1:
            raise ValueError("router integration v3 training messages differ")
        prompts.append(users[0])
    return prompts


def build_receipt() -> dict:
    config = load_config(CONFIG)
    mechanism, parent = parent_config(config)
    cases = build_cases(config)
    expected_count = (
        len(POSITIVE_FAMILIES) * config.positive_cases_per_family
        + len(config.negative_subtypes) * config.negative_cases_per_subtype
    )
    if len(cases) != expected_count or expected_count != 160:
        raise ValueError("router integration v3 case count differs")
    family_counts = Counter(case["family"] for case in cases)
    expected_family_counts = Counter(
        {
            **{
                family: config.positive_cases_per_family
                for family in POSITIVE_FAMILIES
            },
            **{
                subtype: config.negative_cases_per_subtype
                for subtype in config.negative_subtypes
            },
        }
    )
    label_counts = Counter(case["expected_label"] for case in cases)
    if family_counts != expected_family_counts:
        raise ValueError("router integration v3 family counts differ")
    if label_counts != Counter({"A": 16, "B": 16, "C": 128}):
        raise ValueError("router integration v3 label counts differ")
    if any(
        term in case["prompt"].casefold()
        for case in cases
        for term in ("route", "router", "classify", "classification")
    ):
        raise ValueError("router integration v3 prompt leaks classification")

    training_prompts = _training_prompts(Path(config.router_training_data_path))
    v1_cases = build_v1_cases(load_v1_config(V1_CONFIG))
    v2_cases = build_v2_cases(load_v2_config(V2_CONFIG))
    prior_router = build_prior_router_cases(
        load_prior_router_config(PRIOR_ROUTER_CONFIG)
    )
    prior_binary = build_binary_cases(load_binary_config(PRIOR_BINARY_CONFIG))
    fresh_hashes = {_prompt_sha256(case["prompt"]) for case in cases}
    overlap = {
        "router_training_prompts": len(
            fresh_hashes
            & {_prompt_sha256(prompt) for prompt in training_prompts}
        ),
        "integration_v1_prompts": len(
            fresh_hashes
            & {_prompt_sha256(case["prompt"]) for case in v1_cases}
        ),
        "integration_v2_prompts": len(
            fresh_hashes
            & {_prompt_sha256(case["prompt"]) for case in v2_cases}
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
    benchmark_audit = contamination_audit(parent, cases)
    if (
        any(overlap.values())
        or any(benchmark_audit["prior_surface_prompt_overlap"].values())
        or any(benchmark_audit["benchmark_prompt_overlap"].values())
    ):
        raise ValueError("router integration v3 is not history-disjoint")

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
        raise ValueError("router integration v3 context budget differs")

    contract = public_case_contract(cases)
    return {
        "schema_version": (
            "nano_harness_router_adapter_integration_preregister_v3"
        ),
        "experiment_id": config.experiment_id,
        "identity": {
            "code_revision": git_revision(),
            "config_sha256": sha256_file(CONFIG),
            "case_contract_sha256": contract["case_contract_sha256"],
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
            "remapped_adapter_sha256": config.adapter_tree_sha256,
            "remapped_adapter_weights_sha256": (
                config.adapter_weights_sha256
            ),
        },
        "case_contract": contract,
        "freshness": {
            "case_seed": config.case_seed,
            "value_offset": config.value_offset,
            "history_disjoint_prompt_overlap": overlap,
            "prior_surface_prompt_overlap": benchmark_audit[
                "prior_surface_prompt_overlap"
            ],
            "benchmark_prompt_overlap": benchmark_audit[
                "benchmark_prompt_overlap"
            ],
            "benchmark_rows_hashed": benchmark_audit["benchmark_rows_hashed"],
            "training_prompts_hashed": len(training_prompts),
            "integration_v1_prompts_hashed": len(v1_cases),
            "integration_v2_prompts_hashed": len(v2_cases),
            "prior_multiclass_prompts_hashed": len(prior_router),
            "prior_binary_prompts_hashed": len(prior_binary),
            "integration_v1_or_v2_outputs_loaded": False,
            "benchmark_outputs_loaded": False,
            "canary_rows_or_outputs_loaded": False,
            "holdout_rows_or_outputs_loaded": False,
        },
        "surface": {
            "cases": len(cases),
            "family_counts": dict(sorted(family_counts.items())),
            "expected_label_counts": dict(sorted(label_counts.items())),
            "positive_cases": 32,
            "negative_cases": 128,
            "negative_subtypes": list(config.negative_subtypes),
            "classification_instruction_occurrences": 0,
        },
        "router": {
            "served_adapter_name": config.served_adapter_name,
            "adapter_sha256": config.adapter_tree_sha256,
            "system_prompt_sha256": hashlib.sha256(
                ROUTER_SYSTEM.encode()
            ).hexdigest(),
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
        "acceptance": {
            "all_three_arms_complete_and_parseable_160": True,
            "router_outputs_parseable_160": True,
            "router_a_recall_16": True,
            "router_b_recall_16": True,
            "router_c_recall_128": True,
            "each_c_subtype_recall_16": True,
            "negative_false_positive_routes_zero": True,
            "positive_verified_executions_32": True,
            "positive_feedback_result_matches_32": True,
            "fallbacks_zero": True,
            "negative_candidate_exact_direct_parity": True,
            "candidate_vs_four_b_significant": True,
            "candidate_vs_four_b_minimum_wins": config.minimum_harness_wins,
            "candidate_vs_four_b_maximum_losses": (
                config.maximum_harness_losses
            ),
            "candidate_vs_nine_b_significant": True,
            "candidate_vs_nine_b_minimum_wins": config.minimum_harness_wins,
            "candidate_vs_nine_b_maximum_losses": (
                config.maximum_harness_losses
            ),
            "every_family_non_regression": True,
            "benchmark_treatment_preregistration_allowed_after_pass": True,
            "benchmark_generation_allowed_after_pass": False,
            "integration_v1_or_v2_rerun_allowed_after_pass": False,
            "canary_allowed_after_pass": False,
            "holdout_allowed_after_pass": False,
            "training_or_rl_allowed_after_pass": False,
        },
        "decision_policy": {
            "passed": (
                "Publish fresh v3 evidence and separately pre-register one "
                "benchmark-agnostic treatment transfer. Do not generate "
                "benchmark outputs."
            ),
            "failed": (
                "Publish negative evidence and stop this router treatment. "
                "Do not tune or rerun v1, v2, or v3."
            ),
            "forbidden_after_observation": [
                "integration_v1_or_v2_rerun",
                "integration_v3_rerun",
                "case_seed_value_or_wording_change",
                "adapter_namespace_or_weight_change",
                "prompt_parser_or_budget_change",
                "tool_validator_or_feedback_change",
                "gate_change",
                "benchmark_access",
                "canary_access",
                "holdout_access",
                "training",
                "rl",
            ],
        },
        "service_launch": config.service_launch,
        "execution_boundary": config.execution_boundary,
        "claim_boundary": (
            "This pre-registers one history-disjoint synthetic integration "
            "covering A, B, and eight C subtypes. It loads no prior "
            "integration outputs and starts no service or generation."
        ),
    }


def render_markdown(receipt: dict) -> str:
    return f"""# Qwen3.5 Router Adapter Integration v3

## Purpose

Negative-diversity SFT and namespace-remapped serving both pass 1,536/1,536.
V3 tests transfer on 160 new answer-task prompts: A/B plus all eight C
subtypes. It does not rerun V1 or V2.

## Freshness

- seed `{receipt['freshness']['case_seed']}`, value offset
  `{receipt['freshness']['value_offset']}`;
- 160 new prompts, A/B = 16/16 and C = 8 x 16;
- overlap with 7,680 training prompts, V1/V2 prompts, prior multiclass/binary
  prompts, and prior generic surfaces: all zero;
- overlap with complete GSM8K/MMLU/GPQA prompt columns: all zero;
- prior integration outputs, benchmark outputs, canary, and holdout: not loaded.

## Gates

- A 16/16, B 16/16, every C subtype 16/16, zero false routes;
- 32 verified executions, 32 feedback matches, zero fallbacks;
- all 128 negative cases preserve direct output exactly;
- significant zero-loss superiority over direct 4B and 9B;
- every-family non-regression.

Passing permits only a separately pre-registered benchmark-agnostic treatment
transfer. V1/V2/V3 cannot be rerun after observation. Benchmark generation,
canary, holdout, training, and RL remain closed.

## Identity

- config SHA: `{receipt['identity']['config_sha256']}`;
- case contract SHA: `{receipt['identity']['case_contract_sha256']}`;
- remapped adapter SHA: `{receipt['identity']['remapped_adapter_sha256']}`;
- parity report SHA: `{receipt['identity']['parity_report_sha256']}`;
- generation started: false.
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
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
