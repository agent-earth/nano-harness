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
from nano_harness.router_adapter_integration_v2 import (
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
from nano_harness.verified_tool_execution import (
    contamination_audit,
    public_case_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/harness/qwen35_router_adapter_integration_v2.json"
JSON_OUTPUT = (
    ROOT
    / "docs/experiments/qwen35_router_adapter_integration_v2.preregister.json"
)
MARKDOWN_OUTPUT = (
    ROOT / "docs/experiments/qwen35_router_adapter_integration_v2.md"
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
    prompts = []
    for row in document["samples"]:
        users = [
            message["content"]
            for message in row["messages"]
            if message["role"] == "user"
        ]
        if len(users) != 1:
            raise ValueError("router integration v2 training messages differ")
        prompts.append(users[0])
    return prompts


def build_receipt() -> dict:
    config = load_config(CONFIG)
    mechanism, parent = parent_config(config)
    cases = build_cases(config)
    if len(cases) != 128:
        raise ValueError("router integration v2 case count differs")
    family_counts = Counter(case["family"] for case in cases)
    label_counts = Counter(case["expected_label"] for case in cases)
    if family_counts != Counter({family: 32 for family in ALL_FAMILIES}):
        raise ValueError("router integration v2 family counts differ")
    if label_counts != Counter({"A": 32, "B": 32, "C": 64}):
        raise ValueError("router integration v2 label counts differ")
    if any(
        term in case["prompt"].casefold()
        for case in cases
        for term in ("route", "router", "classify", "classification")
    ):
        raise ValueError("router integration v2 prompt leaks classification")

    v1_config = json.loads(
        Path(config.integration_v1_config_path).read_text(encoding="utf-8")
    )
    from nano_harness.router_adapter_integration import (
        build_cases as build_v1_cases,
        load_config as load_v1_config,
    )

    v1_cases = build_v1_cases(load_v1_config(config.integration_v1_config_path))
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
            fresh_hashes
            & {_prompt_sha256(prompt) for prompt in training_prompts}
        ),
        "integration_v1_prompts": len(
            fresh_hashes
            & {_prompt_sha256(case["prompt"]) for case in v1_cases}
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
        raise ValueError("router integration v2 is not history-disjoint")

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
        raise ValueError("router integration v2 context budget differs")

    contract = public_case_contract(cases)
    return {
        "schema_version": (
            "nano_harness_router_adapter_integration_preregister_v2"
        ),
        "experiment_id": config.experiment_id,
        "identity": {
            "code_revision": git_revision(),
            "config_sha256": sha256_file(CONFIG),
            "case_contract_sha256": contract["case_contract_sha256"],
            "mechanism_config_sha256": config.mechanism_config_sha256,
            "router_training_data_sha256": (
                config.router_training_data_sha256
            ),
            "integration_v1_config_sha256": (
                config.integration_v1_config_sha256
            ),
            "integration_v1_preregister_sha256": (
                config.integration_v1_preregister_sha256
            ),
            "integration_v1_report_sha256": (
                config.integration_v1_report_sha256
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
            "prior_multiclass_prompts_hashed": len(prior_router),
            "prior_binary_prompts_hashed": len(prior_binary),
            "integration_v1_outputs_loaded": False,
            "benchmark_outputs_loaded": False,
            "canary_rows_or_outputs_loaded": False,
            "holdout_rows_or_outputs_loaded": False,
        },
        "surface": {
            "cases": len(cases),
            "family_counts": dict(sorted(family_counts.items())),
            "expected_label_counts": dict(sorted(label_counts.items())),
            "positive_cases": 64,
            "negative_cases": 64,
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
            "candidate_vs_four_b_maximum_losses": (
                config.maximum_harness_losses
            ),
            "candidate_vs_nine_b_significant": True,
            "candidate_vs_nine_b_minimum_wins": config.minimum_harness_wins,
            "candidate_vs_nine_b_maximum_losses": (
                config.maximum_harness_losses
            ),
            "every_family_non_regression": True,
            "question_only_scan_preregistration_allowed_after_pass": True,
            "integration_v1_rerun_allowed_after_pass": False,
            "benchmark_allowed_after_pass": False,
            "canary_allowed_after_pass": False,
            "holdout_allowed_after_pass": False,
            "training_or_rl_allowed_after_pass": False,
        },
        "decision_policy": {
            "passed": (
                "Publish fresh v2 evidence and separately pre-register one "
                "question-only scan. Do not generate benchmark outputs."
            ),
            "failed": (
                "Publish negative evidence and stop this remapped router. "
                "Do not tune or rerun v1 or v2."
            ),
            "forbidden_after_observation": [
                "integration_v1_rerun",
                "integration_v2_rerun",
                "case_or_seed_change",
                "value_range_or_wording_change",
                "adapter_or_namespace_change",
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
        "execution_boundary": config.execution_boundary,
        "claim_boundary": (
            "This pre-registers one new history-disjoint synthetic "
            "integration using the content-identical remapped adapter. It "
            "does not rerun or load v1 outputs and starts no generation."
        ),
    }


def render_markdown(receipt: dict) -> str:
    return f"""# Qwen3.5 Router Adapter Integration v2

## Purpose

Serving parity proved that the original PEFT namespace was inert under vLLM,
while the content-identical remap reproduced HF at 192/192. V2 tests transfer
on a **new** 128-case surface; it does not rerun V1.

## Freshness

- seed `{receipt['freshness']['case_seed']}`, value offset
  `{receipt['freshness']['value_offset']}`;
- task prompts ask for answers, not route/classification labels;
- exact normalized overlap with 960 training, 128 V1, 256 multiclass, and 128
  binary prompts: all zero;
- exact overlap with complete GSM8K/MMLU/GPQA prompt columns: all zero;
- V1 outputs, benchmark outputs, canary, and holdout: not loaded.

## Gates

- A 32/32, B 32/32, C 64/64 and zero false routes;
- 64 verified executions, 64 feedback matches, zero fallbacks;
- negative direct preservation;
- significant zero-loss superiority over direct 4B and 9B;
- every-family non-regression.

Passing permits only a separately pre-registered question-only scan. V1 and V2
cannot be rerun after observation. Benchmark, canary, holdout, training, and RL
remain closed.

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
