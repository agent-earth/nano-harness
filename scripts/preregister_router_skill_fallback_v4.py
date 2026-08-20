#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from nano_harness.baseline import sha256_file
from nano_harness.router_adapter_integration import (
    ROUTER_SYSTEM,
    build_cases as build_v1_cases,
    load_config as load_v1_config,
)
from nano_harness.router_adapter_integration_v2 import (
    build_cases as build_v2_cases,
)
from nano_harness.router_adapter_integration_v2 import (
    load_config as load_v2_config,
)
from nano_harness.router_adapter_integration_v3 import (
    build_cases as build_v3_cases,
)
from nano_harness.router_adapter_integration_v3 import (
    load_config as load_v3_config,
)
from nano_harness.router_skill_fallback_v4 import (
    C_FAMILIES,
    C_SKILL_PROMPT,
    POSITIVE_FAMILIES,
    build_cases,
    load_config,
    parent_config,
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
CONFIG = ROOT / "configs/harness/qwen35_router_skill_fallback_v4.json"
JSON_OUTPUT = (
    ROOT
    / "docs/experiments/qwen35_router_skill_fallback_v4.preregister.json"
)
MARKDOWN_OUTPUT = (
    ROOT / "docs/experiments/qwen35_router_skill_fallback_v4.md"
)
V1_CONFIG = ROOT / "configs/harness/qwen35_router_adapter_integration_v1.json"
V2_CONFIG = ROOT / "configs/harness/qwen35_router_adapter_integration_v2.json"
V3_CONFIG = ROOT / "configs/harness/qwen35_router_adapter_integration_v3.json"


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
            raise ValueError("router skill fallback v4 training messages differ")
        prompts.append(users[0])
    return prompts


def build_receipt() -> dict:
    config = load_config(CONFIG)
    mechanism, parent = parent_config(config)
    cases = build_cases(config)
    families = (*POSITIVE_FAMILIES, *C_FAMILIES)
    if len(cases) != 160:
        raise ValueError("router skill fallback v4 case count differs")
    family_counts = Counter(case["family"] for case in cases)
    label_counts = Counter(case["expected_label"] for case in cases)
    if family_counts != Counter({family: 16 for family in families}):
        raise ValueError("router skill fallback v4 family counts differ")
    if label_counts != Counter({"A": 16, "B": 16, "C": 128}):
        raise ValueError("router skill fallback v4 label counts differ")
    if any(
        term in case["prompt"].casefold()
        for case in cases
        for term in ("route", "router", "classify", "classification")
    ):
        raise ValueError("router skill fallback v4 prompt leaks classification")

    training_prompts = _training_prompts(Path(config.router_training_data_path))
    prior_cases = {
        "integration_v1_prompts": build_v1_cases(load_v1_config(V1_CONFIG)),
        "integration_v2_prompts": build_v2_cases(load_v2_config(V2_CONFIG)),
        "integration_v3_prompts": build_v3_cases(load_v3_config(V3_CONFIG)),
    }
    fresh_hashes = {_prompt_sha256(case["prompt"]) for case in cases}
    overlap = {
        "router_training_prompts": len(
            fresh_hashes
            & {_prompt_sha256(prompt) for prompt in training_prompts}
        ),
        **{
            name: len(
                fresh_hashes
                & {_prompt_sha256(case["prompt"]) for case in prior}
            )
            for name, prior in prior_cases.items()
        },
    }
    benchmark_audit = contamination_audit(parent, cases)
    if (
        any(overlap.values())
        or any(benchmark_audit["prior_surface_prompt_overlap"].values())
        or any(benchmark_audit["benchmark_prompt_overlap"].values())
    ):
        raise ValueError("router skill fallback v4 is not history-disjoint")

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
    ab_plan_max_input = {
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
    c_plan_max_input = max(
        len(
            tokenizer.encode(
                tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": C_SKILL_PROMPT},
                        {"role": "user", "content": case["prompt"]},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                    **parent.chat_template_kwargs,
                )
            )
        )
        for case in cases
        if case["family"] in C_FAMILIES
    )
    if (
        route_max_input + config.route_max_tokens > parent.max_model_len
        or max(ab_plan_max_input.values()) + config.plan_max_tokens
        > parent.max_model_len
        or c_plan_max_input + config.plan_max_tokens > parent.max_model_len
    ):
        raise ValueError("router skill fallback v4 context budget differs")

    contract = public_case_contract(cases)
    return {
        "schema_version": "nano_harness_router_skill_fallback_preregister_v4",
        "experiment_id": config.experiment_id,
        "identity": {
            "code_revision": git_revision(),
            "config_sha256": sha256_file(CONFIG),
            "case_contract_sha256": contract["case_contract_sha256"],
            "mechanism_config_sha256": config.mechanism_config_sha256,
            "parent_config_sha256": config.parent_config_sha256,
            "router_training_data_sha256": config.router_training_data_sha256,
            "integration_v3_report_sha256": (
                config.integration_v3_report_sha256
            ),
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
            "integration_v1_prompts_hashed": len(
                prior_cases["integration_v1_prompts"]
            ),
            "integration_v2_prompts_hashed": len(
                prior_cases["integration_v2_prompts"]
            ),
            "integration_v3_prompts_hashed": len(
                prior_cases["integration_v3_prompts"]
            ),
            "integration_v1_v2_v3_outputs_loaded": False,
            "benchmark_outputs_loaded": False,
            "canary_rows_or_outputs_loaded": False,
            "holdout_rows_or_outputs_loaded": False,
        },
        "surface": {
            "cases": len(cases),
            "family_counts": dict(sorted(family_counts.items())),
            "expected_label_counts": dict(sorted(label_counts.items())),
            "positive_cases": 32,
            "c_skill_cases": 128,
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
        "skills": {
            "ab_skill_prompts_sha256": {
                family: hashlib.sha256(
                    SKILL_PROMPTS[family].encode()
                ).hexdigest()
                for family in POSITIVE_FAMILIES
            },
            "ab_tool_regex_by_family": TOOL_REGEX_BY_FAMILY,
            "c_skill_prompt_sha256": hashlib.sha256(
                C_SKILL_PROMPT.encode()
            ).hexdigest(),
            "c_skill_structured_output_regex": (
                config.skill_plan_structured_output_regex
            ),
            "plan_max_tokens": config.plan_max_tokens,
            "plan_retry_limit": config.plan_retry_limit,
            "ab_maximum_plan_input_tokens": ab_plan_max_input,
            "c_maximum_plan_input_tokens": c_plan_max_input,
            "deterministic_result_is_final_output": True,
            "model_cannot_rewrite_c_skill_result": True,
            "selector_uses_case_metadata": False,
            "executor_uses_expected_answer": False,
            "executor_uses_case_correctness": False,
        },
        "acceptance": {
            "all_three_arms_complete_and_parseable_160": True,
            "router_outputs_parseable_and_correct_160": True,
            "ab_verified_executions_32": True,
            "c_skill_verified_executions_128": True,
            "c_skill_result_exact_128": True,
            "fallbacks_zero": True,
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
            "integration_v1_v2_v3_rerun_allowed_after_pass": False,
            "canary_allowed_after_pass": False,
            "holdout_allowed_after_pass": False,
            "training_or_rl_allowed_after_pass": False,
        },
        "decision_policy": {
            "passed": (
                "Publish V4 evidence and separately pre-register a "
                "benchmark-agnostic treatment transfer."
            ),
            "failed": (
                "Publish negative evidence. Do not rerun or tune V1-V4."
            ),
            "forbidden_after_observation": [
                "integration_v1_v2_v3_rerun",
                "v4_rerun",
                "case_seed_value_or_wording_change",
                "adapter_namespace_or_weight_change",
                "router_prompt_parser_or_budget_change",
                "skill_prompt_schema_parser_or_budget_change",
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
            "This pre-registers one history-disjoint synthetic test of an "
            "eight-skill C fallback. It starts no service or generation."
        ),
    }


def render_markdown(receipt: dict) -> str:
    return f"""# Qwen3.5 Router Skill Fallback v4

## Purpose

V3 proved the router (160/160) and A/B verified execution, but rejected
`C -> 4B direct` because it inherited 14 9B-only wins. V4 changes only the C
policy: a model selects one of eight typed skills, a deterministic verifier
checks copied facts, and the verified integer is emitted without model rewrite.

## Freshness

- seed `{receipt['freshness']['case_seed']}`, value offset
  `{receipt['freshness']['value_offset']}`;
- A/B = 16/16 and eight C skills = 8 x 16;
- overlap with 7,680 training prompts, V1/V2/V3 prompts, prior surfaces, and
  complete GSM8K/MMLU/GPQA prompt columns: all zero;
- prior outputs, benchmark outputs, canary, and holdout: not loaded.

## Gates

- router 160/160;
- A/B verified executions 32/32;
- C typed-skill verified executions and exact results 128/128;
- zero fallbacks;
- significant zero-loss superiority over direct 4B and 9B;
- every-family non-regression.

Passing permits only a separately pre-registered benchmark treatment. V1-V4
cannot be rerun after observation. Benchmark generation, canary, holdout,
training, and RL remain closed.

## Identity

- config SHA: `{receipt['identity']['config_sha256']}`;
- case contract SHA: `{receipt['identity']['case_contract_sha256']}`;
- V3 report SHA: `{receipt['identity']['integration_v3_report_sha256']}`;
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
