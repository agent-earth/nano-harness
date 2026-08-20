#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from transformers import AutoTokenizer

from nano_harness.baseline import sha256_file
from nano_harness.semantic_binary_detectors import (
    ALL_FAMILIES,
    DETECTOR_PROMPTS,
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
CONFIG = (
    ROOT / "configs/harness/qwen35_semantic_binary_detectors_v1.json"
)
JSON_OUTPUT = (
    ROOT
    / "docs/experiments/qwen35_semantic_binary_detectors_v1.preregister.json"
)
MARKDOWN_OUTPUT = (
    ROOT / "docs/experiments/qwen35_semantic_binary_detectors_v1.md"
)


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def build_receipt() -> dict:
    config = load_config(CONFIG)
    parent = parent_config(config)
    cases = build_cases(config)
    contract = public_case_contract(cases)
    audit = contamination_audit(parent, cases)
    if not audit["passed"] or len(cases) != 128:
        raise ValueError("semantic binary detector contamination differs")
    counts = {
        family: sum(case["family"] == family for case in cases)
        for family in ALL_FAMILIES
    }
    if counts != {family: 32 for family in ALL_FAMILIES}:
        raise ValueError("semantic binary detector family counts differ")
    tokenizer = AutoTokenizer.from_pretrained(
        parent.four_b_model_path,
        local_files_only=True,
    )
    detector_max_input = {
        family: max(
            len(
                tokenizer.encode(
                    tokenizer.apply_chat_template(
                        [
                            {
                                "role": "system",
                                "content": DETECTOR_PROMPTS[family],
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
        )
        for family in POSITIVE_FAMILIES
    }
    if (
        max(detector_max_input.values()) + config.detector_max_tokens
        > parent.max_model_len
    ):
        raise ValueError("semantic binary detector context differs")
    return {
        "schema_version": (
            "nano_harness_semantic_binary_detectors_preregister_v1"
        ),
        "experiment_id": config.experiment_id,
        "identity": {
            "code_revision": git_revision(),
            "config_sha256": sha256_file(CONFIG),
            "mechanism_config_sha256": config.mechanism_config_sha256,
            "multiclass_report_sha256": config.multiclass_report_sha256,
            "case_contract_sha256": contract["case_contract_sha256"],
        },
        "case_contract": contract,
        "freshness": {
            "prior_surface_prompt_overlap": audit[
                "prior_surface_prompt_overlap"
            ],
            "benchmark_prompt_overlap": audit["benchmark_prompt_overlap"],
            "benchmark_rows_hashed": audit["benchmark_rows_hashed"],
            "benchmark_outputs_loaded": False,
            "canary_rows_or_outputs_loaded": False,
            "holdout_rows_loaded": False,
        },
        "surface": {
            "cases": 128,
            "family_counts": counts,
            "positive_cases": 64,
            "negative_cases": 64,
        },
        "detectors": {
            family: {
                "system_prompt_sha256": hashlib.sha256(
                    DETECTOR_PROMPTS[family].encode()
                ).hexdigest(),
                "regex": config.detector_structured_output_regex,
                "max_tokens": config.detector_max_tokens,
                "maximum_input_tokens": detector_max_input[family],
            }
            for family in POSITIVE_FAMILIES
        },
        "composition": {
            "exactly_one_yes": "select_that_skill",
            "both_no": "NONE",
            "both_yes": "NONE",
            "none_route": "direct_preserve",
            "router_uses_case_metadata": False,
            "router_uses_expected_answer": False,
            "router_uses_case_correctness": False,
        },
        "executor_invariance": {
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
            "feedback_result_equality_required": True,
            "direct_fallback_required": True,
        },
        "acceptance": {
            "all_rows_complete_and_parseable": True,
            "all_detector_outputs_parseable": True,
            "detector_composition_correct_128": True,
            "positive_route_recall_64": True,
            "negative_none_correct_64": True,
            "negative_false_positive_routes_zero": True,
            "conflicts_zero": True,
            "positive_verified_executions_64": True,
            "positive_feedback_matches_64": True,
            "fallbacks_zero": True,
            "negative_candidate_exact_direct_parity": True,
            "candidate_vs_four_b_significant": True,
            "candidate_vs_four_b_minimum_wins": config.minimum_harness_wins,
            "candidate_vs_four_b_maximum_losses": config.maximum_harness_losses,
            "candidate_vs_nine_b_significant": True,
            "candidate_vs_nine_b_minimum_wins": config.minimum_harness_wins,
            "candidate_vs_nine_b_maximum_losses": config.maximum_harness_losses,
            "every_family_non_regression": True,
            "real_question_detector_scan_preregistration_allowed_after_pass": True,
            "benchmark_generation_allowed_after_pass": False,
            "canary_rerun_allowed_after_pass": False,
            "holdout_allowed_after_pass": False,
            "training_allowed_after_pass": False,
        },
        "decision_policy": {
            "forbidden_after_observation": [
                "case_change",
                "detector_prompt_change",
                "detector_regex_change",
                "composition_change",
                "budget_change",
                "skill_prompt_change",
                "tool_schema_change",
                "validator_change",
                "feedback_change",
                "fallback_change",
                "gate_change",
                "arm_rerun",
                "benchmark_access",
                "canary_access",
                "holdout_access",
            ],
            "passed": (
                "Publish local evidence and separately pre-register a question-"
                "only real detector scan. Do not generate benchmark outputs."
            ),
            "failed": (
                "Publish negative evidence and stop this detector composition. "
                "Do not tune or rerun this surface."
            ),
        },
        "execution_boundary": config.execution_boundary,
        "claim_boundary": (
            "This pre-registers a fresh 128-case local detector test. Passing "
            "allows only a separately pre-registered question-only real scan."
        ),
    }


def render_markdown(receipt: dict) -> str:
    return f"""# Qwen3.5 Semantic Binary Detectors v1

## 设计

Multiclass router 对 unsupported tasks 0 false positive，但 implicit-scale recall
为0。新实验拆成两个独立 detector：

- implicit-scale detector：`DETECT: YES/NO`
- strict-profit detector：`DETECT: YES/NO`

组合规则固定：恰好一个 YES 才选该 skill；双 NO 或双 YES 一律 `NONE` 并
direct-preserve。

## Fresh Surface

- 128 cases：4 families × 32；
- positive 64，negative 64；
- prior/benchmark prompt overlap：0；
- benchmark/canary/holdout rows 或 outputs：0。

## Gate

- 256 detector calls 全部 parseable；
- composition correct 128/128；
- positive recall 64/64；
- negative NONE 64/64，false positive 0；
- conflict 0；
- positive 64 verified executions 与 feedback matches；
- negative candidate 与 direct 完全一致；
- candidate vs 4B/9B 均显著、至少12 wins、0 losses。

通过也只允许另行预注册 question-only real detector scan，不允许 benchmark
generation、canary rerun、holdout 或 training。

## Boundary

- config SHA：`{receipt['identity']['config_sha256']}`；
- case contract SHA：`{receipt['identity']['case_contract_sha256']}`；
- model generation：false；
- evaluation started：false；
- benchmark/canary/holdout accessed：false。
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
                "detectors": receipt["detectors"],
                "composition": receipt["composition"],
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
