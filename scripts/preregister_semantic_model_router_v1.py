#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from transformers import AutoTokenizer

from nano_harness.baseline import sha256_file
from nano_harness.semantic_model_router import (
    ALL_FAMILIES,
    NEGATIVE_FAMILIES,
    POSITIVE_FAMILIES,
    ROUTER_PROMPT,
    build_cases,
    load_config,
    parent_config,
)
from nano_harness.semantic_skill_execution import (
    SKILL_PROMPTS,
    TOOL_REGEX_BY_FAMILY,
    load_config as load_mechanism_config,
)
from nano_harness.verified_tool_execution import (
    contamination_audit,
    public_case_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/harness/qwen35_semantic_model_router_v1.json"
JSON_OUTPUT = (
    ROOT
    / "docs/experiments/qwen35_semantic_model_router_v1.preregister.json"
)
MARKDOWN_OUTPUT = (
    ROOT / "docs/experiments/qwen35_semantic_model_router_v1.md"
)


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def build_receipt() -> dict:
    config = load_config(CONFIG)
    mechanism = load_mechanism_config(config.mechanism_config_path)
    parent = parent_config(config)
    cases = build_cases(config)
    contract = public_case_contract(cases)
    audit = contamination_audit(parent, cases)
    if not audit["passed"] or len(cases) != 256:
        raise ValueError("semantic model router contamination differs")
    counts = {
        family: sum(case["family"] == family for case in cases)
        for family in ALL_FAMILIES
    }
    if counts != {family: 64 for family in ALL_FAMILIES}:
        raise ValueError("semantic model router family counts differ")
    if any(
        marker in case["prompt"]
        for case in cases
        for markers in mechanism.route_markers.values()
        for marker in markers
    ):
        raise ValueError("semantic model router positive prompts leak exact markers")

    tokenizer = AutoTokenizer.from_pretrained(
        parent.four_b_model_path,
        local_files_only=True,
    )
    router_max_input = max(
        len(
            tokenizer.encode(
                tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": ROUTER_PROMPT},
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
    plan_max_input = {}
    for family in POSITIVE_FAMILIES:
        selected = [case for case in cases if case["family"] == family]
        plan_max_input[family] = max(
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
            for case in selected
        )
    if (
        router_max_input + config.router_max_tokens > parent.max_model_len
        or max(plan_max_input.values()) + config.plan_max_tokens
        > parent.max_model_len
    ):
        raise ValueError("semantic model router context budget differs")
    return {
        "schema_version": (
            "nano_harness_semantic_model_router_preregister_v1"
        ),
        "experiment_id": config.experiment_id,
        "identity": {
            "code_revision": git_revision(),
            "config_sha256": sha256_file(CONFIG),
            "mechanism_config_sha256": config.mechanism_config_sha256,
            "applicability_report_sha256": (
                config.applicability_report_sha256
            ),
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
            "exact_marker_occurrences": 0,
        },
        "surface": {
            "cases": 256,
            "family_counts": counts,
            "positive_cases": 128,
            "negative_cases": 128,
            "positive_families": list(POSITIVE_FAMILIES),
            "negative_families": list(NEGATIVE_FAMILIES),
        },
        "router": {
            "system_prompt_sha256": hashlib.sha256(
                ROUTER_PROMPT.encode()
            ).hexdigest(),
            "structured_output_regex": config.router_structured_output_regex,
            "routes": [
                "implicit_scale_total",
                "first_strict_profit_period",
                "NONE",
            ],
            "max_tokens": config.router_max_tokens,
            "maximum_input_tokens": router_max_input,
            "uses_case_metadata": False,
            "uses_expected_answer": False,
            "uses_case_correctness": False,
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
            "maximum_plan_input_tokens": plan_max_input,
            "feedback_result_equality_required": True,
            "direct_fallback_on_invalid_plan_or_feedback": True,
        },
        "acceptance": {
            "all_rows_complete_and_parseable": True,
            "router_outputs_parseable_256": True,
            "positive_route_recall_128": True,
            "negative_none_correct_128": True,
            "negative_false_positive_routes_zero": True,
            "positive_verified_executions_128": True,
            "positive_feedback_result_matches_128": True,
            "fallbacks_zero": True,
            "negative_candidate_exact_direct_parity": True,
            "candidate_vs_four_b_significant": True,
            "candidate_vs_four_b_minimum_wins": config.minimum_harness_wins,
            "candidate_vs_four_b_maximum_losses": config.maximum_harness_losses,
            "candidate_vs_nine_b_significant": True,
            "candidate_vs_nine_b_minimum_wins": config.minimum_harness_wins,
            "candidate_vs_nine_b_maximum_losses": config.maximum_harness_losses,
            "every_family_non_regression": True,
            "real_question_model_scan_preregistration_allowed_after_pass": True,
            "benchmark_generation_allowed_after_pass": False,
            "canary_rerun_allowed_after_pass": False,
            "holdout_allowed_after_pass": False,
            "training_allowed_after_pass": False,
        },
        "decision_policy": {
            "forbidden_after_observation": [
                "case_change",
                "positive_or_negative_family_change",
                "router_prompt_change",
                "router_regex_change",
                "router_budget_change",
                "tool_schema_change",
                "skill_prompt_change",
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
                "only real-surface router scan. Do not generate benchmark outputs."
            ),
            "failed": (
                "Publish negative evidence and stop this router. Do not tune "
                "or rerun on the observed local surface."
            ),
        },
        "execution_boundary": config.execution_boundary,
        "claim_boundary": (
            "This pre-registers a fresh local router test. It generates no model "
            "output and does not access benchmark, canary, holdout, or training "
            "rows. Passing authorizes only a separately pre-registered question-"
            "only router scan."
        ),
    }


def render_markdown(receipt: dict) -> str:
    return f"""# Qwen3.5 Constrained Semantic Model Router v1

## 目标

Exact markers 在真实 question surface 上 0/15559 覆盖。下一机制不从这些已扫描
prompts 提炼规则，而是在 fresh non-benchmark surface 上验证 4B enum router：

- `implicit_scale_total`
- `first_strict_profit_period`
- `NONE`

## Fresh Surface

- positive 128：两个 semantic families 各64，使用自然改写且不含旧 exact markers；
- negative 128 unsupported：box total / remaining stock 各64，必须全部选
  `NONE`；
- prior/benchmark prompt overlap：0；
- benchmark/canary/holdout rows 或 outputs：0。

## Frozen Pipeline

1. 4B router 只输出三个枚举之一，16 tokens；
2. `NONE` 直接复用 direct；
3. 正类只暴露一个 parent typed semantic skill；
4. planner、source validator、executor、feedback equality 与 fallback 保持不变；
5. router 或 planner 都不读 expected、correctness 或 case metadata。

## Gate

- 256 router outputs parseable；
- positive route recall 128/128；
- negative NONE 128/128，false positive 0；
- 128 positive verified executions 和 feedback matches；
- negative candidate 与 direct 评分字段完全一致；
- candidate vs 4B / 9B 均显著、至少12 wins、0 losses；
- every-family non-regression。

通过只允许另行预注册 **question-only real router scan**，不能直接生成 benchmark
outputs，也不能重跑 canary、访问 holdout 或训练。

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
                "router": receipt["router"],
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
