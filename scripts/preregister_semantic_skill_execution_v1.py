#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from transformers import AutoTokenizer

from nano_harness.baseline import sha256_file
from nano_harness.semantic_skill_execution import (
    FAMILIES,
    SKILL_PROMPTS,
    TOOL_REGEX_BY_FAMILY,
    build_cases,
    load_config,
    parent_config,
    route_prompt,
)
from nano_harness.verified_tool_execution import (
    contamination_audit,
    public_case_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/harness/qwen35_semantic_skill_execution_v1.json"
JSON_OUTPUT = (
    ROOT
    / "docs/experiments/qwen35_semantic_skill_execution_v1.preregister.json"
)
MARKDOWN_OUTPUT = (
    ROOT / "docs/experiments/qwen35_semantic_skill_execution_v1.md"
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
    if not audit["passed"]:
        raise ValueError("semantic skill contamination audit failed")
    routes = [route_prompt(case["prompt"]) for case in cases]
    if (
        len(cases) != 256
        or any(not route["routed"] for route in routes)
        or any(route["router_uses_case_metadata"] for route in routes)
    ):
        raise ValueError("semantic skill prompt routing differs")

    tokenizer = AutoTokenizer.from_pretrained(
        parent.four_b_model_path,
        local_files_only=True,
    )
    plan_lengths = {}
    feedback_lengths = {}
    for family in FAMILIES:
        selected = [case for case in cases if case["family"] == family]
        plan_lengths[family] = max(
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
        feedback_lengths[family] = max(
            len(
                tokenizer.encode(
                    tokenizer.apply_chat_template(
                        [
                            {
                                "role": "system",
                                "content": (
                                    "Return the verified semantic-tool result "
                                    "without changing it. Return only one "
                                    "FINAL: <integer> line."
                                ),
                            },
                            {
                                "role": "user",
                                "content": (
                                    f"<original_task>\n{case['prompt']}\n"
                                    "</original_task>\n\n"
                                    "<verified_semantic_tool>\n"
                                    f"name={family}\n"
                                    "arguments=<verified arguments>\n"
                                    "result=<verified integer>\n"
                                    "</verified_semantic_tool>\n\n"
                                    "Use the verified result as authoritative. "
                                    "Return only FINAL: <integer>."
                                ),
                            },
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
        max(plan_lengths.values()) + config.plan_max_tokens
        > parent.max_model_len
        or max(feedback_lengths.values()) + config.final_max_tokens
        > parent.max_model_len
    ):
        raise ValueError("semantic skill context budget differs")

    v2 = json.loads(Path(config.v2_report_path).read_text(encoding="utf-8"))
    canary = json.loads(
        Path(config.canary_rejection_path).read_text(encoding="utf-8")
    )
    if (
        v2.get("decision", {}).get("local_harness_admitted") is not True
        or canary.get("decision", {}).get("route_rejected") is not True
        or canary.get("decision", {}).get(
            "further_tuning_or_rerun_on_observed_canary_allowed"
        )
        is not False
    ):
        raise ValueError("semantic skill predecessor decisions differ")
    return {
        "schema_version": (
            "nano_harness_semantic_skill_execution_preregister_v1"
        ),
        "experiment_id": config.experiment_id,
        "identity": {
            "code_revision": git_revision(),
            "config_sha256": sha256_file(CONFIG),
            "parent_config_sha256": config.parent_config_sha256,
            "v2_report_sha256": config.v2_report_sha256,
            "canary_rejection_sha256": config.canary_rejection_sha256,
            "service_receipt_sha256": config.service_receipt_sha256,
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
        "mechanism": {
            "retained": [
                "same Qwen3.5-4B and Qwen3.5-9B services",
                "single applicable skill exposure",
                "typed source-fact validation",
                "one retry",
                "verified-result feedback",
                "direct fallback",
                "temperature and token budgets",
                "paired bootstrap and exact McNemar gates",
            ],
            "changed": [
                "fresh 120000-offset 256-case surface",
                "prompt-marker routing without case metadata",
                "typed implicit double/triple semantics",
                "typed first strictly profitable whole-period semantics",
            ],
            "stopped": [
                "literal-only arithmetic expression grounding",
                "211-case canary access",
            ],
        },
        "skills": {
            family: {
                "markers": config.route_markers[family],
                "plan_regex": TOOL_REGEX_BY_FAMILY[family],
                "system_prompt_sha256": hashlib.sha256(
                    SKILL_PROMPTS[family].encode()
                ).hexdigest(),
                "maximum_plan_input_tokens": plan_lengths[family],
                "maximum_feedback_input_tokens": feedback_lengths[family],
            }
            for family in FAMILIES
        },
        "acceptance": {
            "cases": 256,
            "cases_per_family": 128,
            "all_rows_complete_and_parseable": True,
            "prompt_routes_256": True,
            "single_tool_exposures_256": True,
            "verified_executions_256": True,
            "feedback_result_matches_256": True,
            "executor_contract_failures_zero": True,
            "harness_vs_four_b_significant": True,
            "harness_vs_four_b_minimum_wins": config.minimum_harness_wins,
            "harness_vs_four_b_maximum_losses": config.maximum_harness_losses,
            "harness_vs_nine_b_significant": True,
            "harness_vs_nine_b_minimum_wins": config.minimum_harness_wins,
            "harness_vs_nine_b_maximum_losses": config.maximum_harness_losses,
            "every_family_non_regression_vs_four_b_and_nine_b": True,
            "next_fresh_local_replication_allowed_after_pass": True,
            "canary_allowed_after_pass": False,
            "benchmark_allowed_after_pass": False,
        },
        "inherited_contract": {
            "four_b_model": parent.four_b_model,
            "nine_b_model": parent.nine_b_model,
            "service_receipt_path": parent.service_receipt_path,
            "temperature": parent.temperature,
            "chat_template_kwargs": parent.chat_template_kwargs,
            "direct_max_tokens": config.direct_max_tokens,
            "plan_max_tokens": config.plan_max_tokens,
            "final_max_tokens": config.final_max_tokens,
            "plan_retry_limit": config.plan_retry_limit,
            "bootstrap_samples": config.bootstrap_samples,
            "bootstrap_seed": config.bootstrap_seed,
            "significance_alpha": config.significance_alpha,
            "minimum_harness_wins": config.minimum_harness_wins,
            "maximum_harness_losses": config.maximum_harness_losses,
        },
        "decision_policy": {
            "forbidden_after_observation": [
                "case_change",
                "route_marker_change",
                "skill_schema_change",
                "semantic_executor_change",
                "prompt_change",
                "regex_change",
                "retry_change",
                "budget_change",
                "temperature_change",
                "model_or_service_change",
                "fallback_change",
                "gate_change",
                "arm_rerun",
                "canary_access",
                "benchmark_access",
                "holdout_access",
            ],
            "passed": (
                "Publish local evidence and separately pre-register a fresh "
                "history-disjoint semantic-skill replication. Do not access "
                "the observed canary."
            ),
            "failed": (
                "Publish negative evidence and change mechanism only on a new "
                "surface. Do not tune or rerun this surface."
            ),
        },
        "execution_boundary": config.execution_boundary,
        "claim_boundary": (
            "This pre-registers a fresh synthetic local mechanism test only. "
            "It generates no model output and does not reopen the observed "
            "canary, complete benchmark, independent holdout, or training."
        ),
    }


def render_markdown(receipt: dict) -> str:
    return f"""# Qwen3.5 Typed Semantic Skill Execution v1

## 假设

上一轮 literal calculator canary 保住了 209 行，但两个 recovery 都没变成新增
正确项。失败不是安全执行器算错，而是 tool semantics 太弱：

- `double` 是语言算子，不是题面数字；
- break-even 12 与“第一个严格盈利的整数周期”13不是同一语义。

本实验只在 fresh synthetic local surface 验证两个 typed semantic skills：

1. `implicit_scale_total`：executor 内部把 `double/triple` 映射为 2/3；
2. `first_strict_profit_period`：executor 计算
   `floor(setup_cost / period_net) + 1`，显式实现严格正收益边界。

Router 只读 prompt marker，不读 case family metadata、expected 或 correctness。
每行只暴露一个 skill schema。

## Fresh Surface

- 256 cases，2 families × 128；
- case contract SHA：
  `{receipt['identity']['case_contract_sha256']}`；
- prior choice/tool surface overlap：0；
- complete GSM8K/MMLU/GPQA prompt overlap：0；
- benchmark outputs、canary rows/outputs、holdout rows：0；
- training eligible rows：0。

## Frozen Gate

- 256/256 complete and parseable；
- 256 prompt routes、single-tool exposures、verified executions 和 feedback
  result matches；
- harness vs 4B direct、 dual 9B direct 均 CI lower > 0、McNemar p < 0.05、
  至少 12 wins、0 losses；
- 两个 family 分别对 4B/9B non-regression；
- 通过也只允许另行预注册 fresh local replication，不能访问已观察 canary、
  complete benchmark 或 independent holdout。

## Boundaries

- config SHA：`{receipt['identity']['config_sha256']}`；
- V2 report SHA：`{receipt['identity']['v2_report_sha256']}`；
- canary rejection SHA：
  `{receipt['identity']['canary_rejection_sha256']}`；
- model generation：false；
- evaluation started：false；
- canary accessed：false；
- benchmark accessed：false；
- holdout accessed：false。

观察结果后禁止修改 cases、markers、schema、executor、prompt、regex、retry、
budgets、fallback、gate 或重跑。
"""


def main() -> None:
    receipt = build_receipt()
    JSON_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUTPUT.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    MARKDOWN_OUTPUT.write_text(
        render_markdown(receipt),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "experiment_id": receipt["experiment_id"],
                "identity": receipt["identity"],
                "freshness": receipt["freshness"],
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
