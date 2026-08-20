#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from transformers import AutoTokenizer

from nano_harness.baseline import sha256_file
from nano_harness.grounded_calculator_canary import (
    load_config,
    verify_frozen_inputs,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/harness/qwen35_grounded_calculator_canary_v1.json"
)
JSON_OUTPUT = (
    ROOT
    / "docs/experiments/qwen35_grounded_calculator_canary_v1.preregister.json"
)
MARKDOWN_OUTPUT = (
    ROOT / "docs/experiments/qwen35_grounded_calculator_canary_v1.md"
)


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def build_receipt() -> dict:
    config = load_config(CONFIG)
    frozen = verify_frozen_inputs(config, verify_service=False)
    parent = frozen["parent_config"]
    tokenizer = AutoTokenizer.from_pretrained(
        parent.four_b_model_path,
        local_files_only=True,
    )
    maximum_plan_input = 0
    maximum_feedback_input = 0
    for case in frozen["cases"]:
        if case.case_id not in frozen["eligible_case_ids"]:
            continue
        plan_text = tokenizer.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": (
                        "The GSM8K calculator-recovery skill was selected because "
                        "the existing direct response lacked a parseable final "
                        "line. Translate the original problem into one arithmetic "
                        "expression. Every numeric literal must copy a numeric "
                        "value from the original task; a source value may be "
                        "reused. Use only +, -, *, /, and parentheses. Do not "
                        "answer, estimate, or explain. Return only CALC: "
                        "<expression>."
                    ),
                },
                {"role": "user", "content": case.prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
            **parent.chat_template_kwargs,
        )
        feedback_text = tokenizer.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": (
                        "Return the verified calculator result without changing "
                        "it. Return only one FINAL: <number> line."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"<original_task>\n{case.prompt}\n</original_task>\n\n"
                        "<verified_calculator>\n"
                        "expression=<bounded grounded expression>\n"
                        "result=<verified integer>\n"
                        "</verified_calculator>\n\n"
                        "Use the verified result as authoritative. Return only "
                        "FINAL: <number>."
                    ),
                },
            ],
            tokenize=False,
            add_generation_prompt=True,
            **parent.chat_template_kwargs,
        )
        maximum_plan_input = max(
            maximum_plan_input,
            len(tokenizer.encode(plan_text)),
        )
        maximum_feedback_input = max(
            maximum_feedback_input,
            len(tokenizer.encode(feedback_text)),
        )
    if (
        maximum_plan_input + config.plan_max_tokens > parent.max_model_len
        or maximum_feedback_input + config.final_max_tokens
        > parent.max_model_len
    ):
        raise ValueError("grounded calculator context budget differs")

    return {
        "schema_version": (
            "nano_harness_grounded_calculator_canary_preregister_v1"
        ),
        "experiment_id": config.experiment_id,
        "identity": {
            "code_revision": git_revision(),
            "config_sha256": sha256_file(CONFIG),
            "manifest_sha256": config.manifest_sha256,
            "case_manifest_sha256": config.case_manifest_sha256,
            "four_b_raw_sha256": config.four_b_raw_sha256,
            "nine_b_raw_sha256": config.nine_b_raw_sha256,
            "baseline_report_sha256": config.baseline_report_sha256,
            "v2_config_sha256": config.v2_config_sha256,
            "v2_preregister_sha256": config.v2_preregister_sha256,
            "v2_report_sha256": config.v2_report_sha256,
            "service_receipt_sha256": config.service_receipt_sha256,
            "eligible_case_ids_sha256": frozen[
                "eligible_case_ids_sha256"
            ],
        },
        "surface": {
            "cases": len(frozen["cases"]),
            "benchmark_counts": {
                benchmark: sum(
                    case.benchmark == benchmark for case in frozen["cases"]
                )
                for benchmark in ("gsm8k", "mmlu", "gpqa_diamond")
            },
            "prior_direct_four_b_correct": 163,
            "prior_direct_nine_b_correct": 151,
            "prior_direct_parse_failure_eligible_rows": len(
                frozen["eligible_case_ids"]
            ),
            "case_id_allowlist_used": False,
            "eligibility_rule": config.recovery_eligibility,
        },
        "route": {
            "by_benchmark": config.route_by_benchmark,
            "direct_preserve": (
                "Every MMLU and GPQA row and every parseable GSM8K row reuses "
                "the frozen 4B direct output byte-for-byte for scoring fields."
            ),
            "gsm8k_recovery": [
                "ask 4B for one CALC expression under the frozen regex",
                "parse a bounded Python expression AST without eval",
                "allow only +, -, *, /, parentheses, and unary signs",
                "require every numeric literal value to occur in the prompt",
                "execute with exact rational arithmetic and require an integer",
                "send the verified result to the same 4B for a 32-token FINAL",
                "reuse the frozen direct output on any failure",
            ],
            "plan_regex": config.plan_structured_output_regex,
            "plan_max_tokens": config.plan_max_tokens,
            "final_max_tokens": config.final_max_tokens,
            "plan_retry_limit": config.plan_retry_limit,
            "maximum_expression_chars": config.maximum_expression_chars,
            "maximum_ast_nodes": config.maximum_ast_nodes,
            "maximum_absolute_value": config.maximum_absolute_value,
            "maximum_plan_input_tokens": maximum_plan_input,
            "maximum_feedback_input_tokens": maximum_feedback_input,
        },
        "admission_gates": config.admission_gates,
        "decision_policy": {
            "forbidden_after_observation": [
                "eligibility_rule_change",
                "case_id_allowlist",
                "prompt_change",
                "regex_change",
                "AST_grammar_change",
                "grounding_rule_change",
                "retry_change",
                "budget_change",
                "temperature_change",
                "model_or_service_change",
                "fallback_change",
                "parser_or_scorer_change",
                "admission_gate_change",
                "candidate_rerun",
                "complete_benchmark_access_before_canary_pass",
                "independent_holdout_access",
            ],
            "passed": (
                "Publish the canary result, then separately pre-register the "
                "same GSM8K recovery route on the complete three-task suite."
            ),
            "failed": (
                "Preserve the result as negative evidence. Do not tune or rerun "
                "on the observed 211 rows."
            ),
        },
        "execution_boundary": config.execution_boundary,
        "claim_boundary": (
            "This is a pre-registration based on the already-public 211-case "
            "contract and frozen prior direct raw identities. It generates no "
            "new candidate canary output. A future pass is admission evidence "
            "only and is not a complete benchmark or holdout claim."
        ),
    }


def render_markdown(receipt: dict) -> str:
    surface = receipt["surface"]
    route = receipt["route"]
    gates = receipt["admission_gates"]
    return f"""# Qwen3.5 Grounded Calculator Canary v1

## 具体实验

这次只预注册，不生成新的 canary output。

冻结的 211 cases 包含 GSM8K 96、MMLU 96、GPQA-Diamond 19。4B direct
原结果是 163/211，只有 2 行无法解析；9B direct 是 151/211。

候选 harness 不重新生成 209 行：

- MMLU、GPQA 全部复用原 4B direct output；
- GSM8K 中已有 `prediction` 的行也复用原 output；
- 只有 `benchmark=gsm8k && status=completed && prediction=None` 才进入
  calculator recovery；
- 动态资格条件不读取 expected、score 或 case correctness，也没有 case-ID
  allowlist。

## Grounded Calculator

Recovery 让同一个 4B 输出一条 `CALC: <expression>`。安全执行器：

- 只允许 `+ - * /`、括号和一元正负号；
- 每个数字字面量都必须来自原题面，允许复用题面数字；
- 使用 `Fraction` 精确计算，不使用 `eval`；
- AST 最多 {route['maximum_ast_nodes']} nodes，表达式最多
  {route['maximum_expression_chars']} chars，结果必须是整数；
- 失败时原样回退 direct；
- 成功后把 verified result 回传同一个 4B，只允许 32-token `FINAL`。

Plan budget 是 {route['plan_max_tokens']}，one retry；最大实测 plan input
{route['maximum_plan_input_tokens']} tokens，低于 4096 context。

## 原 Gate 不变

- overall 至少 {gates['minimum_overall_correct']}/211；
- GSM8K 至少 {gates['benchmark_minimum_correct']['gsm8k']}/96；
- MMLU 至少 {gates['benchmark_minimum_correct']['mmlu']}/96；
- GPQA 至少 {gates['benchmark_minimum_correct']['gpqa_diamond']}/19；
- parse failures 最多 {gates['maximum_parse_failures']}，API errors 为 0；
- candidate-only wins 必须多于 base-only wins；
- 所有非 eligible 行的评分字段必须与 frozen direct 完全相同。

## 边界

- config SHA256：`{receipt['identity']['config_sha256']}`；
- case manifest SHA256：`{receipt['identity']['case_manifest_sha256']}`；
- V2 public report SHA256：`{receipt['identity']['v2_report_sha256']}`；
- eligible set 只记录 SHA256，不作为运行 allowlist：
  `{receipt['identity']['eligible_case_ids_sha256']}`；
- candidate generation：false；
- complete benchmark：关闭；
- independent holdout：密封；
- training eligible rows：0。

观察结果后禁止修改 eligibility、prompt、regex、AST grammar、grounding、
retry、budget、fallback、parser、scorer、gate 或重跑。
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
                "schema_version": receipt["schema_version"],
                "experiment_id": receipt["experiment_id"],
                "identity": receipt["identity"],
                "surface": receipt["surface"],
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
