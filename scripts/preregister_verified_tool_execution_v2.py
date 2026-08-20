#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from transformers import AutoTokenizer

from nano_harness.baseline import sha256_file
from nano_harness.verified_tool_execution import (
    build_cases,
    contamination_audit,
    public_case_contract,
)
from nano_harness.verified_tool_execution_v2 import (
    SKILL_PROMPTS,
    TOOL_REGEX_BY_FAMILY,
    load_config,
    parent_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/harness/qwen35_verified_tool_execution_v2.json"
JSON_OUTPUT = (
    ROOT
    / "docs/experiments/qwen35_verified_tool_execution_v2.preregister.json"
)
MARKDOWN_OUTPUT = (
    ROOT / "docs/experiments/qwen35_verified_tool_execution_v2.md"
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
    cases = build_cases(parent)
    contract = public_case_contract(cases)
    parent_audit = contamination_audit(parent, cases)
    v1_raw = json.loads(
        (
            ROOT
            / "results/harness/qwen35-verified-tool-execution-v1/result.json"
        ).read_text(encoding="utf-8")
    )
    normalize = lambda value: " ".join(str(value).casefold().split())
    v2_hashes = {
        hashlib.sha256(normalize(row["prompt"]).encode()).hexdigest()
        for row in cases
    }
    v1_hashes = {
        hashlib.sha256(normalize(row["prompt"]).encode()).hexdigest()
        for row in build_cases(
            __import__(
                "nano_harness.verified_tool_execution",
                fromlist=["load_config"],
            ).load_config(config.parent_config_path)
        )
    }
    if (
        v2_hashes & v1_hashes
        or not parent_audit["passed"]
        or v1_raw["experiment_id"] != "qwen35-verified-tool-execution-v1"
    ):
        raise ValueError("verified tool v2 contamination differs")
    tokenizer = AutoTokenizer.from_pretrained(
        parent.four_b_model_path,
        local_files_only=True,
    )
    plan_lengths = {}
    for family in TOOL_REGEX_BY_FAMILY:
        selected = [row for row in cases if row["family"] == family]
        lengths = []
        for row in selected:
            text = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SKILL_PROMPTS[family]},
                    {"role": "user", "content": row["prompt"]},
                ],
                tokenize=False,
                add_generation_prompt=True,
                **parent.chat_template_kwargs,
            )
            lengths.append(len(tokenizer.encode(text)))
        plan_lengths[family] = max(lengths)
    if max(plan_lengths.values()) + parent.plan_max_tokens > parent.max_model_len:
        raise ValueError("verified tool v2 plan context differs")
    return {
        "schema_version": (
            "nano_harness_verified_tool_execution_preregister_v2"
        ),
        "experiment_id": config.experiment_id,
        "identity": {
            "code_revision": git_revision(),
            "config_sha256": sha256_file(CONFIG),
            "parent_config_sha256": config.parent_config_sha256,
            "prior_v1_report_sha256": config.prior_v1_report_sha256,
            "service_receipt_sha256": config.service_receipt_sha256,
            "case_contract_sha256": contract["case_contract_sha256"],
        },
        "case_contract": contract,
        "freshness": {
            "v1_prompt_overlap": 0,
            "prior_choice_prompt_overlap": parent_audit[
                "prior_surface_prompt_overlap"
            ],
            "benchmark_prompt_overlap": parent_audit[
                "benchmark_prompt_overlap"
            ],
            "benchmark_rows_hashed": parent_audit["benchmark_rows_hashed"],
        },
        "mechanism_delta": {
            "parent_v1_result": {
                "four_b_direct_correct": v1_raw["arms"]["four_b_direct"][
                    "correct"
                ],
                "nine_b_direct_correct": v1_raw["arms"]["nine_b_direct"][
                    "correct"
                ],
                "harness_correct": v1_raw["arms"]["four_b_verified_tool"][
                    "correct"
                ],
                "verified_executions": v1_raw["routing"][
                    "verified_executions"
                ],
                "fallbacks": v1_raw["routing"]["fallbacks"],
            },
            "retained": [
                "typed executor",
                "strict source-fact validator",
                "one retry",
                "verified-result feedback",
                "direct fallback",
                "models and services",
                "temperature and budgets",
                "scorer and statistical gates",
            ],
            "changed": [
                "fresh value_offset 90000",
                "case-family skill route",
                "one applicable tool regex exposed per case",
            ],
            "stopped": ["all four tool schemas exposed at once"],
        },
        "skill_router": {
            "name": config.skill_router,
            "routes": {
                family: {
                    "skill_id": family,
                    "exposed_tools": [family],
                    "plan_regex": TOOL_REGEX_BY_FAMILY[family],
                    "system_prompt_sha256": hashlib.sha256(
                        SKILL_PROMPTS[family].encode()
                    ).hexdigest(),
                    "plan_input_max": plan_lengths[family],
                }
                for family in TOOL_REGEX_BY_FAMILY
            },
            "uses_expected_answer": False,
            "uses_case_correctness": False,
        },
        "inherited_contract": {
            "four_b_model": parent.four_b_model,
            "nine_b_model": parent.nine_b_model,
            "service_receipt_path": parent.service_receipt_path,
            "temperature": parent.temperature,
            "chat_template_kwargs": parent.chat_template_kwargs,
            "direct_max_tokens": parent.direct_max_tokens,
            "plan_max_tokens": parent.plan_max_tokens,
            "final_max_tokens": parent.final_max_tokens,
            "plan_retry_limit": parent.plan_retry_limit,
            "direct_structured_output_regex": (
                parent.direct_structured_output_regex
            ),
            "bootstrap_samples": parent.bootstrap_samples,
            "bootstrap_seed": parent.bootstrap_seed,
            "significance_alpha": parent.significance_alpha,
            "minimum_harness_wins": parent.minimum_harness_wins,
            "maximum_harness_losses": parent.maximum_harness_losses,
        },
        "acceptance": {
            "all_rows_complete_and_parseable": True,
            "skill_routes_256": True,
            "single_tool_exposures_256": True,
            "verified_executions_256": True,
            "executor_contract_failures_zero": True,
            "harness_vs_four_b_significant": True,
            "harness_vs_four_b_minimum_wins": parent.minimum_harness_wins,
            "harness_vs_four_b_maximum_losses": parent.maximum_harness_losses,
            "harness_vs_nine_b_significant": True,
            "harness_vs_nine_b_minimum_wins": parent.minimum_harness_wins,
            "harness_vs_nine_b_maximum_losses": parent.maximum_harness_losses,
            "every_family_non_regression_vs_four_b_and_nine_b": True,
            "canary_allowed_after_pass": True,
            "benchmark_allowed_after_pass": False,
        },
        "decision_policy": {
            "forbidden_after_observation": [
                "case_change",
                "skill_route_change",
                "tool_regex_change",
                "prompt_change",
                "validator_change",
                "retry_change",
                "budget_change",
                "temperature_change",
                "model_change",
                "service_change",
                "gate_change",
                "arm_rerun",
                "benchmark_access",
                "canary_access_before_pass",
                "holdout_access",
            ],
            "passed": (
                "Publish v2 local evidence and separately pre-register the "
                "exact skill-routed harness for the 211-case canary."
            ),
            "failed": (
                "Preserve evidence and change mechanism on a new surface; do "
                "not tune or rerun on this surface."
            ),
        },
        "execution_boundary": config.execution_boundary,
        "claim_boundary": (
            "Passing establishes only fresh synthetic local harness admission "
            "and permits a separately pre-registered 211-case canary. It is "
            "not complete benchmark or final superiority evidence."
        ),
    }


def render_markdown(receipt: dict) -> str:
    return f"""# Qwen3.5 Skill-Routed Verified Tool Execution v2

## 唯一变化

V1 在 fresh 256 cases 上 192/256，显著超过 4B direct 21/256 和 9B direct
13/256，但 64 个 labor cases 因看到全部四种 tool schema 而选错 tool。

V2 只做两项预注册变化：

- fresh value offset：90000；
- case-family skill router 每次只暴露该 family 的一个 typed tool schema。

Typed executor、source-fact validator、one retry、result feedback、direct fallback、
models、services、budgets、temperature、scorer 和全部统计 gate 保持不变。

## Freshness

- 256 cases，4 families × 64；
- case contract SHA：
  `{receipt['identity']['case_contract_sha256']}`；
- 与 V1 prompts overlap：0；
- 与 prior choice v1/v2/v3 prompts overlap：0；
- 与完整 GSM8K/MMLU/GPQA prompts overlap：0。

## Gate

- 256/256 skill routes；
- 256/256 single-tool exposures；
- 256/256 verified executions；
- 0 executor contract failures；
- harness vs 4B direct 和 vs 9B direct 均显著、至少 12 wins、0 losses；
- every-family non-regression；
- 通过只允许另行预注册 211-case canary，完整 benchmark 仍关闭。

## 执行边界

```json
{json.dumps(receipt['execution_boundary'], indent=2, sort_keys=True)}
```
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
                "skill_router": receipt["skill_router"],
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
