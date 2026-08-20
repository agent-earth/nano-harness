#!/usr/bin/env python3

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from transformers import AutoTokenizer

from nano_harness.baseline import sha256_file
from nano_harness.verified_tool_execution import (
    FAMILIES,
    build_cases,
    contamination_audit,
    load_config,
    public_case_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/harness/qwen35_verified_tool_execution_v1.json"
JSON_OUTPUT = (
    ROOT
    / "docs/experiments/"
    "qwen35_verified_tool_execution_v1.preregister.json"
)
MARKDOWN_OUTPUT = (
    ROOT / "docs/experiments/qwen35_verified_tool_execution_v1.md"
)


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def _model_identity(
    model_path: str,
    config_sha256: str,
    index_sha256: str,
    shards: tuple[dict, ...],
) -> dict:
    path = Path(model_path)
    if (
        sha256_file(path / "config.json") != config_sha256
        or sha256_file(path / "model.safetensors.index.json")
        != index_sha256
    ):
        raise ValueError("verified tool model identity mismatch")
    verified = []
    for shard in shards:
        shard_path = path / shard["name"]
        if (
            shard_path.stat().st_size != shard["bytes"]
            or sha256_file(shard_path) != shard["sha256"]
        ):
            raise ValueError("verified tool model shard mismatch")
        verified.append({**shard, "verified": True})
    return {
        "model_config_sha256": config_sha256,
        "model_index_sha256": index_sha256,
        "weight_shards": verified,
    }


def _service_command(config, *, four_b: bool) -> list[str]:
    model_path = (
        config.four_b_model_path if four_b else config.nine_b_model_path
    )
    model = config.four_b_model if four_b else config.nine_b_model
    port = 8000 if four_b else 8001
    gpu = 0 if four_b else 1
    return [
        "env",
        f"CUDA_VISIBLE_DEVICES={gpu}",
        f"TRITON_LIBCUDA_PATH={config.triton_libcuda_path}",
        "../../.venv/bin/vllm",
        "serve",
        model_path,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--served-model-name",
        model,
        "--dtype",
        config.serving_dtype,
        "--max-model-len",
        str(config.max_model_len),
        "--gpu-memory-utilization",
        str(config.gpu_memory_utilization),
        "--enforce-eager",
        "--max-num-batched-tokens",
        str(config.max_num_batched_tokens),
        "--max-num-seqs",
        str(config.max_num_seqs),
    ]


def build_receipt() -> dict:
    config = load_config(CONFIG)
    cases = build_cases(config)
    contract = public_case_contract(cases)
    audit = contamination_audit(config, cases)
    if not audit["passed"]:
        raise ValueError("verified tool contamination detected")
    family_counts = {
        family: sum(row["family"] == family for row in cases)
        for family in FAMILIES
    }
    if set(family_counts.values()) != {config.cases_per_family}:
        raise ValueError("verified tool family counts differ")
    four_identity = _model_identity(
        config.four_b_model_path,
        config.four_b_model_config_sha256,
        config.four_b_model_index_sha256,
        config.four_b_weight_shards,
    )
    nine_identity = _model_identity(
        config.nine_b_model_path,
        config.nine_b_model_config_sha256,
        config.nine_b_model_index_sha256,
        config.nine_b_weight_shards,
    )
    if (
        sha256_file(
            Path(config.triton_libcuda_path) / "libcuda.so.1"
        )
        != config.triton_libcuda_sha256
    ):
        raise ValueError("verified tool libcuda identity mismatch")
    tokenizer = AutoTokenizer.from_pretrained(
        config.four_b_model_path,
        local_files_only=True,
    )
    direct_lengths = []
    plan_lengths = []
    final_feedback_lengths = []
    for row in cases:
        direct = tokenizer.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": (
                        "Solve the arithmetic task exactly. Return only one "
                        "line in the form FINAL: <integer>."
                    ),
                },
                {"role": "user", "content": row["prompt"]},
            ],
            tokenize=False,
            add_generation_prompt=True,
            **config.chat_template_kwargs,
        )
        plan = tokenizer.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": (
                        "Select the one typed arithmetic tool matching the task "
                        "and copy every labeled source fact exactly. Return only "
                        "the TOOL line required by the structured contract. Do "
                        "not calculate."
                    ),
                },
                {"role": "user", "content": row["prompt"]},
            ],
            tokenize=False,
            add_generation_prompt=True,
            **config.chat_template_kwargs,
        )
        feedback = tokenizer.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": (
                        "Use the verified tool result as authoritative and "
                        "return only one FINAL: <integer> line."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"<original_task>\n{row['prompt']}\n</original_task>\n"
                        "<verified_tool>\nname=placeholder\n"
                        "arguments=placeholder\nresult=-999999999999999999\n"
                        "</verified_tool>\nUse the verified result. Return only "
                        "FINAL: <integer>."
                    ),
                },
            ],
            tokenize=False,
            add_generation_prompt=True,
            **config.chat_template_kwargs,
        )
        direct_lengths.append(len(tokenizer.encode(direct)))
        plan_lengths.append(len(tokenizer.encode(plan)))
        final_feedback_lengths.append(len(tokenizer.encode(feedback)))
    context = {
        "direct_input_max": max(direct_lengths),
        "plan_input_max": max(plan_lengths),
        "final_feedback_input_max": max(final_feedback_lengths),
        "direct_input_plus_budget_max": (
            max(direct_lengths) + config.direct_max_tokens
        ),
        "plan_input_plus_budget_max": (
            max(plan_lengths) + config.plan_max_tokens
        ),
        "final_feedback_input_plus_budget_max": (
            max(final_feedback_lengths) + config.final_max_tokens
        ),
    }
    if max(context.values()) > config.max_model_len:
        raise ValueError("verified tool context exceeds serving limit")
    return {
        "schema_version": (
            "nano_harness_verified_tool_execution_preregister_v1"
        ),
        "experiment_id": config.experiment_id,
        "identity": {
            "code_revision": git_revision(),
            "config_sha256": sha256_file(CONFIG),
            "case_contract_sha256": contract["case_contract_sha256"],
            "four_b_model": four_identity,
            "nine_b_model": nine_identity,
            "triton_libcuda_sha256": config.triton_libcuda_sha256,
        },
        "case_contract": contract,
        "family_counts": family_counts,
        "contamination_audit": audit,
        "harness": {
            "stages": [
                "direct_four_b",
                "direct_nine_b",
                "four_b_plan",
                "strict_source_fact_validation",
                "safe_typed_execution",
                "four_b_verified_result_feedback",
            ],
            "typed_tools": list(FAMILIES),
            "plan_retry_limit": config.plan_retry_limit,
            "invalid_plan_policy": "reuse_four_b_direct",
            "executor_uses_expected_answer": False,
            "executor_uses_case_correctness": False,
            "final_feedback_contains": [
                "original_task",
                "verified_tool_name",
                "verified_arguments",
                "verified_result",
            ],
        },
        "inference": {
            "temperature": config.temperature,
            "chat_template_kwargs": config.chat_template_kwargs,
            "direct_max_tokens": config.direct_max_tokens,
            "plan_max_tokens": config.plan_max_tokens,
            "final_max_tokens": config.final_max_tokens,
            "direct_structured_output_regex": (
                config.direct_structured_output_regex
            ),
            "plan_structured_output_regex": (
                config.plan_structured_output_regex
            ),
            "context": context,
        },
        "serving": {
            "vllm_version": config.vllm_version,
            "dtype": config.serving_dtype,
            "max_model_len": config.max_model_len,
            "gpu_memory_utilization": config.gpu_memory_utilization,
            "enforce_eager": config.enforce_eager,
            "max_num_batched_tokens": config.max_num_batched_tokens,
            "max_num_seqs": config.max_num_seqs,
            "commands": {
                "qwen3.5-4b": _service_command(config, four_b=True),
                "qwen3.5-9b": _service_command(config, four_b=False),
            },
        },
        "acceptance": {
            "all_rows_complete_and_parseable": True,
            "harness_accuracy_gt_four_b_direct": True,
            "harness_vs_four_b_bootstrap_ci_lower_gt_zero": True,
            "harness_vs_four_b_mcnemar_p_lt": config.significance_alpha,
            "harness_vs_four_b_minimum_wins": config.minimum_harness_wins,
            "harness_vs_four_b_maximum_losses": config.maximum_harness_losses,
            "harness_accuracy_gt_nine_b_direct": True,
            "harness_vs_nine_b_bootstrap_ci_lower_gt_zero": True,
            "harness_vs_nine_b_mcnemar_p_lt": config.significance_alpha,
            "harness_vs_nine_b_minimum_wins": config.minimum_harness_wins,
            "harness_vs_nine_b_maximum_losses": (
                config.maximum_harness_losses
            ),
            "every_family_non_regression_vs_four_b_and_nine_b": True,
            "verified_execution_count_positive": True,
            "executor_contract_failures_zero": True,
            "canary_allowed_after_pass": True,
            "benchmark_allowed_after_pass": False,
        },
        "uncertainty": {
            "bootstrap_samples": config.bootstrap_samples,
            "bootstrap_seed": config.bootstrap_seed,
            "exact_mcnemar": True,
        },
        "decision_policy": {
            "forbidden_after_observation": [
                "case_change",
                "family_change",
                "prompt_change",
                "tool_schema_change",
                "source_fact_validation_change",
                "retry_change",
                "budget_change",
                "temperature_change",
                "parser_change",
                "model_change",
                "service_change",
                "threshold_change",
                "arm_rerun",
                "benchmark_access",
                "canary_access_before_pass",
                "holdout_access",
            ],
            "passed": (
                "Publish local matched harness evidence and consume only the "
                "exact frozen harness in the existing 211-case canary."
            ),
            "failed": (
                "Preserve negative evidence and change mechanism on a new "
                "surface; do not tune or rerun on this surface."
            ),
        },
        "execution_boundary": {
            "service_started": False,
            "model_generation_started": False,
            "evaluation_started": False,
            "benchmark_accessed": False,
            "canary_accessed": False,
            "training_started": False,
            "this_commit_only_preregisters": True,
        },
        "claim_boundary": (
            "Passing establishes only fresh synthetic harness admission and "
            "permits the frozen 211-case canary. It is not complete benchmark, "
            "holdout, or final 4B/9B superiority evidence."
        ),
    }


def render_markdown(receipt: dict) -> str:
    context = receipt["inference"]["context"]
    return f"""# Qwen3.5 Verified Tool-Execution Harness v1

## 目的

在全新 synthetic tasks 上测试真正的两阶段 agent loop，而不是题面 parser
override：

1. 4B 输出 typed `TOOL:` plan；
2. harness 严格校验 tool、字段、整数类型和 source facts；
3. safe executor 计算 verified result；
4. verified result 回填给同一 4B 输出 `FINAL:`；
5. invalid plan 最多重试一次，仍失败则回退原 4B direct。

## Fresh surface

- 4 families × 64 = 256 cases；
- case contract SHA：
  `{receipt['identity']['case_contract_sha256']}`；
- 与 prior choice matrix v1/v2/v3 prompt overlap：0；
- 与完整 GSM8K/MMLU/GPQA prompt overlap：0；
- benchmark/canary/holdout rows、outputs：0；
- executor 不读取 expected answer 或 correctness。

## Context

- direct input+budget max：`{context['direct_input_plus_budget_max']}`；
- plan input+budget max：`{context['plan_input_plus_budget_max']}`；
- result-feedback input+budget max：
  `{context['final_feedback_input_plus_budget_max']}`；
- serving max_model_len：4096。

## Matched arms

- 4B direct；
- 9B direct；
- 4B verified-tool harness。

所有模型 temperature=0、thinking=false；direct/final 都强制 `FINAL: <integer>`
regex，plan 强制 typed TOOL JSON regex。

## Gate

- all rows complete + parseable；
- harness vs 4B direct 和 harness vs 9B direct 都要求：
  accuracy 提升、CI lower > 0、McNemar p < 0.05、至少 12 wins、0 losses；
- every-family 对 4B/9B 均 non-regression；
- verified execution count > 0；
- executor contract failure = 0。

通过只允许进入已预注册 211-case canary；完整 benchmark 仍关闭。

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
                "family_counts": receipt["family_counts"],
                "contamination_audit": receipt["contamination_audit"],
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
