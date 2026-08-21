#!/usr/bin/env python3

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from nano_harness.baseline import sha256_file


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/campaign/ultimate_distill_final_report_v1.json"
PUBLIC = ROOT / "docs/results/ultimate_distill_final_report_v1.public.json"
MARKDOWN = ROOT / "docs/results/ultimate_distill_final_report_v1.md"


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if (
        config.get("schema_version")
        != "nano_harness_ultimate_distill_final_report_v1"
        or config.get("claim_boundary", {}).get("single_checkpoint_claim")
        is not False
        or config.get("claim_boundary", {}).get(
            "verified_tool_is_public_benchmark"
        )
        is not False
        or config.get("claim_boundary", {}).get(
            "sft_plus_rl_result_exists"
        )
        is not False
    ):
        raise ValueError("final report claim boundary differs")
    for repository in config["repositories"].values():
        revision = repository["revision"]
        if len(revision) != 40 or any(
            character not in "0123456789abcdef" for character in revision
        ):
            raise ValueError("invalid source repository revision")
    for evidence in (
        config["external_evidence"]["data"]
        + config["external_evidence"]["training"]
    ):
        source_sha256 = evidence["source_sha256"]
        if len(source_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in source_sha256
        ):
            raise ValueError(f"invalid external evidence SHA: {evidence['id']}")
    return config


def load_local_sources(
    config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    sources = {}
    for name, source in config["local_sources"].items():
        source_path = ROOT / source["path"]
        if not source_path.is_file():
            raise ValueError(f"missing local source: {name}")
        actual_sha256 = sha256_file(source_path)
        if actual_sha256 != source["sha256"]:
            raise ValueError(
                f"local source identity differs: {name}: {actual_sha256}"
            )
        sources[name] = json.loads(source_path.read_text(encoding="utf-8"))
    return sources


def compact_comparison(comparison: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "cases",
        "candidate_correct",
        "candidate_accuracy",
        "baseline_correct",
        "baseline_accuracy",
        "delta",
        "paired_bootstrap_95_ci",
        "mcnemar_exact_p",
        "paired_counts",
        "candidate_parse_failures",
        "baseline_parse_failures",
    )
    return {key: comparison[key] for key in keys if key in comparison}


def benchmark_rows(
    synthesis: dict[str, Any],
) -> list[dict[str, Any]]:
    ordered = ("mmlu", "gpqa_diamond", "gsm8k", "mbpp")
    rows = []
    for benchmark in ordered:
        source = synthesis["benchmarks"][benchmark]
        comparison = source["comparison_vs_nine_b"]
        holm = next(
            item
            for item in synthesis["holm_bonferroni"]["ordered_tests"]
            if item["benchmark"] == benchmark
        )
        rows.append(
            {
                "benchmark": benchmark,
                "route": source["route"],
                **compact_comparison(comparison),
                "holm_threshold": holm["threshold"],
                "holm_rejected": holm["rejected"],
                "won": source["won"],
            }
        )
    return rows


def build_report() -> dict[str, Any]:
    config = load_config()
    sources = load_local_sources(config)
    synthesis = sources["three_complete_benchmarks"]
    direct = sources["complete_direct_baseline"]
    v5 = sources["complete_v5_ablation"]
    conditional = sources["complete_conditional_majority"]
    mbpp_development = sources["mbpp_development"]
    mbpp_confirmation = sources["mbpp_confirmation"]
    mbpp_replication = sources["mbpp_replication"]
    mbpp_test = sources["mbpp_complete_test"]
    mbpp_27b = sources["mbpp_27b"]
    tool_27b = sources["verified_tool_27b"]
    serving_27b = sources["twenty_seven_b_serving"]

    rows = benchmark_rows(synthesis)
    public_wins = [row["benchmark"] for row in rows if row["won"]]
    public_losses = [row["benchmark"] for row in rows if not row["won"]]

    return {
        "schema_version": "nano_harness_ultimate_distill_final_public_v1",
        "experiment_id": config["experiment_id"],
        "generated_on": config["generated_on"],
        "identity": {
            "render_revision": git_revision(),
            "config_sha256": sha256_file(CONFIG),
            "local_source_sha256": {
                name: source["sha256"]
                for name, source in config["local_sources"].items()
            },
            "repositories": config["repositories"],
        },
        "executive_summary": {
            "goal_met": True,
            "public_benchmarks_won_vs_matched_nine_b": len(public_wins),
            "public_benchmark_wins": public_wins,
            "public_benchmark_losses": public_losses,
            "twenty_seven_b_public_benchmark_parity_met": False,
            "twenty_seven_b_local_capability_suite_exceeded": True,
            "winning_layer": "harness_routing_and_verification",
            "training_quality_gain_established": False,
            "plain_language": (
                "同一个 Qwen3.5-4B 底座按题库走预先冻结的不同答题流程，"
                "在 MMLU、GPQA-Diamond、MBPP 三个完整公开题库上显著超过"
                "同条件 Qwen3.5-9B。GSM8K 仍明显落后。训练数据已经建成，"
                "但完成的 SFT、DPO、RL 和 OPD 尚未证明模型质量稳定提升。"
            ),
        },
        "definitions": {
            "matched_direct": (
                "4B 和 9B 回答完全相同的题目，使用相同提示词、输出长度、"
                "解码与评分器；区别只有模型大小。"
            ),
            "harness": (
                "不改模型权重，只改变答题流程，例如多次采样、保守投票、"
                "执行公开测试、失败后修复，以及不满足条件时退回原答案。"
            ),
            "verified": (
                "答案交给目标不可见的确定性代码或公开测试执行，只有检查"
                "通过才允许覆盖模型原答案。它是 harness 的一个组成部分，"
                "不是另一个训练模型。"
            ),
            "complete_benchmark": (
                "按预注册规则评完该评测 split 的全部样本，不是抽样、scan、"
                "import 检查或 dry-run。"
            ),
            "significant_win": (
                "4B 分数更高、配对 bootstrap 95% 置信区间下界大于 0、"
                "4B 独赢题多于 9B 独赢题，并通过四项完整 benchmark 的 "
                "Holm-Bonferroni 多重检验。"
            ),
        },
        "public_benchmarks": {
            "candidate": synthesis["candidate"],
            "comparisons_vs_nine_b": rows,
            "holm_bonferroni": synthesis["holm_bonferroni"],
            "decision": synthesis["decision"],
            "matched_direct_baseline": {
                "cases": direct["comparison"]["cases"],
                "four_b_correct": direct["comparison"]["overall_micro"][
                    "candidate_correct"
                ],
                "nine_b_correct": direct["comparison"]["overall_micro"][
                    "baseline_correct"
                ],
                "four_b_minus_nine_b": direct["comparison"]["overall_micro"][
                    "delta"
                ],
                "complete_benchmarks_significantly_won": direct["decision"][
                    "complete_benchmarks_significantly_won"
                ],
            },
            "route_ablation": {
                "direct_only": {
                    "complete_benchmarks_significantly_won": direct[
                        "decision"
                    ]["complete_benchmarks_significantly_won"],
                    "result": "one_of_three",
                },
                "mmlu_preserve_direct": {
                    "candidate_correct": 10273,
                    "nine_b_correct": 9066,
                    "effect": "kept the already stronger 4B direct answer",
                },
                "gpqa_conservative_choice_consensus": {
                    "direct_four_b_correct": direct["comparison"][
                        "benchmarks"
                    ]["gpqa_diamond"]["candidate_correct"],
                    "harness_correct": conditional["comparisons"][
                        "versus_nine_b"
                    ]["gpqa_diamond"]["candidate_correct"],
                    "nine_b_correct": conditional["comparisons"][
                        "versus_nine_b"
                    ]["gpqa_diamond"]["baseline_correct"],
                    "effect_vs_direct_four_b_cases": 9,
                },
                "gsm8k_conditional_majority": {
                    "direct_four_b_correct": conditional["comparisons"][
                        "versus_four_b"
                    ]["gsm8k"]["baseline_correct"],
                    "harness_correct": conditional["comparisons"][
                        "versus_four_b"
                    ]["gsm8k"]["candidate_correct"],
                    "effect_vs_direct_four_b_cases": 16,
                    "nine_b_correct": conditional["comparisons"][
                        "versus_nine_b"
                    ]["gsm8k"]["baseline_correct"],
                    "decision": "improved_4b_but_still_lost_to_9b",
                },
                "prior_three_task_harness": {
                    "complete_benchmarks_significantly_won": v5["decision"][
                        "complete_benchmarks_significantly_won"
                    ],
                    "decision": "rejected_due_to_gsm8k_regression",
                },
            },
        },
        "mbpp_evidence_chain": [
            {
                "stage": "development",
                "split": mbpp_development["boundary"]["split"],
                "cases": 120,
                "four_b_harness_correct": 108,
                "nine_b_correct": 97,
                "delta": 0.09166666666666666,
                "status": "method_supported",
                "benchmark_score": False,
            },
            {
                "stage": "fresh_confirmation",
                "split": mbpp_confirmation["boundary"]["split"],
                "cases": 47,
                "four_b_harness_correct": 36,
                "nine_b_correct": 31,
                "delta": 0.10638297872340426,
                "mcnemar_exact_p": 0.1796875,
                "status": "directional_but_not_significant",
                "benchmark_score": False,
            },
            {
                "stage": "independent_exact_replication",
                "split": mbpp_replication["boundary"]["split"],
                "cases": 254,
                "four_b_harness_correct": 211,
                "nine_b_correct": 187,
                "delta": 0.09448818897637795,
                "mcnemar_exact_p": 8.430331945419312e-06,
                "status": "replicated",
                "benchmark_score": False,
            },
            {
                "stage": "complete_sanitized_test",
                "split": mbpp_test["boundary"]["split"],
                "cases": 257,
                "four_b_harness_correct": 219,
                "four_b_direct_correct": 189,
                "nine_b_correct": 198,
                "delta_vs_nine_b": 0.08171206225680934,
                "mcnemar_exact_p": 0.0005082604475319386,
                "replicas_generated": 340,
                "repairs_generated": 116,
                "public_test_improving_overrides": 38,
                "status": "complete_benchmark_win",
                "benchmark_score": True,
            },
        ],
        "twenty_seven_b": {
            "serving": {
                "download": serving_27b["download"],
                "service": serving_27b["service"],
                "smoke": {
                    "passed": serving_27b["smoke"]["passed"],
                    "passed_probes": serving_27b["smoke"]["passed_probes"],
                    "total_probes": serving_27b["smoke"]["total_probes"],
                    "probe_ids": [
                        item["id"] for item in serving_27b["smoke"]["results"]
                    ],
                },
                "rejected_gptq": {
                    key: value
                    for key, value in serving_27b["rejected_gptq"].items()
                    if key != "observed_outputs"
                },
            },
            "mbpp_complete": {
                **compact_comparison(mbpp_27b["comparison"]),
                "noninferiority": mbpp_27b["noninferiority"],
                "decision": "parity_rejected",
            },
            "verified_tool_complete_local_suite": {
                **compact_comparison(tool_27b["comparison"]["overall"]),
                "candidate_correct": round(
                    tool_27b["comparison"]["overall"]["candidate_accuracy"]
                    * tool_27b["comparison"]["overall"]["cases"]
                ),
                "baseline_correct": round(
                    tool_27b["comparison"]["overall"]["baseline_accuracy"]
                    * tool_27b["comparison"]["overall"]["cases"]
                ),
                "families": {
                    family: compact_comparison(comparison)
                    for family, comparison in tool_27b["comparison"][
                        "by_family"
                    ].items()
                },
                "noninferiority": tool_27b["noninferiority"],
                "decision": "four_b_harness_significantly_exceeds_27b",
                "claim_scope": tool_27b["boundary"]["claim_scope"],
            },
        },
        "data_pipeline": config["external_evidence"]["data"],
        "training_ablation": config["external_evidence"]["training"],
        "ablation_contract": config["ablation_contract"],
        "agent_benchmark_feasibility": config[
            "agent_benchmark_feasibility"
        ],
        "conclusions": {
            "proved": [
                (
                    "Frozen benchmark-specific harness routes make one "
                    "Qwen3.5-4B base model significantly outperform matched "
                    "Qwen3.5-9B on complete MMLU, GPQA-Diamond, and MBPP."
                ),
                (
                    "Target-blind verification can produce a very large "
                    "advantage over 27B on a bounded exact capability suite."
                ),
                (
                    "The data pipeline can produce more than 10M checked "
                    "training tokens with provenance, deduplication, split, "
                    "and leakage controls."
                ),
                (
                    "The current training methods are executable and "
                    "reproducible, but none supports attribution of the "
                    "public benchmark wins to changed model weights."
                ),
            ],
            "not_proved": [
                "One fine-tuned 4B checkpoint beats 9B on three benchmarks.",
                "The 4B harness matches or exceeds 27B on complete MBPP.",
                "SFT, DPO, RL, OPD, or SFT+RL produced a stable benchmark gain.",
                "Any official agent benchmark score was obtained on this devbox.",
                "The current system beats 27B broadly outside the named local suite.",
            ],
        },
        "reproducibility": {
            "render_command": (
                "PYTHONPATH=. ../../.venv/bin/python "
                "scripts/render_ultimate_distill_final_report_v1.py"
            ),
            "test_command": (
                "PYTHONPATH=. ../../.venv/bin/python -m unittest discover "
                "-s tests -v"
            ),
            "raw_outputs_policy": "local_and_ignored",
            "public_reports_only": True,
        },
        "claim_boundary": config["claim_boundary"],
    }


def percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def render_markdown(report: dict[str, Any]) -> str:
    benchmark_lines = []
    for item in report["public_benchmarks"]["comparisons_vs_nine_b"]:
        benchmark_lines.append(
            f"| {item['benchmark']} | {item['route']} | "
            f"{item['candidate_correct']}/{item['cases']} "
            f"({percent(item['candidate_accuracy'])}) | "
            f"{item['baseline_correct']}/{item['cases']} "
            f"({percent(item['baseline_accuracy'])}) | "
            f"{item['delta']:+.4f} | "
            f"[{item['paired_bootstrap_95_ci'][0]:+.4f}, "
            f"{item['paired_bootstrap_95_ci'][1]:+.4f}] | "
            f"{item['mcnemar_exact_p']:.6g} | "
            f"{'通过' if item['won'] else '失败'} |"
        )

    mbpp_lines = []
    for item in report["mbpp_evidence_chain"]:
        score = (
            f"{item['four_b_harness_correct']}/{item['cases']} vs "
            f"{item['nine_b_correct']}/{item['cases']}"
        )
        delta = (
            item["delta_vs_nine_b"]
            if "delta_vs_nine_b" in item
            else item["delta"]
        )
        mbpp_lines.append(
            f"| {item['stage']} | {item['split']} | {item['cases']} | "
            f"{score} | {delta:+.4f} | "
            f"{item['status']} |"
        )

    data_lines = [
        (
            f"| {item['id']} | {item['train_rows']:,} | "
            f"{item['dev_rows']:,} | "
            f"{item['train_tokens']:,} | "
            f"{'通过' if item['all_release_checks_passed'] else '失败'} |"
        )
        for item in report["data_pipeline"]
        if "train_tokens" in item
    ]
    data_lines.insert(
        2,
        (
            "| orca-math-preference-v1 | 512 | 192 | — | 通过 |"
        ),
    )

    training_rows = []
    training_labels = {
        "qwen35-scaled-quality-sft-v1": "专项算术 SFT",
        "skill-release-long-sequence-sft-smoke-v1": "10M 数据运行 smoke",
        "skill-release-bounded-dose-sft-v2": "10M 数据小剂量 SFT",
        "skill-release-reasoning-preservation-sft-v4": "推理样本加权 SFT",
        "orca-math-sft-smoke-v1": "Orca Math SFT",
        "orca-math-verifier-dpo-v1": "整段 DPO",
        "orca-math-verifier-dpo-suffix-v2": "只训 FINAL 后缀的 DPO",
        "execution-target-paired-consistency-v1": "过程到答案一致性训练",
        "qwen35-rl-opd-admission-v1": "RL / OPD 实现 smoke",
        "qwen35-synthetic-quality-v1": "RL / OPD 质量检查",
        "qwen35-router-classification-sft-smoke-v1": "三分类 router SFT",
        "qwen35-router-negative-diversity-sft-v2": "八类负例 router SFT",
    }
    for item in report["training_ablation"]:
        before = item.get(
            "before_correct", item.get("before_verified", item.get("base_correct"))
        )
        after = item.get(
            "after_correct",
            item.get(
                "after_verified",
                (
                    f"RL {item.get('rl_correct')} / OPD "
                    f"{item.get('opd_correct')}"
                    if "rl_correct" in item
                    else None
                ),
            ),
        )
        if item["id"] == "qwen35-rl-opd-admission-v1":
            outcome = "实现前未运行 → 两种方法均完成 2-step smoke"
        elif item["id"] == "qwen35-synthetic-quality-v1":
            outcome = "Base 3 → RL 3 / OPD 3"
        else:
            outcome = f"{before} → {after}"
        used = (
            item.get("train_rows_used")
            or item.get("unique_train_rows_used")
            or item.get("train_pairs_used")
            or item.get("trajectory_rows_per_method")
            or "见配置"
        )
        training_rows.append(
            f"| {training_labels[item['id']]} | {used} | "
            f"{item.get('optimizer_steps', item.get('optimizer_steps_per_method', '—'))} | "
            f"{item.get('evaluation_cases', '—')} | {outcome} | "
            f"{item['decision']} |"
        )

    verified_tool = report["twenty_seven_b"][
        "verified_tool_complete_local_suite"
    ]
    mbpp_27b = report["twenty_seven_b"]["mbpp_complete"]
    direct = report["public_benchmarks"]["matched_direct_baseline"]
    return f"""# Ultimate Distill 最终全栈实验报告 v1

生成日期：{report['generated_on']}

## 一句话结论

同一个 Qwen3.5-4B 底座，按 benchmark 走预先冻结的不同答题流程后，
在 **MMLU、GPQA-Diamond、MBPP 三个完整公开 benchmark** 上显著超过
同条件 Qwen3.5-9B。GSM8K 明显落后，必须保留为负结果。对 Qwen3.5-27B，
4B 在完整 MBPP 上没有达到预设的 -2 个百分点非劣门槛；在另一个 256 题
本地精确工具能力集上，4B harness 为 256/256，27B 直答为 63/256。

这不是“一个训练后的 4B checkpoint 全面超过 9B/27B”。公开 benchmark
的提升来自 **harness 路由和验证**，不是 SFT、DPO、RL 或 OPD。

## 先把几个词说清楚

- **同条件直接回答（matched direct）**：4B 和 9B 回答同一批题，提示词、
  输出长度、解码与评分器相同，只换模型。
- **Harness**：不改模型权重，只改变答题步骤。例如多次采样、保守投票、
  执行公开测试、失败后修复，以及不满足条件时退回原答案。
- **Verified**：让目标不可见的确定性代码或公开测试检查候选答案。只有检查
  通过才允许覆盖原答案。它是 harness 的一个迭代组件，不是训练方法。
- **完整 benchmark**：该 split 的全部样本都按预注册规则跑完。scan、
  import check、dry-run 和小样本开发集都不算正式分数。

## 完整公开 Benchmark

| Benchmark | 4B 使用的冻结流程 | 4B Harness | 9B 直接回答 | 分数差 | 95% CI | McNemar p | 结论 |
| --- | --- | ---: | ---: | ---: | --- | ---: | --- |
{chr(10).join(benchmark_lines)}

四项完整 benchmark 被放在同一个 Holm-Bonferroni 检验族中，familywise
alpha 为 0.05。MMLU、MBPP、GSM8K、GPQA-Diamond 的阈值依次为
0.0125、0.016667、0.025、0.05。四项差异都显著，但方向是三胜一负：
MMLU、GPQA-Diamond、MBPP 胜，GSM8K 负。

### 提升具体来自哪里

- 完整三任务直接回答基线共有 {direct['cases']:,} 题：4B
  {direct['four_b_correct']:,} 题正确，9B {direct['nine_b_correct']:,} 题
  正确。直接回答只能在一个 benchmark 上形成显著胜利。
- **MMLU** 没增加模型调用，保留 4B 直接答案。4B 为 10,273/14,042，
  9B 为 9,066/14,042。9B 有 3,322 个严格格式解析失败，所以这里证明的是
  当前冻结输出合同下的完整任务表现，不应外推成裸模型知识能力差距。
- **GPQA-Diamond** 先保留 4B 的直接答案，再独立检查四个选项；只有两个
  独立检查和确认步骤同意同一个非原选项时才覆盖。4B 从 76/198 提高到
  85/198，9B 为 69/198。
- **GSM8K** 只在触发条件满足时做 5 次采样和多数判断，否则退回 4B 原答案。
  它把 4B 从 1,204/1,319 提到 1,220/1,319，但仍低于 9B 的
  1,243/1,319，所以不能算胜利。
- **MBPP** 先让 4B 写代码并运行公开测试。直接答案失败时再生成 5 个候选，
  按公开测试通过数选最好的一份；仍未全过才最多修复 3 轮。完整 test 共
  生成 340 个候选和 116 次修复，产生 38 次严格改善且没有使用隐藏参考解。

## MBPP 为什么可信

| 阶段 | 数据 split | 题数 | 4B Harness vs 9B | 分数差 | 结论 |
| --- | --- | ---: | ---: | ---: | --- |
{chr(10).join(mbpp_lines)}

47 题确认集的方向是正的，但统计不显著，所以当时明确冻结为负证据，没有
直接打开 test。随后在另一个 254 题集合上用完全相同策略复现，达到显著性
门槛，才一次性运行 257 题 sanitized test。最终 4B harness 为 219/257，
9B 为 198/257；4B 直接回答只有 189/257。

## 与 27B 的比较

### 完整 MBPP：没有追平

4B harness 为 {mbpp_27b['candidate_correct']}/257，27B 直接回答为
{mbpp_27b['baseline_correct']}/257，差
{percent(mbpp_27b['delta'])}，95% CI
[{percent(mbpp_27b['paired_bootstrap_95_ci'][0])},
{percent(mbpp_27b['paired_bootstrap_95_ci'][1])}]。预注册要求置信区间下界
不低于 -2%，实际下界是
{percent(mbpp_27b['paired_bootstrap_95_ci'][0])}，因此判定失败。

### 256 题本地精确工具能力集：明显超过

这套题只有四类可由确定性代码精确求解的问题。4B harness 为
{verified_tool['candidate_correct']}/256，27B 直接回答为
{verified_tool['baseline_correct']}/256，差
{percent(verified_tool['delta'])}，95% CI
[{percent(verified_tool['paired_bootstrap_95_ci'][0])},
{percent(verified_tool['paired_bootstrap_95_ci'][1])}]。四个题型都通过 -2%
非劣门槛，整体显著超过 27B。

边界必须保留：这是完整的本地合成能力集，不是公开 benchmark，不能用它
宣称“4B 全面超过 27B”。

### 27B 服务验证

- 模型通过 `BUCKET=ai-infra oniond download model Qwen3.5-27B` 下载。
- vLLM 0.19.1，BF16/FP16 权重，2 张 V100，tensor parallel 2，
  4,096 context，1 GiB 显式 KV cache，3/3 确定性 smoke 通过。
- GPTQ-Int4 在 V100 上输出连续 `!`，vLLM 也警告 4-bit `gptq_gemm`
  有数值问题；Marlin 不支持 compute capability 7.0。该量化服务被排除，
  没有拿来评分。

## 数据生成：生成了多少，实际训练了多少

| 数据版本 | Train rows | Dev rows | Train tokens | 数据门禁 |
| --- | ---: | ---: | ---: | --- |
{chr(10).join(data_lines)}

最大的 skill 数据池确实达到了 **15,888 条 train / 11,425,166 train
tokens**。它由 55 个并行 shard 生成，每个 shard 都有 generator 和 critic
调用；全局去重删除了 11,872 条跨 shard 重复。

但这不等于已经用 11.43M tokens 完整训练一次。已完成的对应训练是：

- 长序列运行 smoke：从数据池取 10 条训练样本，4 steps，4/5 → 4/5；
- bounded-dose SFT：取 80 条，20 steps，17/20 → 17/20；
- reasoning-preservation SFT：取 80 条，20 steps，16/20 → 16/20。

三次都证明训练、保存、重载链路可运行，但没有证明质量提升。完整 10M-token
训练没有启动，因为小规模门禁没有通过，继续扩大只会放大成本和风险。

Orca Math 数据池有 32,768 条 train、1,024 条 dev、12,820,576 train
tokens，并通过与固定 GSM8K/MMLU/GPQA 语料零精确/近似重叠检查。实际
Orca SFT smoke 只使用 160 条训练样本、40 steps；结果从 100/192 降到
58/192，因此停止。

## 训练消融

| 实验 | 实际训练样本/对 | Steps | 评测题数 | 前 → 后 | 决策 |
| --- | ---: | ---: | ---: | --- | --- |
{chr(10).join(training_rows)}

结论很直接：

- 标准 SFT 有过明显回归，也有 0/96 → 2/96 的微小正向结果，但后者
  McNemar p=0.5，不显著。
- 两个 DPO 实验分别是 91/192 → 91/192、83/192 → 83/192。第二个目标
  更聚焦，改了 4 个输出，但没有改对任何一题。
- RL 和 OPD 各跑了 2 个 optimizer step，证明实现能训练、adapter 能重载、
  logits 会变化；随后在 96 题上两者都是 3/96，与 base 相同。
- 一致性目标在 80 题上把 verifier 分数从 51 提到 55、exact 从 32 提到
  33，首次修好一个 process-to-final pair；p=0.21875，只能算方向性机制
  证据。
- Router SFT 在本地合成分类上从 112/192 提到 192/192，扩展负例版本从
  1,127/1,536 提到 1,536/1,536。这证明 router 行为可学，但不等于真实
  benchmark 提升，且后者还要求 serving namespace remap。
- `SFT+RL` 没有运行。原因是没有 SFT 或 RL 候选先通过质量门禁，组合两个
  未通过的组件没有可辩护的实验依据。

## Agent Benchmark 可跑性

SkillBench、ClawBench、WildClawBench、Terminal-Bench 2、SWE-bench
Lite 的 5/5 本地检查通过：包括真实 skill scan、319/319 ClawBench
定义、60/60 WildClawBench 任务、89/89 Terminal-Bench 2 manifest，以及
300 行 SWE-bench Lite parquet。

这些都是可运行性检查，不是分数。本机缺少官方执行需要的 container mount
namespace 权限；Docker、rootless Podman 和 Buildah 路径都在运行测试前
失败。因此本报告的 agent benchmark 正式得分仍为 0 项。

## 最终消融判断

| 层 | 结果 |
| --- | --- |
| 4B direct baseline | 完整三任务只显著赢 1 项 |
| Harness-only | 完整四项中赢 3 项；这是主结果 |
| Data-only | 规模、去重、来源和泄漏检查通过；不代表模型提升 |
| SFT | 局部正向、无变化和显著回归都出现；无公开 benchmark 增益 |
| DPO | 两次均无正确率变化 |
| RL | 实现通过，96 题质量无提升 |
| OPD | 实现通过，96 题质量无提升 |
| SFT+RL | 未执行；前置组件未通过质量门禁 |
| Verifier / tool | 在 MBPP 和 256 题精确工具集上有效 |

因此，当前能复现的三项公开 benchmark 提升应归因于 **harness 路由与
验证**，不能归因于训练。

## 证明了什么，没证明什么

**已经证明**

1. 一个 Qwen3.5-4B 底座配合预先冻结的 benchmark-specific harness，
   在完整 MMLU、GPQA-Diamond、MBPP 上显著超过 matched 9B。
2. 目标不可见的确定性执行器在适用题型上能让 4B 大幅超过 27B 直答。
3. 数据管线能生成超过 10M tokens 的可追踪、可去重、可做泄漏检查的数据。
4. 当前训练框架可稳定完成 SFT、DPO、RL、OPD smoke 并重载 adapter。

**没有证明**

1. 一个 fine-tuned 4B checkpoint 同时赢下三个 benchmark。
2. 4B 在完整 MBPP 上追平 27B；实际结果是 219/257 对 226/257。
3. SFT、DPO、RL、OPD 或 SFT+RL 带来了稳定 benchmark 提升。
4. 本机取得任何正式 agent benchmark 分数。
5. 4B 在本地精确工具能力集以外广泛超过 27B。

## 复现与证据

- 主仓库：`https://github.com/steven-kid/nano-harness.git`
- 分支：`fullstack/campaign-v1`
- 报告生成基线提交：`{report['identity']['repositories']['nano_harness']['revision']}`
- 数据仓库提交：`{report['identity']['repositories']['nano_data_pipeline']['revision']}`
- 训练仓库提交：`{report['identity']['repositories']['nano_train']['revision']}`
- 主结论 JSON SHA256：
  `{report['identity']['local_source_sha256']['three_complete_benchmarks']}`
- 27B 工具能力 JSON SHA256：
  `{report['identity']['local_source_sha256']['verified_tool_27b']}`
- 27B MBPP 负结果 JSON SHA256：
  `{report['identity']['local_source_sha256']['mbpp_27b']}`

重新生成：

```bash
cd ultimate-distill-workspace/worktrees/nano-harness-fullstack-traex-03
PYTHONPATH=. ../../.venv/bin/python scripts/render_ultimate_distill_final_report_v1.py
```

完整回归：

```bash
PYTHONPATH=. ../../.venv/bin/python -m unittest discover -s tests -v
```

原始生成结果、模型权重和日志保留在本地 ignored 路径；GitHub 只提交配置、
测试、预注册收据和 public-safe 报告。
"""


def main() -> None:
    report = build_report()
    PUBLIC.parent.mkdir(parents=True, exist_ok=True)
    PUBLIC.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    MARKDOWN.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
