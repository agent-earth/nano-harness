#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nano_harness.benchmark_blind_treatment import build_treatment_receipt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT / "configs/campaign/qwen35_benchmark_blind_treatment_v1.json"
)
DEFAULT_JSON = (
    ROOT
    / "docs/experiments/qwen35_benchmark_blind_treatment_v1.preregister.json"
)
DEFAULT_MARKDOWN = (
    ROOT / "docs/experiments/qwen35_benchmark_blind_treatment_v1.md"
)


def render_markdown(receipt: dict) -> str:
    peer = receipt["peer_result"]
    readiness = receipt["readiness"]
    arm_rows = []
    for arm in receipt["arms"]:
        arm_rows.append(
            f"| `{arm['arm_id']}` | `{arm['variable_changed']}` | "
            f"{arm['intervention']} |"
        )
    evidence_rows = []
    for evidence in receipt["prior_evidence"]:
        assertion_summary = ", ".join(
            f"`{row['pointer']}` = `{json.dumps(row['actual'])}`"
            for row in evidence["assertions"]
        )
        evidence_rows.append(
            f"- `{evidence['evidence_id']}`：{assertion_summary}。"
        )
    canary = receipt["admission_gates"]["canary"]
    complete = receipt["admission_gates"]["complete"]
    return f"""# Qwen3.5 Benchmark-Blind Treatment v1

## 这次具体做了什么

这次只冻结下一轮 4B treatment 的实验设计，没有跑新的训练、canary 或
完整 benchmark。

- treatment：`{receipt['treatment_id']}`
- config SHA256：`{receipt['identity']['config_sha256']}`
- peer 预注册 commit：`{receipt['identity']['peer_revision']}`
- peer replication result 是否存在：`{str(peer['exists']).lower()}`
- 当前是否允许跑 canary：`{str(readiness['canary_generation_allowed']).lower()}`
- 当前是否允许跑完整 treatment benchmark：
  `{str(readiness['complete_treatment_generation_allowed']).lower()}`

当前结论：{readiness['reason']}

## 为什么不能直接跑完整 benchmark

完整 direct 基线已经证明 4B 只通过 1/3 个 benchmark gate。下一次实验必须
先用与 benchmark 无关的训练信号证明一致性 adapter 稳定有效，再用冻结的
211-case canary 检查是否破坏 GSM8K、MMLU、GPQA。peer replication 缺少任一
显著性、reload、finite 或 JSON 非回归 gate，三种 treatment 全部禁止生成。

## 三个固定消融臂

| Arm | 唯一变化 | 实验内容 |
| --- | --- | --- |
{chr(10).join(arm_rows)}

`arbiter_only` 不是已晋级的 harness。它在 GPQA dev8 上 2 次覆盖均修正答案，
但在后续 72-case holdout5 上 1 赢 1 输，最终 gate 未通过。因此它只作为固定
ablation 继续验证，不能单独写成“verified 提升”。

## 已冻结的历史证据

{chr(10).join(evidence_rows)}

这些值只用于说明为什么选择“一致性 adapter + GPQA arbiter”做机制消融，
不能把完整 benchmark 的错题、答案或模型输出送进训练、reward、verifier 或
数据生成。

## Canary 准入

三个 arm 都必须独立跑完，不因前一个结果好坏而跳过：

- 固定 211 cases；case、dataset、prompt、parser、scorer 完全一致；
- overall 至少 `{canary['minimum_overall_correct']}/{canary['cases']}`；
- GSM8K 至少 `{canary['benchmark_minimum_correct']['gsm8k']}/96`；
- MMLU 至少 `{canary['benchmark_minimum_correct']['mmlu']}/96`；
- GPQA 至少 `{canary['benchmark_minimum_correct']['gpqa_diamond']}/19`；
- API error 为 0，parse failure 最多
  `{canary['maximum_parse_failures']}`；
- 相对 base 4B 的 candidate-only wins 必须多于 base-only wins。

任一 arm 失败就保留为负证据，不能在这 211 rows 上改 prompt、budget、route、
adapter weight 或训练参数修复。

## 完整 benchmark 准入

只有独立通过 canary 的 arm 才能跑完整 GSM8K、MMLU、GPQA。每个 admitted
arm 都要执行，不能看第一个完整结果后只挑最好看的 arm。最终 4B 超过 9B 的
正式 gate 仍是：

- 每个 benchmark candidate accuracy 更高；
- paired bootstrap 95% CI 下界 > 0；
- exact McNemar `p < {complete['alpha']}`；
- candidate-only wins > 9B-only wins；
- 相对 direct 4B 每个 benchmark 不回退；
- 三个完整 benchmark 全部通过；
- strict score 是唯一正式分数，loose-format 只做非评分诊断。

## 禁止事项

观察任何 treatment 输出后禁止：

{chr(10).join(f"- `{item}`" for item in receipt['decision_policy']['forbidden_after_any_treatment_observation'])}

独立 holdout 继续密封。RL/OPD 也不会因为这份预注册自动开放。

## 下一步

{receipt['next_action']}
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--json-output", default=str(DEFAULT_JSON))
    parser.add_argument("--markdown-output", default=str(DEFAULT_MARKDOWN))
    args = parser.parse_args()

    receipt = build_treatment_receipt(args.config)
    json_output = Path(args.json_output)
    markdown_output = Path(args.markdown_output)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_output.write_text(
        render_markdown(receipt),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "schema_version": receipt["schema_version"],
                "treatment_id": receipt["treatment_id"],
                "checks": receipt["checks"],
                "peer_result": receipt["peer_result"],
                "readiness": receipt["readiness"],
                "next_action": receipt["next_action"],
                "json_output": str(json_output),
                "markdown_output": str(markdown_output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
