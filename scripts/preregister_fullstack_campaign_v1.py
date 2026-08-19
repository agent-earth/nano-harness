#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nano_harness.fullstack_campaign import build_campaign_receipt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT / "configs/campaign/ultimate_distill_fullstack_v1.json"
)
DEFAULT_JSON = (
    ROOT / "docs/experiments/ultimate_distill_fullstack_v1.preregister.json"
)
DEFAULT_MARKDOWN = (
    ROOT / "docs/experiments/ultimate_distill_fullstack_v1.md"
)


def render_markdown(receipt: dict) -> str:
    models = {
        row["model_id"]: row for row in receipt["inventory"]["models"]
    }
    benchmarks = receipt["inventory"]["complete_benchmarks"]
    capabilities = receipt["inventory"]["capabilities"]
    prior = receipt["prior_evidence"]
    checks = receipt["checks"]
    return f"""# Ultimate Distill Full-Stack Campaign v1

## 这次做了什么

这次只做全栈现状审计和下一轮实验预注册，没有启动模型推理、训练、
benchmark 跑分、RL 或 OPD。

- campaign：`{receipt['campaign_id']}`
- base revision：`{receipt['identity']['base_revision']}`
- config SHA256：`{receipt['identity']['config_sha256']}`
- 所有 {len(receipt['identity']['artifacts'])} 个依赖文件身份已重算并通过

## 当前可以直接做什么

- 本机有两张 32GB V100；Qwen3.5-4B 和 Qwen3.5-9B 权重完整且身份通过。
- 已冻结三个完整 benchmark：
{chr(10).join(f"  - {row['name']}: {row['rows']:,} rows, {row['scorer']}" for row in benchmarks)}
- 下一步先跑三个完整 benchmark 的 matched 4B/9B direct baseline。
- skill harness、普通 SFT 和 paired consistency 已有版本化实现。

## 当前还不能做什么

- Qwen3.5-27B-FP8 未安装。`oniond` 可见该模型，但预注册时只剩
  39 GiB 磁盘；未获得单独清理决策前不删除现有模型、数据或 evidence。
- RL 和 OPD 目前没有版本化实现或 smoke receipt，因此继续关闭。
- SWE-bench Lite、ClawBench、WildClawBench、Terminal-Bench 2 等当前只有
  scan/parse/dry-run 证据；本机缺少可用容器 mount namespace，不能把 scan
  写成正式模型分数。

## 已复核的历史结论

{chr(10).join(f"- `{row['evidence_id']}`：{row['claim_boundary']}" for row in prior)}

skill 自进化 synthetic contract 复算结果：

- parent：{receipt['skill_evolution']['parent']['passed']}/{receipt['skill_evolution']['parent']['total']}
- candidate：{receipt['skill_evolution']['candidate']['passed']}/{receipt['skill_evolution']['candidate']['total']}
- promoted：`{str(receipt['skill_evolution']['promotion']['promoted']).lower()}`
- 这只证明 frozen synthetic skill contract 改进，不是 benchmark 提升。

## 候选阶梯

{chr(10).join(
    f"{index}. `{stage['stage_id']}`：{stage['treatment']} "
    f"停止条件：{stage['stop_rule']}"
    for index, stage in enumerate(receipt['candidate_ladder'], start=1)
)}

RL/OPD 不会因为出现在路线图里就自动获准。必须先补实现、污染审计、
finite smoke、reload、固定 config 和 no-post-hoc-search gate。

## 最终验收

- 在完整 GSM8K、MMLU、GPQA-Diamond 上分别与 matched 9B 做 paired 比较。
- 每个 benchmark 都要求 candidate accuracy 更高、bootstrap 95% CI 下界大于
  0、exact McNemar `p<0.05`、case/prompt/parser/scorer 完全一致、零 API error。
- 至少 3 个完整 benchmark 同时通过，才允许声称“4B 显著超过 9B”。
- 27B parity 只在完整 GSM8K 和 MMLU 上判断；预注册 non-inferiority margin
  为 0.02，两项都要通过。

## 审计结果

{chr(10).join(f"- {name}: `{str(value).lower()}`" for name, value in checks.items())}

## 执行边界

```json
{json.dumps(receipt['execution_boundary'], indent=2, sort_keys=True)}
```

下一可执行切片：{receipt['next_executable_slice']['action']}
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--json-output", default=str(DEFAULT_JSON))
    parser.add_argument("--markdown-output", default=str(DEFAULT_MARKDOWN))
    args = parser.parse_args()

    receipt = build_campaign_receipt(args.config)
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
                "campaign_id": receipt["campaign_id"],
                "checks": receipt["checks"],
                "readiness": receipt["readiness"],
                "execution_boundary": receipt["execution_boundary"],
                "next_executable_slice": receipt[
                    "next_executable_slice"
                ],
                "json_output": str(json_output),
                "markdown_output": str(markdown_output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
