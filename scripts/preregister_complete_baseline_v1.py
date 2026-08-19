#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nano_harness.complete_baseline import build_receipt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT / "configs/campaign/qwen35_complete_direct_v1.preregister.json"
)
DEFAULT_RECEIPT = (
    ROOT / "docs/experiments/qwen35_complete_direct_v1.preregister.json"
)
DEFAULT_REPORT = (
    ROOT / "docs/experiments/qwen35_complete_direct_v1.md"
)
DEFAULT_CASES = (
    ROOT / "configs/generated/qwen35_complete_direct_v1_cases.public.json"
)


def _hours(seconds: float) -> str:
    return f"{seconds / 3600:.2f}"


def render_markdown(receipt: dict) -> str:
    costs = receipt["historical_cost_projection"]["total_per_model"]
    context = receipt["context"]["by_benchmark"]
    return f"""# Qwen3.5 Complete Direct Baseline v1 Pre-Registration

## 范围

这次只冻结完整 matched direct baseline，没有启动 vLLM、模型生成、跑分、
训练、RL 或 OPD。

- experiment：`{receipt['experiment_id']}`
- config SHA256：`{receipt['identity']['config_sha256']}`
- suite SHA256：`{receipt['identity']['suite_manifest_sha256']}`
- public case contract SHA256：`{receipt['identity']['case_contract_sha256']}`
- case IDs SHA256：`{receipt['identity']['case_ids_sha256']}`

## 为什么需要新的 case ID

旧 runner 的 `content_stable_v1` 会把相同题目文本视为同一 case。完整
MMLU 14,042 行里存在重复题面，旧逻辑会少算 174 行。新 suite 显式使用
`row_stable_v2`，把 source row index 加入 identity；旧实验默认仍使用
`content_stable_v1`，历史结果不变。

## 完整数据

- GSM8K：{receipt['cases']['by_benchmark']['gsm8k']:,} 行；
- MMLU：{receipt['cases']['by_benchmark']['mmlu']:,} 行；
- GPQA-Diamond：{receipt['cases']['by_benchmark']['gpqa_diamond']:,} 行；
- 总计：{receipt['cases']['total']:,} 行，case ID 全部唯一。

公开 case contract 只包含 case ID、benchmark、source index、字符数、输出
预算、prompt/system SHA 和 scorer，不包含题目、答案或模型输出。

## Context

| Benchmark | max input | p99 input | max input + output budget |
| --- | ---: | ---: | ---: |
{chr(10).join(
    f"| {name} | {row['input_max']} | {row['input_p99']} | "
    f"{row['input_plus_budget_max']} |"
    for name, row in context.items()
)}

服务 context 冻结为 4096。此前 1024 context 无法覆盖最长 MMLU/GPQA 行。

## 分片与恢复

- 16 个逻辑 shard，算法为 `sha256(case_id) mod 16`；
- shard 行数范围：{receipt['sharding']['minimum']}–
  {receipt['sharding']['maximum']}；
- 每个模型各自写 ignored JSONL shard；
- 重跑按稳定 case ID 跳过 completed rows，error rows 会重试；
- merge 必须恰好覆盖全部 {receipt['cases']['total']:,} 个 case IDs，重复、
  缺失或多余均 fail closed。

## 服务

- GPU0 / `127.0.0.1:8000`：Qwen3.5-4B；
- GPU1 / `127.0.0.1:8001`：Qwen3.5-9B；
- vLLM 0.19.1、FP16、eager、`max_model_len=4096`、
  `gpu_memory_utilization=0.85`、`max_num_seqs=1`；
- 当前容器的 `ldconfig` 不暴露驱动库，但
  `/usr/lib/x86_64-linux-gnu/libcuda.so.1` 存在。服务 argv 只为 Triton
  增加 `TRITON_LIBCUDA_PATH=/usr/lib/x86_64-linux-gnu`，并冻结目标库 SHA；
  模型和推理参数不变；
- 启动后必须读取 `/v1/models`，确认 served model name 才能执行 shard。

完整 argv 已保存在 JSON receipt，避免 shell quoting 漂移。

## 时间估计

基于已完成 211-case 与 512-case direct runs 的单请求墙钟时间：

- 4B：约 {_hours(costs['four_b']['projected_wall_seconds_min'])}–
  {_hours(costs['four_b']['projected_wall_seconds_max'])} 小时；
- 9B：约 {_hours(costs['nine_b']['projected_wall_seconds_min'])}–
  {_hours(costs['nine_b']['projected_wall_seconds_max'])} 小时。

这是计划估算，不是正式测量。正式 run 必须记录真实 wall time、token、
parse failure、truncation、API error 和 raw SHA。

## 数据边界

- benchmark rows 不可训练；
- outputs 不可进入 SFT、DPO、RL、reward 或 verifier training；
- raw JSONL 不提交；
- 观察完整结果后禁止搜索 prompt、budget、parser 或 scorer。

## 执行边界

```json
{json.dumps(receipt['execution_boundary'], indent=2, sort_keys=True)}
```
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--receipt", default=str(DEFAULT_RECEIPT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--case-contract", default=str(DEFAULT_CASES))
    args = parser.parse_args()

    receipt, cases = build_receipt(args.config)
    receipt_path = Path(args.receipt)
    report_path = Path(args.report)
    cases_path = Path(args.case_contract)
    for path in (receipt_path, report_path, cases_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(render_markdown(receipt), encoding="utf-8")
    cases_path.write_text(
        json.dumps(cases, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "schema_version": receipt["schema_version"],
                "experiment_id": receipt["experiment_id"],
                "cases": receipt["cases"],
                "sharding": receipt["sharding"],
                "context": receipt["context"],
                "checks": receipt["checks"],
                "execution_boundary": receipt["execution_boundary"],
                "receipt": str(receipt_path),
                "report": str(report_path),
                "case_contract": str(cases_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
