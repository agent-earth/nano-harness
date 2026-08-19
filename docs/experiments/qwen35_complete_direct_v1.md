# Qwen3.5 Complete Direct Baseline v1 Pre-Registration

## 范围

这次只冻结完整 matched direct baseline，没有启动 vLLM、模型生成、跑分、
训练、RL 或 OPD。

- experiment：`qwen35-complete-direct-v1`
- config SHA256：`3d8ba23272528747afcfdc5273ca376598dfa57d003357c12485e9eb89b5298f`
- suite SHA256：`6ec49d522892975e4532954d3bac7c7e5ed9b24e2c698700d5f8a61667753e90`
- public case contract SHA256：`6dbfe39cc941e17438315cd61d3900bbeb1d6a60615cf3a9f14f01cd1573990c`
- case IDs SHA256：`d38ee8c3eabbefaf7381253f6a69ba87fa63d9ee25fa4b8aeaa5f2afd73b0c63`

## 为什么需要新的 case ID

旧 runner 的 `content_stable_v1` 会把相同题目文本视为同一 case。完整
MMLU 14,042 行里存在重复题面，旧逻辑会少算 174 行。新 suite 显式使用
`row_stable_v2`，把 source row index 加入 identity；旧实验默认仍使用
`content_stable_v1`，历史结果不变。

## 完整数据

- GSM8K：1,319 行；
- MMLU：14,042 行；
- GPQA-Diamond：198 行；
- 总计：15,559 行，case ID 全部唯一。

公开 case contract 只包含 case ID、benchmark、source index、字符数、输出
预算、prompt/system SHA 和 scorer，不包含题目、答案或模型输出。

## Context

| Benchmark | max input | p99 input | max input + output budget |
| --- | ---: | ---: | ---: |
| gpqa_diamond | 2798 | 916 | 2830 |
| gsm8k | 264 | 203 | 864 |
| mmlu | 1054 | 528 | 1086 |

服务 context 冻结为 4096。此前 1024 context 无法覆盖最长 MMLU/GPQA 行。

## 分片与恢复

- 16 个逻辑 shard，算法为 `sha256(case_id) mod 16`；
- shard 行数范围：916–
  1040；
- 每个模型各自写 ignored JSONL shard；
- 重跑按稳定 case ID 跳过 completed rows，error rows 会重试；
- merge 必须恰好覆盖全部 15,559 个 case IDs，重复、
  缺失或多余均 fail closed。

## 服务

- GPU0 / `127.0.0.1:8000`：Qwen3.5-4B；
- GPU1 / `127.0.0.1:8001`：Qwen3.5-9B；
- vLLM 0.19.1、FP16、eager、`max_model_len=4096`、
  `gpu_memory_utilization=0.85`、`max_num_seqs=1`；
- 启动后必须读取 `/v1/models`，确认 served model name 才能执行 shard。

完整 argv 已保存在 JSON receipt，避免 shell quoting 漂移。

## 时间估计

基于已完成 211-case 与 512-case direct runs 的单请求墙钟时间：

- 4B：约 4.12–
  4.35 小时；
- 9B：约 3.81–
  4.05 小时。

这是计划估算，不是正式测量。正式 run 必须记录真实 wall time、token、
parse failure、truncation、API error 和 raw SHA。

## 数据边界

- benchmark rows 不可训练；
- outputs 不可进入 SFT、DPO、RL、reward 或 verifier training；
- raw JSONL 不提交；
- 观察完整结果后禁止搜索 prompt、budget、parser 或 scorer。

## 执行边界

```json
{
  "benchmark_scoring_started": false,
  "model_generation_started": false,
  "opd_started": false,
  "rl_started": false,
  "this_commit_only_preregisters": true,
  "training_started": false
}
```
