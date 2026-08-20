# Qwen3.5 Skill-Routed Verified Tool Execution v2

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
  `cd5037a2574254ad87ff13abc3d8af51670d9b565d4433b041df4968c5eb2d71`；
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
{
  "benchmark_accessed": false,
  "canary_accessed": false,
  "evaluation_started": false,
  "model_generation_started": false,
  "service_reused": true,
  "this_commit_only_preregisters": true,
  "training_started": false
}
```
