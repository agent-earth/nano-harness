# Qwen3.5 Verified Tool-Execution Harness v1

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
  `433de81f03060ae89f197b19b6525d7bc9f414bc2fa0501534441ebb121718e9`；
- 与 prior choice matrix v1/v2/v3 prompt overlap：0；
- 与完整 GSM8K/MMLU/GPQA prompt overlap：0；
- benchmark/canary/holdout rows、outputs：0；
- executor 不读取 expected answer 或 correctness。

## Context

- direct input+budget max：`117`；
- plan input+budget max：`194`；
- result-feedback input+budget max：
  `178`；
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
{
  "benchmark_accessed": false,
  "canary_accessed": false,
  "evaluation_started": false,
  "model_generation_started": false,
  "service_started": false,
  "this_commit_only_preregisters": true,
  "training_started": false
}
```
