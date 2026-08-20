# Qwen3.5 Grounded Calculator Canary v1 Result

## 结论

**未通过，不允许跑完整 benchmark，也不允许在这 211 行上修改后重跑。**

- candidate：163/211；
- frozen 4B direct：163/211；
- frozen 9B direct：151/211；
- candidate vs 4B：0 wins / 0 losses，delta
  +0.0000；
- 209 个非 eligible 行评分字段完全复用，0 regression；
- 2 个 GSM8K parse failures 中，1 个安全执行、1 个 fail-close；
- parse failures 2→1，但正确数仍是 163，没有达到 164 gate。

## 分任务

| Benchmark | Correct | Parse failures | API errors |
| --- | ---: | ---: | ---: |
| gpqa_diamond | 6/19 | 0 | 0 |
| gsm8k | 90/96 | 1 | 0 |
| mmlu | 67/96 | 0 | 0 |

## 两个 Recovery

1. 一个 plan 两次输出相同的隐含 multiplier。数字 `2` 不在题面中，strict
   source grounding 两次拒绝，原样回退 direct。这证明 fail-close 有效，但
   字面 numeric grounding 不能表示 “double” 这类语言算子。
2. 另一个 plan 通过 grounding 并精确执行出 break-even 结果 12，formatter
   也原样返回 12；题目要求的是第一个**严格盈利**的整数年份 13。问题不在
   算术执行，而在缺少离散边界语义。

不公开 prompt 或 raw output；plan/output 只记录 SHA 和失败类型。

## Gate

```json
{
  "candidate_api_errors_zero": true,
  "candidate_only_gt_base_only": false,
  "direct_preservation_exact": true,
  "exact_211_case_identity": true,
  "gpqa_diamond_at_least_6": true,
  "gsm8k_at_least_90": true,
  "mmlu_at_least_67": true,
  "overall_correct_at_least_164": false,
  "parse_failures_at_most_2": true,
  "recovery_api_errors_zero": true,
  "unsafe_executions_zero": true
}
```

失败项：

- overall 163/211 < 164/211；
- relative to direct 4B，candidate-only 0 不大于 base-only 0。

## Evidence

- prereg commit：`860f6fd01aeee47ea92bc7c8731059eef60e16fd`；
- config SHA：`3c4049a89ff34895989a82450d78b1d74c3e5889ac9d24307e2b93f2dfe230a2`；
- prereg SHA：`859beba62fa8caffcc24e010f2824d456ce9acbd150bad2fbef4a120b78cce9d`；
- raw result SHA：`006bf6cd46f92b092138ae7cf659d931e95e80355a3538fdf403226379760c36`；
- candidate raw SHA：`8f4b2d743b5008de2763af4452c7891641d2065f0293f90ec9497fe5da272c37`；
- case manifest SHA：`eafbe4d42487a225322dd3b3bdc1d805c065fb15f0f8b968e65ccf747f96976f`。

## 决策

- canary passed：`false`；
- complete benchmark preregistration：`false`；
- complete benchmark generation：`false`；
- independent holdout：`false`；
- observed-canary tuning/rerun：`false`。

下一步只能在**新的、互不重叠的 local synthetic surface**测试 typed semantic
skills：显式处理语言算子和离散边界语义。当前 211-case canary 已观察，永久
关闭对它的 route/prompt/grammar/budget 修改和重跑。
