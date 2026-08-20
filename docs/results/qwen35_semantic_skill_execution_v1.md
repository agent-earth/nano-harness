# Qwen3.5 Typed Semantic Skill Execution v1 Result

## 结论

- local semantic skill admitted：
  `true`；
- fresh local replication preregistration allowed：
  `true`；
- fresh replication generation allowed：`false`；
- canary / benchmark / holdout / training：全部 `false`。

## 具体结果

| Arm | Correct | Accuracy | Parseable |
| --- | ---: | ---: | ---: |
| four_b_direct | 0/256 | 0.0000 | 256/256 |
| four_b_semantic_skills | 256/256 | 1.0000 | 256/256 |
| nine_b_direct | 0/256 | 0.0000 | 256/256 |

两个 family 均为 128 cases：

- implicit double/triple total：harness 128/128；
- first strictly profitable whole period：harness 128/128；
- 4B direct 和 9B direct 在两个 family 都是 0/128。

这说明 typed semantic skill 能补足“隐含语言算子”和“严格离散边界”，但这是
刻意构造的 local mechanism surface，**不是 benchmark 分数**。Direct 两臂为 0
也意味着下一步必须用 paraphrase 和不同数值分布做 fresh replication，不能直接
推广到真实任务。

## Routing

```json
{
  "fallbacks": 0,
  "feedback_result_matches": 256,
  "final_feedback_calls": 256,
  "plan_retries": 0,
  "prompt_routes": 256,
  "single_tool_exposures": 256,
  "verified_executions": 256
}
```

256 prompt-only routes、single-tool exposures、verified executions 和 feedback
result matches；0 retry、0 fallback。Router 不读 case metadata，executor 不读
expected 或 correctness。

## Paired Comparisons

| Comparison | Delta | 95% CI | Wins | Losses | McNemar p |
| --- | ---: | --- | ---: | ---: | ---: |
| harness_vs_four_b | +1.0000 | [+1.0000, +1.0000] | 256 | 0 | 1.72723e-77 |
| harness_vs_nine_b | +1.0000 | [+1.0000, +1.0000] | 256 | 0 | 1.72723e-77 |
| four_b_vs_nine_b | +0.0000 | [+0.0000, +0.0000] | 0 | 0 | 1 |

## Frozen Gates

```json
{
  "all_rows_complete_and_parseable": true,
  "every_family_non_regression_vs_four_b_and_nine_b": true,
  "executor_contract_failures_zero": true,
  "feedback_result_matches_256": true,
  "harness_vs_four_b_accuracy_positive": true,
  "harness_vs_four_b_ci_lower_gt_zero": true,
  "harness_vs_four_b_maximum_losses": true,
  "harness_vs_four_b_mcnemar_p_lt_005": true,
  "harness_vs_four_b_minimum_wins": true,
  "harness_vs_nine_b_accuracy_positive": true,
  "harness_vs_nine_b_ci_lower_gt_zero": true,
  "harness_vs_nine_b_maximum_losses": true,
  "harness_vs_nine_b_mcnemar_p_lt_005": true,
  "harness_vs_nine_b_minimum_wins": true,
  "prompt_routes_256": true,
  "single_tool_exposures_256": true,
  "verified_executions_256": true
}
```

## Interrupted Preflights

两次启动都在模型请求前 fail-close：

- GitHub HTTP/2 framing error；
- localhost health 被错误送到代理而返回 403。

两次均确认 result artifact 不存在，随后用 HTTP/1.1 和
`NO_PROXY=127.0.0.1,localhost` 完成唯一正式 run。它们不是额外实验臂。

## Evidence

- prereg commit：`53d07c610759db60b050ffd990de0d4c3a5c9a66`；
- config SHA：`4e5785b9b4fc9dd5b3a76a95b438c2f2c3d505a2fae6231b81cfbf53ebbd6ad9`；
- prereg SHA：`fad4dff56233edc35d10d14f6ff5922c03f054d7d656e779e23c0f1dc102f7fc`；
- raw result SHA：`345b1d16d32fbf42bccab4129533e4072fa52b8b8f7f8a519e52e26328482bf4`；
- case contract SHA：`913c64a4bcfdce4d04299b9958a291937d3c7bfd6425ee956e835042e29651bf`；
- canary rejection SHA：
  `6f0fcebabd0bfb8099ec34e6465362c1c884524605484aec47251068e9f5b056`。

下一步只允许另行预注册 fresh history-disjoint paraphrase/numerical-regime
replication。已观察的 211-case canary 不重跑，complete benchmark、independent
holdout 和 training 继续关闭。
