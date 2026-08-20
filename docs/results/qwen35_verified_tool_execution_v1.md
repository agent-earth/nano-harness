# Qwen3.5 Verified Tool-Execution Harness v1 Result

## 结论

- local harness admitted：
  `false`；
- 211-case canary allowed：
  `false`；
- complete benchmark allowed：`false`；
- tuning/rerun on observed cases：`false`。

## Arms

| Arm | Correct | Accuracy | Parseable |
| --- | ---: | ---: | ---: |
| four_b_direct | 21/256 | 0.0820 | 256/256 |
| four_b_verified_tool | 192/256 | 0.7500 | 256/256 |
| nine_b_direct | 13/256 | 0.0508 | 256/256 |

## Paired comparisons

| Comparison | Delta | 95% CI | Wins | Losses | McNemar p |
| --- | ---: | --- | ---: | ---: | ---: |
| harness_vs_four_b | +0.6680 | [+0.6094, +0.7266] | 171 | 0 | 6.68191e-52 |
| harness_vs_nine_b | +0.6992 | [+0.6406, +0.7539] | 179 | 0 | 2.61012e-54 |
| four_b_vs_nine_b | +0.0312 | [-0.0078, +0.0742] | 19 | 11 | 0.200488 |

## Routing

```json
{
  "executor_contract_failures": 64,
  "fallbacks": 64,
  "final_feedback_calls": 192,
  "final_reason_counts": {
    "tool_name_mismatch": 64,
    "verified_execution": 192
  },
  "plan_retries": 64,
  "verified_executions": 192
}
```

192 cases completed plan → safe execute → verified-result feedback. 64
`labor_total` cases exhausted the frozen one-retry plan contract and fell back to
4B direct, so `executor_contract_failures_zero` fails even though the harness
has a large significant net gain.

## Frozen gates

```json
{
  "all_rows_complete_and_parseable": true,
  "every_family_non_regression_vs_four_b_and_nine_b": true,
  "executor_contract_failures_zero": false,
  "harness_accuracy_gt_four_b_direct": true,
  "harness_accuracy_gt_nine_b_direct": true,
  "harness_vs_four_b_bootstrap_ci_lower_gt_zero": true,
  "harness_vs_four_b_maximum_losses": true,
  "harness_vs_four_b_mcnemar_p_lt_005": true,
  "harness_vs_four_b_minimum_wins": true,
  "harness_vs_nine_b_bootstrap_ci_lower_gt_zero": true,
  "harness_vs_nine_b_maximum_losses": true,
  "harness_vs_nine_b_mcnemar_p_lt_005": true,
  "harness_vs_nine_b_minimum_wins": true,
  "verified_execution_count_positive": true
}
```

## Evidence

- config SHA：`538cbcde51ccb8ad43e4f91db4201a2ffd835c1493a5b2bd58177db7dcab3cd3`；
- prereg SHA：`97c2b54ca691708149b9e987e65dc88df829a953bf33edb4ed0c0c78f6fd0dfb`；
- service receipt SHA：
  `895bd40d886d870f99230441baecdf1feb926e2992b6a151e54d8165145f1c0d`；
- raw result SHA：`a2f9bdda490f4b3d86631fa057abe98c8987cad7e39e802e0a75d4cef43b7b86`；
- case contract SHA：`433de81f03060ae89f197b19b6525d7bc9f414bc2fa0501534441ebb121718e9`。

公开报告只包含聚合、case IDs、reason counts 和 SHA；不公开 prompt、facts、
expected、model outputs 或 full tool trajectories。Canary 和完整 benchmark 继续关闭。
