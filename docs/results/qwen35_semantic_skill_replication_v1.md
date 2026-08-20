# Qwen3.5 Typed Semantic Skill Replication v1 Result

## 结论

- replication admitted：`true`；
- real-task transfer preregistration allowed：
  `true`；
- real-task generation allowed：`false`；
- canary rerun / benchmark generation / holdout / training：全部 `false`。

## Fresh Replication

- unseen compact-display / kiosk contexts；
- small-integer numerical regime；
- parent case ID / prompt / source-fact overlap：0 / 0 / 0；
- prior benchmark prompt overlap：0；
- mechanism、models、services、budgets、retry、fallback 和 gates 不变。

| Arm | Correct | Accuracy | Parseable |
| --- | ---: | ---: | ---: |
| four_b_direct | 5/256 | 0.0195 | 256/256 |
| four_b_semantic_skills | 256/256 | 1.0000 | 256/256 |
| nine_b_direct | 4/256 | 0.0156 | 256/256 |

两个 semantic families 的 harness 都是 128/128。4B direct 为 5/256，
9B direct 为 4/256；因此这次不再是 parent 的双零 direct surface。

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

256 prompt routes、single-tool exposures、verified executions 和 feedback
matches，0 retry、0 fallback。

## Paired Comparisons

| Comparison | Delta | 95% CI | Wins | Losses | McNemar p |
| --- | ---: | --- | ---: | ---: | ---: |
| harness_vs_four_b | +0.9805 | [+0.9609, +0.9961] | 251 | 0 | 5.52715e-76 |
| harness_vs_nine_b | +0.9844 | [+0.9688, +0.9961] | 252 | 0 | 2.76357e-76 |
| four_b_vs_nine_b | +0.0039 | [-0.0195, +0.0273] | 5 | 4 | 1 |

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

## Evidence

- prereg commit：`ea13026b1e4f8692a564ed5e5153083971e7ad60`；
- config SHA：`90774e3be32504637ee5cca27c45a8c577c7b9803e9c46c7bb68381b81a79501`；
- prereg SHA：`2f42337b0dc6ecfddaa0290141f59c76a475f2d79f8e1926cd59187cf95c9a38`；
- raw result SHA：`1673d71a2dd68f9ebe74b281649534c71c2fad002164405fc2ec6c5f92de0f9b`；
- case contract SHA：`e863bb74a2e58bb8287f004603c7e450c18743a597ba72ad1cd6fa4989ca1e74`；
- parent report SHA：`fe53a512cbf0b6ada65ed3ae27c5f3dc90165e367cfecdb58307dd030d017d5f`。

通过只允许**另行预注册** real-task transfer。必须先冻结 real-task case
identities、prompt-only eligibility、direct-preserve 范围、fallback、budgets 和
gates；当前不能生成 benchmark outputs，也不能重跑已观察 211-case canary。
