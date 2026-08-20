# Qwen3.5 Skill-Routed Verified Tool Execution v2 Result

## 结论

- local harness admitted：
  `true`；
- canary pre-registration allowed：
  `true`；
- canary generation allowed：`false`；
- complete benchmark allowed：`false`。

## Arms

| Arm | Correct | Accuracy | Parseable |
| --- | ---: | ---: | ---: |
| four_b_direct | 30/256 | 0.1172 | 256/256 |
| four_b_skill_verified_tool | 256/256 | 1.0000 | 256/256 |
| nine_b_direct | 19/256 | 0.0742 | 256/256 |

## Routing

```json
{
  "fallbacks": 0,
  "final_feedback_calls": 256,
  "plan_retries": 0,
  "single_tool_exposures": 256,
  "skill_routes": 256,
  "verified_executions": 256
}
```

V2 完成 256/256 skill routes、single-tool exposures、verified executions 和
result-feedback calls，0 retry、0 fallback、0 contract failure。

## Paired comparisons

| Comparison | Delta | 95% CI | Wins | Losses | McNemar p |
| --- | ---: | --- | ---: | ---: | ---: |
| harness_vs_four_b | +0.8828 | [+0.8438, +0.9219] | 226 | 0 | 1.8546e-68 |
| harness_vs_nine_b | +0.9258 | [+0.8906, +0.9570] | 237 | 0 | 9.05568e-72 |
| four_b_vs_nine_b | +0.0430 | [+0.0000, +0.0859] | 22 | 11 | 0.0801433 |

## Frozen gates

```json
{
  "all_rows_complete_and_parseable": true,
  "every_family_non_regression_vs_four_b_and_nine_b": true,
  "executor_contract_failures_zero": true,
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
  "single_tool_exposures_256": true,
  "skill_routes_256": true,
  "verified_executions_256": true
}
```

## Evidence

- config SHA：`ae6740e37da66b393f0732e7d86b785148e9d6fc663cbbdea0c8554d68f5ae0f`；
- prereg SHA：`1c312a575b4c4bc4000e64495f3157ceff529361d5c0dc43112b318faf1c0797`；
- raw result SHA：`3d4c987f8f949289e50d97bdb7f00dd08036eec511a8d77261cc0b24ddbb8047`；
- case contract SHA：`cd5037a2574254ad87ff13abc3d8af51670d9b565d4433b041df4968c5eb2d71`；
- parent V1 report SHA：
  `6baa2f1e5fc30efa1e07f169588847d09f41fca79ead5af235a75f643e0deb07`。

通过只允许**另行预注册** 211-case canary；当前仍不能生成 canary outputs，
也不能访问完整 benchmark 或 independent holdout。
