# Qwen3.5 Constrained Semantic Model Router v1 Result

## 结论

**拒绝，不允许 real question model scan。**

- unsupported negative：128/128 选 `NONE`，false positive 0；
- strict-profit positive：64/64 路由正确并执行成功；
- implicit-scale positive：0/64 路由正确，全部选 `NONE`；
- positive recall：64/128；
- candidate：64/256；4B direct：0/256；9B direct：0/256。

Router 的保守性方向成立，但召回 gate 明确失败。不能根据这些已观察样例改
prompt、枚举、预算或重跑。

## Confusion

```json
[
  {
    "cases": 128,
    "expected_route": "NONE",
    "selected_route": "NONE"
  },
  {
    "cases": 64,
    "expected_route": "first_strict_profit_period",
    "selected_route": "first_strict_profit_period"
  },
  {
    "cases": 64,
    "expected_route": "implicit_scale_total",
    "selected_route": "NONE"
  }
]
```

## Routing

```json
{
  "cases": 256,
  "fallbacks": 0,
  "negative_cases": 128,
  "negative_false_positive_routes": 0,
  "negative_none_correct": 128,
  "positive_cases": 128,
  "positive_route_correct": 64,
  "router_correct": 192,
  "verified_executions": 64
}
```

所有 negative `NONE` rows 与 direct 保持一致；64 个 strict-profit rows 完成
verified execution。Implicit-scale rows安全地 direct-preserve，但没有得到预期
增益。

## Paired Comparisons

| Comparison | Delta | 95% CI | Wins | Losses | McNemar p |
| --- | ---: | --- | ---: | ---: | ---: |
| candidate_vs_four_b | +0.2500 | [+0.1992, +0.3047] | 64 | 0 | 1.0842e-19 |
| candidate_vs_nine_b | +0.2500 | [+0.1992, +0.3047] | 64 | 0 | 1.0842e-19 |
| four_b_vs_nine_b | +0.0000 | [+0.0000, +0.0000] | 0 | 0 | 1 |

Aggregate 对 direct 显著，但不能覆盖 route recall 失败，故仍拒绝。

## Frozen Gates

```json
{
  "all_rows_complete_and_parseable": true,
  "candidate_vs_four_b_maximum_losses": true,
  "candidate_vs_four_b_minimum_wins": true,
  "candidate_vs_four_b_significant": true,
  "candidate_vs_nine_b_maximum_losses": true,
  "candidate_vs_nine_b_minimum_wins": true,
  "candidate_vs_nine_b_significant": true,
  "every_family_non_regression": true,
  "fallbacks_zero": true,
  "negative_candidate_exact_direct_parity": true,
  "negative_false_positive_routes_zero": true,
  "negative_none_correct_128": true,
  "positive_route_recall_128": false,
  "positive_verified_executions_128": false,
  "router_outputs_parseable_256": true
}
```

## Evidence

- prereg commit：`6a30f64e32cb01957dcd03292180aece4e7a0cfb`；
- config SHA：`8c6f0215cb2a0f805e04fe6c00a28fc3b5847d0c2a14575a90b3ed2a6586ebce`；
- prereg SHA：`963c93ca46493b91b6bf59a3933b0d40fbaf5eaabf8f8a3748bad4c74d1abf14`；
- raw result SHA：`8d3842984e6643a8c86aa29038bfb1280fa12a56deb3010f1151f03ed54a33a3`；
- case contract SHA：`6bb1f7ff3079fa1e3c75d033e26d3c6f6109b0abdf96c8949edebea2761c31ca`。

下一机制只能在 fresh surface 测试两个独立 binary skill detectors，再用
`NONE` 默认组合；typed executors、source validation、feedback equality 和
fallback 保持不变。Benchmark generation、canary、holdout、training 继续关闭。
