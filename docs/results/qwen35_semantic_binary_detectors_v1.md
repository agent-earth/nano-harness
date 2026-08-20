# Qwen3.5 Semantic Binary Detectors v1 Result

## 结论

**拒绝。两个 detector 对128条全部输出 NO。**

- negative NONE：64/64；
- negative false positive：0；
- positive recall：0/64；
- verified executions：0；
- candidate / 4B direct / 9B direct：均 0/128。

零误报来自恒 NO，不是可用 precision-recall tradeoff。按冻结 gate 不允许
real question scan，也不能调 prompt 或重跑。

## Confusion

```json
[
  {
    "cases": 64,
    "expected_route": "NONE",
    "selected_route": "NONE"
  },
  {
    "cases": 32,
    "expected_route": "first_strict_profit_period",
    "selected_route": "NONE"
  },
  {
    "cases": 32,
    "expected_route": "implicit_scale_total",
    "selected_route": "NONE"
  }
]
```

## Detector Outputs

```json
[
  {
    "cases": 128,
    "detector": "first_strict_profit_period",
    "yes": false
  },
  {
    "cases": 128,
    "detector": "implicit_scale_total",
    "yes": false
  }
]
```

## Frozen Gates

```json
{
  "all_detector_outputs_parseable": true,
  "all_rows_complete_and_parseable": true,
  "candidate_vs_four_b_maximum_losses": true,
  "candidate_vs_four_b_minimum_wins": false,
  "candidate_vs_four_b_significant": false,
  "candidate_vs_nine_b_maximum_losses": true,
  "candidate_vs_nine_b_minimum_wins": false,
  "candidate_vs_nine_b_significant": false,
  "conflicts_zero": true,
  "detector_composition_correct_128": false,
  "every_family_non_regression": true,
  "fallbacks_zero": true,
  "negative_candidate_exact_direct_parity": true,
  "negative_false_positive_routes_zero": true,
  "negative_none_correct_64": true,
  "positive_route_recall_64": false,
  "positive_verified_executions_64": false
}
```

## Evidence

- prereg commit：`83835e3f9a13ad5145fd95205b49e2ff66ee6a31`；
- config SHA：`32c1d877a1cecdb9041fc88226fa2e52890390712f449a8552be0a1202d88748`；
- prereg SHA：`27a3a425a79852666069345d51c494e981b5ae0664a5c9aa41f17140b842e832`；
- raw result SHA：`5915d5e8c3e05aaae353eeee514a9145f4dc58113a268a94290b0098df543df6`；
- case contract SHA：`a3993974c923b5cab4b8570642226076a8b9c51ead24175773681a8c84b624a5`。

## 下一步

不再搜索 inference prompt。改为另行预注册 synthetic router SFT
classification objective：大量 paraphrase positives + unsupported negatives，typed
executors、validators、feedback equality 和 fallback 全部冻结。真实 benchmark、
canary、holdout、training generation 在该 SFT/data contract 提交前继续关闭。
