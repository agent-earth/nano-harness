# Qwen3.5 Router Adapter Integration v2 Result

## Verdict

**REJECT.**

V2 uses the content-identical namespace-remapped adapter on 128 new prompts.
Neither V1 rows nor V1 outputs were loaded or regenerated.

## Arms

```json
{
  "four_b_direct": {
    "accuracy": 0.0,
    "by_family": {
      "box_total": {
        "cases": 32,
        "correct": 0,
        "parseable": 32
      },
      "first_strict_profit_period": {
        "cases": 32,
        "correct": 0,
        "parseable": 32
      },
      "implicit_scale_total": {
        "cases": 32,
        "correct": 0,
        "parseable": 32
      },
      "remaining_stock": {
        "cases": 32,
        "correct": 0,
        "parseable": 32
      }
    },
    "cases": 128,
    "correct": 0,
    "parseable": 128
  },
  "four_b_router_adapter_v2": {
    "accuracy": 0.5,
    "by_family": {
      "box_total": {
        "cases": 32,
        "correct": 0,
        "parseable": 32
      },
      "first_strict_profit_period": {
        "cases": 32,
        "correct": 32,
        "parseable": 32
      },
      "implicit_scale_total": {
        "cases": 32,
        "correct": 32,
        "parseable": 32
      },
      "remaining_stock": {
        "cases": 32,
        "correct": 0,
        "parseable": 32
      }
    },
    "cases": 128,
    "correct": 64,
    "parseable": 128
  },
  "nine_b_direct": {
    "accuracy": 0.0,
    "by_family": {
      "box_total": {
        "cases": 32,
        "correct": 0,
        "parseable": 32
      },
      "first_strict_profit_period": {
        "cases": 32,
        "correct": 0,
        "parseable": 32
      },
      "implicit_scale_total": {
        "cases": 32,
        "correct": 0,
        "parseable": 32
      },
      "remaining_stock": {
        "cases": 32,
        "correct": 0,
        "parseable": 32
      }
    },
    "cases": 128,
    "correct": 0,
    "parseable": 128
  }
}
```

## Routing

```json
{
  "cases": 128,
  "correct": 96,
  "fallbacks": 32,
  "negative_c_correct": 32,
  "negative_cases": 64,
  "negative_false_positive_routes": 32,
  "positive_cases": 64,
  "positive_correct": 64,
  "verified_executions": 64
}
```

## Routing By Family

```json
{
  "box_total": {
    "cases": 32,
    "fallbacks": 32,
    "route_correct": 0,
    "selected_labels": {
      "A": 32
    },
    "verified_executions": 0
  },
  "first_strict_profit_period": {
    "cases": 32,
    "fallbacks": 0,
    "route_correct": 32,
    "selected_labels": {
      "B": 32
    },
    "verified_executions": 32
  },
  "implicit_scale_total": {
    "cases": 32,
    "fallbacks": 0,
    "route_correct": 32,
    "selected_labels": {
      "A": 32
    },
    "verified_executions": 32
  },
  "remaining_stock": {
    "cases": 32,
    "fallbacks": 0,
    "route_correct": 32,
    "selected_labels": {
      "C": 32
    },
    "verified_executions": 0
  }
}
```

## Paired Comparisons

| Comparison | Delta | 95% CI | Wins | Losses | McNemar p |
| --- | ---: | --- | ---: | ---: | ---: |
| candidate_vs_four_b | +0.5000 | [+0.4141, +0.5859] | 64 | 0 | 1.0842e-19 |
| candidate_vs_nine_b | +0.5000 | [+0.4141, +0.5859] | 64 | 0 | 1.0842e-19 |
| four_b_vs_nine_b | +0.0000 | [+0.0000, +0.0000] | 0 | 0 | 1 |

## Frozen Gates

```json
{
  "all_three_arms_complete_and_parseable_128": true,
  "candidate_vs_four_b_maximum_losses": true,
  "candidate_vs_four_b_minimum_wins": true,
  "candidate_vs_four_b_significant": true,
  "candidate_vs_nine_b_maximum_losses": true,
  "candidate_vs_nine_b_minimum_wins": true,
  "candidate_vs_nine_b_significant": true,
  "every_family_non_regression": true,
  "fallbacks_zero": false,
  "negative_candidate_exact_direct_parity": true,
  "negative_false_positive_routes_zero": false,
  "positive_feedback_result_matches_64": true,
  "positive_verified_executions_64": true,
  "router_a_recall_32": true,
  "router_b_recall_32": true,
  "router_c_precision_64": false,
  "router_outputs_parseable_128": true
}
```

## Boundaries

V1 and V2 cannot be rerun. Passing allows only a separately pre-registered
question-only scan. Benchmark, canary, holdout, training, and RL stay closed.

## Evidence

- prereg SHA: `1a23a1bb391a3ebac7e70aecd5e2d855ef624e825c26e6bfe9ed942d07cc9e2e`;
- service SHA: `4bf5328d4256db8428d7fd8db96a6346a07abd6fc93c9920ff3e7bbb51c1e1fb`;
- raw SHA: `2758925bc575dab994c70cf94f7e43a8d269d02d1759b10e05b20694cdac234a`;
- remapped adapter SHA: `fbaa39dcb3fcf34e9aab280308cb5a5416094c1968e4ac3a69cd739a806ecc49`.
