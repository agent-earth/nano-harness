# Qwen3.5 Router Adapter Integration v3 Result

## Verdict

**REJECT.**

V3 uses the new negative-diversity adapter on 160 history-disjoint answer-task
prompts: A/B plus all eight C subtypes. V1/V2 were not rerun.

## Arms

```json
{
  "four_b_direct": {
    "accuracy": 0.18125,
    "by_family": {
      "box_total": {
        "cases": 16,
        "correct": 0,
        "parseable": 16
      },
      "first_strict_profit_period": {
        "cases": 16,
        "correct": 0,
        "parseable": 16
      },
      "implicit_scale_total": {
        "cases": 16,
        "correct": 0,
        "parseable": 16
      },
      "paired_average": {
        "cases": 16,
        "correct": 6,
        "parseable": 16
      },
      "percentage_change": {
        "cases": 16,
        "correct": 14,
        "parseable": 16
      },
      "quotient_remainder": {
        "cases": 16,
        "correct": 2,
        "parseable": 16
      },
      "remaining_stock": {
        "cases": 16,
        "correct": 0,
        "parseable": 16
      },
      "single_operation": {
        "cases": 16,
        "correct": 7,
        "parseable": 16
      },
      "time_conversion": {
        "cases": 16,
        "correct": 0,
        "parseable": 16
      },
      "weighted_total": {
        "cases": 16,
        "correct": 0,
        "parseable": 16
      }
    },
    "cases": 160,
    "correct": 29,
    "parseable": 160
  },
  "four_b_router_adapter_v3": {
    "accuracy": 0.38125,
    "by_family": {
      "box_total": {
        "cases": 16,
        "correct": 0,
        "parseable": 16
      },
      "first_strict_profit_period": {
        "cases": 16,
        "correct": 16,
        "parseable": 16
      },
      "implicit_scale_total": {
        "cases": 16,
        "correct": 16,
        "parseable": 16
      },
      "paired_average": {
        "cases": 16,
        "correct": 6,
        "parseable": 16
      },
      "percentage_change": {
        "cases": 16,
        "correct": 14,
        "parseable": 16
      },
      "quotient_remainder": {
        "cases": 16,
        "correct": 2,
        "parseable": 16
      },
      "remaining_stock": {
        "cases": 16,
        "correct": 0,
        "parseable": 16
      },
      "single_operation": {
        "cases": 16,
        "correct": 7,
        "parseable": 16
      },
      "time_conversion": {
        "cases": 16,
        "correct": 0,
        "parseable": 16
      },
      "weighted_total": {
        "cases": 16,
        "correct": 0,
        "parseable": 16
      }
    },
    "cases": 160,
    "correct": 61,
    "parseable": 160
  },
  "nine_b_direct": {
    "accuracy": 0.25625,
    "by_family": {
      "box_total": {
        "cases": 16,
        "correct": 0,
        "parseable": 16
      },
      "first_strict_profit_period": {
        "cases": 16,
        "correct": 0,
        "parseable": 16
      },
      "implicit_scale_total": {
        "cases": 16,
        "correct": 0,
        "parseable": 16
      },
      "paired_average": {
        "cases": 16,
        "correct": 16,
        "parseable": 16
      },
      "percentage_change": {
        "cases": 16,
        "correct": 16,
        "parseable": 16
      },
      "quotient_remainder": {
        "cases": 16,
        "correct": 0,
        "parseable": 16
      },
      "remaining_stock": {
        "cases": 16,
        "correct": 0,
        "parseable": 16
      },
      "single_operation": {
        "cases": 16,
        "correct": 8,
        "parseable": 16
      },
      "time_conversion": {
        "cases": 16,
        "correct": 1,
        "parseable": 16
      },
      "weighted_total": {
        "cases": 16,
        "correct": 0,
        "parseable": 16
      }
    },
    "cases": 160,
    "correct": 41,
    "parseable": 160
  }
}
```

## Routing By Family

```json
{
  "box_total": {
    "cases": 16,
    "fallbacks": 0,
    "route_correct": 16,
    "selected_labels": {
      "C": 16
    },
    "verified_executions": 0
  },
  "first_strict_profit_period": {
    "cases": 16,
    "fallbacks": 0,
    "route_correct": 16,
    "selected_labels": {
      "B": 16
    },
    "verified_executions": 16
  },
  "implicit_scale_total": {
    "cases": 16,
    "fallbacks": 0,
    "route_correct": 16,
    "selected_labels": {
      "A": 16
    },
    "verified_executions": 16
  },
  "paired_average": {
    "cases": 16,
    "fallbacks": 0,
    "route_correct": 16,
    "selected_labels": {
      "C": 16
    },
    "verified_executions": 0
  },
  "percentage_change": {
    "cases": 16,
    "fallbacks": 0,
    "route_correct": 16,
    "selected_labels": {
      "C": 16
    },
    "verified_executions": 0
  },
  "quotient_remainder": {
    "cases": 16,
    "fallbacks": 0,
    "route_correct": 16,
    "selected_labels": {
      "C": 16
    },
    "verified_executions": 0
  },
  "remaining_stock": {
    "cases": 16,
    "fallbacks": 0,
    "route_correct": 16,
    "selected_labels": {
      "C": 16
    },
    "verified_executions": 0
  },
  "single_operation": {
    "cases": 16,
    "fallbacks": 0,
    "route_correct": 16,
    "selected_labels": {
      "C": 16
    },
    "verified_executions": 0
  },
  "time_conversion": {
    "cases": 16,
    "fallbacks": 0,
    "route_correct": 16,
    "selected_labels": {
      "C": 16
    },
    "verified_executions": 0
  },
  "weighted_total": {
    "cases": 16,
    "fallbacks": 0,
    "route_correct": 16,
    "selected_labels": {
      "C": 16
    },
    "verified_executions": 0
  }
}
```

## Paired Comparisons

| Comparison | Delta | 95% CI | Wins | Losses | McNemar p |
| --- | ---: | --- | ---: | ---: | ---: |
| candidate_vs_four_b | +0.2000 | [+0.1375, +0.2625] | 32 | 0 | 4.65661e-10 |
| candidate_vs_nine_b | +0.1250 | [+0.0437, +0.2062] | 34 | 14 | 0.0055152 |
| four_b_vs_nine_b | -0.0750 | [-0.1250, -0.0312] | 2 | 14 | 0.00418091 |

## Frozen Gates

```json
{
  "all_three_arms_complete_and_parseable_160": true,
  "candidate_vs_four_b_maximum_losses": true,
  "candidate_vs_four_b_minimum_wins": true,
  "candidate_vs_four_b_significant": true,
  "candidate_vs_nine_b_maximum_losses": false,
  "candidate_vs_nine_b_minimum_wins": true,
  "candidate_vs_nine_b_significant": true,
  "each_c_subtype_recall_16": true,
  "every_family_non_regression": false,
  "fallbacks_zero": true,
  "negative_candidate_exact_direct_parity": true,
  "negative_false_positive_routes_zero": true,
  "positive_feedback_result_matches_32": true,
  "positive_verified_executions_32": true,
  "router_a_recall_16": true,
  "router_b_recall_16": true,
  "router_c_recall_128": true,
  "router_outputs_parseable_160": true
}
```

## Boundaries

V1/V2/V3 cannot be rerun. Passing allows only a separately pre-registered
benchmark-agnostic treatment transfer. Benchmark generation, canary, holdout,
training, and RL remain closed.

## Evidence

- prereg SHA: `ff8a811e787e486241e24461f3b7a8f3c292a38bec82d7496d56a5733c38559b`;
- service SHA: `9a72fcb04391a62e4c1526a51112bc516de5c7251c6b5a00a085365a23e7bdb8`;
- raw SHA: `2f714c345778ab885859f35d7a06a06abbc7664caca2691517a7fe9b7d24984f`;
- remapped adapter SHA: `cea357d281ed100437268e213564fc5a5c00e6024b0c7a4be207cc686453e3f9`.
