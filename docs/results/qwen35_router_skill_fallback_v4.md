# Qwen3.5 Router Skill Fallback v4 Result

## Verdict

**REJECT.**

V4 keeps the admitted router and A/B verifier, replacing only `C -> 4B direct`
with eight typed, deterministic skills on 160 history-disjoint prompts.

## Arms

```json
{
  "four_b_direct": {
    "accuracy": 0.25,
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
        "correct": 15,
        "parseable": 16
      },
      "percentage_change": {
        "cases": 16,
        "correct": 15,
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
        "correct": 8,
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
    "correct": 40,
    "parseable": 160
  },
  "four_b_router_skill_v4": {
    "accuracy": 0.8875,
    "by_family": {
      "box_total": {
        "cases": 16,
        "correct": 16,
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
        "correct": 2,
        "parseable": 16
      },
      "remaining_stock": {
        "cases": 16,
        "correct": 16,
        "parseable": 16
      },
      "single_operation": {
        "cases": 16,
        "correct": 12,
        "parseable": 16
      },
      "time_conversion": {
        "cases": 16,
        "correct": 16,
        "parseable": 16
      },
      "weighted_total": {
        "cases": 16,
        "correct": 16,
        "parseable": 16
      }
    },
    "cases": 160,
    "correct": 142,
    "parseable": 160
  },
  "nine_b_direct": {
    "accuracy": 0.275,
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
        "correct": 15,
        "parseable": 16
      },
      "percentage_change": {
        "cases": 16,
        "correct": 15,
        "parseable": 16
      },
      "quotient_remainder": {
        "cases": 16,
        "correct": 6,
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
    "correct": 44,
    "parseable": 160
  }
}
```

## Routing And Skills

```json
{
  "box_total": {
    "cases": 16,
    "fallbacks": 0,
    "router_correct": 16,
    "verified_executions": 16
  },
  "first_strict_profit_period": {
    "cases": 16,
    "fallbacks": 0,
    "router_correct": 16,
    "verified_executions": 16
  },
  "implicit_scale_total": {
    "cases": 16,
    "fallbacks": 0,
    "router_correct": 16,
    "verified_executions": 16
  },
  "paired_average": {
    "cases": 16,
    "fallbacks": 0,
    "router_correct": 16,
    "verified_executions": 16
  },
  "percentage_change": {
    "cases": 16,
    "fallbacks": 0,
    "router_correct": 16,
    "verified_executions": 16
  },
  "quotient_remainder": {
    "cases": 16,
    "fallbacks": 16,
    "router_correct": 16,
    "verified_executions": 0
  },
  "remaining_stock": {
    "cases": 16,
    "fallbacks": 0,
    "router_correct": 16,
    "verified_executions": 16
  },
  "single_operation": {
    "cases": 16,
    "fallbacks": 4,
    "router_correct": 16,
    "verified_executions": 12
  },
  "time_conversion": {
    "cases": 16,
    "fallbacks": 0,
    "router_correct": 16,
    "verified_executions": 16
  },
  "weighted_total": {
    "cases": 16,
    "fallbacks": 0,
    "router_correct": 16,
    "verified_executions": 16
  }
}
```

## C Skill Failures

```json
{
  "box_total": {
    "cases": 16,
    "executed": 16,
    "failure_reasons": {},
    "fallbacks": 0
  },
  "paired_average": {
    "cases": 16,
    "executed": 16,
    "failure_reasons": {},
    "fallbacks": 0
  },
  "percentage_change": {
    "cases": 16,
    "executed": 16,
    "failure_reasons": {},
    "fallbacks": 0
  },
  "quotient_remainder": {
    "cases": 16,
    "executed": 0,
    "failure_reasons": {
      "source_facts_mismatch": 16
    },
    "fallbacks": 16
  },
  "remaining_stock": {
    "cases": 16,
    "executed": 16,
    "failure_reasons": {},
    "fallbacks": 0
  },
  "single_operation": {
    "cases": 16,
    "executed": 12,
    "failure_reasons": {
      "source_facts_mismatch": 4
    },
    "fallbacks": 4
  },
  "time_conversion": {
    "cases": 16,
    "executed": 16,
    "failure_reasons": {},
    "fallbacks": 0
  },
  "weighted_total": {
    "cases": 16,
    "executed": 16,
    "failure_reasons": {},
    "fallbacks": 0
  }
}
```

The router remains correct on all 160 cases and A/B verified execution remains
32/32. The shared eight-skill selector executes 108/128 C cases; all 16
quotient cases and four single-operation cases fall back after typed
source-fact mismatch. The deterministic executors are not the failed
component.

## Paired Comparisons

| Comparison | Delta | 95% CI | Wins | Losses | McNemar p |
| --- | ---: | --- | ---: | ---: | ---: |
| candidate_vs_four_b | +0.6375 | [+0.5625, +0.7125] | 102 | 0 | 3.9443e-31 |
| candidate_vs_nine_b | +0.6125 | [+0.5250, +0.6937] | 102 | 4 | 1.27359e-25 |
| four_b_vs_nine_b | -0.0250 | [-0.0563, +0.0000] | 1 | 5 | 0.21875 |

## Frozen Gates

```json
{
  "ab_verified_executions_32": true,
  "all_three_arms_complete_and_parseable_160": true,
  "c_skill_result_exact_128": false,
  "c_skill_verified_executions_128": false,
  "candidate_vs_four_b_maximum_losses": true,
  "candidate_vs_four_b_minimum_wins": true,
  "candidate_vs_four_b_significant": true,
  "candidate_vs_nine_b_maximum_losses": false,
  "candidate_vs_nine_b_minimum_wins": true,
  "candidate_vs_nine_b_significant": true,
  "every_family_non_regression": false,
  "fallbacks_zero": false,
  "router_outputs_parseable_and_correct_160": true
}
```

## Boundaries

V1-V4 cannot be rerun. Passing permits only a separately pre-registered
benchmark treatment. Benchmark generation, canary, holdout, training, and RL
remain closed.

```json
{
  "ab_verified_execution_succeeded": true,
  "dominant_failures": {
    "quotient_remainder": 16,
    "single_operation": 4
  },
  "failed_c_skill_cases": 20,
  "post_observation_skill_prompt_or_schema_tuning_allowed": false,
  "router_transfer_succeeded": true,
  "shared_c_skill_selector_admitted": false
}
```

## Evidence

- prereg SHA: `00a1db518d93219b8fff86651adbc0bb5589522d19f1ebf81d4050103eb7f2ae`;
- raw SHA: `b92bcf4c4bbb56c5247aa2bacf54a9161b041e9252e344d41a9af74271359354`;
- V3 report SHA: `cfbd0edbc74739eb1d0a860c19cca2c07edfd52093d7ccf7f86e114f33a2ac03`.
