# Qwen3.5 Router Adapter Integration v1 Result

## Verdict

**REJECT.**

This report evaluates the exact SFT router adapter on a fresh 128-case
history-disjoint integration. The adapter only chooses A/B/C; the unchanged
base 4B performs typed execution, while C preserves direct output.

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
  "four_b_router_adapter": {
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
  "correct": 64,
  "fallbacks": 64,
  "negative_c_correct": 0,
  "negative_cases": 64,
  "negative_false_positive_routes": 64,
  "positive_cases": 64,
  "positive_correct": 64,
  "verified_executions": 64
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

## Evidence

- config SHA: `4eb7000201530ecb2ced96f4b1d490d115f4c1e1c6a6cb008cf64d5dc403d4c4`;
- prereg SHA: `ed5c4e6800385e7a4bfce0aed027bd1f81a6854bb1ed5b3f6aa0cc6e808491f3`;
- service receipt SHA: `e1cc845c5763c93918ac3a746f097f8f18ba46cd5da7bbf3116580908b439ead`;
- raw result SHA: `78a3a9564b444731cf409176b8fbec5983e8550b0ba9f4023ccccb80e9b4258a`;
- adapter SHA: `48d72666cf269f64825791801c71aad5ae1d9e97cbc70574c216b12974e46e63`.

Next action: Reject this adapter integration and publish the failed gates. Do not tune or rerun on these observed cases.
