# Orca Math Self-Consistency v1 Result

## Verdict

**REJECT.**

| Comparison | Candidate | Baseline | Delta | 95% CI | McNemar p | Wins / Losses |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| vs 4B direct | 48/96 | 47/96 | +0.0104 | [+0.0000, +0.0312] | 1 | 1 / 0 |
| vs 9B direct | 48/96 | 36/96 | +0.1250 | [+0.0417, +0.2083] | 0.00753784 | 15 / 3 |

## Harness Behavior

- 5 full-solve replicas per case;
- require 4 agreeing numeric finals before override;
- overrides: 2/96;
- fallbacks to frozen 4B direct:
  51/96.

No rerun, prompt, replica-count, threshold, temperature, seed, parser, scorer,
or generation-budget change is allowed after this result.

## Boundary

Fresh non-benchmark local development only. This is not GSM8K, MMLU, GPQA,
9B-complete, 27B, or agent-benchmark evidence.
