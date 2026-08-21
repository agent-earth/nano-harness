# MBPP Full-Validation Confirmation v2

## Verdict

**REJECT.**

| Comparison | Candidate | Baseline | Delta | 95% CI | McNemar p | Wins / Losses |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| vs 4B direct | 36/47 | 27/47 | +0.1915 | [+0.0851, +0.3191] | 0.00390625 | 9 / 0 |
| vs 9B direct | 36/47 | 31/47 | +0.1064 | [-0.0213, +0.2340] | 0.179688 | 7 / 2 |

The frozen v2 harness generated 100 replicas
and 34 repairs, making
9 strictly test-improving overrides. Reference
solutions remained hidden.

This result covers 47 full-validation tasks disjoint from all prior MBPP
development rows. It is not the 257-case sanitized test score. No rerun or
post-observation tuning is allowed.
