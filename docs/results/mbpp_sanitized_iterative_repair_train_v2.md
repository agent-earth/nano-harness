# MBPP Sanitized Iterative Repair Train v2

## Verdict

**SUPPORTED.**

This is method-development evidence on the 120-case train split, not an MBPP
validation or test score.

| Comparison | Candidate | Baseline | Delta | 95% CI | McNemar p | Wins / Losses |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| vs 4B direct | 108/120 | 95/120 | +0.1083 | [+0.0583, +0.1667] | 0.000244141 | 13 / 0 |
| vs 9B direct | 108/120 | 97/120 | +0.0917 | [+0.0417, +0.1500] | 0.000976562 | 11 / 0 |

The harness generated 125 replicas and
38 repairs, making
18 strictly test-improving overrides. Reference
solutions remained hidden; only official public assertions and deterministic
sandbox outcomes were used.

No rerun or tuning is allowed on this train surface.
