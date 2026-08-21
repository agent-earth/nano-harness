# MBPP Sanitized Test v2

## Verdict

**COMPLETE BENCHMARK SUPERIORITY.**

| Comparison | Candidate | Baseline | Delta | 95% CI | McNemar p | Wins / Losses |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| vs 4B direct | 219/257 | 189/257 | +0.1167 | [+0.0778, +0.1556] | 1.86265e-09 | 30 / 0 |
| vs 9B direct | 219/257 | 198/257 | +0.0817 | [+0.0389, +0.1284] | 0.00050826 | 28 / 7 |

The frozen v2 harness generated 340 replicas
and 116 repairs, making
38 strictly public-test-improving overrides. Reference
solutions remained hidden.

This is the complete 257-case MBPP sanitized-test score under the one-shot
pre-registered protocol. Test rows and outputs remain forbidden from training,
reward, verifier fitting, and post-observation tuning.
