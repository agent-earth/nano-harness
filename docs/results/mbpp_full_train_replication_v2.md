# MBPP Full-Train Replication v2

## Verdict

**ADMIT TO TEST PRE-REGISTRATION.**

| Comparison | Candidate | Baseline | Delta | 95% CI | McNemar p | Wins / Losses |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| vs 4B direct | 211/254 | 180/254 | +0.1220 | [+0.0827, +0.1654] | 9.31323e-10 | 31 / 0 |
| vs 9B direct | 211/254 | 187/254 | +0.0945 | [+0.0551, +0.1378] | 8.43033e-06 | 27 / 3 |

The unchanged frozen v2 harness generated
370 replicas and
131 repairs, making
42 strictly test-improving overrides. Reference
solutions remained hidden.

This result covers 254 full-train tasks excluded from prior sanitized-train
development and disjoint from validation, test, and few-shot rows. It is not
the 257-case sanitized test score. No rerun or post-observation tuning is
allowed.
