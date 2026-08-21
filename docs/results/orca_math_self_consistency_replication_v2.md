# Orca Math Self-Consistency Replication v2

## Verdict

**REJECT.**

| Comparison | Candidate | Baseline | Delta | 95% CI | McNemar p | Wins / Losses |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| vs 4B direct | 76/160 | 73/160 | +0.0187 | [+0.0000, +0.0437] | 0.25 | 3 / 0 |
| vs 9B direct | 76/160 | 65/160 | +0.0688 | [+0.0125, +0.1250] | 0.0346897 | 17 / 6 |

## Gate Failure

The exact frozen policy preserves 4B: 3 wins, 0 losses, non-negative bootstrap
lower bound, and no 4B stratum regression. It also beats 9B overall
significantly. However, the long stratum scores 6/40 versus 9B 8/40, so the
pre-registered per-stratum non-regression gate fails. Complete benchmark access
remains closed.

## Descriptive Pool

Across v1 plus replication, candidate scores
124/256 versus 9B
101/256, delta
+0.0898, 95% CI
[+0.0430,
+0.1367], McNemar
`p=0.000430857`. This pooled view was not a
pre-registered replication gate and cannot override the formal rejection.

No rerun or tuning is allowed on either observed surface.
