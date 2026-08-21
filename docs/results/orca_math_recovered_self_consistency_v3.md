# Orca Math Recovered Self-Consistency v3

## Verdict

**REJECT.**

| Comparison | Candidate | Baseline | Delta | 95% CI | McNemar p | Wins / Losses |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| vs recovered 4B direct | 61/96 | 61/96 | +0.0000 | [-0.0417, +0.0417] | 1 | 2 / 2 |
| vs recovered 9B direct | 61/96 | 52/96 | +0.0938 | [+0.0104, +0.1771] | 0.0635681 | 14 / 5 |

## What Changed

All arms use the same target-blind parser: strict `FINAL:` first, otherwise the
last numeric token from the final 1,500 characters. This removed parse failures
without using references. The candidate stayed level with 4B direct, but its
paired bootstrap interval versus 4B still crosses below zero.

Against 9B, candidate leads by 9 cases and has a positive bootstrap interval,
but exact McNemar is `p=0.0635681` and the short stratum
regresses by one case. The strict gate therefore fails.

The 672 model requests completed before a metadata-only return-field error.
All three raw arms and receipts contain the exact 96 pre-registered IDs.
Finalization read those existing files and made no additional model request.

No rerun or tuning is allowed on this surface.
