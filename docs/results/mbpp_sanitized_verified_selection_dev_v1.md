# MBPP Sanitized Verified Selection Dev v1

## Verdict

**REJECT.**

| Comparison | Candidate | Baseline | Delta | 95% CI | McNemar p | Wins / Losses |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| vs 4B direct | 36/43 | 32/43 | +0.0930 | [+0.0233, +0.1860] | 0.125 | 4 / 0 |
| vs 9B direct | 36/43 | 35/43 | +0.0233 | [+0.0000, +0.0698] | 1 | 1 / 0 |

## What Ran

Direct 4B passed 32/43 and direct 9B
passed 35/43. Only the
11 direct-4B failures entered
the verifier-selection route. It generated
33 replicas and
7 aggregate-feedback repairs, then made
5 improvements over direct 4B.

The model saw the public MBPP assertions, matching the benchmark protocol, but
never saw reference solutions. Every assertion ran in a no-network bubblewrap
sandbox with a read-only root filesystem, isolated Python mode, per-test
timeout, CPU, address-space, file-size, and open-file limits. Repair received
the same public assertions plus only aggregate pass count and failure classes.

## Decision Boundary

This result covers all 43 sanitized validation tasks. The 257-case sanitized
test split remains untouched unless every pre-registered validation gate
passes. No validation rerun or post-observation tuning is allowed.
