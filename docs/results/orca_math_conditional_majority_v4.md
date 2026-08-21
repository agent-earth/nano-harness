# Orca Math Conditional-Majority v4

## Verdict

**ADMIT TO PRE-REGISTRATION.**

This is a fresh local development gate, not a complete benchmark score.

| Comparison | Candidate | Baseline | Delta | 95% CI | McNemar p | Wins / Losses |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| vs recovered 4B direct | 58/96 | 54/96 | +0.0417 | [+0.0104, +0.0833] | 0.125 | 4 / 0 |
| vs recovered 9B direct | 58/96 | 46/96 | +0.1250 | [+0.0521, +0.2083] | 0.00418091 | 14 / 2 |

## What The Harness Does

The parser first reads a strict `FINAL:` answer and otherwise recovers the last
numeric token from the final 1,500 characters. The candidate then asks the 4B
model for five stochastic solutions:

- if the direct answer had no strict `FINAL:`, a 3-of-5 agreement may replace
  the recovered direct answer;
- if the direct answer already had a strict `FINAL:`, replacement requires
  unanimous 5-of-5 agreement;
- without the required agreement, the candidate keeps the recovered direct
  answer.

This policy made 6 actual answer replacements and
fell back to direct on 42 cases. It routed
41 strict-parse failures through the
3-vote threshold and 55 strict-parseable
cases through the 5-vote threshold.

## Why It Passed

Against matched 4B direct, the candidate gains four cases and loses none. Its
paired bootstrap lower bound is positive, and every length stratum improves:
short +0.0417, medium
+0.0417, long
+0.0417.

Against matched 9B direct, it gains 12 net cases, with
14 wins and
2 losses. The paired interval excludes
zero and exact McNemar is `p=0.00418091`. No stratum
regresses: short +0.0000, medium
+0.2083, long
+0.0833.

All pre-registered preservation and superiority gates pass. The existing raw
files were rendered offline; no model request was repeated.

## Decision Boundary

The frozen parser and conditional-majority policy may now be pre-registered
for one matched complete benchmark treatment. This result does not itself
prove superiority on GSM8K, MMLU, GPQA, SWE-bench, or any other complete
benchmark. No rerun or tuning is allowed on these 96 observed cases.
