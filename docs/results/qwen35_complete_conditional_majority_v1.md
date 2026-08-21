# Qwen3.5 Complete Conditional-Majority v1 Result

## Verdict

**REJECT.**

| Benchmark | Candidate | Direct 4B | Direct 9B | Delta vs 9B | 95% CI | McNemar p |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| gsm8k | 1220/1319 | 1204/1319 | 1243/1319 | -0.0174 | [-0.0288, -0.0061] | 0.0051524 |
| mmlu | 10273/14042 | 10273/14042 | 9066/14042 | +0.0860 | [+0.0786, +0.0936] | 1.24172e-112 |
| gpqa_diamond | 85/198 | 76/198 | 69/198 | +0.0808 | [+0.0051, +0.1566] | 0.0479403 |

## What Ran

Only GSM8K received new model calls: five 4B samples for each of 1,319 cases.
MMLU preserved frozen 4B direct output, and GPQA reused the frozen V5
conservative-consensus endpoint. GSM8K had 21 answer
replacements and 256 fallbacks.

GSM8K uses the target-blind recovered parser for candidate, 4B, and 9B.
Its second complete treatment attempt is judged at Bonferroni
`alpha=0.025`. The final three-benchmark family uses Holm-Bonferroni at
familywise `alpha=0.05`.

## Decision Boundary

This is sequential evidence. MMLU and GPQA were not rerun, and this is not an
independent three-benchmark replication or a 27B comparison. Raw outputs stay
local and may not enter training, reward, or verifier data. No rerun or
post-observation tuning is allowed.
