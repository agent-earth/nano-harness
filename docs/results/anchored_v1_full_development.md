# Anchored v1 Full Development Result

| Benchmark | Anchored v1 | Base 4B | 9B |
| --- | ---: | ---: | ---: |
| GSM8K | 91/96 | 90/96 | 89/96 |
| MMLU | 66/96 | 67/96 | 58/96 |
| GPQA-Diamond | 7/19 | 6/19 | 4/19 |

Anchored v1 scores 164/211, versus base 4B at 163/211 and 9B at
151/211.

Versus base 4B, micro delta is
+0.0047 with 95% CI
[-0.0142,
+0.0237] and
McNemar p=1.000. MMLU is one
case lower, so the frozen per-task non-regression gate fails.

Versus 9B, micro delta is +0.0616, 95% CI
[+0.0047,
+0.1232], and McNemar
p=0.066. The CI is above zero, but p does not pass the
pre-registered 0.05 gate.

Do not open the independent holdout. Merge, scale, and RL remain forbidden.
