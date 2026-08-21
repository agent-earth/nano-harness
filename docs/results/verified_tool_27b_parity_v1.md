# Verified-Tool 27B Parity v1

## Verdict

**PARITY ADMITTED.**

| Scope | 4B Harness | 27B Direct | Delta | 95% CI |
| --- | ---: | ---: | ---: | --- |
| overall | 1.0000 | 0.2461 | +0.7539 | [+0.6992, +0.8047] |
| box_total | 1.0000 | 0.0000 | +1.0000 | [+1.0000, +1.0000] |
| remaining_stock | 1.0000 | 0.0000 | +1.0000 | [+1.0000, +1.0000] |
| paired_average | 1.0000 | 0.9844 | +0.0156 | [+0.0000, +0.0469] |
| labor_total | 1.0000 | 0.0000 | +1.0000 | [+1.0000, +1.0000] |

The frozen 4B harness result was reused without model generation. Only the 27B
direct arm was generated. Parity requires the overall and every-family 95%
paired-bootstrap lower bounds to be at least -0.02.

This is a complete local synthetic capability benchmark, not an external
benchmark score. It contains no MMLU, GSM8K, GPQA, MBPP, canary, or holdout
rows or outputs.
