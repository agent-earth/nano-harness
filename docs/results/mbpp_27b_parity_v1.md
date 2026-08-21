# MBPP 27B Parity v1

## Verdict

**PARITY REJECTED.**

| Arm | Correct | Accuracy |
| --- | ---: | ---: |
| Frozen 4B harness | 219/257 | 0.8521 |
| Qwen3.5-27B BF16 direct | 226/257 | 0.8794 |

- paired delta, 4B harness minus 27B: -0.0272;
- paired-bootstrap 95% CI:
  [-0.0700,
  +0.0156];
- noninferiority margin: -0.0200;
- candidate-only / 27B-only: 11 /
  18;
- exact McNemar p: 0.264931.

This is a complete 257-case MBPP parity comparison. The 4B arm is the frozen
one-shot test output and was not regenerated. The 27B arm is direct generation
from the validated BF16 TP=2 service. No benchmark row or output may enter
training, reward, verifier fitting, or post-observation tuning.
