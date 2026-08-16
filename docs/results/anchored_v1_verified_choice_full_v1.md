# Anchored-v1 Verified Choice Full Development v1 Result

## Result

The executor preserves all old-suite outputs but does not transfer:

| Benchmark | Candidate | Base 4B | 9B |
| --- | ---: | ---: | ---: |
| GSM8K | 91/96 | 90/96 | 89/96 |
| MMLU | 66/96 | 67/96 | 58/96 |
| GPQA-Diamond | 7/19 | 6/19 | 4/19 |

Candidate remains 164/211. All 115 choice prompts are
outside parser v1's narrow explicit arithmetic-average contract, so there are
zero overrides, 211/211 fallback parity, and zero regressions.

Versus base 4B, micro delta is +0.0047, 95% CI
[-0.0142,
+0.0237], and McNemar
p=1.000. MMLU remains one case below base, so
the frozen per-task non-regression gate fails.

Versus 9B, micro delta is +0.0616, 95% CI
[+0.0047,
+0.1232], and McNemar
p=0.066. The dual significance criterion still
does not pass.

## Decision

Do not open the independent holdout. Do not expand parser v1 from observed
benchmark prompts. Preserve the local verified-execution signal and replan
from fresh generic data. Merge, scale, and RL remain forbidden.

## Identity

- pre-registration revision: `33fa95b`;
- config SHA256: `27202f9c43325366c8cf35070e9d040b8f5ee52ce8745f1e15b6f3f7d732c1dd`;
- suite manifest SHA256: `08c71cae463bd3b0a0031e95d6339136d0c445beecaac631c4f5843e0b14d4c1`;
- anchored-v1 raw SHA256: `a8f6a731042c7b81c97196abd60d6c632006b7c59da1bbdb2328ab73c539def0`;
- canary pass SHA256: `e9edf209ace2d52516b38602d942812525f997543463a6abd2a7463340f68ccc`;
- raw applicator result SHA256: `93d6572570ff7c15c5f7827099e9c7128c9704f0a84e4db5fc8a25755254c5a2`.
