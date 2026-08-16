# V6 Matched Adapter Evaluation Result

## Task Results

| Benchmark | V6 adapter | Base 4B | 9B |
| --- | ---: | ---: | ---: |
| gsm8k | 73/96 | 90/96 | 89/96 |
| mmlu | 68/96 | 67/96 | 58/96 |
| gpqa_diamond | 4/19 | 6/19 | 4/19 |

## Candidate Versus Base 4B

- candidate macro: 0.5598;
- base 4B macro: 0.6504;
- micro delta: -0.0853;
- paired 95% CI:
  [-0.1280,
  -0.0474];
- exact McNemar p: 0.000121;
- task non-regression: False;
- parse non-regression: False.

## Candidate Versus 9B

- candidate macro: 0.5598;
- 9B macro: 0.5806;
- micro delta: -0.0284;
- paired 95% CI:
  [-0.0948,
  +0.0379];
- exact McNemar p: 0.496617;
- task non-regression: False.

## GSM8K Failure Diagnostic

Official candidate GSM8K failures:
23/96.

- official parse failures:
  9;
- non-scoring inline `FINAL:` values matching the reference:
  8;
- parseable but numerically wrong outputs:
  14;
- base-4B-only correct cases:
  17.

The inline diagnostic does not change the official score. It separates output
contract regressions from genuine numeric or modeling regressions.

## Decision

- passes base 4B non-regression:
  False;
- significantly exceeds matched 9B:
  False;
- API errors across all arms: 0.

Regardless of outcome, this evaluation does not directly authorize merge,
scale-up, or RL. Preserve task-level discordances and serving parity evidence
for the next separately pre-registered ablation.

## Reproduction Identity

- pre-registration revision: `2250ed0`;
- serving parity revision: `f7875fe`;
- candidate raw SHA256:
  `a1e0d7dad5ed02b6a881357ccb7f315406d84477f489cd3c363570af7b379721`;
- base 4B raw SHA256:
  `c59383d3fd3d6087025d6e1ff649979d9d5a9e8dc73b5429a4f8e9fa41b6b8c7`;
- 9B raw SHA256:
  `ffae93774d51b87a2e29258d170a84f8b165f996e2e78eedd102271dfc260044`;
- source adapter weights SHA256:
  `1b2065129f368f6d3b72bbf875bbd0a2d2b7b97ab8b7c4ec7ca10c8155f343ea`;
- serving adapter weights SHA256:
  `057ec7aa2214e5fb35d8bb6afd88ec71b3d9e468dc12bd69ccc7a9e5c1c43d4d`.
