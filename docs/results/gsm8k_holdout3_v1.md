# GSM8K Dual-Solve Holdout3 Result

## Result

On 96 unseen GSM8K cases:

- 4B draft-verify: 0.9167;
- 4B dual-solve: 0.9583;
- 9B direct: 0.9792.

Dual-solve improves over draft-verify by +0.0417, with 95%
bootstrap CI [+0.0104,
+0.0833]. It has
4 dual-only wins and
0 draft-only wins.

Against 9B, dual-solve is -0.0208, 95% CI
[-0.0521,
+0.0000]. It scores 92/96 versus 94/96.

## Decision

Dual-solve is retained as evidence that independent re-solving repairs some
draft errors, but holdout acceptance fails:

- its point estimate remains below 9B;
- the CI lower bound (-0.0521) is slightly below the -0.05 non-inferiority
  margin;
- it uses 103551 tokens versus
  28769 for 9B direct.

The next experiment must use a fresh slice. Benchmark-aware routing is the next
falsifiable direction: preserve direct/reasoning behavior for math while using
draft-verify only where it has repeatedly improved MMLU and GPQA. No holdout3
tuning is allowed.

## Reproduction Identity

- Code revision: `1e95fe78b9db6df9facc853962d5d5757ae30651`
- Draft raw SHA256: `173a6810a854865e65e96e07c80fd521b3a620a4ee474a861dd9d23ad16cffaf`
- Dual raw SHA256: `22be7e0e62df58e6e1719875c8c4432450c5f1b406ac75843a572306c35d1647`
- 9B raw SHA256: `4044300c9cb7946c15d7d03bd7e8fb0cbe158af4644741ed7e0cdfbf5fef4c8e`

Raw stage texts remain local and ignored.
