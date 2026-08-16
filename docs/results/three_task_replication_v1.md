# Three-Task 4B/9B Replication Result

## Result

| Benchmark | Qwen3.5-4B | Qwen3.5-9B | Delta | Paired 95% CI | McNemar p |
| --- | ---: | ---: | ---: | ---: | ---: |
| gsm8k | 90/96 (0.9375) | 89/96 (0.9271) | +0.0104 | [-0.0417, +0.0625] | 1 |
| mmlu | 67/96 (0.6979) | 58/96 (0.6042) | +0.0938 | [-0.0104, +0.2083] | 0.136 |
| gpqa_diamond | 6/19 (0.3158) | 4/19 (0.2105) | +0.1053 | [-0.1579, +0.3684] | 0.6875 |

- 4B macro accuracy: 0.6504;
- 9B macro accuracy: 0.5806;
- macro delta: +0.0698.

Across all 211 matched cases, 4B scores
163/211 (0.7725) and
9B scores 151/211
(0.7156). The paired micro delta is
+0.0569, with 95% bootstrap CI
[+0.0000,
+0.1185] and exact McNemar
`p=0.088430`.

There are 27 4B-only wins and
15 9B-only wins. Every task point
estimate favors 4B, so the holdout5 direction replicates without task
regression. The confidence interval lower bound is exactly zero and the
McNemar p-value exceeds 0.05, so the pre-registered significance rule does not
pass.

## Parse And Cost

- 4B: 2 parse failures,
  2 length truncations,
  55313 tokens,
  910.8s summed request latency;
- 9B: 33 parse failures,
  3 length truncations,
  54028 tokens,
  840.2s summed request latency;
- both arms: zero API errors.

## Contract Audit

Both arms contain exactly the committed 211 unique case IDs. Prompt hashes,
direct-stage input hashes, strategies, dataset versions, scorers, and budgets
match the pre-registration. Raw outputs remain local and ignored.

## Decision

The replication does not satisfy every pre-registered superiority condition.

The result is a replicated directional 4B advantage, not statistically
significant superiority. Preserve the 27 4B-only and 15 9B-only discordances
as versioned data/verifier inputs for training, rather than continuing
post-hoc prompt search on this sample.

## Reproduction Identity

- Pre-registration/code revision: `2206fe7bc490ffd5d8689380ed01da38256b8ddd`
- 4B raw SHA256: `c59383d3fd3d6087025d6e1ff649979d9d5a9e8dc73b5429a4f8e9fa41b6b8c7`
- 9B raw SHA256: `ffae93774d51b87a2e29258d170a84f8b165f996e2e78eedd102271dfc260044`
