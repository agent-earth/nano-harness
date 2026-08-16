# Large Independent 4B/9B Confirmation Result

## Official Result

| Benchmark | Qwen3.5-4B | Qwen3.5-9B | Delta | Paired 95% CI | McNemar p |
| --- | ---: | ---: | ---: | ---: | ---: |
| gsm8k | 235/256 (0.9180) | 241/256 (0.9414) | -0.0234 | [-0.0508, +0.0000] | 0.146 |
| mmlu | 180/256 (0.7031) | 162/256 (0.6328) | +0.0703 | [+0.0195, +0.1211] | 0.01135 |

- 4B macro accuracy: 0.8105;
- 9B macro accuracy: 0.7871;
- macro delta: +0.0234.

Across all 512 matched cases, 4B scores
415/512 (0.8105) and
9B scores 403/512
(0.7871). The paired micro delta is
+0.0234, with 95% bootstrap CI
[-0.0059,
+0.0527] and exact McNemar
`p=0.148006`.

There are 35 4B-only wins and
23 9B-only wins. MMLU significantly
favors 4B under the official strict answer contract, while GSM8K favors 9B.
The aggregate interval crosses zero and the per-task non-regression rule
fails.

## Non-Scoring Format Diagnostic

The official results above are unchanged. All
57 9B MMLU parse failures have the form
`FINAL <letter>` without the required colon. Of those letters,
29 match the reference and
28 do not.

If a colon-only normalization were applied hypothetically, 9B MMLU would be
191/256
(0.7461), compared with
the official 4B result of 180/256. This value is diagnostic only and is not
used in any score, confidence interval, p-value, or decision. It shows that
the official MMLU advantage primarily measures answer-contract compliance,
not stable semantic superiority.

## Parse And Cost

- 4B: 5 parse failures,
  5 length truncations,
  131258 tokens,
  2254.4s summed request latency;
- 9B: 63 parse failures,
  6 length truncations,
  128156 tokens,
  2065.7s summed request latency;
- both arms: zero API errors.

## Contract Audit

Both arms contain exactly the committed 512 unique case IDs. Prompt hashes,
direct-stage input hashes, strategies, dataset versions, scorers, and budgets
match the pre-registration. Raw outputs remain local and ignored.

## Decision

The confirmation fails the pre-registered superiority rule: aggregate
significance and GSM8K non-regression both fail. Stop direct-only confirmation.
Separate format-compliance examples from semantic discordances before any
harness or training ablation.

## Reproduction Identity

- Pre-registration/code revision: `8776f7b610f8cbb9e8c2716f7cd87b4e828978cd`
- 4B raw SHA256: `977a99e978936fbcfc99d45d859a8324746215b3ba518ed3bc5c4a2b5990b33e`
- 9B raw SHA256: `07b442f620245379444f02d633c76502032df84dfe5ad0b74d88996d3c0103bd`
