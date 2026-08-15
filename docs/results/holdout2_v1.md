# Draft-Verify Holdout2 Result

## Evidence Integrity Correction

The aggregate 4B-versus-9B conclusion below is invalid. Code revision
`cf2a00e` accidentally sent `case.draft_prompt` through the direct runner
while the validator still checked `case.prompt`. For answer-only MMLU and
GPQA, both direct arms therefore used a reasoning prompt with a 32-token
budget and truncated all 48 outputs. The treatment observations and raw hashes
are retained as negative evidence, but they cannot establish uplift over a
matched direct control.

The GSM8K comparison remains valid because `prompt` and `draft_prompt` are
identical reasoning contracts for GSM8K. It shows 4B draft-verify at 22/24 and
9B direct at 23/24. A corrected direct runner and fresh cases are required.

## Primary Result

The originally reported, now invalid aggregate comparison was: 4B draft-verify
scores
0.6806 versus 9B direct at
0.3194. The paired micro delta is
+0.3611, 95% bootstrap CI
[+0.2500, +0.4722], exact McNemar
`p=0.0000002161`.

| Benchmark | 4B direct | 4B draft-verify | 9B direct |
| --- | ---: | ---: | ---: |
| GSM8K | 0.9167 | 0.9167 | 0.9583 |
| MMLU | 0.0000 | 0.6667 | 0.0000 |
| GPQA-Diamond | 0.0000 | 0.4583 | 0.0000 |
| Macro | 0.3056 | 0.6806 | 0.3194 |

Do not interpret this aggregate lead as model or harness quality. It is
explained by the direct-control prompt/budget mismatch.

## Acceptance Decision

Harness-stage acceptance is not yet satisfied because the pre-registered
task-group non-regression criterion fails on GSM8K: 4B draft-verify scores
22/24 while 9B direct scores 23/24. The observed -1 case delta has a bootstrap
interval including zero, but the criterion cannot be relaxed after seeing
holdout2.

The corrected next experiment is benchmark-aware routing on fresh cases using
the repaired runner and a validator-matched prompt contract.

## Cost

- 4B direct: 20200 tokens,
  268.9s.
- 4B draft-verify: 36697 tokens,
  173.1s.
- 9B direct: 19225 tokens,
  224.0s.

## Reproduction Identity

- Code revision: `33912848bbdd3ff7339c45018688c95f6af48925`
- 4B direct raw SHA256: `9cb709f0019210ae8bb608d231b58a2482f3e6a876f9a622cc855a2a7685e874`
- 4B treatment raw SHA256: `179c57f50c8b1ff0773b28ea72eb87f87464f0dcb3dfda3f15ccbf45a96654a8`
- 9B direct raw SHA256: `63f0741d3fbabee2e935adcecb2f63b0b6b87ef86b2b53aef9115512ac70bb11`

Raw outputs remain local and ignored.
