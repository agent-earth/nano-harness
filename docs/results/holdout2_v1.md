# Draft-Verify Holdout2 Result

## Primary Result

On the pre-registered 72-case holdout2, 4B draft-verify scores
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

The overall lead is large and significant. MMLU and GPQA improve
significantly, and treatment final parse failures are zero.

## Acceptance Decision

Harness-stage acceptance is not yet satisfied because the pre-registered
task-group non-regression criterion fails on GSM8K: 4B draft-verify scores
22/24 while 9B direct scores 23/24. The observed -1 case delta has a bootstrap
interval including zero, but the criterion cannot be relaxed after seeing
holdout2.

The strategy remains frozen. The next experiment is a larger unseen
GSM8K-only confirmation, not another policy change.

## Cost

- 4B direct: 20200 tokens,
  268.9s.
- 4B draft-verify: 36697 tokens,
  173.1s.
- 9B direct: 19225 tokens,
  224.0s.

## Reproduction Identity

- Code revision: `49c39d90939304d559b4631532070044b6620c26`
- 4B direct raw SHA256: `9cb709f0019210ae8bb608d231b58a2482f3e6a876f9a622cc855a2a7685e874`
- 4B treatment raw SHA256: `179c57f50c8b1ff0773b28ea72eb87f87464f0dcb3dfda3f15ccbf45a96654a8`
- 9B direct raw SHA256: `63f0741d3fbabee2e935adcecb2f63b0b6b87ef86b2b53aef9115512ac70bb11`

Raw outputs remain local and ignored.
