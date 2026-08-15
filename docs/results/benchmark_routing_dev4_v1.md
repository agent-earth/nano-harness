# Benchmark Routing Dev4 Result

## Result

| Benchmark | 4B direct | 4B routed | 9B direct |
| --- | ---: | ---: | ---: |
| GSM8K | 0.8333 | 0.8333 | 1.0000 |
| MMLU | 0.8333 | 0.5000 | 0.6667 |
| GPQA-Diamond | 0.1667 | 0.3333 | 0.5000 |
| Macro | 0.6111 | 0.5556 | 0.7222 |

Against 9B direct, routed 4B has paired micro delta -0.1667, 95% bootstrap CI [-0.3889, +0.0556], and exact McNemar `p=0.37500000`.

## Contract Audit

All case identities, selected routes, and actual direct/draft stage input hashes match the committed manifests. Raw outputs remain local and ignored.

## Decision

Dev4 rejects the routed policy; holdout4 must not run.

## Failure Analysis

Relative to 4B direct, routed execution has 1 GPQA-only win and 2 MMLU direct-only wins. GPQA draft truncations are 6/6; MMLU draft truncations are 1/6.

The next fresh-slice hypothesis keeps GSM8K and MMLU direct and tests a larger reasoning draft only for GPQA. Holdout4 remains unread because dev4 failed its pre-registered promotion rule.

## Reproduction Identity

- Code revision: `7945a73a3c1714fc6f55182443286ef163224fe3`
- 4B direct raw SHA256: `eb02d12b1641dc8f932b871788d465a56cb3b7dcdfb45ccfc15c1af2d60fb9fe`
- 4B routed raw SHA256: `de8794b7fd2be7101dcff0dcc9fb24fcb19b609d60c56e1ad9642a9e70f1cffc`
- 9B direct raw SHA256: `c9fdbdb9077fb6cc5af4b2618d7ddda136b1e1652b77da6389c8f8fb64beffd3`
