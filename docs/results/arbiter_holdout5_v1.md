# Conservative Arbiter Holdout5 Result

## Result

| Benchmark | 4B direct | 4B routed | 9B direct |
| --- | ---: | ---: | ---: |
| GSM8K | 0.9583 | 0.9583 | 1.0000 |
| MMLU | 0.6667 | 0.6667 | 0.4583 |
| GPQA-Diamond | 0.4583 | 0.4583 | 0.3750 |
| Macro | 0.6944 | 0.6944 | 0.6111 |

Routed 4B versus 9B direct has paired micro delta
+0.0833, 95% bootstrap CI
[-0.0139,
+0.1806], exact McNemar
`p=0.14599609`.

The GPQA arbiter overrides protected direct on
2 cases:
1 improve correctness and
1 reduce correctness.

## Contract Audit

All 72 identities and routed stages match the committed protocol. Routed
GSM8K/MMLU predictions and GPQA protected-direct predictions match the 4B
direct control before arbitration. Raw outputs remain local and ignored.

## Decision

Holdout5 does not satisfy every pre-registered harness acceptance rule.

## Reproduction Identity

- Code revision: `6a3f5881d9d5a77e1ec1a167fb3ce5044db8fb0c`
- 4B direct raw SHA256: `99f1ed96b35d12fb00c3030b1fceb8f035277862320b60551a670dd83e98c9d2`
- 4B routed raw SHA256: `0f93cd131b71ae6d04be156f25f85956a8fcf51a76888b33c85b0a42556ab857`
- 9B direct raw SHA256: `cdf5dffaca1d8e699c4dfbea269462d933e03bdff7ef056d4bd07c160f669e2b`
