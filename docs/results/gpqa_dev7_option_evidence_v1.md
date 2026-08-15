# GPQA Dev7 Normalized Option Evidence Result

## Result

- 4B direct: 0.2500;
- 4B normalized option evidence: 0.2500;
- 9B direct: 0.2500.

Treatment versus 4B direct is +0.0000, 95% bootstrap CI
[-0.2500,
+0.2500], with
1 treatment-only win and
1 direct-only loss.

Treatment uses 26706 tokens and
193.3s. It completes
48 option evaluator calls, has
0 final parse failures, and applies strict
bare-letter normalization 1 time.

## Contract Audit

All case identities, strategies, evaluator inputs, reconstructed selector
inputs, raw selector hashes, and normalizer settings match the committed
protocol. Raw outputs remain local and ignored.

## Decision

Dev7 fails the directional promotion rule. Formatting reliability is repaired,
but the dev6 correctness uplift does not replicate: one direct-wrong case is
fixed and one direct-correct case is broken.

The next fresh hypothesis must preserve the direct answer as an explicit
candidate and allow option evidence to override it only through a separate
arbiter.

## Reproduction Identity

- Code revision: `ad38abec766de48d68cf012fea8cbb4181e165b1`
- 4B direct raw SHA256: `5d9dea271b6e49347a49cc6b905e6b1300215f6494a7b2d75066e522ed5ae684`
- 4B treatment raw SHA256: `06ae25180fa3b539cceca78da1efc41f8f5ea6e33be67744ffd2dff7eebf1977`
- 9B direct raw SHA256: `7abb4b9d79c58964d84b81a2f6cea84bfe33c19e84f912b58c6c1c29652f1941`
