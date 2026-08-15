# GPQA Dev8 Conservative Arbiter Result

## Result

- 4B direct: 0.4167;
- 4B conservative arbiter: 0.5833;
- 9B direct: 0.3333.

Treatment versus 4B direct is +0.1667, 95% bootstrap CI
[+0.0000,
+0.4167], with
2 treatment-only wins and
0 direct-only losses.

The arbiter overrides protected direct on 2 cases:
2 improve correctness,
0 reduce correctness, and
0 are neutral.

Treatment uses 32235 tokens and
199.0s, with
0 final parse failures.

## Contract Audit

Protected-direct predictions match the independent 4B direct arm for all
cases. Case IDs, all six stage inputs, raw arbiter hashes, and strategy match
the committed protocol. Raw outputs remain local and ignored.

## Decision

Dev8 satisfies every directional promotion rule.

## Reproduction Identity

- Code revision: `ae25af62af9446144cf94807071ae5635ff26a5b`
- 4B direct raw SHA256: `d46d73a00f04f1a9cb15fcf1cf9354d0813e289deafceebdd88f7e23b86a2d8b`
- 4B treatment raw SHA256: `a80cc0d1e0e9963b3935e8c1e41937b0a81cfc341ada2ec1b25fa60b51364c88`
- 9B direct raw SHA256: `e8801abeff2593f339c43bab93ebdb5cd75d9eb2c0d462093111b099624ca914`
