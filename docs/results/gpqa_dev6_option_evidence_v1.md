# GPQA Dev6 Option Evidence Result

## Result

- 4B direct: 0.2500;
- 4B option evidence: 0.5000;
- 9B direct: 0.4167.

Treatment versus 4B direct is +0.2500, 95% bootstrap CI
[+0.0000,
+0.5000], with
3 treatment-only wins and
0 direct-only losses.

Treatment uses 27820 tokens and
195.6s. It completed
48 option evaluator calls with
48 truncations.

## Contract Audit

All case identities, strategies, four evaluator stages per case, actual option
input hashes, and selector input hashes match the committed protocol. Raw
outputs remain local and ignored.

## Decision

Dev6 fails at least one directional promotion rule.

The only treatment parse failure is a stopped selector that returned the bare
letter `D`; it matches that case's reference but is scored wrong under the
frozen `FINAL:` contract. No post-hoc rescoring is applied. The next fresh
experiment keeps all option prompts and budgets unchanged and adds only strict
deterministic normalization for a selector output that is exactly one choice
letter.

## Reproduction Identity

- Code revision: `3462d03d8013228665871c1c2200ff53bb227210`
- 4B direct raw SHA256: `2455bea3d6dbc5d117d2d65e08449b5395cc25a4267fe9d6c5166cbfb7765afc`
- 4B treatment raw SHA256: `63c7c1f10fdfbe833cc6d42a221b9081b25ace5b4468b2e238d9e985f05ac15e`
- 9B direct raw SHA256: `fba7f94e15ad5c86129511f226114c6291e18a853bd95833ad99872cf4582776`
