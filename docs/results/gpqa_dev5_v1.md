# GPQA Dev5 v1 Result

## Result

- 4B direct: 0.2500;
- 4B 384-token draft-verify: 0.2500;
- 9B direct: 0.3333.

Treatment versus 4B direct is +0.0000, 95% bootstrap CI
[+0.0000,
+0.0000], with
0 treatment-only wins and
0 direct-only losses.

Treatment uses 15077 tokens and
180.0s. Draft truncations are
10/12; final parse
failures and API errors are 0 and
0.

Treatment and 4B direct have identical correctness on
12/12 cases and identical predictions on
11/12. The only changed prediction is
wrong in both arms. Neither of the two non-truncated drafts changes the direct
prediction.

## Contract Audit

All three arms match committed case identities, selected strategies, and
actual direct/draft stage input hashes. Raw outputs remain local and ignored.

## Decision

Dev5 fails at least one directional promotion rule.

The next action is: test independent per-option evidence on fresh cases rather
than increasing a monolithic draft budget again.

## Reproduction Identity

- Code revision: `9b8acb9a7728644f501b0420b21ba6eb17de7833`
- 4B direct raw SHA256: `05477661a5473aeaa5281cf5f6d867a5a0fc536c8cb4ee932190c3ca25182c69`
- 4B treatment raw SHA256: `a7c513481dfb03b9841895e59d25b61bdcd4a83c5a3d900d5c493e3b5a897c8a`
- 9B direct raw SHA256: `d8ca194643c369f72e9831b9c514930efdae17c93a19daac034c9b1eda308f1f`
