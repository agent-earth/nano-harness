# GSM8K Dev9 Protected Re-solve Result

## Result

- 4B direct: 0.8750;
- 4B protected re-solve: 0.7083;
- 9B direct: 0.9583.

Treatment versus 4B direct is -0.1667, 95% bootstrap CI
[-0.3750,
+0.0000], with
1 treatment-only wins and
5 direct-only losses.

The independent re-solve disagrees with direct on
4 cases. The arbiter overrides on
8 cases:
1 wins,
5 losses, and
2 neutral.

Treatment uses 29549 tokens and
470.6s.

All 7 final parse failures are 64-token
arbiter length truncations. Their protected direct answers are correct on
5 cases; the independent re-solve
is correct on 5 cases; both are wrong
on 1 case.

## Contract Audit

Protected-direct predictions match the independent 4B direct arm for all
cases. Re-solve independence, stage inputs, and raw arbiter hashes match the
committed protocol. Raw outputs remain local and ignored.

## Decision

Dev9 fails at least one directional promotion rule.

No dev9 output is rescored. The next fresh experiment keeps all prompts and
budgets unchanged and deterministically falls back to the protected direct
answer only when the arbiter final is unparseable.

## Reproduction Identity

- Code revision: `928211ca570badd04bec6ead524946149e9e05ed`
- 4B direct raw SHA256: `ef146720b9bd8bd32a8496a7987f3595f6cb57d14ef2ec5bbe6b516fb5248f41`
- 4B treatment raw SHA256: `6b614b5c43b2e7528140dbbd9236c1c183ae6aced15344809339e78a2a23db61`
- 9B direct raw SHA256: `e42d8f96a4ab45ddf46c1deb9f7ee2c3f84359807f9733d05343fe5a23b6a0b1`
