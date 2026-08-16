# Anchored-v1 Verified Choice Executor v1 Result

## Result

The target-blind verified executor passes every frozen local gate:

- baseline strict / semantic: 22/32 /
  25/32;
- candidate strict / semantic: 23/32 /
  26/32;
- candidate numeric / choice / process semantic:
  11/16,
  7/8,
  8/8;
- one choice fix, zero regressions;
- 27 fallback outputs remain byte-identical to direct;
- zero model calls and no target use during parsing.

## Mechanism Evidence

The fixed row contains expressions `27 * 4` and
`26 * 7`. Exact `Fraction` evaluation yields
108 and 182, whose
average is 145. Exactly one option has that value, so the
executor selects `B`. Fractional results with no
exact option remain on direct output; no rounding or nearest-option heuristic
is allowed.

## Decision

Passing authorizes only the old sealed 40-case regression canary. The old
211-case development suite, independent holdout, merge, scale, and RL remain
blocked.

## Reproduction Identity

- pre-registration revision: `41441c3`;
- config SHA256: `b6aa27605fdbf25db5a470cfe95f0fee704f1c249c9069ef6b12401afa5d5178`;
- dataset SHA256: `4657e96af9f9d1b81bfdb5fac6a29c31baf24b23d7753508aafdea5603ffd80d`;
- baseline result SHA256: `7f9ed333fbf75beea925d3de261587d5225521fa63fecc4bb68c267e9f6cc57e`;
- raw local result SHA256: `528fdbc6eaf03eeef87dc5dfa6dce4880184136a7fd14c86dc07755a15c2c0b0`.
