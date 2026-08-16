# Anchored-v1 Choice Calculation Selector v1 Result

## Result

The harness is contract-safe but fails its frozen local improvement gate.

- baseline and candidate strict / semantic: 22/32 /
  25/32 and 22/32 /
  25/32;
- candidate numeric / choice / process semantic:
  11/16,
  6/8,
  8/8;
- selector regex compliance: 8/8;
- non-choice direct-output reuse: 24/24;
- choice fixes / regressions: 0 / 0;
- wall time: 107.4 seconds.

The only failed gate is choice >=7/8: choice remains 6/8.

## Mechanism Evidence

All eight calculation stages return only a `FINAL: <letter>` line instead of
the requested explicit arithmetic. The original answer-only instruction
dominates the calculation-stage system prompt. The selector therefore receives
no independent calculation to verify. One wrong choice moves from one wrong
option to another; no case is fixed or regressed.

This is evidence against the exact prompt protocol. Do not alter its prompt,
budget, regex, or retry policy after observing this development result.

## Decision

Reject this local harness and preserve anchored-v1. The sealed canary, old
full-development suite, and independent holdout were not run. The holdout
remains unread.

## Reproduction Identity

- pre-registration revision: `f747afe`;
- config SHA256: `8727fb9f3d97c860f551fac2046822feec64898e9fe5005271f64df8e247674a`;
- dataset SHA256: `4657e96af9f9d1b81bfdb5fac6a29c31baf24b23d7753508aafdea5603ffd80d`;
- serving receipt SHA256: `2549527942acfe53a1eb352453649a9ea3cc31d68bb9790c865553ee95c2f578`;
- serving adapter weights SHA256:
  `9ce7be3954f8e0f3d245fe846d6e35275243b7f0caf66cb847fd716173658649`;
- raw local result SHA256: `7f9ed333fbf75beea925d3de261587d5225521fa63fecc4bb68c267e9f6cc57e`.
