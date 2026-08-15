# GSM8K Dev9 Protected Re-solve Protocol

## Hypothesis

Direct 4B math is usually strong but still trails 9B on some fresh slices.
Preserve the full direct path as a protected candidate. Independently re-solve
with 384 tokens, then use a 64-token conservative math arbiter that overrides
only for a specific arithmetic, unit, rate, or interpretation contradiction.

The independent re-solve never sees direct reasoning. The arbiter sees only
the protected direct numeric answer and the independent re-solve, avoiding a
second copy of long direct reasoning.

## Fresh Slice

GSM8K dev9 uses `start: 336, limit: 24`, the first continuous 24-case window
with unique IDs and zero overlap against all committed historical and sealed
manifests.

Three matched arms:

1. Qwen3.5-4B direct;
2. Qwen3.5-4B protected direct plus independent re-solve and arbiter;
3. Qwen3.5-9B direct.

## Directional Decision

Promote only if treatment:

- has higher point accuracy than 4B direct;
- has at least one treatment-only win and zero direct-only losses;
- has no API or final parse errors;
- has protected-direct predictions identical to the 4B direct arm;
- passes all case, strategy, and stage-input hash audits.

Report override count, wins, losses, truncations, tokens, and wall time. Do not
tune on dev9.
