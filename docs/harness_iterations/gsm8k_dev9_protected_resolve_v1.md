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

## Result

Protected re-solve scores 17/24 versus 4B direct at 21/24 and 9B direct at
23/24. Relative to 4B direct it has one treatment-only win and five
direct-only losses, delta -0.1667, bootstrap 95% CI [-0.3750, 0.0000].

The arbiter produces seven unparseable length-truncated finals. Protected
direct is correct in five of those cases, the independent re-solve is correct
in one protected-wrong case, and both are wrong in one case. Across all cases,
the arbiter makes eight overrides: one win, five losses, and two neutral.

Dev9 fails. Do not rescore it. On fresh dev10, keep prompts and budgets
unchanged and add deterministic protected-direct fallback only when the arbiter
final is unparseable.

- [`docs/results/gsm8k_dev9_protected_resolve_v1.md`](../results/gsm8k_dev9_protected_resolve_v1.md)
- [`docs/results/gsm8k_dev9_protected_resolve_v1.public.json`](../results/gsm8k_dev9_protected_resolve_v1.public.json)
