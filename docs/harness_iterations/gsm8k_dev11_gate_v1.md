# GSM8K Dev11 Decision Gate Protocol

## Hypothesis

Keep dev10 direct and independent re-solve prompts and 600/384 budgets frozen.
Replace the explanatory numeric arbiter with an 8-token decision gate:

- exact `USE_RESOLVE` selects a parseable re-solve number;
- exact `KEEP`, any other text, truncation, or missing re-solve number selects
  protected direct;
- code deterministically emits `FINAL: <selected number>`.

This separates model judgment from final formatting and fails closed to direct.

## Fresh Slice

GSM8K dev11 uses `start: 384, limit: 24`, the first continuous 24-case window
with unique IDs and zero overlap against all historical/sealed manifests.

Three matched arms:

1. Qwen3.5-4B direct;
2. Qwen3.5-4B protected direct plus re-solve and decision gate;
3. Qwen3.5-9B direct.

## Directional Decision

Promote only if treatment:

- has higher point accuracy than 4B direct;
- has at least one treatment-only win and zero direct-only losses;
- has no API or final parse errors;
- has protected-direct predictions identical to direct;
- passes case, strategy, stage-input, gate-output, and deterministic-selection
  audits.

Report gate decisions, invalid gate outputs, wins, losses, tokens, and time.
Do not tune on dev11.
