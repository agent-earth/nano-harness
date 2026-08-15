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

## Result

Decision gate scores 18/24 versus both 4B direct and 9B direct at 20/24.
Relative to 4B direct it has zero treatment-only wins and two direct-only
losses, delta -0.0833, bootstrap 95% CI [-0.2083, 0.0000].

The gate emits valid `KEEP`/`USE_RESOLVE` output on all cases. It selects
`USE_RESOLVE` three times: zero repairs, two regressions, and one neutral.
Thus formatting separation works, but LLM arbitration is not a reliable
correctness signal.

Stop all learned math arbitration. Fresh dev12 must use deterministic numeric
majority over protected direct and two independent re-solves; select a
non-direct answer only when at least two numeric predictions agree, otherwise
keep direct.

- [`docs/results/gsm8k_dev11_gate_v1.md`](../results/gsm8k_dev11_gate_v1.md)
- [`docs/results/gsm8k_dev11_gate_v1.public.json`](../results/gsm8k_dev11_gate_v1.public.json)
