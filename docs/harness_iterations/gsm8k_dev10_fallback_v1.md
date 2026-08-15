# GSM8K Dev10 Protected Fallback Protocol

## Frozen Change

Keep dev9 direct, independent re-solve, arbiter prompts, and 600/384/64 token
budgets unchanged. Add one deterministic fallback:

- score the arbiter output under the numeric `FINAL:` contract;
- if it is unparseable and protected direct is parseable, return
  `FINAL: <protected direct answer>`;
- otherwise keep the arbiter output unchanged.

Record raw arbiter output/hash and whether fallback fired. Do not rescore dev9.

## Fresh Slice

GSM8K dev10 uses `start: 360, limit: 24`, the first continuous 24-case window
with unique IDs and zero overlap against all historical/sealed manifests.

Three matched arms:

1. Qwen3.5-4B direct;
2. Qwen3.5-4B protected re-solve with strict fallback;
3. Qwen3.5-9B direct.

## Directional Decision

Promote only if treatment:

- has higher point accuracy than 4B direct;
- has at least one treatment-only win and zero direct-only losses;
- has no API or final parse errors;
- has protected-direct predictions identical to the direct arm;
- passes all case, strategy, stage-input, and raw-output audits.

Report fallback count separately. Do not tune on dev10.

## Result

Treatment and 4B direct both score 20/24; 9B direct scores 21/24. Treatment and
4B direct have identical correctness and predictions on all 24 cases. The
strict fallback fires 10 times and eliminates final parse failures, but creates
no repair.

Among the fallback cases, protected direct is correct in seven and the
independent re-solve is correct in eight. In two cases protected is wrong while
re-solve is correct; the raw arbiter text identifies the protected
contradiction but truncates before a numeric `FINAL:`. Thus unconditional
protected fallback is safe but discards real repair evidence.

Dev10 fails. Fresh dev11 must separate decision from formatting: an 8-token
gate outputs only `KEEP` or `USE_RESOLVE`, and deterministic code emits the
selected numeric final. Do not rescore dev10.

- [`docs/results/gsm8k_dev10_fallback_v1.md`](../results/gsm8k_dev10_fallback_v1.md)
- [`docs/results/gsm8k_dev10_fallback_v1.public.json`](../results/gsm8k_dev10_fallback_v1.public.json)
