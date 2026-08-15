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
