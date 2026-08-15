# GPQA Dev7 Option Evidence Protocol

## Frozen Change

Retain dev6 option decomposition exactly:

- four independent A-D evaluators;
- 96 tokens per evaluator;
- unchanged evaluator prompts;
- unchanged 64-token selector and selector prompt.

The only change is deterministic output normalization. If the stripped
selector output is exactly one letter A-D, convert it to `FINAL: <letter>`.
Do not normalize explanations, multiple letters, or any other text. Record the
raw output hash and whether normalization fired.

## Fresh Slice

GPQA dev7 uses `start: 120, limit: 12`, the first continuous 12-case window
with unique IDs and zero overlap against all committed historical manifests.

Three arms use identical cases:

1. repaired Qwen3.5-4B direct;
2. Qwen3.5-4B option evidence with strict normalizer;
3. repaired Qwen3.5-9B direct.

## Directional Decision

Promote to a newly pre-registered three-task holdout only if treatment:

- has higher point accuracy than 4B direct;
- has more treatment-only wins than direct-only losses;
- has no API or final parse errors;
- completes all 48 option evaluator calls;
- passes case, strategy, stage-input, and selector-input hash audits.

Do not tune on dev7. The previously frozen holdout4 stays sealed because its
route differs from this strategy.
