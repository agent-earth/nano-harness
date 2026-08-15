# GPQA Dev8 Conservative Arbiter Protocol

## Hypothesis

Unconditional option selection can both repair and break direct answers.
Generate the direct answer inside the treatment as a protected candidate, then
run the same four 96-token option evaluators. A 64-token conservative arbiter
preserves direct by default and overrides only when evidence identifies a
specific contradiction and a clearly stronger alternative.

Strict bare-letter normalization remains enabled. All protected-direct,
option-evaluator, and arbiter inputs and outputs are hashed and recorded.

## Fresh Slice

GPQA dev8 uses `start: 132, limit: 12`, the first continuous 12-case window
with unique IDs and zero overlap against all committed historical manifests.

Three matched arms:

1. repaired Qwen3.5-4B direct;
2. Qwen3.5-4B protected direct plus option evidence and arbiter;
3. repaired Qwen3.5-9B direct.

## Directional Decision

Promote only if treatment:

- has higher point accuracy than 4B direct;
- has more treatment-only wins than direct-only losses;
- has no API or final parse errors;
- completes protected direct, all 48 option calls, and all 12 arbiters;
- passes case, strategy, and all stage-input hash audits.

Report how often the arbiter overrides its protected direct candidate. Do not
tune on dev8.
