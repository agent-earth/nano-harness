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

## Result

The conservative arbiter scores 7/12 versus 4B direct at 5/12 and 9B direct
at 4/12. Relative to 4B direct it has two treatment-only wins, zero
direct-only losses, delta +0.1667, and bootstrap 95% CI
[0.0000, 0.4167].

Protected-direct predictions match the independent direct arm on all 12 cases.
The arbiter overrides exactly two times; both overrides repair wrong direct
answers and neither breaks a correct answer. There are no API or final parse
errors. Dev8 satisfies every directional promotion rule.

Freeze this GPQA policy and pre-register a new three-task holdout. Keep GSM8K
and MMLU direct, route only GPQA through the conservative arbiter, and do not
reuse the sealed holdout4.

- [`docs/results/gpqa_dev8_conservative_arbiter_v1.md`](../results/gpqa_dev8_conservative_arbiter_v1.md)
- [`docs/results/gpqa_dev8_conservative_arbiter_v1.public.json`](../results/gpqa_dev8_conservative_arbiter_v1.public.json)
