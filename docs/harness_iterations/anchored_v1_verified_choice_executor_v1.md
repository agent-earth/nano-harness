# Anchored-v1 Verified Choice Executor v1

## Hypothesis

The natural-language calculation-selector harness fails because the original
answer-only user instruction causes all 8 calculation stages to emit only a
`FINAL` line. A deterministic executor outside the generation contract can
repair explicit arithmetic-choice cases without prompt search or model-weight
changes.

## Frozen Eligibility

The executor may override anchored-v1 direct output only when all conditions
are proven from the prompt, without reading the target:

- format family is `final_choice`;
- prompt asks for the average of two quantities;
- exactly two explicit integer binary expressions are present before options;
- operators are one of `+`, `-`, `*`, `/`;
- exactly four unique options `A` through `D` have unique numeric values;
- exact rational evaluation of the two expressions and their average matches
  exactly one option value.

All arithmetic uses `fractions.Fraction`. There is no rounding, tolerance,
approximate match, nearest-option selection, answer lookup, or model call.
Every other row reuses anchored-v1 direct output byte-for-byte.

## Frozen Local Prediction

The generic choice development split contains 8 participant-average cases.
Eligibility is determined mechanically, but two known boundary shapes motivate
the policy:

- an exact integer average with a unique matching option may be overridden;
- a fractional result with no exact option must fall back, even if one option
  appears close.

No target or prior correctness label enters parsing or routing. The result is
still evaluated against targets only after candidate outputs are frozen.

## Identity

- dataset SHA256:
  `4657e96af9f9d1b81bfdb5fac6a29c31baf24b23d7753508aafdea5603ffd80d`;
- anchored-v1 local baseline result SHA256:
  `7f9ed333fbf75beea925d3de261587d5225521fa63fecc4bb68c267e9f6cc57e`;
- parser version: `explicit_two_expression_average_v1`;
- config SHA256:
  `b6aa27605fdbf25db5a470cfe95f0fee704f1c249c9069ef6b12401afa5d5178`.

## Local Gate

Baseline must reproduce 22/32 strict, 25/32 semantic, and numeric / choice /
process semantic 11/16 / 6/8 / 8/8.

Candidate must satisfy:

- strict >=22/32 and semantic >=25/32;
- numeric >=11/16 and process =8/8;
- choice >=7/8;
- all non-overridden outputs are byte-identical to direct;
- every override has a complete safe-evaluation receipt;
- parser never reads target/reference content;
- no benchmark, canary, or independent-holdout row is loaded.

Passing authorizes only the old sealed canary. Full development requires the
canary pass, and the independent holdout remains closed until old-suite
per-task base non-regression.

## Reproduction

```bash
PYTHONPATH=. ../.venv/bin/python -m nano_harness.cli verified-choice \
  --config configs/harness/anchored_v1_verified_choice_executor_v1.json
```
