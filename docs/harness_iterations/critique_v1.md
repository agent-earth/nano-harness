# Critique v1 And Holdout Protocol

## Evidence Integrity Correction

The dev2 comparison changed both the critique stage and the draft prompt: the
incumbent used `case.prompt`, while the composite critique arm used
`case.draft_prompt`. It compares two composite strategies and cannot isolate
critique-stage causality.

The direct holdout arms used `case.draft_prompt` under answer-only MMLU/GPQA
budgets. Their aggregate comparisons are invalid; only the matched GSM8K
subset remains usable. The significant holdout claim is withdrawn and raw
artifacts remain available by digest.

## Dev2 Critique Test

The fresh dev2 slice uses `start: 30, limit: 6` for each benchmark. It has no
overlap with v5 evaluation or the first development slice.

The incumbent is draft-verify with a 256-token draft and a 32-token verifier.
The treatment inserts a 192-token independent critique before the same strict
formatter.

The composite critique strategy is rejected:

- incumbent: 14/18, macro 0.7778;
- critique: 12/18, macro 0.6667;
- paired delta: -0.1111, bootstrap 95% CI [-0.2778, 0.0000];
- both lost cases are GPQA;
- critique uses 26,365 tokens versus 8,418 and 249s versus 41s;
- draft and critique stages truncate 10 and 9 times respectively.

No further tuning is allowed on dev2.

## Untouched Holdout

Before reading holdout results, the retained policy is fixed as draft-verify
with a 256-token draft and a 32-token verifier.

The holdout uses `start: 36, limit: 6` for GSM8K, MMLU, and GPQA-Diamond under
the same seeded hash order. It must have zero overlap with:

- the 72 fixed v5 evaluation cases;
- the 18 first-development cases;
- the 18 dev2 cases.

Three arms run on the same 18 holdout identities:

1. Qwen3.5-4B direct, for treatment attribution;
2. Qwen3.5-4B draft-verify, the frozen selected policy;
3. Qwen3.5-9B direct, the primary comparison.

The primary question is whether 4B draft-verify exceeds 9B direct without a
task-group regression, API errors, or final parse failures. Holdout results are
confirmatory evidence only for this frozen policy; no prompt or budget changes
may follow from inspecting these cases.

## Result

The composite critique strategy scores 12/18 versus draft-verify at 14/18.
The two lost cases are GPQA. It uses 26,365 tokens and 249 seconds versus
8,418 tokens and 41 seconds, but the draft prompt also changed.

The untouched holdout observed 4B draft-verify at 13/18 while both direct arms
scored 6/18. The aggregate result is invalid because the MMLU/GPQA direct arms
used a mismatched prompt/budget contract.

A corrected fresh confirmation is required.

- [`docs/results/critique_v1_holdout.md`](../results/critique_v1_holdout.md)
- [`docs/results/critique_v1_holdout.public.json`](../results/critique_v1_holdout.public.json)
