# GPQA Dev6 Option Evidence Protocol

## Hypothesis

Monolithic GPQA drafts preserved direct predictions even when given more
tokens. Generate a structurally different signal: four independent evaluators
assess options A-D separately with 96-token budgets, then a 64-token selector
chooses from explicit support and rejection evidence.

Evaluators do not see each other's outputs. Only the selector sees all four
evidence blocks. Every evaluator and selector records input/output hashes,
usage, and finish reason.

## Frozen Slice

GPQA dev6 uses `start: 108, limit: 12`, the first continuous 12-case window
with unique IDs and zero overlap against all committed historical manifests.

Three matched arms run on identical case IDs:

1. repaired Qwen3.5-4B direct;
2. Qwen3.5-4B option-evidence plus selector;
3. repaired Qwen3.5-9B direct.

## Directional Decision

Promote only if treatment:

- has higher point accuracy than 4B direct;
- has more treatment-only wins than direct-only losses;
- has no API or final parse errors;
- completes all four option evaluators for every case;
- passes case, strategy, and stage-input hash audits.

The 12-case slice cannot establish significance. Failure stops this strategy;
do not tune it on dev6.

## Validation

```bash
PYTHONPATH=. ../.venv/bin/python scripts/validate_gpqa_dev6.py
PYTHONPATH=. ../.venv/bin/python scripts/validate_qwen35_baseline.py \
  --manifest configs/harness/qwen35_gpqa_dev6_option_evidence_v1.yaml \
  --case-manifest configs/generated/qwen35_gpqa_dev6_option_evidence_v1_cases.json
```

## Result

Option evidence scores 6/12 versus 4B direct at 3/12 and 9B direct at 5/12.
Relative to 4B direct it has three treatment-only wins, zero direct-only
losses, delta +0.2500, and bootstrap 95% CI [0.0000, 0.5000].

The pre-registered promotion rule fails because one selector returned bare
`D` instead of `FINAL: D`. It stopped normally and the bare letter matches the
reference, but no post-hoc rescoring is allowed. All 48 option evaluators
reached the 96-token limit, yet the decomposition still produces net
corrective evidence.

Retain the mechanism but do not promote it. On fresh dev7 keep prompts and
budgets unchanged and add only strict deterministic normalization when the
selector output is exactly one choice letter.

- [`docs/results/gpqa_dev6_option_evidence_v1.md`](../results/gpqa_dev6_option_evidence_v1.md)
- [`docs/results/gpqa_dev6_option_evidence_v1.public.json`](../results/gpqa_dev6_option_evidence_v1.public.json)
