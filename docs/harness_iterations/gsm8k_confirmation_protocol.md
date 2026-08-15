# GSM8K Confirmation Protocol

The draft-verify policy remains frozen at a 256-token draft and a 32-token
verifier. The only unresolved holdout2 acceptance item is GSM8K, where 4B
draft-verify scored 22/24 and 9B direct scored 23/24.

This confirmation uses the same seeded order with `start: 66, limit: 96`.
These cases are unseen and have zero overlap with all prior evaluation,
development, and holdout slices.

Two arms run on identical case IDs:

1. Qwen3.5-4B draft-verify;
2. Qwen3.5-9B direct.

Pre-registered confirmation succeeds if:

- 4B point accuracy is at least 9B accuracy;
- the paired 95% bootstrap lower bound is above -0.05;
- 4B has no model API errors or final parse failures.

The non-inferiority margin handles finite-sample variation, but the point
estimate requirement preserves the original no-regression intent. No policy,
prompt, budget, or scorer change is allowed after reading this result.

## Result

The confirmation fails both criteria:

- 4B draft-verify: 84/96, accuracy 0.8750;
- 9B direct: 90/96, accuracy 0.9375;
- paired delta -0.0625;
- bootstrap 95% CI [-0.1146, -0.0208];
- exact McNemar p=0.03125;
- six 9B-only wins and zero 4B-only wins.

Among the six 9B-only cases, four 4B drafts hit the 256-token limit and two
completed with incorrect reasoning. The strict verifier corrected none of
them, confirming that it acts as a formatter rather than an independent math
reasoner.

Harness-stage acceptance remains open. The next experiment must use fresh dev3
cases to test an independent math re-solve verifier; no tuning is allowed on
these 96 cases.

- [`docs/results/gsm8k_confirmation_v1.md`](../results/gsm8k_confirmation_v1.md)
- [`docs/results/gsm8k_confirmation_v1.public.json`](../results/gsm8k_confirmation_v1.public.json)
