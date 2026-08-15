# Holdout2 Protocol

Draft-verify is frozen at a 256-token draft and a 32-token verifier. No prompt,
budget, strategy, or scorer changes are allowed before or after reading this
confirmation set.

Holdout2 uses the same seeded ordering as all prior experiments, with
`start: 42` and `limit: 24` for GSM8K, MMLU, and GPQA-Diamond. Its 72 case IDs
must have zero overlap with:

- fixed v5 evaluation (`start: 0, limit: 24`);
- dev1 (`start: 24, limit: 6`);
- dev2 (`start: 30, limit: 6`);
- small holdout (`start: 36, limit: 6`).

Three arms run on identical case identities:

1. Qwen3.5-4B direct;
2. frozen Qwen3.5-4B draft-verify;
3. Qwen3.5-9B direct.

Primary acceptance requires 4B draft-verify to exceed 9B direct with:

- a paired 95% bootstrap interval above zero;
- exact McNemar p < 0.05;
- no task group below the corresponding 9B direct accuracy;
- no model API errors or treatment final-answer parse failures.

The 4B direct arm attributes treatment uplift. Holdout2 is confirmation only;
its result cannot trigger another policy change. A failed acceptance must
produce an evidence-backed replan instead of holdout tuning.

## Result

4B draft-verify scores 49/72 (macro 0.6806), compared with 4B direct at
22/72 and 9B direct at 23/72. Relative to 9B, the paired delta is +0.3611,
bootstrap 95% CI [+0.2500, +0.4722], exact McNemar p=2.16e-7. MMLU and GPQA
improve significantly, and treatment final parse failures are zero.

The pre-registered task-group criterion fails on GSM8K: 4B draft-verify scores
22/24 while 9B direct scores 23/24. Harness-stage acceptance therefore remains
open despite the large significant overall win. The strategy stays frozen; the
next experiment is a larger unseen GSM8K-only confirmation.

- [`docs/results/holdout2_v1.md`](../results/holdout2_v1.md)
- [`docs/results/holdout2_v1.public.json`](../results/holdout2_v1.public.json)
