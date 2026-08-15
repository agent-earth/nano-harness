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
