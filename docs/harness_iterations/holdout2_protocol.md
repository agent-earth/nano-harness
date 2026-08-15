# Holdout2 Protocol

## Evidence Integrity Correction

After publication, runner review found that revision `cf2a00e` sent
`case.draft_prompt` through direct arms while validation modeled
`case.prompt`. The answer-only MMLU and GPQA controls therefore used a
reasoning prompt with only 32 output tokens and truncated every response. The
aggregate confirmation and significant-win claim are invalid.

Treatment observations and raw digests remain negative evidence. The GSM8K
subset remains matched because its prompt fields are equivalent. A repaired
runner and fresh cases are required for a new aggregate comparison.

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

The invalid aggregate observation was 4B draft-verify at 49/72 (macro 0.6806),
compared with 4B direct at
22/72 and 9B direct at 23/72. Relative to 9B, the paired delta is +0.3611,
bootstrap 95% CI [+0.2500, +0.4722], exact McNemar p=2.16e-7. This must not be
interpreted as uplift because the direct controls were mismatched.

The pre-registered task-group criterion fails on GSM8K: 4B draft-verify scores
22/24 while 9B direct scores 23/24. Harness-stage acceptance therefore remains
open. The next aggregate experiment must use repaired, validator-matched
controls on fresh cases.

- [`docs/results/holdout2_v1.md`](../results/holdout2_v1.md)
- [`docs/results/holdout2_v1.public.json`](../results/holdout2_v1.public.json)
