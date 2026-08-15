# GSM8K Dual-Solve Holdout3 Protocol

Dual-solve is frozen after a fresh dev3 directional result of 22/24 versus
draft-verify at 19/24. The dev delta was not significant and cannot support a
claim by itself.

Holdout3 uses `start: 186, limit: 96` under the established seeded ordering. It
must have zero overlap with all prior GSM8K evaluation, development, holdout,
and confirmation cases.

Three arms run on identical case IDs:

1. Qwen3.5-4B draft-verify (256-token draft, 32-token verifier);
2. Qwen3.5-4B dual-solve (256-token first solve, 384-token independent second
   solve, 32-token selector);
3. Qwen3.5-9B direct.

Pre-registered acceptance requires:

- dual-solve point accuracy at least draft-verify accuracy;
- dual-solve point accuracy at least 9B direct accuracy;
- dual-solve versus 9B paired 95% bootstrap lower bound above -0.05;
- no dual-solve model API errors or final parse failures.

Token and wall-clock cost are reported but are not a hard gate. No strategy,
prompt, budget, or scorer changes are allowed after reading holdout3.

## Result

On holdout3:

- 4B draft-verify: 88/96, accuracy 0.9167;
- 4B dual-solve: 92/96, accuracy 0.9583;
- 9B direct: 94/96, accuracy 0.9792.

Dual-solve significantly improves over draft-verify by +0.0417, bootstrap 95%
CI [+0.0104, +0.0833]. It still fails the 9B confirmation: point delta
-0.0208 and CI [-0.0521, 0.0000], with two 9B-only wins and zero dual-only
wins.

The holdout acceptance is not satisfied. Dual-solve is useful mechanism
evidence but is too costly and still below 9B. The next experiment must use a
fresh slice and test benchmark-aware routing; no tuning is allowed on holdout3.

- [`docs/results/gsm8k_holdout3_v1.md`](../results/gsm8k_holdout3_v1.md)
- [`docs/results/gsm8k_holdout3_v1.public.json`](../results/gsm8k_holdout3_v1.public.json)
