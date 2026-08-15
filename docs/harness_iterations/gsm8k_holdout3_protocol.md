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
