# Qwen3.5 Router Adapter Integration v3

## Purpose

Negative-diversity SFT and namespace-remapped serving both pass 1,536/1,536.
V3 tests transfer on 160 new answer-task prompts: A/B plus all eight C
subtypes. It does not rerun V1 or V2.

## Freshness

- seed `20260828`, value offset
  `8000000`;
- 160 new prompts, A/B = 16/16 and C = 8 x 16;
- overlap with 7,680 training prompts, V1/V2 prompts, prior multiclass/binary
  prompts, and prior generic surfaces: all zero;
- overlap with complete GSM8K/MMLU/GPQA prompt columns: all zero;
- prior integration outputs, benchmark outputs, canary, and holdout: not loaded.

## Gates

- A 16/16, B 16/16, every C subtype 16/16, zero false routes;
- 32 verified executions, 32 feedback matches, zero fallbacks;
- all 128 negative cases preserve direct output exactly;
- significant zero-loss superiority over direct 4B and 9B;
- every-family non-regression.

Passing permits only a separately pre-registered benchmark-agnostic treatment
transfer. V1/V2/V3 cannot be rerun after observation. Benchmark generation,
canary, holdout, training, and RL remain closed.

## Identity

- config SHA: `0f7f2aedf12d63a651c79f4b06417de45cc5bd83b48a9be5ab396282baed048a`;
- case contract SHA: `b15bf58f89faff6212b5a16133c07bdfe9e83547be1ef6a6bda33454ec51d410`;
- remapped adapter SHA: `cea357d281ed100437268e213564fc5a5c00e6024b0c7a4be207cc686453e3f9`;
- parity report SHA: `28fe9e86c4ab80fc83618d34c4d9bb9a0e9ac158ac640781685ff23d778b5d3c`;
- generation started: false.
