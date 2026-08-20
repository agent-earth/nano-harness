# Qwen3.5 Router Adapter Integration v2

## Purpose

Serving parity proved that the original PEFT namespace was inert under vLLM,
while the content-identical remap reproduced HF at 192/192. V2 tests transfer
on a **new** 128-case surface; it does not rerun V1.

## Freshness

- seed `20260826`, value offset
  `25000`;
- task prompts ask for answers, not route/classification labels;
- exact normalized overlap with 960 training, 128 V1, 256 multiclass, and 128
  binary prompts: all zero;
- exact overlap with complete GSM8K/MMLU/GPQA prompt columns: all zero;
- V1 outputs, benchmark outputs, canary, and holdout: not loaded.

## Gates

- A 32/32, B 32/32, C 64/64 and zero false routes;
- 64 verified executions, 64 feedback matches, zero fallbacks;
- negative direct preservation;
- significant zero-loss superiority over direct 4B and 9B;
- every-family non-regression.

Passing permits only a separately pre-registered question-only scan. V1 and V2
cannot be rerun after observation. Benchmark, canary, holdout, training, and RL
remain closed.

## Identity

- config SHA: `e9b4af568af1c54242b17442d361a0e0bc8548013cf54ddd716092ec6f6a8c35`;
- case contract SHA: `d1b03b8f5fae8e9a0a5e8d3a0b3d4f5d7f8fa78a53c767fb8ece3e404c8be486`;
- remapped adapter SHA: `fbaa39dcb3fcf34e9aab280308cb5a5416094c1968e4ac3a69cd739a806ecc49`;
- parity report SHA: `539517c890e53f2a0e4034c724d1324df6cc828186d9621f77c106c08d4a1c01`;
- generation started: false.
