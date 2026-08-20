# Qwen3.5 V5 Complete Treatment v1

This freezes one matched treatment over the existing 15,559 complete direct
case set. It starts no generation.

- GSM8K 1,319: three grounded calculator plans; override direct only when at
  least two safe executions agree.
- MMLU 14,042: preserve frozen 4B direct exactly.
- GPQA-Diamond 198: override direct only when two option reviews and one
  confirmation agree on the same non-direct option.
- every failure or disagreement preserves frozen 4B direct.
- config SHA: `b083d320e7103cb0809b5c22e6f8ebbc9330a3eb96d3910fd17d34c8f9a52f10`;
- case IDs SHA: `d38ee8c3eabbefaf7381253f6a69ba87fa63d9ee25fa4b8aeaa5f2afd73b0c63`;
- V5 report SHA: `2b1e6be49e3e00de5168f377c8ce2ab097777e7d4a55976fdd423b4a5b74e48f`.

Passing requires strict superiority over 9B on each complete benchmark,
positive paired bootstrap CI, McNemar p<0.05, and no regression versus direct
4B. Observation freezes the treatment permanently.
