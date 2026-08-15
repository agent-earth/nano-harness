# Qwen3.5 Local Baseline v5

## Result

| Benchmark | Qwen3.5-4B | Qwen3.5-9B | 4B - 9B |
| --- | ---: | ---: | ---: |
| gsm8k | 0.9167 (22/24) | 0.9167 (22/24) | +0.0000 |
| mmlu | 0.7500 (18/24) | 0.7917 (19/24) | -0.0417 |
| gpqa_diamond | 0.4167 (10/24) | 0.4167 (10/24) | +0.0000 |
| Macro average | 0.6944 | 0.7083 | -0.0139 |

Across all 72 paired cases, 4B scored 0.6944 and 9B scored 0.7083. The paired delta is -0.0139, with a fixed-seed 95% bootstrap interval [-0.1111, +0.0694] and exact McNemar `p=1.0000`. This baseline does not establish a significant model-quality difference.

## Failure Evidence

- 4B parse categories: `{'parsed': 72, 'length_truncation': 0, 'missing_final_line': 0, 'malformed_final_line': 0}`.
- 9B parse categories: `{'parsed': 62, 'length_truncation': 3, 'missing_final_line': 7, 'malformed_final_line': 0}`.
- v1 used a 256-token reasoning budget and truncated every GPQA output.
- v2 used a 600-token reasoning budget but still truncated 39/48 GPQA outputs.
- v3 answer-only removed truncation but reduced GSM8K accuracy sharply.
- v4 used one global 600-token budget; it allowed answer-only contract drift.
- v5 uses a 600-token reasoning budget for GSM8K and a 32-token answer-only budget for MMLU and GPQA, identically for both models.

## Reproduction Identity

- Code revision: `3334053b1a1abfb71bfdf0bfb656859a45edfbcf`
- Suite manifest SHA256: `7b0dc4e1b3d54236648ceab7bb299144e924bd769f654dfd807e7486bba9fb38`
- Case manifest SHA256: `d9061e8fd4d52e9f16e04a58ef26f5e00c6c79c89ed9563fea41da3b51751e4c`
- 4B raw result SHA256: `0831f9ac5af5a96349b2ef91d6c02d17d90a3434e6ced83a69d3fd49026da61b`
- 9B raw result SHA256: `bddb885f092d9d520715877ff137e4d504ec811db6e6b118ab63a58d85ec63af`

Raw case outputs remain local and ignored. The public JSON report includes aggregate metrics and case IDs for paired failures, not task bodies or model response text.
