# Qwen3.5 Local Baseline

This baseline compares local Qwen3.5-4B and Qwen3.5-9B services under one
matched, deterministic contract.

## Suite

`configs/baselines/qwen35_local_v3.yaml` selects 24 cases from each benchmark:

- GSM8K: numeric exact match after a required `FINAL:` line;
- MMLU: choice-letter exact match;
- GPQA-Diamond: choice-letter exact match.

Case selection sorts content-derived IDs by a seeded hash. GPQA source
questions longer than 1200 characters are excluded before selection because they do not fit the
verified 1024-token service budget without changing task semantics. The
committed case manifest contains IDs, source indices, labels, and compact
metadata, but not task bodies.

The suite passes `chat_template_kwargs={"enable_thinking": false}` to both
models. A real pre-run smoke showed that default thinking mode consumed the
full 256-token budget on trivial arithmetic before completing the required
answer line. With thinking disabled, both models returned the correct `FINAL:`
line in 6-16 completion tokens. This is a matched inference setting, not a
harness treatment.

All three suite versions keep identical case IDs:

- v1 used a 256-token reasoning budget. All 48 GPQA attempts hit the limit and
  failed to emit a parseable final answer.
- v2 raised the reasoning budget to 600. GPQA still produced 39/48 length
  truncations, so the resulting macro score remained confounded.
- v3 uses one matched answer-only contract for all three benchmarks and a
  32-token output budget. A real GPQA pre-run smoke completed in four tokens for
  both models with parseable predictions.

v1 and v2 are retained as negative configuration evidence and are not used as
the valid three-benchmark baseline.

This 72-case suite is a directional baseline. It is not large enough by itself
to establish the final statistical significance required by the project goal.

## Validation

From the repository root:

```bash
PYTHONPATH=. ../.venv/bin/python scripts/validate_qwen35_baseline.py
PYTHONPATH=. ../.venv/bin/python -m unittest discover -s tests -v
```

The suite validator checks dataset SHA256 values, exact case selection,
per-benchmark counts, unique IDs, and Qwen3.5 tokenizer context usage. The
current maximum is 420 input tokens and 452 tokens including the 32-token
output budget.

## Model Service

Start one model at a time on port 8000. Keep all settings matched except the
model path and served model name:

```bash
CUDA_VISIBLE_DEVICES=0 ../.venv/bin/vllm serve ../../models/Qwen3.5-4B \
  --host 127.0.0.1 --port 8000 \
  --served-model-name qwen3.5-4b \
  --dtype float16 \
  --max-model-len 1024 \
  --gpu-memory-utilization 0.85 \
  --enforce-eager \
  --max-num-batched-tokens 1024 \
  --max-num-seqs 1
```

Use `../../models/Qwen3.5-9B` and `qwen3.5-9b` for the 9B service. Confirm
`/v1/models` before starting a run.

The 9B service cannot allocate KV cache at `--gpu-memory-utilization 0.60`
after loading its 17.66 GiB of weights; the measured cache budget is -1.46
GiB. The matched 0.85 setting provides 4.46 GiB of 9B KV cache and 15.72 GiB
for 4B on 32 GiB V100 GPUs.

## Run

```bash
PYTHONPATH=. ../.venv/bin/python -m nano_harness.cli baseline \
  --manifest configs/baselines/qwen35_local_v3.yaml \
  --dataset-root ../../datasets \
  --model qwen3.5-4b \
  --base-url http://127.0.0.1:8000/v1 \
  --output results/baselines/qwen35-local-v3/4b/cases.jsonl

PYTHONPATH=. ../.venv/bin/python -m nano_harness.cli baseline-summary \
  results/baselines/qwen35-local-v3/4b/cases.jsonl
```

Use a separate `9b/cases.jsonl` output for Qwen3.5-9B. The runner resumes by
stable case ID and records output, normalized prediction, reference, score,
latency, and API token usage per case.

Do not change prompts, case IDs, context budget, output budget, temperature, or
scorers between model runs. Any changed treatment requires a new suite ID.
