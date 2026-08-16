# V6 Matched Adapter Evaluation v1

## Purpose

Arithmetic process SFT v6 passes its local process-contract gate, but baseline
and post-SFT final-answer accuracy are both 32/32. This evaluation tests the
unchanged adapter on the previously frozen three-task replication cases before
any merge, scale-up, or RL decision.

## Frozen Contract

The candidate reuses all 211 cases from
`qwen35-three-task-replication-v1`:

- GSM8K: 96 cases, concise reasoning, 600 output tokens, numeric exact;
- MMLU: 96 cases, answer-only, 32 output tokens, choice exact;
- GPQA-Diamond: 19 cases, answer-only, 32 output tokens, choice exact.

Case IDs, prompts, system prompts, scorers, dataset versions, temperature,
`enable_thinking: false`, and output budgets are field-identical to the frozen
base 4B and 9B arms.

The v6 adapter is served through vLLM 0.19.1 as model
`qwen3.5-4b-process-v6` with:

```text
--enable-lora
--lora-modules qwen3.5-4b-process-v6=<local adapter path>
```

The adapter tree is immutable and SHA-bound. Raw candidate results remain
local and ignored.

Qwen3.5 is exposed by vLLM as a multimodal wrapper whose text modules use the
`language_model.model.layers.*` namespace. PEFT stores the same tensors under
`model.layers.*`. Before serving, a deterministic local-only conversion:

- changes only that namespace in safetensors keys;
- preserves all 224 tensor dtype, shape, and content hashes;
- writes its receipt and converted weights under ignored `results/serving/`;
- requires base/candidate first-token logits to differ;
- requires a known validation case to fail under base and pass exact plus
  semantic verification under the adapter.

The 211-case evaluation is forbidden unless these serving parity gates pass.

## Frozen Identity

- case manifest SHA256:
  `eafbe4d42487a225322dd3b3bdc1d805c065fb15f0f8b968e65ccf747f96976f`;
- frozen suite manifest SHA256:
  `88f6e832d38e739c6b622a30633a2737077fc081037e6e1543cb5763b169a7b9`;
- base 4B raw SHA256:
  `c59383d3fd3d6087025d6e1ff649979d9d5a9e8dc73b5429a4f8e9fa41b6b8c7`;
- 9B raw SHA256:
  `ffae93774d51b87a2e29258d170a84f8b165f996e2e78eedd102271dfc260044`;
- v6 adapter tree SHA256:
  `49f08829e06aa75c1cf6e5f16891bf79378011b8fe874fde4e392f5fcb5aa083`;
- v6 training revision: `faa2d56`.

## Pre-Registered Decision

Report candidate versus base 4B and candidate versus 9B separately.

The adapter passes benchmark non-regression only if:

- all 211 cases complete with zero API errors;
- case IDs, prompt hashes, direct-stage hashes, budgets, and scorers match;
- candidate accuracy is not below base 4B on GSM8K, MMLU, or GPQA-Diamond;
- candidate macro and micro accuracy are not below base 4B;
- no task has more parse failures than base 4B.

The adapter significantly exceeds 9B only if:

- candidate macro accuracy exceeds 9B;
- paired micro bootstrap 95% CI lower bound is above zero;
- exact McNemar p-value is below 0.05;
- no task point estimate is below 9B;
- neither arm has API errors.

Report all task metrics, paired intervals, McNemar p-values, discordant IDs,
parse failures, length truncations, token use, latency, and raw SHA256 values.
Failure preserves evidence and blocks merge, scale-up, and RL. Passing
non-regression alone still does not authorize RL; it only justifies a separate
decision on the next training ablation.

## Reproduction

```bash
PYTHONPATH=. ../.venv/bin/python scripts/validate_v6_matched_adapter.py
../.venv/bin/python scripts/build_qwen35_vllm_adapter.py \
  --source ../nano-train/artifacts/arithmetic-process-sft-smoke-v6/adapter \
  --output results/serving/qwen35-v6-vllm-adapter \
  --receipt results/serving/qwen35-v6-vllm-adapter.receipt.json
PYTHONPATH=. ../.venv/bin/python -m nano_harness.cli baseline \
  --manifest configs/harness/qwen35_v6_matched_adapter_v1.yaml \
  --dataset-root ../../datasets \
  --model qwen3.5-4b-process-v6 \
  --base-url http://127.0.0.1:8003/v1 \
  --output results/harness/qwen35-v6-matched-adapter-v1/candidate/cases.jsonl
```

## Result

The v6 adapter fails benchmark non-regression:

- GSM8K: 73/96 versus base 4B 90/96 and 9B 89/96;
- MMLU: 68/96 versus base 4B 67/96 and 9B 58/96;
- GPQA-Diamond: 4/19 versus base 4B 6/19 and 9B 4/19;
- micro: 145/211 versus base 4B 163/211 and 9B 151/211;
- macro: 0.5598 versus base 4B 0.6504 and 9B 0.5806.

Against base 4B, paired micro delta is -0.0853, 95% CI
[-0.1280, -0.0474], exact McNemar p=0.000121. There are 20 base-only
wins and 2 adapter-only wins. GSM8K regresses significantly by 17 cases;
GPQA regresses by 2; MMLU improves by 1.

Candidate GSM8K has 9 official parse failures. A non-scoring inline-final
diagnostic finds 8 contain the correct numeric value but violate the required
standalone `FINAL:` line; 14 other GSM8K failures are parseable but numerically
wrong. Official scores are unchanged.

All 211 cases complete with zero API errors. Case, prompt, stage-input, budget,
scorer, source-result, adapter-namespace, logits, and known-case serving audits
pass. The adapter is rejected for merge, scale-up, and RL.

Public result:

- `docs/results/v6_matched_adapter_v1.md`;
- `docs/results/v6_matched_adapter_v1.public.json`.
