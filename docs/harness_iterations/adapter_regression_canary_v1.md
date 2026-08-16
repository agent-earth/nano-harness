# Adapter Regression Canary v1

## Purpose

The v6 process adapter passes synthetic validation but regresses the frozen
211-case suite. Future adapters need a cheaper regression gate before another
full evaluation.

This canary is calibrated after observing v6. It is valid only as a future
adapter regression gate. It is not independent evidence for v6 and cannot
support a quality or superiority claim.

## Selection

The canary takes deterministic prefixes from the already frozen three-task
replication order:

- GSM8K: first 16 of 96;
- MMLU: first 16 of 96;
- GPQA-Diamond: first 8 of 19.

No result value influences case selection. Prompts, budgets, scorers,
temperature, dataset versions, and `enable_thinking: false` remain identical
to the full suite.

## Leakage Boundary

All 40 rows remain sealed evaluation cases:

- `source_split: sealed_eval_canary`;
- `training_eligible: false`;
- no direct or derived SFT/DPO/RL/reward/verifier training;
- no tuning on per-case canary outputs;
- no quality claim from canary performance.

Only aggregate pass/fail and failure families may guide whether to continue to
the full suite or redesign non-evaluation data.

## Calibration

On the frozen raw results:

| Task | Base 4B | Rejected v6 |
| --- | ---: | ---: |
| GSM8K | 14/16 | 13/16 |
| MMLU | 13/16 | 13/16 |
| GPQA-Diamond | 3/8 | 2/8 |
| Total | 30/40 | 28/40 |

The canary reproduces v6 rejection without using the full 211 cases.

## Future Gate

A future adapter may proceed to the full suite only if:

- GSM8K is at least 14/16;
- MMLU is at least 13/16;
- GPQA-Diamond is at least 3/8;
- total is at least 30/40;
- no task has more parse failures than base 4B;
- all 40 calls complete without API errors;
- case, prompt, budget, scorer, model, and serving parity audits pass.

Passing does not establish uplift. It only permits the unchanged adapter to
run the full frozen evaluation.

## Reproduction

```bash
PYTHONPATH=. ../.venv/bin/python scripts/validate_adapter_regression_canary.py
PYTHONPATH=. ../.venv/bin/python -m nano_harness.cli baseline \
  --manifest configs/harness/qwen35_adapter_regression_canary_v1.yaml \
  --dataset-root ../../datasets \
  --model <candidate-model-id> \
  --base-url <candidate-openai-endpoint> \
  --output results/harness/<candidate-canary>/cases.jsonl
```
