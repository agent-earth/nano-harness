# GSM8K Dev16 Constrained Recovery Confirmation Protocol

## Frozen Hypothesis

Dev15 validated vLLM regex-constrained numeric recovery, but its 48 direct
outputs were all parseable and recovery never ran. Dev16 tests whether the
unchanged conditional mechanism converts rare direct parse failures without
changing parseable direct answers or adding material cost.

The policy is frozen from dev15:

- run the same 600-token 4B direct path;
- invoke recovery only when direct has no parseable numeric final;
- give recovery 32 tokens;
- enforce
  `FINAL: [-+]?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)` through vLLM
  `structured_outputs.regex`;
- use deterministic recovery selection;
- do not tune or rescore after observing dev16.

## Mechanical Fresh-Window Selection

The exclusion set is the union of GSM8K case IDs from every committed
`configs/generated/*cases.json` file before dev16: 41 manifests and 576 unique
IDs. Case IDs hash the source question, so overlap detection is independent of
prompt or strategy.

Scanning the deterministically sorted GSM8K test set for the first continuous
96-case window with unique IDs and zero exclusion-set overlap gives
`start: 576, limit: 96`. The immediately preceding window at `start: 575`
overlaps one historical ID. `scripts/validate_gsm8k_dev16.py` reproduces this
selection and fails if the result changes.

## Matched Arms

1. Qwen3.5-4B direct at `http://127.0.0.1:8000/v1`;
2. Qwen3.5-4B conditional constrained recovery at
   `http://127.0.0.1:8002/v1`;
3. Qwen3.5-9B direct at `http://127.0.0.1:8001/v1`.

All arms use the same 96 case IDs, source data SHA256, system prompt,
temperature 0, `enable_thinking: false`, scorer, and 600-token direct budget.
Separate 4B services allow the direct control and treatment to run
concurrently without sharing result files.

## Pre-Registered Decision

Accept constrained recovery on dev16 only if:

- recovery triggers and creates at least one treatment-only win;
- there are zero direct-only losses;
- treatment parse failures are fewer than direct parse failures;
- every recovery call carries the committed regex and every recovery output
  full-matches it;
- direct/treatment case identity, protected-direct parity, conditional-call,
  prompt-hash, and deterministic-selection audits all pass;
- treatment/direct token ratio is below 1.2x;
- no treatment API errors occur.

Report paired accuracy delta, 10,000-sample paired bootstrap 95% CI, exact
McNemar p-value, parse failures, recovery calls/wins/truncations, API errors,
tokens, wall time, raw artifact SHA256 values, and case-level discordances.
This is a mechanism confirmation, not evidence that 4B beats 9B across the
project benchmark suite.

## Reproduction

```bash
PYTHONPATH=. ../.venv/bin/python scripts/validate_gsm8k_dev16.py
PYTHONPATH=. ../.venv/bin/python scripts/validate_qwen35_baseline.py \
  --manifest configs/harness/qwen35_gsm8k_dev16_constrained_v1.yaml \
  --case-manifest configs/generated/qwen35_gsm8k_dev16_constrained_v1_cases.json
PYTHONPATH=. ../.venv/bin/python -m pytest -q
```
