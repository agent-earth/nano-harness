# Large Independent 4B/9B Confirmation Protocol

## Hypothesis

The fresh 211-case replication directionally favored Qwen3.5-4B on GSM8K,
MMLU, and GPQA-Diamond, but its paired confidence interval started at zero and
McNemar p was 0.0884. This experiment tests the same frozen direct contract on
a larger independent GSM8K/MMLU sample.

GPQA-Diamond is not rerun because the prior replication exhausted all 19
remaining fresh cases under the frozen 1200-character filter. Its positive
direction remains separate evidence and is not pooled into this experiment's
significance calculation.

## Frozen Contract

The prompt, dataset versions, scorer, temperature, `enable_thinking: false`,
and output budgets are field-identical to the preceding direct replication:

- GSM8K: visible concise reasoning, 600 output tokens, numeric exact score;
- MMLU: answer-only, 32 output tokens, choice exact score.

Qwen3.5-4B and Qwen3.5-9B use the same committed case manifest. No prior result
changes selection, prompts, normalization, or scoring.

## Mechanical Fresh Selection

The historical exclusion set is the union of all 52 committed generated case
manifests before this experiment. The selected windows are:

- GSM8K `start: 768, limit: 256`;
- MMLU `start: 2801, limit: 256`.

Each is the first continuous 256-case window with unique IDs and zero
historical overlap. The preceding GSM8K window overlaps one historical ID.
The preceding MMLU window has no historical overlap but contains one duplicate
case ID. `scripts/validate_large_confirmation.py` reconstructs these claims.

## Pre-Registered Decision

Conclude that the direct 4B advantage is statistically confirmed on this
independent two-task sample only if every condition holds:

- 4B macro accuracy is greater than 9B macro accuracy;
- paired micro 4B-minus-9B bootstrap 95% CI has a lower bound above zero;
- exact paired McNemar p-value is below 0.05;
- 4B point estimates are not lower on GSM8K or MMLU;
- both arms contain exactly the committed 512 unique case IDs;
- prompt hashes and direct-stage input hashes match the manifest;
- neither arm has API errors.

Report per-task and aggregate paired counts, uncertainty, case-level
discordances, parse failures, truncations, token use, wall time, and raw
artifact SHA256 values. Keep format failures distinct from semantic failures.
Failure of any condition preserves directional evidence but does not support a
significant-superiority claim or training expansion.

## Reproduction

```bash
PYTHONPATH=. ../.venv/bin/python scripts/validate_large_confirmation.py
PYTHONPATH=. ../.venv/bin/python scripts/validate_qwen35_baseline.py \
  --manifest configs/harness/qwen35_large_confirmation_v1.yaml \
  --case-manifest configs/generated/qwen35_large_confirmation_v1_cases.json
PYTHONPATH=. ../.venv/bin/python -m pytest -q
```
