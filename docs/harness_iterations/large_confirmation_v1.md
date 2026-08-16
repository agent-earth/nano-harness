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

## Result

Qwen3.5-4B scores 415/512 and Qwen3.5-9B scores 403/512. The paired
4B-minus-9B micro delta is +0.0234, with 95% bootstrap CI
[-0.0059, +0.0527] and exact McNemar p=0.1480. Macro accuracy is 0.8105
versus 0.7871.

Task results diverge:

- GSM8K: 4B 235/256 versus 9B 241/256, delta -0.0234;
- MMLU: 4B 180/256 versus 9B 162/256, delta +0.0703, CI
  [+0.0195, +0.1211], p=0.0114.

The aggregate interval crosses zero and GSM8K regresses, so the confirmation
fails the pre-registered superiority rule.

Under the official strict scorer, 9B has 57 MMLU parse failures and 4B has
zero. Every failed 9B output is `FINAL <letter>` without a colon. A
non-scoring diagnostic finds 29 of those letters match the reference; a
hypothetical colon-only normalization would produce 191/256 for 9B, above
4B's official 180/256. Official scores and paired statistics remain unchanged.
The result shows that the official MMLU advantage is primarily format-contract
compliance rather than stable semantic superiority.

Stop direct-only confirmation. Separate format and semantic discordances
before designing the next harness or training ablation.

- [`docs/results/large_confirmation_v1.md`](../results/large_confirmation_v1.md)
- [`docs/results/large_confirmation_v1.public.json`](../results/large_confirmation_v1.public.json)
