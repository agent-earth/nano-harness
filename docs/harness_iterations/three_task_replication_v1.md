# Three-Task 4B/9B Replication Protocol

## Hypothesis

Holdout5 produced a directional 4B advantage over 9B on the matched
GSM8K/MMLU/GPQA-Diamond suite: 50/72 versus 44/72, paired micro delta
+0.0833, 95% CI [-0.0139, +0.1806], and exact McNemar p=0.146. MMLU and
GPQA favored 4B, while GSM8K favored 9B by one case.

This replication tests whether the frozen direct prompt and answer contracts
reproduce a statistically supported 4B advantage on a larger, entirely fresh
sample. It does not add or tune a treatment.

## Frozen Contract

The prompt, scorer, temperature, `enable_thinking: false`, and per-task output
budgets are field-identical to the holdout5 direct arm:

- GSM8K: visible concise reasoning, 600 output tokens, numeric exact score;
- MMLU: answer-only, 32 output tokens, choice exact score;
- GPQA-Diamond: answer-only, 32 output tokens, choice exact score, source
  length at most 1200 characters.

Qwen3.5-4B and Qwen3.5-9B use the same manifest and case IDs. No result from
earlier experiments changes prompts, scores, or selections.

## Mechanical Fresh Selection

The exclusion set is the union of case IDs from all 51 committed generated
case manifests before this experiment. The selected deterministic windows are:

- GSM8K `start: 672, limit: 96`;
- MMLU `start: 237, limit: 96`;
- GPQA-Diamond `start: 144, limit: 19` after the frozen 1200-character filter.

Each is the first continuous window of its requested size that has unique case
IDs and zero historical overlap. The immediately preceding GSM8K and GPQA
windows each overlap one historical ID. The preceding MMLU window contains one
duplicate case ID. GPQA has exactly 19 fresh eligible cases remaining, so this
window exhausts the filtered fresh remainder.

`scripts/validate_three_task_replication.py` reconstructs these claims from the
datasets and committed manifests.

## Pre-Registered Decision

Conclude that 4B significantly exceeds the matched 9B direct baseline on this
suite only if all conditions hold:

- 4B macro accuracy is greater than 9B macro accuracy;
- paired micro 4B-minus-9B bootstrap 95% CI has a lower bound above zero;
- exact paired McNemar p-value is below 0.05;
- the 4B point estimate is not lower on GSM8K, MMLU, or GPQA-Diamond;
- both arms contain exactly the committed 211 unique case IDs;
- prompt hashes and stage inputs match the committed manifest;
- neither arm has API errors.

Report macro and micro estimates separately because task sizes are unequal.
Also report per-task paired deltas, uncertainty, discordant case IDs, parse
failures, truncations, token use, wall time, and raw artifact SHA256 values.
Failure of any condition preserves the result as evidence but does not support
a superiority claim.

## Reproduction

```bash
PYTHONPATH=. ../.venv/bin/python scripts/validate_three_task_replication.py
PYTHONPATH=. ../.venv/bin/python scripts/validate_qwen35_baseline.py \
  --manifest configs/harness/qwen35_three_task_replication_v1.yaml \
  --case-manifest configs/generated/qwen35_three_task_replication_v1_cases.json
PYTHONPATH=. ../.venv/bin/python -m pytest -q
```

## Result

Qwen3.5-4B scores 163/211 and Qwen3.5-9B scores 151/211. The
4B-minus-9B paired micro delta is +0.0569, with 95% bootstrap CI
[0.0000, +0.1185] and exact McNemar p=0.0884. Macro accuracy is 0.6504
versus 0.5806.

All three task point estimates favor 4B:

- GSM8K: 90/96 versus 89/96;
- MMLU: 67/96 versus 58/96;
- GPQA-Diamond: 6/19 versus 4/19.

There are 27 4B-only wins and 15 9B-only wins. The 4B arm has two parse
failures, while 9B has 33, including 25 on MMLU. This is valid end-to-end
contract behavior, but downstream data must distinguish format failures from
semantic answer failures.

The directional holdout5 advantage replicates without task regression, but
the confidence interval lower bound is zero and p exceeds 0.05. The result
therefore fails the pre-registered significance rule and does not establish
statistically significant superiority.

- [`docs/results/three_task_replication_v1.md`](../results/three_task_replication_v1.md)
- [`docs/results/three_task_replication_v1.public.json`](../results/three_task_replication_v1.public.json)
