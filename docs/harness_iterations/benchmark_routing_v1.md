# Benchmark Routing v1 Protocol

## Motivation

Evidence review found that old three-benchmark holdout controls violated their
validated prompt contract. The direct and draft paths are repaired before this
experiment. Prior aggregate holdout claims are not used to select or justify
the route.

Valid GSM8K evidence shows direct reasoning is stronger and cheaper than the
draft-verify policy, while dual solving remains below 9B and is expensive.
The next falsifiable policy is:

- GSM8K: direct reasoning;
- MMLU: 256-token reasoning draft, then a 32-token verifier;
- GPQA-Diamond: 256-token reasoning draft, then a 32-token verifier.

Each result records the selected strategy and the SHA256 of the actual direct
or draft input.

## Frozen Slices

Dev4 is directional only:

- GSM8K: `start: 282, limit: 6`;
- MMLU: `start: 66, limit: 6`;
- GPQA-Diamond: `start: 66, limit: 6`.

Holdout4 is frozen before reading dev4:

- GSM8K: `start: 288, limit: 24`;
- MMLU: `start: 96, limit: 24`;
- GPQA-Diamond: `start: 72, limit: 24`.

The validator requires both slices to have zero case overlap with every prior
committed suite and with each other. Direct and routed arms must have identical
case identities.

MMLU positions 72-95 are intentionally skipped because that continuous window
contains a duplicate content-derived case ID. Position 96 is the first
24-case continuous window with unique IDs and zero historical overlap, found
before any inference.

## Arms

Both slices use three matched arms:

1. repaired Qwen3.5-4B direct;
2. Qwen3.5-4B benchmark routing;
3. repaired Qwen3.5-9B direct.

All arms use temperature 0, `enable_thinking: false`, identical datasets,
scorers, case identities, and per-benchmark final contracts. The route is the
only treatment.

## Decision Rules

Dev4 supports running the already-frozen holdout4 only if routed execution has
no API errors, no final parse failures, and does not trail 4B direct overall.
Because holdout4 is already frozen, no prompt, route, budget, case, or scorer
may change after dev4.

Holdout4 acceptance requires:

- routed 4B point accuracy at least 9B direct on every benchmark;
- routed 4B macro accuracy above 9B direct;
- paired micro 95% bootstrap lower bound above zero;
- exact McNemar p-value below 0.05;
- no routed API errors or final parse failures.

Cost and latency are reported but are not hard gates. Failure produces an
evidence-backed replan and no holdout tuning.

## Validation

```bash
PYTHONPATH=. ../.venv/bin/python scripts/validate_routing_holdout4.py
PYTHONPATH=. ../.venv/bin/python scripts/validate_qwen35_baseline.py \
  --manifest configs/harness/qwen35_routing_dev4_v1.yaml \
  --case-manifest configs/generated/qwen35_routing_dev4_v1_cases.json
PYTHONPATH=. ../.venv/bin/python scripts/validate_qwen35_baseline.py \
  --manifest configs/harness/qwen35_routing_holdout4_v1.yaml \
  --case-manifest configs/generated/qwen35_routing_holdout4_v1_cases.json
```
