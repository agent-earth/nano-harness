# GPQA Dev5 v1 Protocol

## Hypothesis

Broad benchmark routing failed dev4 because MMLU direct-correct answers were
changed and all six 256-token GPQA drafts truncated. Keep GSM8K and MMLU
direct. Test whether a 384-token reasoning draft plus 32-token verifier adds
net GPQA correct cases.

## Frozen Slice

GPQA dev5 uses `start: 96, limit: 12` after the established seeded ordering and
1200-character eligibility filter. It is the first 12-case continuous window
with unique content-derived IDs and zero overlap with all committed historical
manifests.

Three arms use identical case IDs:

1. repaired Qwen3.5-4B direct;
2. repaired Qwen3.5-4B draft-verify, 384-token draft and 32-token verifier;
3. repaired Qwen3.5-9B direct.

Temperature, scorer, chat-template kwargs, and final answer contract are
matched. The treatment changes only the 4B reasoning route.

## Decision Rule

Promote GPQA-only routing to a new pre-registered three-task holdout only if:

- treatment point accuracy exceeds 4B direct;
- treatment has more treatment-only wins than direct-only losses;
- no API errors or final parse failures occur;
- draft truncation is below the dev4 rate of 6/6;
- the extra cost is reported.

This directional slice is too small to establish significance. Failure stops
this route; do not tune on these 12 cases.

## Validation

```bash
PYTHONPATH=. ../.venv/bin/python scripts/validate_qwen35_baseline.py \
  --manifest configs/harness/qwen35_gpqa_dev5_draft_verify_v1.yaml \
  --case-manifest configs/generated/qwen35_gpqa_dev5_draft_verify_v1_cases.json
```
