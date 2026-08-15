# Evidence Integrity Correction v1

## Finding

Revision `cf2a00e` introduced two prompt-contract mismatches:

1. direct runs sent `case.draft_prompt`, while validation modeled
   `case.prompt`;
2. draft-verify runs sent `case.prompt`, while validation and documentation
   modeled `case.draft_prompt`.

For answer-only MMLU and GPQA, the direct mismatch combined a reasoning prompt
with a 32-token output budget. Every MMLU and GPQA output in the holdout1 and
holdout2 direct arms truncated and failed parsing.

## Evidence Scope

- `qwen35-local-v5`: valid; it ran before the regression.
- draft-verify fixed-v5 treatment: valid observation of the actual
  answer-only-draft implementation, but not evidence for the stated
  reasoning-draft mechanism.
- dev1 treatment versus direct: invalid because the direct MMLU/GPQA control
  was mismatched.
- dev2 critique comparison: valid only as a comparison of composite
  strategies; critique-only attribution is invalid because the draft prompt
  also changed.
- holdout1 and holdout2 aggregate comparisons: invalid because the direct
  MMLU/GPQA controls were mismatched.
- holdout1 and holdout2 GSM8K subsets: valid because GSM8K `prompt` and
  `draft_prompt` express the same reasoning contract.
- GSM8K confirmation, dev3, and holdout3: valid and unaffected.

Raw artifacts and hashes are retained. Invalid conclusions are not deleted;
their public reports now carry machine-readable `validity` fields and visible
corrections.

## Reproduction

```bash
git show cf2a00e^:nano_harness/baseline.py | sed -n '300,370p'
git show cf2a00e:nano_harness/baseline.py | sed -n '400,500p'
PYTHONPATH=. ../.venv/bin/python -m nano_harness.cli baseline-summary \
  results/harness/qwen35-holdout2-direct-v1/9b/cases.jsonl
PYTHONPATH=. ../.venv/bin/python scripts/validate_qwen35_baseline.py
PYTHONPATH=. ../.venv/bin/python -m pytest -q
```

The corrected runner sends `case.prompt` to direct execution and
`case.draft_prompt` to the draft stage. Regression tests assert both contracts,
and benchmark routing uses the same strategy selector as the context-budget
validator.
