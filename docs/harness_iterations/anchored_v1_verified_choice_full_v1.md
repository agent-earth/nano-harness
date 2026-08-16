# Anchored-v1 Verified Choice Full Development v1

## Purpose

The target-blind verified-choice executor passes the synthetic local gate and
the old sealed canary. This stage applies the exact parser to the old matched
211-case development suite without new model inference.

## Frozen Application

- rebuild 96 GSM8K, 96 MMLU, and 19 GPQA-Diamond cases from the frozen
  manifest;
- require the committed 211-case manifest to match exactly;
- require every anchored-v1 raw row to match case, prompt/system hash, source
  index, expected identity, model, direct strategy, and output budget;
- call parser `explicit_two_expression_average_v1` only for `choice_exact`;
- override only for an exact rational result matching one unique numeric
  option;
- reuse every unsupported or ambiguous direct output;
- score only after routing is frozen.

The parser sees only case prompts. Expected answers and prior scores are not
parser inputs.

## Frozen Identity

- suite manifest SHA256:
  `08c71cae463bd3b0a0031e95d6339136d0c445beecaac631c4f5843e0b14d4c1`;
- 211-case manifest SHA256:
  `eafbe4d42487a225322dd3b3bdc1d805c065fb15f0f8b968e65ccf747f96976f`;
- anchored-v1 raw SHA256:
  `a8f6a731042c7b81c97196abd60d6c632006b7c59da1bbdb2328ab73c539def0`;
- anchored-v1 public receipt SHA256:
  `413a292246ee368000a484d82b68ccf1b37998f7bf3cd91ed2940b15356174a0`;
- verified-choice canary pass SHA256:
  `e9edf209ace2d52516b38602d942812525f997543463a6abd2a7463340f68ccc`;
- base 4B raw SHA256:
  `c59383d3fd3d6087025d6e1ff649979d9d5a9e8dc73b5429a4f8e9fa41b6b8c7`;
- 9B raw SHA256:
  `ffae93774d51b87a2e29258d170a84f8b165f996e2e78eedd102271dfc260044`.
- config SHA256:
  `27202f9c43325366c8cf35070e9d040b8f5ee52ce8745f1e15b6f3f7d732c1dd`.

## Full Gate

Relative to base 4B, candidate must:

- score at least 90/96 GSM8K, 67/96 MMLU, and 6/19 GPQA-Diamond;
- have micro >=163/211 and macro >= base macro;
- have no significantly negative paired micro result;
- introduce zero baseline-correct regressions outside proof-backed overrides;
- preserve case, prompt, strategy, budget, scorer, and API contracts.

Relative to 9B, report task scores, micro/macro delta, paired bootstrap 95%
CI, exact McNemar p, and discordance counts. Significant superiority still
requires CI lower bound above zero and p <0.05 with no task below 9B.

Only a full per-task base-4B non-regression pass may authorize the previously
frozen independent holdout. It does not authorize merge, scale, or RL.

## Reproduction

```bash
PYTHONPATH=. ../.venv/bin/python -m nano_harness.cli verified-choice-full \
  --config configs/harness/anchored_v1_verified_choice_full_v1.json
```
