# Anchored-v1 Verified Choice Canary v1

## Purpose

The target-blind verified-choice executor passes its synthetic local gate at
23/32 strict, 26/32 semantic, and 7/8 choice. This stage applies the exact
parser to the old sealed 40-case adapter regression canary without new model
inference.

The canary remains a post-v6-calibrated regression gate. It cannot establish
independent quality uplift and no canary case payload may enter training.

## Frozen Application

- rebuild the 40 cases from the frozen manifest and dataset identities;
- require the committed case manifest to match byte-for-byte;
- require every anchored-v1 raw row to match case ID, prompt/system hash,
  source index, output budget, expected identity, model, suite, and direct
  strategy;
- apply parser `explicit_two_expression_average_v1` only to `choice_exact`
  prompts;
- override only on one exact `Fraction` result matching one unique numeric
  option;
- reuse direct output for GSM8K and every unsupported or ambiguous choice;
- score only after all routed outputs are frozen.

The parser receives only `case.prompt`. It does not receive expected answers,
prior scores, or benchmark labels beyond the generic scorer type.

## Frozen Identity

- canary manifest SHA256:
  `e213985897e1da260c24e8f383e80d02a3f9c880a09f45e6cc2cc27f51dcf0f8`;
- committed case manifest SHA256:
  `03122cb316732e006dd196ef886ca76796c54a2eefa5f94ef9554dd4b2013b53`;
- anchored-v1 raw SHA256:
  `c42b266b234b3f77f9b22ff8f70e5f2c7de6503a9fa96282b7fcfa4522c74b91`;
- anchored-v1 public receipt SHA256:
  `d84268672681583b3f4bbc4ed85abf6c7f59abf89cab2b7b56f916eb832a2cdf`;
- verified-choice local pass receipt SHA256:
  `f226f12956d3a72fdc3f6923a069c1786dbfa1e08496e4502eb2561a424009c2`.
- config SHA256:
  `3cc141f3b1505f7e9fd222c7fad21e4f220ece8ed16e73952d3c67e082a36c2b`.

## Canary Gate

Anchored-v1 baseline must reproduce:

- GSM8K 15/16;
- MMLU 13/16;
- GPQA-Diamond 4/8;
- total 32/40;
- zero API errors, parse failures, and length truncations.

Candidate must complete the same 40 cases with:

- no benchmark below the anchored-v1 point estimate;
- total >=32/40;
- zero baseline-correct to candidate-wrong regressions;
- all fallback outputs byte-identical to direct;
- every override backed by a reproducible target-blind receipt;
- no parse, API, prompt, case, budget, scorer, or strategy drift.

Passing authorizes only the old 211-case development suite. The independent
holdout remains closed until that suite passes per-task base-4B
non-regression. Merge, scale, and RL remain forbidden.

## Reproduction

```bash
PYTHONPATH=. ../.venv/bin/python -m nano_harness.cli \
  verified-choice-canary \
  --config configs/harness/anchored_v1_verified_choice_canary_v1.json
```
