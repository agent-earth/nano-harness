# Critique v1 And Holdout Result

## Evidence Integrity Correction

The dev2 comparison is a valid observation of two different composite
strategies, but it does not isolate the critique stage: the incumbent draft
received `case.prompt`, while the critique arm received `case.draft_prompt`.
The original claim that critique alone caused the regression is withdrawn.

The 18-case holdout aggregate confirmation is invalid: code revision
`cf2a00e` sent `case.draft_prompt` through direct arms while the validator
checked `case.prompt`. MMLU and GPQA direct controls used a reasoning prompt
with a 32-token answer-only budget and truncated every output.

Raw artifacts and the originally observed numbers remain below for audit, but
the holdout cannot establish a significant 4B advantage. Its GSM8K subset is
still matched because the two GSM8K prompt fields are equivalent.

## Critique Decision

The composite draft-reasoning/critique strategy is rejected. On fresh dev2 it scores
0.6667 versus draft-verify at
0.7778, a -0.1111
delta. Both lost cases are GPQA. This does not isolate critique because the
draft prompt also changed. The composite strategy uses
26365 tokens versus
8418 and
249.4s versus
41.2s.

## Untouched Holdout Confirmation

The selected draft-verify policy was frozen before reading the 18 holdout
cases. The holdout has zero overlap with fixed v5 evaluation, dev1, or dev2.

| Benchmark | 4B direct | 4B draft-verify | 9B direct |
| --- | ---: | ---: | ---: |
| GSM8K | 1.0000 | 1.0000 | 1.0000 |
| MMLU | 0.0000 | 0.6667 | 0.0000 |
| GPQA-Diamond | 0.0000 | 0.5000 | 0.0000 |
| Macro | 0.3333 | 0.7222 | 0.3333 |

Against 9B direct, 4B draft-verify has seven treatment-only wins and zero
9B-only wins. The paired delta is +0.3889,
95% bootstrap CI [+0.1667, +0.6111], exact McNemar
`p=0.015625`.

This originally appeared significant, but the aggregate confirmation is
invalid because its direct controls were mismatched. A corrected fresh
confirmation is required.

## Reproduction Identity

- Code revision: `33912848bbdd3ff7339c45018688c95f6af48925`
- Dev2 incumbent raw SHA256: `9b81723336f9c8f3aadb71996fb083939e32365f1ebe2d7b75d984d405167905`
- Dev2 critique raw SHA256: `1149331a7c44fa439d8cde7d9f54578a37a5db5f3a16b440b8b72cf2c21451b6`
- Holdout 4B direct raw SHA256: `e1ba84e4ed3c3305eabdfc353060b0b859f2829933a79252c79e924a0f9adcc2`
- Holdout 4B treatment raw SHA256: `2c84bb7d9ea066ab6b787c9ab89618a23fbf05fb0f3f771a2d2e5a6c8786b220`
- Holdout 9B direct raw SHA256: `d8507f80daca4f46e707d26039d9cf78508017deb162e1075904e61d5b63ad67`

Raw outputs remain local and ignored.
