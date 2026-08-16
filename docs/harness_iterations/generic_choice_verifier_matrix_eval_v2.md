# Generic Choice Verifier Matrix Evaluation v2

## Purpose

The first fresh matrix establishes significant verified-executor gains for
explicit arithmetic averages, while host-count and verbal-average direct
families remain weak. This stage evaluates two additional target-blind proof
rules on a new history-disjoint matrix.

## Frozen Parser v2

Parser v2 preserves explicit-average parser v1 and adds:

- host count: one coordinator plus `invited` delegates plus
  `invited * guests_per_invitee`;
- verbal average: exact rational average of two explicitly named depot
  counts.

Each rule requires a unique prompt grammar match, exactly four unique numeric
options A-D, and one exact `Fraction` result matching one option. There is no
rounding, tolerance, nearest-option mapping, target access, or model call.
No-exact-option and duplicate-option cases must fall back to constrained 4B
direct output.

## Frozen Matrix

Use the 48-case evaluation-only generic choice verifier matrix v2:

- host exact / no-exact / duplicate option: 8 each;
- verbal average exact / no-exact / duplicate option: 8 each;
- 16 scored exact cases;
- 32 ambiguity cases;
- 0 training-eligible cases.

Every SFT, preference, RL, reward-model, verifier, and case-feedback training
use is forbidden.

## Frozen Arms

- anchored-v1 4B constrained direct;
- Qwen3.5-9B constrained direct;
- anchored-v1 constrained direct plus parser
  `host_count_and_verbal_average_v2`.

Both direct arms use temperature 0, thinking disabled, 32 tokens, and
structured regex `FINAL: [A-D]`. The executor arm is derived from the frozen
4B constrained outputs.

## Identity

- matrix SHA256:
  `70330f730f144c1fb05d50a27e566321561451a962d0bdf736f27b8faa2f79b0`;
- prior matched matrix report SHA256:
  `f225e9625701686757601fa615699301cd1b9374c982451e6a6f7998fd13bf11`;
- 4B model config SHA256:
  `ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670`;
- 9B model config SHA256:
  `d0883072e01861ed0b2d47be3c16c36a8e81c224c7ffaa310c6558fb3f932b05`;
- anchored-v1 serving receipt SHA256:
  `2549527942acfe53a1eb352453649a9ea3cc31d68bb9790c865553ee95c2f578`;
- config SHA256:
  `00e5b2125edee0c8e54c771746952375d6227bb69aa518bb195b642e61dd1ef1`.

## Required Evidence

- 48/48 parseable outputs for both direct arms;
- 16/16 exact cases routed to verified override;
- 32/32 ambiguity cases routed to fallback with direct parity;
- scored and family metrics for all three arms;
- paired executor versus 4B and executor versus 9B statistics;
- reproducible proof receipts and expected-route agreement;
- data, model, serving, token, and latency identities.

This matrix is mechanism evidence, not benchmark or independent-holdout
quality evidence. No result authorizes holdout, merge, scale, or RL.

## Reproduction

```bash
PYTHONPATH=. NANO_HARNESS_API_KEY=local-vllm ../.venv/bin/python \
  -m nano_harness.cli choice-verifier-matrix-eval \
  --config configs/harness/generic_choice_verifier_matrix_eval_v2.json
```
