# Generic Choice Capability Matrix Evaluation v2

## Purpose

Matrix v1 supports the verified executor relative to anchored-4B direct, but
its frozen 9B arm is invalid: 48/48 outputs spend the full 32-token budget on
reasoning and truncate before `FINAL`. V2 repairs only that comparison
contract.

## Frozen Change

Reuse the exact v1 matrix, prompts, model endpoints, temperature, thinking
setting, and 32-token budget. Add the same vLLM structured-output grammar to
both direct arms:

```text
FINAL: [A-D]
```

The executor arm is derived from the new constrained 4B direct outputs, not
from v1 outputs. V1 9B generations are neither reused nor normalized.

## Frozen Arms

- anchored-v1 4B constrained direct;
- Qwen3.5-9B constrained direct;
- anchored-v1 constrained direct plus parser
  `explicit_two_expression_average_v1`.

All three arms share the same 48 evaluation-only cases. The 32 scored cases
and 16 ambiguity cases remain separate.

## Identity

- matrix SHA256:
  `5db7561b95f6b951ef7fb45293e24a39276b69b5b43e04c63712f8450e37b933`;
- v1 public receipt SHA256:
  `11787551bc9dfa49d6f0e7cdee9409fe359e555d7b0304909b023ba700f11e62`;
- 4B model config SHA256:
  `ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670`;
- 9B model config SHA256:
  `d0883072e01861ed0b2d47be3c16c36a8e81c224c7ffaa310c6558fb3f932b05`;
- anchored-v1 serving receipt SHA256:
  `2549527942acfe53a1eb352453649a9ea3cc31d68bb9790c865553ee95c2f578`;
- config SHA256:
  `47776628053eb2f3fa34e2a6cfc7b6682734a702a03aab8bff834c48d11d9c6c`.

## Required Evidence

- 48/48 parseable outputs for both direct arms;
- scored accuracy and family metrics for all three arms;
- executor fixes/regressions versus constrained 4B direct;
- paired 4B executor versus 9B direct comparison on 32 scored cases;
- ambiguity override and fallback parity audits;
- expected-route agreement and reproducible executor receipts;
- token, latency, model, serving, and matrix identities.

The matrix is not independent benchmark quality evidence and remains forbidden
for all training uses. No result authorizes the independent holdout, merge,
scale, or RL.

## Reproduction

```bash
PYTHONPATH=. NANO_HARNESS_API_KEY=local-vllm ../.venv/bin/python \
  -m nano_harness.cli choice-matrix-eval-v2 \
  --config configs/harness/generic_choice_capability_matrix_eval_v2.json
```
