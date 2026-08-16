# Generic Choice Capability Matrix Evaluation v1

## Purpose

The narrow verified-choice executor improves one synthetic development case
but produces zero overrides on the old matched benchmark suite. This fresh,
history-disjoint matrix measures where the capability exists without tuning on
benchmark prompts.

## Frozen Arms

Run the same 48 evaluation-only cases through:

- anchored-v1 4B direct;
- frozen Qwen3.5-9B direct;
- anchored-v1 4B direct plus parser
  `explicit_two_expression_average_v1`.

Both model arms use temperature 0, thinking disabled, a 32-token budget, and
the same system/user prompt. The executor arm is derived from frozen 4B direct
outputs and adds no model calls.

## Matrix Contract

Six families contain 8 rows each:

- explicit average with one exact option;
- explicit average with no exact option;
- verbal average without explicit expressions;
- host count;
- sequential remaining fraction;
- duplicate-option ambiguity.

There are 32 scored cases and 16 ambiguity cases. Null-reference ambiguity
cases are never counted as incorrect; they measure parser override safety,
fallback parity, and output parseability.

The matrix is deterministic, has zero prompt/data overlap with v1-v11, and is
explicitly forbidden for SFT, preference training, RL, reward-model training,
verifier training, or case-level feedback training.

## Frozen Identity

- matrix SHA256:
  `5db7561b95f6b951ef7fb45293e24a39276b69b5b43e04c63712f8450e37b933`;
- anchored-v1 serving receipt SHA256:
  `2549527942acfe53a1eb352453649a9ea3cc31d68bb9790c865553ee95c2f578`;
- anchored-v1 serving weights SHA256:
  `9ce7be3954f8e0f3d245fe846d6e35275243b7f0caf66cb847fd716173658649`;
- 4B model: `qwen3.5-4b-anchor-v1`;
- 9B model: `qwen3.5-9b`;
- config SHA256:
  `f40be9cc5f1a98fb101724bae43a3ede1f477f4cf2d7a4956fa7efc42a2d7257`.

## Required Evidence

Report:

- scored accuracy by arm and family;
- 4B executor fixes/regressions relative to 4B direct;
- 4B/9B paired scored-case comparison;
- parser expected-route agreement;
- ambiguity overrides, fallback parity, and parse failures;
- token use, latency, model and data identities;
- exact failure families and next mechanism decision.

No matrix result alone authorizes the independent holdout, merge, scale, or
RL. It only informs a separately pre-registered next intervention.

## Reproduction

```bash
PYTHONPATH=. NANO_HARNESS_API_KEY=local-vllm ../.venv/bin/python \
  -m nano_harness.cli choice-matrix-eval \
  --config configs/harness/generic_choice_capability_matrix_eval_v1.json
```
