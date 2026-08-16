# Generic Choice Exact Replication v3 Pre-registration

## Question

Test whether the target-blind verified-choice v2 executor significantly
outperforms matched Qwen3.5-9B constrained direct decoding on a larger, fresh
exact-only replication of the host-count and verbal-average mechanisms.

This is mechanism evidence only. It cannot establish GSM8K, MMLU, GPQA,
SWE-bench, agent-benchmark, or independent-holdout superiority.

## Frozen Inputs

- dataset:
  `nano-data-pipeline/datasets/generic_choice_exact_replication_matrix_v3.json`;
- dataset SHA256:
  `0962a82e02151a7af1f3b498bab0f50d8004e630e851401687084e4d77fc8276`;
- 32 scored exact cases:
  16 host-count and 16 verbal-average;
- zero ambiguity rows because the separate v2 matrix already validated 32
  ambiguity safety cases with zero overrides;
- anchored-v1 serving adapter weights SHA256:
  `9ce7be3954f8e0f3d245fe846d6e35275243b7f0caf66cb847fd716173658649`;
- Qwen3.5-4B config SHA256:
  `ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670`;
- Qwen3.5-9B config SHA256:
  `d0883072e01861ed0b2d47be3c16c36a8e81c224c7ffaa310c6558fb3f932b05`;
- parser:
  unchanged `host_count_and_verbal_average_v2`.

The matrix builder audited all v1-v11 datasets plus matrices v1 and v2.
Case IDs, exact hashes, semantic hashes, prompt hashes, and source signatures
have zero overlap. Every row is evaluation-only and explicitly forbidden for
SFT, preference optimization, RL, reward-model training, verifier training,
or case-level feedback training.

## Frozen Decoding

All direct arms use:

- temperature `0.0`;
- `max_tokens=32`;
- `enable_thinking=false`;
- structured output regex `FINAL: [A-D]`;
- the same system prompt and per-case user prompt.

The executor reuses the anchored-v1 direct output and overrides it only when
the unchanged target-blind verifier v2 produces one unique exact option proof.
References are never passed to the parser.

## Frozen Statistics And Decision

- paired bootstrap samples: `10000`;
- paired bootstrap seed: `choice-exact-replication-v3`;
- exact two-sided McNemar alpha: `0.05`;
- significant executor superiority over 9B requires:
  - executor-minus-9B paired bootstrap 95% CI lower bound above zero;
  - exact two-sided McNemar `p < 0.05`;
  - at least six executor-only wins;
  - zero 9B-only losses.

The six-win threshold is the minimum that passes two-sided exact McNemar at
zero losses. A positive point estimate or CI without the McNemar criterion is
insufficient.

No prompt, parser, option offsets, numerical ranges, case count, seed,
budget, adapter, route, or decision threshold may change after observing this
frozen result. Any new hypothesis requires a new history-disjoint dataset and
pre-registration.

Regardless of outcome, the independent holdout remains unread. Benchmark
promotion still requires local, sealed-canary, and prior full per-task
non-regression on a separately justified intervention.

## Command

```bash
PYTHONPATH=. ../.venv/bin/python -m nano_harness.cli \
  choice-exact-replication-eval \
  --config configs/harness/generic_choice_exact_replication_eval_v3.json
```
