# Nemotron Nano Harness Experiment Report

## Scope

This report records only observed results. Synthetic contract tasks are used to
debug harness mechanics; they are not substitutes for SWE-bench, Tau-bench, or
CL-bench scores.

## Environment

- Model endpoint: OpenRouter
- Model: `nvidia/nemotron-nano-9b-v2:free`
- Date: 2026-08-05
- Python: workspace shared Python 3.12 environment
- Docker: unavailable
- SWE data: SWE-bench Lite, 300 rows cached through `hf-mirror.com`
- CL data: direct Hugging Face unavailable; mirror metadata reachable, full
  1,899-task JSONL downloaded through `hf-mirror.com`
- Tau source: fixed revision `59a200c6d575d595120f1cb70fea53cef0632f6b`;
  107 runtime files downloaded with historical trajectories and few-shot
  demonstrations explicitly excluded

## Synthetic Iterations

The two tasks validate a strict JSON contract and a tool-backed state report.

| Iteration | Base score | Optimized score | Observation |
| --- | ---: | ---: | --- |
| v1 | 0.0 | 0.0 | Exact-string scorer incorrectly rejected semantically correct evidence. |
| v2 | 1.0 | 0.5 | Unconditional audit preserved a wrong JSON field type. |
| v3 | 1.0 | 0.0 | Unconditional audit removed a useful observed tool fact; validator remained too rigid. |
| v4 | 1.0 | 1.0 | Machine-triggered contract audit removed the regression and reduced unnecessary rewriting. |

The improvement from optimized v3 to v4 is `+1.0` absolute on this two-task
debug set. Optimized v4 only matches base, so this is not evidence that the
optimized harness outperforms base on a benchmark.

Raw trajectories are stored under `results/iterations/`,
`results/baselines/`, and `results/optimized/`.

## Real SWE-bench Lite Failure Analysis

Task: `psf__requests-2317`.

Optimized SWE v1:

- explored `requests/` and read `requests/sessions.py`;
- made two tool calls;
- returned a plausible diff after three model steps;
- did not call `apply_patch`;
- did not run a test;
- left the repository worktree unchanged.

This is a false-success pattern: the model reported a patch that did not exist.
The generated patch is therefore invalid evidence and has no SWE-bench score.

The harness was changed at the root cause:

1. A SWE task cannot finish until `apply_patch` succeeds.
2. A validation command must run after the latest patch and return exit code 0.
3. The prediction sent to the evaluator is always read from the worktree with
   `git diff --binary`; model-reported diff text is retained only as diagnostic
   metadata.
4. Empty repository diffs are classified as `empty_repository_diff`.
5. Repository preparation uses retry-validated commit snapshots and isolated
   checkouts instead of a slow full-history clone.

A v2 retry reached an OpenRouter free-route response-body stall before any
mutation. The run was interrupted after eight minutes. Request timeout/retry
settings were reduced to 120 seconds and two attempts for the smoke config.

## Full-flow Start Evidence

- SWE-bench Verified: all 500 tasks cached; `psf__requests-2317` optimized probe
  started and stopped with structured `provider_daily_quota`.
- CL-bench: all 1,899 tasks downloaded; shard 0 of 20 started. Before the
  stop-on-quota fix it wrote 95 retryable quota records. Subsequent runs stop
  after the first quota record and retry it after reset.
- Tau-bench: retail (115 tasks, 16 tools) and airline (50 tasks, 14 tools)
  environments loaded offline. Retail optimized shard 0 of 10 started and wrote
  a structured `provider_daily_quota` blocker from the user simulator.
- CL single-task base inference completed on a 76k-character multi-turn task:
  14,534 prompt tokens, 1,633 completion tokens. The optimized counterpart was
  blocked by the daily quota before inference.

The CL adapter was corrected to preserve historical assistant turns. The
context compressor now pins all original task messages and only compacts later
agent/tool trajectory.

## Current Conclusions

- The base and optimized clients, tool loop, context ledger, conditional audit,
  sharding, resume, merge, and three benchmark schemas are implemented.
- Unconditional reflection is harmful for this 9B model; narrow,
  machine-triggered verification is more stable and cheaper.
- A plausible textual patch is not evidence of coding success. Repository state
  and test observations must be authoritative.
- No claim is made that optimized 9B beats base 9B, 30B, 120B, or 550B on any
  full benchmark yet.

## Next Experiments

1. Retry the gated SWE task and require a non-empty worktree diff plus successful
   focused validation.
2. Run identical base and optimized SWE subsets with fixed IDs.
3. Complete Tau-bench source setup and run retail/airline shards.
4. Complete CL-bench data and judge setup.
5. Run the full four-model base matrix and optimized 9B matrix with shard resume.
