# Full Benchmark Comparison

## Matrix

| Harness | Model | SWE-bench | Tau-bench | CL-bench |
| --- | --- | --- | --- | --- |
| base | Nemotron Nano 9B v2 | pending | pending | pending |
| base | Nemotron 3 Nano 30B A3B | pending | pending | pending |
| base | Nemotron 3 Super 120B A12B | pending | pending | pending |
| base | Nemotron 3 Ultra 550B A55B | pending | pending | pending |
| optimized | Nemotron Nano 9B v2 | generation smoke only | pending | pending |

No pending cell is interpreted as zero, failure, or success.

## Full-run Protocol

- Fixed dataset revision and task IDs per comparison.
- Same API provider and model sampling settings.
- One JSONL record per task with raw trajectory and usage.
- Atomic append and completed-ID resume.
- `num_shards` and `shard_id` partitioning.
- Deterministic task-ID merge.
- Official benchmark evaluators produce final scores.
- Relative improvement is computed only after both compared runs complete:
  `(optimized - base) / base`.
- Failure distributions are reported alongside aggregate score.

## Current Blockers

- This host has no Docker executable, so official local SWE-bench scoring cannot
  run here. Generation can proceed and scoring can move to Docker, Modal, or
  `sb-cli`.
- Official CL-bench grading requires a judge model credential separate from the
  tested OpenRouter model credential.
- Free OpenRouter routes can stall or throttle long tool-use requests. The
  runner records task completion atomically and uses bounded retries so full
  shards can be resumed.

## Reproducible Start

```bash
DRY_RUN=1 SHARD_ID=0 NUM_SHARDS=20 ./scripts/run_full_benchmark_matrix.sh
```

Remove `DRY_RUN=1` only after the model key, benchmark data, and scoring runtime
are available.
