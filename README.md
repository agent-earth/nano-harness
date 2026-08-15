# nano-harness

`nano-harness` is a resumable agent evaluation harness for
`nvidia/nemotron-nano-9b-v2:free`. It keeps a minimal base strategy and an
optimized strategy so harness changes can be compared without changing the
model, task, or output schema.

The optimized strategy implements:

- explicit plan, act, observe, and verify guidance;
- a persistent state ledger and bounded context compaction;
- one-tool-call-at-a-time execution with schema and error recovery;
- machine-triggered output audits instead of unconditional self-reflection;
- SWE-bench repository exploration, bounded coding tools, real mutation gates,
  post-patch validation gates, and patches read only from `git diff`;
- Tau-bench policy/state reminders and an `Env`/`Action` bridge;
- CL-bench context/constraint tracking and rubric-oriented final audit;
- atomic JSONL checkpoints, task-ID resume, sharding, and deterministic merge.

## Status

The repository is an active long-running experiment, not a finished performance
claim. Current evidence is documented in
[`docs/nemotron_nano_harness_report.md`](docs/nemotron_nano_harness_report.md).

The current local Qwen3.5-4B/9B matched baseline contract is documented in
[`docs/qwen35_baseline.md`](docs/qwen35_baseline.md).

- Unit tests: 26 passing.
- OpenRouter synthetic smoke: completed for nano base and optimized.
- Real SWE-bench Lite task generation: completed once, then correctly rejected
  as an invalid self-reported patch because no repository mutation or test was
  observed. A stricter completion gate is now implemented.
- Official SWE-bench Docker scoring, full Tau-bench, and CL-bench judge scoring
  still require the documented external runtime/data dependencies.

## Environment

Use the shared workspace environment when available:

```bash
PYTHON=../.venv/bin/python
$PYTHON -m pip install -i https://bytedpypi.byted.org/simple/ -e .
```

Set the OpenRouter key only in the process environment:

```bash
export OPENROUTER_API_KEY=...
```

Do not commit API keys. The key used for the recorded smoke runs is not stored
in this repository.

## Quick Verification

```bash
PYTHONPATH=. ../.venv/bin/python -m unittest discover -s tests -v
```

Run the built-in contract smoke:

```bash
PYTHONPATH=. ../.venv/bin/python -m nano_harness.cli run \
  --config configs/benchmarks/synthetic_base.yaml

PYTHONPATH=. ../.venv/bin/python -m nano_harness.cli run \
  --config configs/benchmarks/synthetic_optimized.yaml
```

Rerunning either command skips completed task IDs. Delete or move the shard file
to intentionally start a fresh trial.

## SWE-bench

Download data through `oniond` with the verified `ai-infra` bucket:

```bash
export BUCKET=ai-infra
oniond download dataset SWE-bench_Lite --dir ../../datasets/SWE-bench_Lite
```

The template defaults to SWE-bench Verified and a 20-way shard:

```bash
SHARD_ID=0 NUM_SHARDS=20 ./scripts/run_swebench_nano.sh
```

Repository snapshots are cached by `repo@base_commit`; each instance gets a
clean isolated checkout. Agent-visible coding tools are restricted to repository
paths and an executable allowlist.

Merge shards:

```bash
PYTHONPATH=. ../.venv/bin/python -m nano_harness.cli merge \
  --input 'results/full/swebench_*/shard-*.jsonl' \
  --output results/full/swebench_merged.jsonl
```

Official scoring requires Docker or the official cloud evaluator:

```bash
python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Verified \
  --predictions_path results/full/swebench_merged.jsonl \
  --max_workers 8 \
  --run_id nano-harness
```

## Tau-bench

Use the official compatible source checkout and set `TAU_BENCH_ROOT`:

```bash
TAU_BENCH_ROOT=/path/to/tau-bench \
SHARD_ID=0 NUM_SHARDS=10 \
./scripts/run_taubench_nano.sh
```

Tau-bench user strategies are model-backed, including `verify`. The matrix
therefore pins both the agent model and a separate user-simulator model. The
default is `openrouter/nvidia/nemotron-3-super-120b-a12b:free`; override
`USER_MODEL`, `USER_PROVIDER`, and `USER_STRATEGY` explicitly and keep them
identical across comparisons. Outputs preserve reward, info, trajectory, trial,
and NanoHarness metadata.

## CL-bench

Place the official `CL-bench.jsonl` at `data/CL-bench.jsonl`, then run:

```bash
SHARD_ID=0 NUM_SHARDS=20 ./scripts/run_clbench_nano.sh
```

Merge inference shards, then grade using the official Tencent-Hunyuan evaluator.
Official grading uses GPT-5.1 by default and needs a separate judge credential;
model inference scores must not be claimed before this grading step completes.

## Full Matrix

The required generation matrix is encoded in:

```bash
DRY_RUN=1 SHARD_ID=0 NUM_SHARDS=20 ./scripts/run_full_benchmark_matrix.sh
```

It includes base harness runs for 9B, 30B, 120B, and 550B, plus the optimized
9B run. Tau-bench is launched separately because it owns a live stateful
environment.

```bash
DRY_RUN=1 SHARD_ID=0 NUM_SHARDS=10 ./scripts/run_full_taubench_matrix.sh
```

See `results/full/run_status.json` before launching. With the current OpenRouter
free tier, the minimum SWE + CL request count alone is 11,995, so a 50
request/day key cannot finish the required matrix in a practical interval.

## Layout

```text
harnesses/                         Reserved for additional harness variants
nano_harness/                      Runtime, prompts, tools, adapters
configs/nemotron/                  Model matrix
configs/benchmarks/                Benchmark templates
scripts/                           Sharded runners and result utilities
results/baselines/                 Recorded base evidence
results/optimized/                 Recorded optimized evidence
results/iterations/                Failure-driven iteration history
docs/                              Reports and full-run status
```
