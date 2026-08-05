#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/../.venv/bin/python}"
TAU_BENCH_ROOT="${TAU_BENCH_ROOT:-$ROOT/benchmark-sources/tau-bench}"
MODEL="${MODEL:-nvidia/nemotron-nano-9b-v2:free}"
STRATEGY="${STRATEGY:-optimized}"
ENVIRONMENT="${ENVIRONMENT:-retail}"
SHARD_ID="${SHARD_ID:-0}"
NUM_SHARDS="${NUM_SHARDS:-10}"

export PYTHONPATH="$ROOT:$TAU_BENCH_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
"$PYTHON" -m nano_harness.tau_runner \
  --env "$ENVIRONMENT" \
  --model "$MODEL" \
  --strategy "$STRATEGY" \
  --num-shards "$NUM_SHARDS" \
  --shard-id "$SHARD_ID" \
  --output "results/full/taubench_${ENVIRONMENT}_${STRATEGY}/shard-${SHARD_ID}-of-${NUM_SHARDS}.jsonl"
