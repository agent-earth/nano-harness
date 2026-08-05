#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/../.venv/bin/python}"
TAU_BENCH_ROOT="${TAU_BENCH_ROOT:-$ROOT/benchmark-sources/tau-bench}"
: "${OPENROUTER_API_KEY:?OPENROUTER_API_KEY must be set for the agent and user simulator}"
MODEL="${MODEL:-nvidia/nemotron-nano-9b-v2:free}"
STRATEGY="${STRATEGY:-optimized}"
ENVIRONMENT="${ENVIRONMENT:-retail}"
USER_MODEL="${USER_MODEL:-openrouter/nvidia/nemotron-3-super-120b-a12b:free}"
USER_PROVIDER="${USER_PROVIDER:-openrouter}"
USER_STRATEGY="${USER_STRATEGY:-llm}"
SHARD_ID="${SHARD_ID:-0}"
NUM_SHARDS="${NUM_SHARDS:-10}"
MODEL_SLUG="$(printf '%s' "$MODEL" | tr '/:' '__')"

export PYTHONPATH="$ROOT:$TAU_BENCH_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
"$PYTHON" -m nano_harness.tau_runner \
  --env "$ENVIRONMENT" \
  --model "$MODEL" \
  --strategy "$STRATEGY" \
  --user-model "$USER_MODEL" \
  --user-provider "$USER_PROVIDER" \
  --user-strategy "$USER_STRATEGY" \
  --num-shards "$NUM_SHARDS" \
  --shard-id "$SHARD_ID" \
  --output "results/full/taubench_${ENVIRONMENT}_${MODEL_SLUG}_${STRATEGY}/shard-${SHARD_ID}-of-${NUM_SHARDS}.jsonl"
