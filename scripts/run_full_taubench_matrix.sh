#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAU_BENCH_ROOT="${TAU_BENCH_ROOT:-$ROOT/benchmark-sources/tau-bench-59a200c6}"
SHARD_ID="${SHARD_ID:-0}"
NUM_SHARDS="${NUM_SHARDS:-10}"
DRY_RUN="${DRY_RUN:-0}"
USER_MODEL="${USER_MODEL:-openrouter/nvidia/nemotron-3-super-120b-a12b:free}"
USER_PROVIDER="${USER_PROVIDER:-openrouter}"
USER_STRATEGY="${USER_STRATEGY:-llm}"

MODELS=(
  "nvidia/nemotron-nano-9b-v2:free"
  "nvidia/nemotron-3-nano-30b-a3b:free"
  "nvidia/nemotron-3-super-120b-a12b:free"
  "nvidia/nemotron-3-ultra-550b-a55b:free"
)

run_one() {
  local model="$1"
  local strategy="$2"
  local environment="$3"
  printf '+ MODEL=%q STRATEGY=%q ENVIRONMENT=%q SHARD_ID=%q NUM_SHARDS=%q ' \
    "$model" "$strategy" "$environment" "$SHARD_ID" "$NUM_SHARDS"
  printf 'USER_MODEL=%q USER_PROVIDER=%q USER_STRATEGY=%q ' \
    "$USER_MODEL" "$USER_PROVIDER" "$USER_STRATEGY"
  printf 'TAU_BENCH_ROOT=%q ./scripts/run_taubench_nano.sh\n' "$TAU_BENCH_ROOT"
  if [[ "$DRY_RUN" != "1" ]]; then
    MODEL="$model" \
    STRATEGY="$strategy" \
    ENVIRONMENT="$environment" \
    SHARD_ID="$SHARD_ID" \
    NUM_SHARDS="$NUM_SHARDS" \
    USER_MODEL="$USER_MODEL" \
    USER_PROVIDER="$USER_PROVIDER" \
    USER_STRATEGY="$USER_STRATEGY" \
    TAU_BENCH_ROOT="$TAU_BENCH_ROOT" \
      "$ROOT/scripts/run_taubench_nano.sh"
  fi
}

for environment in retail airline; do
  for model in "${MODELS[@]}"; do
    run_one "$model" base "$environment"
  done
  run_one "nvidia/nemotron-nano-9b-v2:free" optimized "$environment"
done
