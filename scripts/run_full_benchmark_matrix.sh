#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/../.venv/bin/python}"
SHARD_ID="${SHARD_ID:-0}"
NUM_SHARDS="${NUM_SHARDS:-20}"
DRY_RUN="${DRY_RUN:-0}"

MODELS=(
  "nvidia/nemotron-nano-9b-v2:free"
  "nvidia/nemotron-3-nano-30b-a3b:free"
  "nvidia/nemotron-3-super-120b-a12b:free"
  "nvidia/nemotron-3-ultra-550b-a55b:free"
)

run_command() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [[ "$DRY_RUN" != "1" ]]; then
    "$@"
  fi
}

for model in "${MODELS[@]}"; do
  slug="$(printf '%s' "$model" | tr '/:' '__')"
  for benchmark in swebench clbench; do
    template="$ROOT/configs/benchmarks/${benchmark}_template.yaml"
    generated="$ROOT/results/full/configs/${benchmark}_${slug}_base_${SHARD_ID}.yaml"
    mkdir -p "$(dirname "$generated")"
    "$PYTHON" "$ROOT/scripts/render_run_config.py" \
      --template "$template" \
      --output "$generated" \
      --model "$model" \
      --strategy base \
      --shard-id "$SHARD_ID" \
      --num-shards "$NUM_SHARDS" \
      --run-id "${benchmark}_${slug}_base"
    run_command "$PYTHON" -m nano_harness.cli run --config "$generated"
  done
done

model="nvidia/nemotron-nano-9b-v2:free"
slug="$(printf '%s' "$model" | tr '/:' '__')"
for benchmark in swebench clbench; do
  template="$ROOT/configs/benchmarks/${benchmark}_template.yaml"
  generated="$ROOT/results/full/configs/${benchmark}_${slug}_optimized_${SHARD_ID}.yaml"
  "$PYTHON" "$ROOT/scripts/render_run_config.py" \
    --template "$template" \
    --output "$generated" \
    --model "$model" \
    --strategy optimized \
    --shard-id "$SHARD_ID" \
    --num-shards "$NUM_SHARDS" \
    --run-id "${benchmark}_${slug}_optimized"
  run_command "$PYTHON" -m nano_harness.cli run --config "$generated"
done

echo "Tau-bench uses scripts/run_taubench_nano.sh because it owns a live stateful environment."
