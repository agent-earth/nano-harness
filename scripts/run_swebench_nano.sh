#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/../.venv/bin/python}"
CONFIG="${1:-$ROOT/configs/benchmarks/swebench_template.yaml}"

cd "$ROOT"
"$PYTHON" -m nano_harness.cli run --config "$CONFIG"

echo "Generation is complete. Score merged predictions with the official evaluator:"
echo "python -m swebench.harness.run_evaluation --dataset_name princeton-nlp/SWE-bench_Verified --predictions_path <merged.jsonl> --max_workers <n> --run_id <id>"
