#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/../.venv/bin/python}"
CONFIG="${1:-$ROOT/configs/benchmarks/clbench_template.yaml}"

cd "$ROOT"
"$PYTHON" -m nano_harness.cli run --config "$CONFIG"

echo "Inference is complete. Grade merged output with Tencent-Hunyuan/CL-bench eval.py."
echo "The official judge defaults to GPT-5.1 and requires a separate compatible API key."
