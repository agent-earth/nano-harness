#!/usr/bin/env python3

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from nano_harness.baseline import sha256_file
from nano_harness.verified_tool_execution import load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/harness/qwen35_verified_tool_execution_v1.json"
PREREG = (
    ROOT
    / "docs/experiments/qwen35_verified_tool_execution_v1.preregister.json"
)
RAW_ROOT = (
    ROOT / "results/harness/qwen35-verified-tool-execution-v1/services"
)
FOUR_HEALTH = RAW_ROOT / "4b.models.json"
NINE_HEALTH = RAW_ROOT / "9b.models.json"
OUTPUT = (
    ROOT
    / "docs/experiments/"
    "qwen35_verified_tool_execution_services_v1.public.json"
)


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def _validate_health(
    path: Path,
    *,
    model_id: str,
    model_root: str,
    max_model_len: int,
) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    rows = document.get("data", [])
    if (
        len(rows) != 1
        or rows[0].get("id") != model_id
        or rows[0].get("root") != model_root
        or rows[0].get("max_model_len") != max_model_len
        or rows[0].get("owned_by") != "vllm"
    ):
        raise ValueError(f"verified tool service health differs: {model_id}")
    return {
        "served_model": model_id,
        "root": model_root,
        "max_model_len": max_model_len,
        "health_sha256": sha256_file(path),
    }


def build_receipt() -> dict:
    config = load_config(CONFIG)
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    if (
        prereg["schema_version"]
        != "nano_harness_verified_tool_execution_preregister_v1"
        or prereg["identity"]["config_sha256"] != sha256_file(CONFIG)
        or prereg["execution_boundary"]["service_started"] is not False
        or prereg["execution_boundary"]["model_generation_started"] is not False
    ):
        raise ValueError("verified tool preregistration boundary differs")
    four = _validate_health(
        FOUR_HEALTH,
        model_id=config.four_b_model,
        model_root=config.four_b_model_path,
        max_model_len=config.max_model_len,
    )
    nine = _validate_health(
        NINE_HEALTH,
        model_id=config.nine_b_model,
        model_root=config.nine_b_model_path,
        max_model_len=config.max_model_len,
    )
    if (
        sha256_file(
            Path(config.triton_libcuda_path) / "libcuda.so.1"
        )
        != config.triton_libcuda_sha256
    ):
        raise ValueError("verified tool service libcuda differs")
    return {
        "schema_version": "nano_harness_verified_tool_services_v1",
        "experiment_id": config.experiment_id,
        "identity": {
            "service_revision": git_revision(),
            "config_sha256": sha256_file(CONFIG),
            "preregister_sha256": sha256_file(PREREG),
            "four_b_model_config_sha256": (
                config.four_b_model_config_sha256
            ),
            "four_b_model_index_sha256": config.four_b_model_index_sha256,
            "nine_b_model_config_sha256": (
                config.nine_b_model_config_sha256
            ),
            "nine_b_model_index_sha256": config.nine_b_model_index_sha256,
        },
        "models": {
            config.four_b_model: {
                **four,
                "port": 8000,
                "gpu_index": 0,
            },
            config.nine_b_model: {
                **nine,
                "port": 8001,
                "gpu_index": 1,
            },
        },
        "serving": {
            "vllm_version": config.vllm_version,
            "dtype": config.serving_dtype,
            "max_model_len": config.max_model_len,
            "gpu_memory_utilization": config.gpu_memory_utilization,
            "enforce_eager": config.enforce_eager,
            "max_num_batched_tokens": config.max_num_batched_tokens,
            "max_num_seqs": config.max_num_seqs,
            "triton_libcuda_path": config.triton_libcuda_path,
            "triton_libcuda_sha256": config.triton_libcuda_sha256,
        },
        "recorded_unix": int(time.time()),
        "generation_started": False,
        "evaluation_started": False,
        "benchmark_accessed": False,
        "canary_accessed": False,
        "claim_boundary": (
            "This receipt establishes only exact local serving readiness. It "
            "is not model generation, harness quality, benchmark, canary, or "
            "holdout evidence."
        ),
    }


def main() -> None:
    receipt = build_receipt()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "experiment_id": receipt["experiment_id"],
                "identity": receipt["identity"],
                "models": receipt["models"],
                "serving": receipt["serving"],
                "generation_started": receipt["generation_started"],
                "output": str(OUTPUT),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
