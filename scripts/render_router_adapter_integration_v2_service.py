#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

from nano_harness.baseline import sha256_file
from nano_harness.router_adapter_integration_v2 import load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/harness/qwen35_router_adapter_integration_v2.json"
PREREG = (
    ROOT
    / "docs/experiments/qwen35_router_adapter_integration_v2.preregister.json"
)
PARITY_SERVICE = (
    ROOT
    / "docs/experiments/"
    "qwen35_router_serving_parity_service_v1.public.json"
)
RAW_HEALTH = (
    ROOT
    / "results/harness/qwen35-router-adapter-integration-v2/services/"
    "models.json"
)
OUTPUT = (
    ROOT
    / "docs/experiments/"
    "qwen35_router_adapter_integration_v2_service.public.json"
)


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def committed_preregister_sha256() -> str:
    content = subprocess.check_output(
        [
            "git",
            "show",
            "HEAD:docs/experiments/"
            "qwen35_router_adapter_integration_v2.preregister.json",
        ],
        cwd=ROOT,
    )
    return hashlib.sha256(content).hexdigest()


def build_receipt() -> dict:
    config = load_config(CONFIG)
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    parity_service = json.loads(PARITY_SERVICE.read_text(encoding="utf-8"))
    health = json.loads(RAW_HEALTH.read_text(encoding="utf-8"))
    if (
        prereg.get("schema_version")
        != "nano_harness_router_adapter_integration_preregister_v2"
        or prereg.get("identity", {}).get("config_sha256")
        != sha256_file(CONFIG)
        or committed_preregister_sha256() != sha256_file(PREREG)
        or prereg.get("execution_boundary", {}).get(
            "model_generation_started"
        )
        is not False
        or prereg.get("execution_boundary", {}).get("integration_v1_rerun")
        is not False
        or parity_service.get("schema_version")
        != "nano_harness_router_serving_parity_service_v1"
        or parity_service.get("healthy") is not True
        or parity_service.get("models")
        != {
            "base": config.service_models["base"],
            "original": config.service_models["original_unused"],
            "remapped": config.service_models["remapped_router"],
        }
        or parity_service.get("remapped_adapter_sha256")
        != config.adapter_tree_sha256
    ):
        raise ValueError("router integration v2 prereg or parity service differs")
    rows = health.get("data", [])
    by_id = {row.get("id"): row for row in rows}
    expected = set(config.service_models.values())
    if (
        set(by_id) != expected
        or any(row.get("owned_by") != "vllm" for row in rows)
        or by_id[config.service_models["base"]].get("parent") is not None
        or by_id[config.service_models["remapped_router"]].get("parent")
        != config.service_models["base"]
        or by_id[config.service_models["original_unused"]].get("parent")
        != config.service_models["base"]
    ):
        raise ValueError("router integration v2 service health differs")
    return {
        "schema_version": "nano_harness_router_adapter_integration_v2_service",
        "experiment_id": config.experiment_id,
        "service_revision": git_revision(),
        "config_sha256": sha256_file(CONFIG),
        "preregister_sha256": sha256_file(PREREG),
        "parity_service_sha256": sha256_file(PARITY_SERVICE),
        "health_sha256": sha256_file(RAW_HEALTH),
        "models": config.service_models,
        "model_health": {
            model_id: {
                "root": row.get("root"),
                "parent": row.get("parent"),
                "max_model_len": row.get("max_model_len"),
                "owned_by": row.get("owned_by"),
            }
            for model_id, row in sorted(by_id.items())
        },
        "remapped_adapter_sha256": config.adapter_tree_sha256,
        "remapped_adapter_weights_sha256": config.adapter_weights_sha256,
        "healthy": True,
        "recorded_unix": int(time.time()),
        "v2_generation_started": False,
        "integration_v1_rerun": False,
        "benchmark_accessed": False,
        "canary_accessed": False,
        "holdout_accessed": False,
        "claim_boundary": (
            "This receipt proves only readiness for the new V2 case contract. "
            "It does not rerun V1 or establish transfer quality."
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
                "models": receipt["models"],
                "healthy": receipt["healthy"],
                "v2_generation_started": receipt["v2_generation_started"],
                "integration_v1_rerun": receipt["integration_v1_rerun"],
                "output": str(OUTPUT),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
