#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

from nano_harness.baseline import sha256_file
from nano_harness.router_serving_parity_v2 import load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/harness/qwen35_router_serving_parity_v2.json"
PREREG = (
    ROOT
    / "docs/experiments/qwen35_router_serving_parity_v2.preregister.json"
)
RAW_HEALTH = (
    ROOT
    / "results/harness/qwen35-router-serving-parity-v2/services/models.json"
)
OUTPUT = (
    ROOT
    / "docs/experiments/"
    "qwen35_router_serving_parity_service_v2.public.json"
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
            "qwen35_router_serving_parity_v2.preregister.json",
        ],
        cwd=ROOT,
    )
    return hashlib.sha256(content).hexdigest()


def build_receipt() -> dict:
    config = load_config(CONFIG)
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    health = json.loads(RAW_HEALTH.read_text(encoding="utf-8"))
    if (
        prereg.get("schema_version")
        != "nano_harness_router_serving_parity_preregister_v2"
        or prereg.get("identity", {}).get("config_sha256")
        != sha256_file(CONFIG)
        or committed_preregister_sha256() != sha256_file(PREREG)
        or prereg.get("execution_boundary", {}).get(
            "parity_service_started"
        )
        is not False
        or prereg.get("execution_boundary", {}).get(
            "model_generation_started"
        )
        is not False
    ):
        raise ValueError("router serving parity v2 preregistration differs")
    rows = health.get("data", [])
    by_id = {row.get("id"): row for row in rows}
    expected = set(config.served_models.values())
    if (
        set(by_id) != expected
        or any(row.get("owned_by") != "vllm" for row in rows)
        or by_id[config.served_models["base"]].get("root")
        != "../../../models/Qwen3.5-4B"
        or by_id[config.served_models["base"]].get("parent") is not None
        or by_id[config.served_models["remapped"]].get("parent")
        != config.served_models["base"]
        or by_id[config.served_models["remapped"]].get("root")
        != config.remapped_adapter_path
    ):
        raise ValueError("router serving parity v2 health differs")
    return {
        "schema_version": "nano_harness_router_serving_parity_service_v2",
        "experiment_id": config.experiment_id,
        "service_revision": git_revision(),
        "config_sha256": sha256_file(CONFIG),
        "preregister_sha256": sha256_file(PREREG),
        "health_sha256": sha256_file(RAW_HEALTH),
        "models": config.served_models,
        "model_health": {
            model_id: {
                "root": row.get("root"),
                "parent": row.get("parent"),
                "max_model_len": row.get("max_model_len"),
                "owned_by": row.get("owned_by"),
            }
            for model_id, row in sorted(by_id.items())
        },
        "remapped_adapter_sha256": config.remapped_adapter_tree_sha256,
        "remap_receipt_sha256": config.remap_receipt_sha256,
        "serving": config.service_launch,
        "healthy": True,
        "recorded_unix": int(time.time()),
        "generation_started": False,
        "evaluation_started": False,
        "benchmark_accessed": False,
        "canary_accessed": False,
        "holdout_accessed": False,
        "fresh_integration_accessed": False,
        "claim_boundary": (
            "This receipt proves only base plus remapped LoRA readiness. It "
            "contains no model generation or quality evidence."
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
                "generation_started": receipt["generation_started"],
                "output": str(OUTPUT),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
