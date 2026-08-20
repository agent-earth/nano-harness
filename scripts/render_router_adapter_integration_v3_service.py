#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

from nano_harness.baseline import sha256_file
from nano_harness.router_adapter_integration_v3 import load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/harness/qwen35_router_adapter_integration_v3.json"
PREREG = (
    ROOT
    / "docs/experiments/qwen35_router_adapter_integration_v3.preregister.json"
)
RAW_FOUR_HEALTH = (
    ROOT
    / "results/harness/qwen35-router-adapter-integration-v3/services/"
    "four.models.json"
)
RAW_NINE_HEALTH = (
    ROOT
    / "results/harness/qwen35-router-adapter-integration-v3/services/"
    "nine.models.json"
)
OUTPUT = (
    ROOT
    / "docs/experiments/"
    "qwen35_router_adapter_integration_v3_service.public.json"
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
            "qwen35_router_adapter_integration_v3.preregister.json",
        ],
        cwd=ROOT,
    )
    return hashlib.sha256(content).hexdigest()


def build_receipt() -> dict:
    config = load_config(CONFIG)
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    four_health = json.loads(RAW_FOUR_HEALTH.read_text(encoding="utf-8"))
    nine_health = json.loads(RAW_NINE_HEALTH.read_text(encoding="utf-8"))
    if (
        prereg.get("schema_version")
        != "nano_harness_router_adapter_integration_preregister_v3"
        or prereg.get("identity", {}).get("config_sha256")
        != sha256_file(CONFIG)
        or committed_preregister_sha256() != sha256_file(PREREG)
        or prereg.get("execution_boundary", {}).get(
            "model_generation_started"
        )
        is not False
        or prereg.get("execution_boundary", {}).get(
            "integration_v1_or_v2_rerun"
        )
        is not False
    ):
        raise ValueError("router integration v3 preregistration differs")
    rows = [*four_health.get("data", []), *nine_health.get("data", [])]
    by_id = {row.get("id"): row for row in rows}
    expected = set(config.service_models.values())
    if (
        set(by_id) != expected
        or any(row.get("owned_by") != "vllm" for row in rows)
        or by_id[config.service_models["four_b_base"]].get("root")
        != "../../../models/Qwen3.5-4B"
        or by_id[config.service_models["four_b_base"]].get("parent") is not None
        or by_id[config.service_models["remapped_router"]].get("root")
        != config.adapter_path
        or by_id[config.service_models["remapped_router"]].get("parent")
        != config.service_models["four_b_base"]
        or by_id[config.service_models["nine_b_base"]].get("root")
        != "../../../models/Qwen3.5-9B"
        or by_id[config.service_models["nine_b_base"]].get("parent") is not None
    ):
        raise ValueError("router integration v3 service health differs")
    return {
        "schema_version": "nano_harness_router_adapter_integration_v3_service",
        "experiment_id": config.experiment_id,
        "service_revision": git_revision(),
        "config_sha256": sha256_file(CONFIG),
        "preregister_sha256": sha256_file(PREREG),
        "four_health_sha256": sha256_file(RAW_FOUR_HEALTH),
        "nine_health_sha256": sha256_file(RAW_NINE_HEALTH),
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
        "serving": config.service_launch,
        "healthy": True,
        "recorded_unix": int(time.time()),
        "v3_generation_started": False,
        "integration_v1_or_v2_rerun": False,
        "benchmark_accessed": False,
        "canary_accessed": False,
        "holdout_accessed": False,
        "claim_boundary": (
            "This receipt proves only 4B, 9B, and remapped-router service "
            "readiness for the frozen V3 contract. It contains no generation."
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
                "v3_generation_started": receipt["v3_generation_started"],
                "integration_v1_or_v2_rerun": receipt[
                    "integration_v1_or_v2_rerun"
                ],
                "output": str(OUTPUT),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
