#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

from nano_harness.baseline import sha256_file
from nano_harness.router_skill_registry_v5 import load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/harness/qwen35_router_skill_registry_v5.json"
PREREG = (
    ROOT / "docs/experiments/qwen35_router_skill_registry_v5.preregister.json"
)
RAW_FOUR = (
    ROOT
    / "results/harness/qwen35-router-skill-registry-v5/services/"
    "four.models.json"
)
RAW_NINE = (
    ROOT
    / "results/harness/qwen35-router-skill-registry-v5/services/"
    "nine.models.json"
)
OUTPUT = (
    ROOT
    / "docs/experiments/qwen35_router_skill_registry_v5_service.public.json"
)


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def committed_preregister_sha256() -> str:
    content = subprocess.check_output(
        [
            "git",
            "show",
            "HEAD:docs/experiments/"
            "qwen35_router_skill_registry_v5.preregister.json",
        ],
        cwd=ROOT,
    )
    return hashlib.sha256(content).hexdigest()


def build_receipt() -> dict:
    config = load_config(CONFIG)
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    rows = [
        *json.loads(RAW_FOUR.read_text(encoding="utf-8")).get("data", []),
        *json.loads(RAW_NINE.read_text(encoding="utf-8")).get("data", []),
    ]
    by_id = {row.get("id"): row for row in rows}
    if (
        prereg.get("schema_version")
        != "nano_harness_router_skill_registry_preregister_v5"
        or prereg.get("identity", {}).get("config_sha256")
        != sha256_file(CONFIG)
        or committed_preregister_sha256() != sha256_file(PREREG)
        or prereg.get("execution_boundary", {}).get(
            "model_generation_started"
        )
        is not False
        or set(by_id) != set(config.service_models.values())
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
        raise ValueError("router skill registry V5 service identity differs")
    return {
        "schema_version": "nano_harness_router_skill_registry_v5_service",
        "experiment_id": config.experiment_id,
        "service_revision": git_revision(),
        "config_sha256": sha256_file(CONFIG),
        "preregister_sha256": sha256_file(PREREG),
        "four_health_sha256": sha256_file(RAW_FOUR),
        "nine_health_sha256": sha256_file(RAW_NINE),
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
        "healthy": True,
        "recorded_unix": int(time.time()),
        "v5_generation_started": False,
        "v1_v2_v3_v4_rerun": False,
        "benchmark_accessed": False,
        "canary_accessed": False,
        "holdout_accessed": False,
        "claim_boundary": (
            "This receipt proves only V5 service readiness and contains no "
            "generation or quality evidence."
        ),
    }


def main() -> None:
    receipt = build_receipt()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
