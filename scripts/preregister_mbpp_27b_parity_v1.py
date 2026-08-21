#!/usr/bin/env python3

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from nano_harness.baseline import sha256_file
from nano_harness.mbpp_27b_parity import (
    case_ids_sha256,
    load_config,
    load_policy_config,
)
from nano_harness.mbpp_iterative_repair import load_few_shots, select_shard
from nano_harness.mbpp_sanitized_test import load_test_cases


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/campaign/mbpp_27b_parity_v1.json"
PUBLIC = ROOT / "docs/experiments/mbpp_27b_parity_v1.preregister.json"
MARKDOWN = ROOT / "docs/experiments/mbpp_27b_parity_v1.md"


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def build_receipt() -> dict[str, Any]:
    config = load_config(CONFIG)
    policy = load_policy_config(config, ROOT)
    candidate = config["candidate"]
    raw_path = ROOT / candidate["raw_path"]
    report_path = ROOT / candidate["report_path"]
    if (
        not raw_path.is_file()
        or sha256_file(raw_path) != candidate["raw_sha256"]
        or not report_path.is_file()
        or sha256_file(report_path) != candidate["report_sha256"]
    ):
        raise ValueError("MBPP parity candidate identity differs")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("decision", {}).get("complete_benchmark_superiority")
        is not True
        or report.get("decision", {}).get("rerun_or_tuning_allowed")
        is not False
    ):
        raise ValueError("MBPP parity candidate gate differs")
    service_path = ROOT / config["twenty_seven_b"]["serving_report_path"]
    if (
        not service_path.is_file()
        or sha256_file(service_path)
        != config["twenty_seven_b"]["serving_report_sha256"]
    ):
        raise ValueError("MBPP parity serving report identity differs")
    service = json.loads(service_path.read_text(encoding="utf-8"))
    if (
        service.get("decision", {}).get("bf16_tp2_service_ready") is not True
        or service.get("decision", {}).get("parity_preregistration_allowed")
        is not True
        or service.get("service", {}).get("max_model_len") != 4096
    ):
        raise ValueError("MBPP parity serving gate differs")
    cases = load_test_cases(config, ROOT)
    few_shots = load_few_shots(policy, ROOT)
    shards = [
        select_shard(
            cases,
            num_shards=config["execution"]["num_shards"],
            shard_id=shard_id,
        )
        for shard_id in range(config["execution"]["num_shards"])
    ]
    ids = [case.case_id for shard in shards for _, case in shard]
    if len(ids) != 257 or len(set(ids)) != 257:
        raise ValueError("MBPP parity shard coverage differs")
    return {
        "schema_version": "nano_harness_mbpp_27b_parity_preregister_v1",
        "experiment_id": config["experiment_id"],
        "identity": {
            "code_revision": git_revision(),
            "config_sha256": sha256_file(CONFIG),
            "test_sha256": config["dataset"]["test_sha256"],
            "test_case_ids_sha256": case_ids_sha256(ids),
            "candidate_raw_sha256": candidate["raw_sha256"],
            "candidate_report_sha256": candidate["report_sha256"],
            "policy_config_sha256": config["policy_source"]["config_sha256"],
            "serving_report_sha256": config["twenty_seven_b"][
                "serving_report_sha256"
            ],
            "model_config_sha256": config["twenty_seven_b"][
                "model_config_sha256"
            ],
            "model_index_sha256": config["twenty_seven_b"][
                "model_index_sha256"
            ],
            "few_shot_task_ids": [
                example.task_id for example in few_shots
            ],
        },
        "surface": {
            "benchmark": "mbpp",
            "split": "sanitized_test",
            "complete_benchmark": True,
            "cases": len(cases),
            "shard_counts": [len(shard) for shard in shards],
            "four_b_generation_repeated": False,
            "nine_b_generation_repeated": False,
            "parity_generation_started": False,
        },
        "comparison": {
            "candidate": candidate["model"],
            "baseline": config["twenty_seven_b"]["model"],
            "metric": "public_test_full_pass",
            "paired": True,
            "noninferiority_margin": config["statistics"][
                "noninferiority_margin"
            ],
            "gate": (
                "paired bootstrap 95% CI lower bound for candidate minus "
                "27B is at least -0.02"
            ),
        },
        "direct": config["direct"],
        "statistics": config["statistics"],
        "execution": config["execution"],
        "execution_boundary": config["execution_boundary"],
        "policy": config["policy"],
        "claim_boundary": (
            "This pre-registers one complete 257-case MBPP parity run. It "
            "reuses the frozen 4B candidate and generates only the 27B "
            "direct arm. No parity or superiority result exists yet."
        ),
    }


def render_markdown(receipt: dict[str, Any]) -> str:
    return f"""# MBPP 27B Parity v1

This freezes one complete 257-case MBPP parity comparison before any 27B
benchmark generation.

- 4B candidate: reuse only, raw SHA
  `{receipt['identity']['candidate_raw_sha256']}`;
- 27B: validated BF16 TP=2 vLLM service, 4096-token context;
- case IDs SHA: `{receipt['identity']['test_case_ids_sha256']}`;
- config SHA: `{receipt['identity']['config_sha256']}`;
- shards: `{receipt['surface']['shard_counts']}`;
- noninferiority margin: 2 percentage points.

Parity passes only when the paired-bootstrap 95% lower confidence bound for
4B candidate minus 27B is at least -0.02. The run is one-shot and its rows or
outputs may not enter training, reward, verifier fitting, or tuning.
"""


def main() -> None:
    receipt = build_receipt()
    PUBLIC.parent.mkdir(parents=True, exist_ok=True)
    PUBLIC.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    MARKDOWN.write_text(render_markdown(receipt), encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
