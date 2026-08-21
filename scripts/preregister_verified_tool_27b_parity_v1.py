#!/usr/bin/env python3

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from nano_harness.baseline import sha256_file
from nano_harness.verified_tool_27b_parity import (
    load_config,
    select_shard,
)
from nano_harness.verified_tool_execution import (
    build_cases,
    public_case_contract,
)
from nano_harness.verified_tool_execution_v2 import (
    load_config as load_source_config,
    parent_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/campaign/verified_tool_27b_parity_v1.json"
PUBLIC = (
    ROOT / "docs/experiments/verified_tool_27b_parity_v1.preregister.json"
)
MARKDOWN = ROOT / "docs/experiments/verified_tool_27b_parity_v1.md"


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def build_receipt() -> dict:
    config = load_config(CONFIG)
    source = config["source"]
    source_config_path = ROOT / source["config_path"]
    source_raw_path = ROOT / source["raw_path"]
    source_report_path = ROOT / source["report_path"]
    if (
        sha256_file(source_config_path) != source["config_sha256"]
        or sha256_file(source_raw_path) != source["raw_sha256"]
        or sha256_file(source_report_path) != source["report_sha256"]
    ):
        raise ValueError("verified-tool parity source identity differs")
    source_config = load_source_config(source_config_path)
    parent = parent_config(source_config)
    cases = build_cases(parent)
    contract = public_case_contract(cases)
    source_report = json.loads(
        source_report_path.read_text(encoding="utf-8")
    )
    if (
        contract["case_contract_sha256"]
        != source["case_contract_sha256"]
        or len(cases) != source["cases"]
        or source_report.get("arms", {})
        .get("four_b_skill_verified_tool", {})
        .get("correct")
        != source["harness_correct"]
        or source_report.get("decision", {}).get("local_harness_admitted")
        is not True
    ):
        raise ValueError("verified-tool parity source contract differs")
    service_path = ROOT / config["twenty_seven_b"]["serving_report_path"]
    if (
        sha256_file(service_path)
        != config["twenty_seven_b"]["serving_report_sha256"]
    ):
        raise ValueError("verified-tool parity service report differs")
    service = json.loads(service_path.read_text(encoding="utf-8"))
    if (
        service.get("decision", {}).get("bf16_tp2_service_ready") is not True
        or service.get("decision", {}).get("parity_preregistration_allowed")
        is not True
    ):
        raise ValueError("verified-tool parity service gate differs")
    shards = [
        select_shard(
            cases,
            num_shards=config["execution"]["num_shards"],
            shard_id=shard_id,
        )
        for shard_id in range(config["execution"]["num_shards"])
    ]
    ids = [case["case_id"] for shard in shards for _, case in shard]
    if len(ids) != 256 or len(set(ids)) != 256:
        raise ValueError("verified-tool parity shard coverage differs")
    return {
        "schema_version": (
            "nano_harness_verified_tool_27b_parity_preregister_v1"
        ),
        "experiment_id": config["experiment_id"],
        "identity": {
            "code_revision": git_revision(),
            "config_sha256": sha256_file(CONFIG),
            "source_config_sha256": source["config_sha256"],
            "source_raw_sha256": source["raw_sha256"],
            "source_report_sha256": source["report_sha256"],
            "case_contract_sha256": contract["case_contract_sha256"],
            "harness_rows_sha256": source["harness_rows_sha256"],
            "serving_report_sha256": config["twenty_seven_b"][
                "serving_report_sha256"
            ],
            "model_config_sha256": config["twenty_seven_b"][
                "model_config_sha256"
            ],
            "model_index_sha256": config["twenty_seven_b"][
                "model_index_sha256"
            ],
        },
        "surface": {
            "cases": len(cases),
            "families": {
                family: sum(case["family"] == family for case in cases)
                for family in sorted({case["family"] for case in cases})
            },
            "case_contract": contract,
            "shard_counts": [len(shard) for shard in shards],
            "four_b_generation_repeated": False,
            "nine_b_generation_repeated": False,
            "parity_generation_started": False,
        },
        "comparison": {
            "candidate": "qwen3.5-4b+verified-tool-v2",
            "baseline": config["twenty_seven_b"]["model"],
            "metric": "exact_integer_accuracy",
            "paired": True,
            "noninferiority_margin": config["statistics"][
                "noninferiority_margin"
            ],
            "gate": (
                "overall and every-family paired-bootstrap 95% CI lower "
                "bounds are at least -0.02"
            ),
        },
        "direct": config["direct"],
        "statistics": config["statistics"],
        "execution": config["execution"],
        "execution_boundary": config["execution_boundary"],
        "policy": config["policy"],
        "claim_boundary": (
            "This pre-registers one complete 256-case synthetic verified-tool "
            "parity run. It reuses the frozen 4B harness result and generates "
            "only the 27B direct arm. It is not an external benchmark score."
        ),
    }


def render_markdown(receipt: dict) -> str:
    return f"""# Verified-Tool 27B Parity v1

This freezes one complete 256-case, four-family parity comparison before any
27B evaluation generation.

- frozen 4B harness: 256/256, raw SHA
  `{receipt['identity']['source_raw_sha256']}`;
- 27B arm: direct constrained generation from the validated BF16 TP=2 service;
- case contract SHA: `{receipt['identity']['case_contract_sha256']}`;
- config SHA: `{receipt['identity']['config_sha256']}`;
- shard counts: `{receipt['surface']['shard_counts']}`;
- noninferiority margin: 2 percentage points.

Parity requires the paired-bootstrap 95% lower bound to be at least -0.02 both
overall and in every one of the four families. The run is one-shot, and it
uses no benchmark rows or outputs.
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
