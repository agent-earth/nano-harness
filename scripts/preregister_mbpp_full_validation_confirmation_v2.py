#!/usr/bin/env python3

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from datasets import Dataset

from nano_harness.baseline import sha256_file
from nano_harness.mbpp_confirmation import (
    case_ids_sha256,
    load_config,
    load_confirmation_cases,
)
from nano_harness.mbpp_iterative_repair import load_few_shots, select_shard


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/campaign/mbpp_full_validation_confirmation_v2.json"
PUBLIC = (
    ROOT
    / "docs/experiments/mbpp_full_validation_confirmation_v2.preregister.json"
)
MARKDOWN = ROOT / "docs/experiments/mbpp_full_validation_confirmation_v2.md"


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def build_receipt() -> dict[str, Any]:
    config = load_config(CONFIG)
    predecessor_path = ROOT / config["predecessor"]["v2_result_path"]
    if (
        not predecessor_path.is_file()
        or sha256_file(predecessor_path)
        != config["predecessor"]["v2_result_sha256"]
    ):
        raise ValueError("MBPP confirmation predecessor identity differs")
    predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
    if (
        predecessor.get("decision", {}).get("method_supported") is not True
        or predecessor.get("decision", {}).get(
            "fresh_validation_preregistration_allowed"
        )
        is not True
        or predecessor.get("decision", {}).get("rerun_or_tuning_allowed")
        is not False
    ):
        raise ValueError("MBPP confirmation predecessor gate differs")
    for key in ("test_path",):
        path = (ROOT / config["dataset"][key]).resolve()
        digest = config["dataset"][key.removesuffix("_path") + "_sha256"]
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError("MBPP confirmation test identity differs")
    cases = load_confirmation_cases(config, ROOT)
    few_shots = load_few_shots(config, ROOT)
    train_path = (
        ROOT / config["dataset"]["sanitized_train_path"]
    ).resolve()
    if (
        not train_path.is_file()
        or sha256_file(train_path)
        != config["dataset"]["sanitized_train_sha256"]
    ):
        raise ValueError("MBPP confirmation train identity differs")
    train_task_ids = {
        int(row["task_id"])
        for row in Dataset.from_parquet(str(train_path))
    }
    confirmation_task_ids = {case.task_id for case in cases}
    if confirmation_task_ids & train_task_ids:
        raise ValueError("MBPP confirmation overlaps sanitized train")
    shards = [
        select_shard(
            cases,
            num_shards=config["execution"]["num_shards"],
            shard_id=shard_id,
        )
        for shard_id in range(config["execution"]["num_shards"])
    ]
    ids = [case.case_id for shard in shards for _, case in shard]
    if len(ids) != 47 or len(set(ids)) != 47:
        raise ValueError("MBPP confirmation shard coverage differs")
    return {
        "schema_version": "nano_harness_mbpp_confirmation_preregister_v2",
        "experiment_id": config["experiment_id"],
        "identity": {
            "code_revision": git_revision(),
            "config_sha256": sha256_file(CONFIG),
            "full_validation_sha256": config["dataset"][
                "full_validation_sha256"
            ],
            "excluded_sanitized_validation_sha256": config["dataset"][
                "excluded_sanitized_validation_sha256"
            ],
            "test_sha256": config["dataset"]["test_sha256"],
            "confirmation_case_ids_sha256": case_ids_sha256(cases),
            "few_shot_task_ids": [
                example.task_id for example in few_shots
            ],
            "v2_result_sha256": config["predecessor"]["v2_result_sha256"],
        },
        "surface": {
            "cases": len(cases),
            "task_id_min": min(case.task_id for case in cases),
            "task_id_max": max(case.task_id for case in cases),
            "overlap_with_sanitized_validation": 0,
            "overlap_with_sanitized_train": len(
                confirmation_task_ids & train_task_ids
            ),
            "shard_counts": [len(shard) for shard in shards],
            "test_generation_started": False,
        },
        "candidate": config["candidate"],
        "prompt": config["prompt"],
        "parser": config["parser"],
        "sandbox": config["sandbox"],
        "statistics": config["statistics"],
        "execution": config["execution"],
        "decision_rule": {
            "confirmation_admitted": (
                "all four_b_preservation and nine_b_superiority gates pass"
            ),
            "complete_test_preregistration_allowed": (
                "confirmation_admitted"
            ),
            "rerun_or_tuning_allowed": False,
        },
        "execution_boundary": config["execution_boundary"],
        "claim_boundary": (
            "This pre-registers one fresh 47-case confirmation. It starts "
            "no generation, does not rerun prior train or validation rows, "
            "and establishes no MBPP test score."
        ),
    }


def render_markdown(receipt: dict[str, Any]) -> str:
    return f"""# MBPP Full-Validation Confirmation v2

This freezes the v2 policy on 47 full-validation tasks that are absent from
sanitized validation and sanitized train. It starts no model generation.

- policy: official three-shot, five candidates after direct failure, up to
  three deterministic public-test repair rounds;
- reference solution: hidden;
- sandbox: no network, read-only root, isolated Python, bounded CPU, memory,
  file size, open files, and wall time;
- case IDs SHA:
  `{receipt['identity']['confirmation_case_ids_sha256']}`;
- config SHA: `{receipt['identity']['config_sha256']}`;
- shard counts: `{receipt['surface']['shard_counts']}`.

Passing requires non-regression versus direct 4B and significant superiority
over matched 9B: positive paired bootstrap lower bound, exact McNemar p<0.05,
at least six candidate-only wins, and more wins than losses. Only then may the
257-case sanitized test be separately pre-registered.
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
