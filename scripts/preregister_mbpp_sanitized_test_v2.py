#!/usr/bin/env python3

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from datasets import Dataset

from nano_harness.baseline import sha256_file
from nano_harness.mbpp_iterative_repair import load_few_shots, select_shard
from nano_harness.mbpp_sanitized_test import (
    case_ids_sha256,
    load_config,
    load_test_cases,
    verify_unchanged_policy,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/campaign/mbpp_sanitized_test_v2.json"
PUBLIC = ROOT / "docs/experiments/mbpp_sanitized_test_v2.preregister.json"
MARKDOWN = ROOT / "docs/experiments/mbpp_sanitized_test_v2.md"


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def task_ids(path: Path) -> set[int]:
    return {
        int(row["task_id"])
        for row in Dataset.from_parquet(str(path))
    }


def build_receipt() -> dict[str, Any]:
    config = load_config(CONFIG)
    verify_unchanged_policy(config, ROOT)
    predecessor_path = ROOT / config["predecessor"]["replication_result_path"]
    if (
        not predecessor_path.is_file()
        or sha256_file(predecessor_path)
        != config["predecessor"]["replication_result_sha256"]
    ):
        raise ValueError("MBPP sanitized-test predecessor identity differs")
    predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
    if (
        predecessor.get("decision", {}).get("replication_admitted") is not True
        or predecessor.get("decision", {}).get(
            "sanitized_test_preregistration_allowed"
        )
        is not True
        or predecessor.get("decision", {}).get("rerun_or_tuning_allowed")
        is not False
    ):
        raise ValueError("MBPP sanitized-test predecessor gate differs")
    dataset = config["dataset"]
    cases = load_test_cases(config, ROOT)
    few_shots = load_few_shots(config, ROOT)
    test_task_ids = {case.task_id for case in cases}
    overlaps = {}
    for name in ("sanitized_train", "sanitized_validation", "prompt"):
        path = (ROOT / dataset[f"{name}_path"]).resolve()
        digest = dataset[f"{name}_sha256"]
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"MBPP sanitized-test {name} identity differs")
        overlaps[name] = len(test_task_ids & task_ids(path))
    if any(overlaps.values()):
        raise ValueError("MBPP sanitized-test overlaps development rows")
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
        raise ValueError("MBPP sanitized-test shard coverage differs")
    return {
        "schema_version": "nano_harness_mbpp_sanitized_test_preregister_v2",
        "experiment_id": config["experiment_id"],
        "identity": {
            "code_revision": git_revision(),
            "config_sha256": sha256_file(CONFIG),
            "test_sha256": dataset["test_sha256"],
            "test_case_ids_sha256": case_ids_sha256(cases),
            "few_shot_task_ids": [
                example.task_id for example in few_shots
            ],
            "v2_config_sha256": config["predecessor"]["v2_config_sha256"],
            "replication_result_sha256": config["predecessor"][
                "replication_result_sha256"
            ],
        },
        "surface": {
            "split": "sanitized_test",
            "cases": len(cases),
            "task_id_min": min(case.task_id for case in cases),
            "task_id_max": max(case.task_id for case in cases),
            "overlap_with_sanitized_train": overlaps["sanitized_train"],
            "overlap_with_sanitized_validation": overlaps[
                "sanitized_validation"
            ],
            "overlap_with_few_shot_prompt": overlaps["prompt"],
            "shard_counts": [len(shard) for shard in shards],
            "test_generation_started": False,
        },
        "policy_identity": {
            "unchanged_from_v2": True,
            "compared_keys": [
                "models",
                "prompt",
                "parser",
                "direct",
                "candidate",
                "sandbox",
            ],
        },
        "candidate": config["candidate"],
        "prompt": config["prompt"],
        "parser": config["parser"],
        "sandbox": config["sandbox"],
        "statistics": config["statistics"],
        "execution": config["execution"],
        "decision_rule": {
            "complete_benchmark_superiority": (
                "all four_b_preservation and nine_b_superiority gates pass"
            ),
            "rerun_or_tuning_allowed": False,
        },
        "execution_boundary": config["execution_boundary"],
        "claim_boundary": (
            "This pre-registers the one allowed 257-case MBPP sanitized-test "
            "run. The reference solution remains hidden, outputs are "
            "training-forbidden, and no post-observation rerun is allowed."
        ),
    }


def render_markdown(receipt: dict[str, Any]) -> str:
    return f"""# MBPP Sanitized Test v2

This freezes one complete 257-case MBPP sanitized-test run. It starts no model
generation.

- policy: unchanged from the admitted v2 development and replication runs;
- reference solution: hidden;
- overlap with sanitized train, validation, and few-shot rows: zero;
- test case IDs SHA: `{receipt['identity']['test_case_ids_sha256']}`;
- config SHA: `{receipt['identity']['config_sha256']}`;
- shard counts: `{receipt['surface']['shard_counts']}`.

A complete-benchmark superiority claim requires non-regression versus direct
4B and significant superiority over matched 9B under every pre-registered
gate. The test is run once; no rerun or post-observation tuning is allowed.
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
