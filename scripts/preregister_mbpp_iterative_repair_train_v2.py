#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from nano_harness.baseline import sha256_file
from nano_harness.mbpp_iterative_repair import (
    load_config,
    load_few_shots,
    load_train_cases,
    select_shard,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/campaign/mbpp_sanitized_iterative_repair_train_v2.json"
)
PUBLIC = (
    ROOT
    / "docs/experiments/"
    "mbpp_sanitized_iterative_repair_train_v2.preregister.json"
)
MARKDOWN = (
    ROOT
    / "docs/experiments/mbpp_sanitized_iterative_repair_train_v2.md"
)


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def build_receipt() -> dict[str, Any]:
    config = load_config(CONFIG)
    predecessor_path = ROOT / config["predecessor"]["v1_report_path"]
    if (
        not predecessor_path.is_file()
        or sha256_file(predecessor_path)
        != config["predecessor"]["v1_report_sha256"]
    ):
        raise ValueError("MBPP v2 predecessor identity differs")
    predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
    if (
        predecessor.get("decision", {}).get("validation_admitted")
        is not False
        or predecessor.get("decision", {}).get(
            "rerun_or_tuning_allowed"
        )
        is not False
    ):
        raise ValueError("MBPP v2 predecessor gate differs")
    for key in ("validation_path", "test_path"):
        path = (ROOT / config["dataset"][key]).resolve()
        digest = config["dataset"][key.removesuffix("_path") + "_sha256"]
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"MBPP v2 {key} identity differs")
    cases = load_train_cases(config, ROOT)
    few_shots = load_few_shots(config, ROOT)
    shards = [
        select_shard(
            cases,
            num_shards=config["execution"]["num_shards"],
            shard_id=shard_id,
        )
        for shard_id in range(config["execution"]["num_shards"])
    ]
    all_ids = [case.case_id for shard in shards for _, case in shard]
    if len(all_ids) != 120 or len(set(all_ids)) != 120:
        raise ValueError("MBPP v2 shard coverage differs")
    return {
        "schema_version": (
            "nano_harness_mbpp_iterative_repair_preregister_v2"
        ),
        "experiment_id": config["experiment_id"],
        "identity": {
            "code_revision": git_revision(),
            "config_sha256": sha256_file(CONFIG),
            "train_sha256": config["dataset"]["train_sha256"],
            "prompt_sha256": config["dataset"]["prompt_sha256"],
            "validation_sha256": config["dataset"]["validation_sha256"],
            "test_sha256": config["dataset"]["test_sha256"],
            "train_case_ids_sha256": hashlib.sha256(
                "\n".join(case.case_id for case in cases).encode()
            ).hexdigest(),
            "few_shot_task_ids": [
                example.task_id for example in few_shots
            ],
            "v1_report_sha256": config["predecessor"]["v1_report_sha256"],
        },
        "surface": {
            "split": "train",
            "cases": len(cases),
            "shard_counts": [len(shard) for shard in shards],
            "validation_v1_rerun": False,
            "validation_rows_loaded_by_v2": False,
            "test_generation_started": False,
        },
        "candidate": config["candidate"],
        "prompt": config["prompt"],
        "parser": config["parser"],
        "sandbox": config["sandbox"],
        "statistics": config["statistics"],
        "execution": config["execution"],
        "decision_rule": {
            "method_supported": (
                "all four_b_preservation and nine_b_directional gates pass"
            ),
            "next_allowed": (
                "separately pre-register one frozen validation v2 run"
            ),
            "rerun_or_tuning_allowed": False,
        },
        "execution_boundary": config["execution_boundary"],
        "claim_boundary": (
            "This pre-registers one train-split method-development run. "
            "It starts no generation, does not reopen validation v1, and "
            "does not establish an MBPP validation or test score."
        ),
    }


def render_markdown(receipt: dict[str, Any]) -> str:
    return f"""# MBPP Sanitized Iterative Repair Train v2

This freezes one method-development run on all 120 sanitized train tasks. It
starts no generation and does not reopen the observed validation split.

## Changes From v1

- use the official three-shot examples with task IDs 2, 3, and 4;
- accept the first Python fence anywhere, `[BEGIN]`/`[DONE]`, or parseable
  plain Python;
- generate five independent candidates after a direct failure;
- perform up to three deterministic repair rounds from the current best
  candidate;
- show repair the failed public-test indices and failure classes;
- preserve passing direct 4B and override only on a strictly higher public-test
  pass count.

The evaluation reference implementation remains hidden. Generated code runs
in the same no-network, read-only-root bubblewrap sandbox.

## Identity

- config SHA: `{receipt['identity']['config_sha256']}`;
- train SHA: `{receipt['identity']['train_sha256']}`;
- train case IDs SHA: `{receipt['identity']['train_case_ids_sha256']}`;
- frozen v1 report SHA: `{receipt['identity']['v1_report_sha256']}`.

The 43-case validation v1 and 257-case test are not generated or scored here.
No post-observation tuning or rerun is allowed on this train surface.
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
