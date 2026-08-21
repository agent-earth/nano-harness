#!/usr/bin/env python3

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from datasets import Dataset

from nano_harness.baseline import sha256_file
from nano_harness.mbpp_full_train_replication import (
    case_ids_sha256,
    load_config,
    load_replication_cases,
    verify_unchanged_policy,
)
from nano_harness.mbpp_iterative_repair import load_few_shots, select_shard


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/campaign/mbpp_full_train_replication_v2.json"
PUBLIC = (
    ROOT / "docs/experiments/mbpp_full_train_replication_v2.preregister.json"
)
MARKDOWN = ROOT / "docs/experiments/mbpp_full_train_replication_v2.md"


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
    for key in ("v2_result_path", "confirmation_result_path"):
        path = ROOT / config["predecessor"][key]
        digest = config["predecessor"][key.removesuffix("_path") + "_sha256"]
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError("MBPP replication predecessor identity differs")
    confirmation = json.loads(
        (
            ROOT / config["predecessor"]["confirmation_result_path"]
        ).read_text(encoding="utf-8")
    )
    if (
        confirmation.get("decision", {}).get("confirmation_admitted")
        is not False
        or confirmation.get("decision", {}).get("rerun_or_tuning_allowed")
        is not False
    ):
        raise ValueError("MBPP replication predecessor gate differs")
    dataset = config["dataset"]
    cases = load_replication_cases(config, ROOT)
    few_shots = load_few_shots(config, ROOT)
    case_task_ids = {case.task_id for case in cases}
    comparisons = {}
    for name in (
        "sanitized_validation",
        "full_validation",
        "sanitized_test",
        "prompt",
    ):
        path = (ROOT / dataset[f"{name}_path"]).resolve()
        digest = dataset[f"{name}_sha256"]
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"MBPP replication {name} identity differs")
        comparisons[name] = len(case_task_ids & task_ids(path))
    if any(comparisons.values()):
        raise ValueError("MBPP full-train replication overlaps frozen rows")
    shards = [
        select_shard(
            cases,
            num_shards=config["execution"]["num_shards"],
            shard_id=shard_id,
        )
        for shard_id in range(config["execution"]["num_shards"])
    ]
    ids = [case.case_id for shard in shards for _, case in shard]
    if len(ids) != 254 or len(set(ids)) != 254:
        raise ValueError("MBPP full-train replication shard coverage differs")
    return {
        "schema_version": "nano_harness_mbpp_full_train_preregister_v2",
        "experiment_id": config["experiment_id"],
        "identity": {
            "code_revision": git_revision(),
            "config_sha256": sha256_file(CONFIG),
            "full_train_sha256": dataset["full_train_sha256"],
            "excluded_sanitized_train_sha256": dataset[
                "excluded_sanitized_train_sha256"
            ],
            "replication_case_ids_sha256": case_ids_sha256(cases),
            "few_shot_task_ids": [
                example.task_id for example in few_shots
            ],
            "v2_config_sha256": config["predecessor"][
                "v2_config_sha256"
            ],
            "v2_result_sha256": config["predecessor"][
                "v2_result_sha256"
            ],
            "confirmation_result_sha256": config["predecessor"][
                "confirmation_result_sha256"
            ],
        },
        "surface": {
            "split": "full_train_minus_sanitized_train",
            "cases": len(cases),
            "task_id_min": min(case.task_id for case in cases),
            "task_id_max": max(case.task_id for case in cases),
            "overlap_with_sanitized_validation": comparisons[
                "sanitized_validation"
            ],
            "overlap_with_full_validation": comparisons["full_validation"],
            "overlap_with_sanitized_test": comparisons["sanitized_test"],
            "overlap_with_few_shot_prompt": comparisons["prompt"],
            "shard_counts": [len(shard) for shard in shards],
            "sanitized_test_generation_started": False,
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
            "replication_admitted": (
                "all four_b_preservation and nine_b_superiority gates pass"
            ),
            "sanitized_test_preregistration_allowed": (
                "replication_admitted"
            ),
            "rerun_or_tuning_allowed": False,
        },
        "execution_boundary": config["execution_boundary"],
        "claim_boundary": (
            "This pre-registers one 254-case exact unchanged-policy "
            "replication on full MBPP train rows excluded from prior "
            "sanitized-train development. It starts no model generation, "
            "uses no validation or test rows, and establishes no benchmark "
            "test score."
        ),
    }


def render_markdown(receipt: dict[str, Any]) -> str:
    return f"""# MBPP Full-Train Replication v2

This freezes the unchanged v2 harness on 254 full-train tasks absent from
sanitized train. It starts no model generation and is not a test score.

- policy: byte-for-byte-equivalent model, prompt, parser, direct, candidate,
  and sandbox sections from the frozen v2 config;
- reference solution: hidden;
- prior overlap: zero with sanitized/full validation, sanitized test, and
  official few-shot rows;
- case IDs SHA:
  `{receipt['identity']['replication_case_ids_sha256']}`;
- config SHA: `{receipt['identity']['config_sha256']}`;
- shard counts: `{receipt['surface']['shard_counts']}`.

Passing requires non-regression versus direct 4B and significant superiority
over matched 9B: positive paired-bootstrap lower bound, exact McNemar p<0.05,
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
