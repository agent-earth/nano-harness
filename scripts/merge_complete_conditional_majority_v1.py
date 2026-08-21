#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from nano_harness.baseline import load_cases, load_manifest, sha256_file
from nano_harness.complete_conditional_majority import load_config
from nano_harness.v5_complete_treatment import jsonl_rows
from scripts.run_complete_conditional_majority_shard_v1 import (
    EXECUTION,
    load_execution,
    select_shard,
)


ROOT = Path(__file__).resolve().parents[1]


def merge() -> dict:
    execution = load_execution(EXECUTION)
    config_path = ROOT / execution["parent_config_path"]
    config = load_config(config_path)
    prefix_path = ROOT / execution["completed_prefix"]["path"]
    prefix = jsonl_rows(prefix_path)
    prefix_ids = {row["case_id"] for row in prefix}
    manifest = load_manifest(ROOT / config["baseline"]["suite_manifest_path"])
    cases = [
        case
        for case in load_cases(
            manifest,
            (ROOT / "../../../datasets").resolve(),
        )
        if case.benchmark == "gsm8k"
    ]
    expected = {case.case_id for case in cases}
    parts = list(prefix)
    shard_receipts = []
    for shard_id in range(execution["sharding"]["num_shards"]):
        path = ROOT / execution["output"]["shard_pattern"].replace(
            "<shard_id>", str(shard_id)
        )
        rows = jsonl_rows(path)
        selected_ids = {
            case.case_id
            for _, case in select_shard(
                cases,
                prefix_ids=prefix_ids,
                num_shards=execution["sharding"]["num_shards"],
                shard_id=shard_id,
            )
        }
        if (
            len(rows) != len(selected_ids)
            or {row["case_id"] for row in rows} != selected_ids
        ):
            raise ValueError("complete conditional majority shard differs")
        parts.extend(rows)
        shard_receipts.append(
            {
                "shard_id": shard_id,
                "rows": len(rows),
                "sha256": sha256_file(path),
            }
        )
    by_id = {row["case_id"]: row for row in parts}
    if len(parts) != 1_319 or len(by_id) != 1_319 or set(by_id) != expected:
        raise ValueError("complete conditional majority merge differs")

    merged_path = ROOT / execution["output"]["merged_gsm8k_path"]
    with merged_path.open("w", encoding="utf-8") as handle:
        for case_id in sorted(by_id):
            handle.write(json.dumps(by_id[case_id], sort_keys=True) + "\n")
    receipts_path = ROOT / config["output"]["gsm8k_receipts_path"]
    with receipts_path.open("w", encoding="utf-8") as handle:
        for case_id in sorted(by_id):
            handle.write(
                json.dumps(
                    {
                        "case_id": case_id,
                        "receipt": by_id[case_id]["treatment_receipt"],
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    four_b = jsonl_rows(ROOT / config["baseline"]["four_b_raw_path"])
    prior = jsonl_rows(
        ROOT / config["predecessors"]["prior_complete_candidate_path"]
    )
    complete = list(by_id.values())
    complete.extend(
        {
            **row,
            "suite_id": config["experiment_id"],
            "model": "qwen3.5-4b-complete-conditional-majority",
            "treatment_route": "mmlu_direct_preserve",
        }
        for row in four_b
        if row["benchmark"] == "mmlu"
    )
    complete.extend(
        {
            **row,
            "suite_id": config["experiment_id"],
            "model": "qwen3.5-4b-complete-conditional-majority",
            "treatment_route": "gpqa_frozen_v5_reuse",
        }
        for row in prior
        if row["benchmark"] == "gpqa_diamond"
    )
    complete.sort(key=lambda row: row["case_id"])
    if (
        len(complete) != 15_559
        or len({row["case_id"] for row in complete}) != 15_559
    ):
        raise ValueError("complete conditional majority composition differs")
    complete_path = ROOT / config["output"]["complete_candidate_path"]
    with complete_path.open("w", encoding="utf-8") as handle:
        for row in complete:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    result = {
        "schema_version": (
            "nano_harness_complete_conditional_majority_raw_v1"
        ),
        "experiment_id": config["experiment_id"],
        "identity": {
            "config_sha256": sha256_file(config_path),
            "execution_sha256": sha256_file(EXECUTION),
            "prefix_sha256": execution["completed_prefix"]["sha256"],
            "shards": shard_receipts,
            "gsm8k_candidate_sha256": sha256_file(merged_path),
            "gsm8k_receipts_sha256": sha256_file(receipts_path),
            "complete_candidate_sha256": sha256_file(complete_path),
        },
        "surface": {
            "generated_benchmark": "gsm8k",
            "generated_cases": len(by_id),
            "mmlu_model_requests": 0,
            "gpqa_diamond_model_requests": 0,
            "execution_shards": 2,
        },
        "evaluation_boundary": {
            "benchmark_rows_training_eligible": False,
            "expected_answer_used_during_generation": False,
            "case_correctness_used_during_generation": False,
            "scoring_applied_after_generation": True,
            "raw_outputs_committed": False,
        },
    }
    result_path = ROOT / config["output"]["result_path"]
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    print(json.dumps(merge(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
