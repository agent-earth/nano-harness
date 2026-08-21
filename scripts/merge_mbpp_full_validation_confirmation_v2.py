#!/usr/bin/env python3

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from nano_harness.baseline import sha256_file
from nano_harness.mbpp_confirmation import (
    load_config,
    load_confirmation_cases,
)
from nano_harness.mbpp_iterative_repair import select_shard


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/campaign/mbpp_full_validation_confirmation_v2.json"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def merge() -> dict:
    config = load_config(CONFIG)
    cases = load_confirmation_cases(config, ROOT)
    output_root = ROOT / config["output_dir"]
    rows = []
    shards = []
    for shard_id in range(config["execution"]["num_shards"]):
        path = output_root / f"shard-{shard_id}.jsonl"
        result_path = output_root / f"shard-{shard_id}.result.json"
        shard_rows = read_jsonl(path)
        shard_result = json.loads(result_path.read_text(encoding="utf-8"))
        expected_ids = {
            case.case_id
            for _, case in select_shard(
                cases,
                num_shards=config["execution"]["num_shards"],
                shard_id=shard_id,
            )
        }
        actual_ids = {row["case_id"] for row in shard_rows}
        if (
            len(shard_rows) != len(expected_ids)
            or len(actual_ids) != len(expected_ids)
            or actual_ids != expected_ids
            or shard_result.get("identity", {}).get("raw_sha256")
            != sha256_file(path)
            or shard_result.get("surface", {}).get("shard_id") != shard_id
            or shard_result.get("surface", {}).get("num_shards")
            != config["execution"]["num_shards"]
        ):
            raise ValueError("MBPP confirmation shard case set differs")
        rows.extend(shard_rows)
        shards.append(
            {
                "shard_id": shard_id,
                "rows": len(shard_rows),
                "sha256": sha256_file(path),
                "service_sha256": shard_result["identity"]["service_sha256"],
            }
        )
    expected_ids = {case.case_id for case in cases}
    actual_ids = {row["case_id"] for row in rows}
    if (
        len(rows) != 47
        or len(actual_ids) != 47
        or actual_ids != expected_ids
    ):
        raise ValueError("MBPP confirmation merged case set differs")
    rows.sort(key=lambda row: row["case_id"])
    merged = output_root / "cases.jsonl"
    with merged.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    result = {
        "schema_version": "nano_harness_mbpp_confirmation_merged_v2",
        "experiment_id": config["experiment_id"],
        "identity": {
            "config_sha256": sha256_file(CONFIG),
            "raw_sha256": sha256_file(merged),
            "shards": shards,
        },
        "surface": {
            "split": "full_validation_minus_sanitized_validation",
            "cases": len(rows),
            "sanitized_validation_v1_rerun": False,
            "train_v2_rerun": False,
            "test_generation_started": False,
        },
        "diagnostics": {
            "four_b_direct_full_pass": sum(
                row["four_b_direct"]["test_result"]["full_pass"]
                for row in rows
            ),
            "nine_b_direct_full_pass": sum(
                row["nine_b_direct"]["test_result"]["full_pass"]
                for row in rows
            ),
            "candidate_full_pass": sum(
                row["candidate"]["test_result"]["full_pass"] for row in rows
            ),
            "overrides": sum(row["receipt"]["override"] for row in rows),
            "replicas_generated": sum(
                row["receipt"]["replicas_generated"] for row in rows
            ),
            "repair_rounds_generated": sum(
                row["receipt"]["repair_rounds_generated"] for row in rows
            ),
            "selected_source_counts": dict(
                sorted(
                    Counter(
                        row["receipt"]["selected_source"] for row in rows
                    ).items()
                )
            ),
        },
    }
    (output_root / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    print(json.dumps(merge(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
