#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

from nano_harness.baseline import sha256_file
from nano_harness.mbpp_verified_selection import (
    load_cases,
    load_config,
    select_shard,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/campaign/mbpp_sanitized_verified_selection_dev_v1.json"
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def merge() -> dict:
    config = load_config(CONFIG)
    cases = load_cases(config, ROOT)
    output_root = ROOT / config["output_dir"]
    rows = []
    shards = []
    service_sha = None
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
            raise ValueError("MBPP shard case set differs")
        if service_sha is None:
            service_sha = shard_result["identity"]["service_sha256"]
        elif service_sha != shard_result["identity"]["service_sha256"]:
            raise ValueError("MBPP shard service identities differ")
        rows.extend(shard_rows)
        shards.append(
            {
                "shard_id": shard_id,
                "rows": len(shard_rows),
                "sha256": sha256_file(path),
            }
        )
    expected_ids = {case.case_id for case in cases}
    actual_ids = {row["case_id"] for row in rows}
    if (
        len(rows) != 43
        or len(actual_ids) != 43
        or actual_ids != expected_ids
    ):
        raise ValueError("MBPP merged case set differs")
    rows.sort(key=lambda row: row["case_id"])
    merged = output_root / "cases.jsonl"
    with merged.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    result = {
        "schema_version": "nano_harness_mbpp_verified_selection_raw_v1",
        "experiment_id": config["experiment_id"],
        "identity": {
            "config_sha256": sha256_file(CONFIG),
            "raw_sha256": sha256_file(merged),
            "shards": shards,
            "service_sha256": service_sha or {},
        },
        "surface": {
            "split": config["dataset"]["split"],
            "cases": len(rows),
            "test_feasibility_probe_rows": 1,
            "test_content_used_for_policy_design": False,
        },
        "evaluation_boundary": {
            "public_tests_visible_to_model": True,
            "reference_solution_used": False,
            "test_outcome_used_by_verifier": True,
            "benchmark_rows_training_eligible": False,
            "raw_outputs_committed": False,
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
