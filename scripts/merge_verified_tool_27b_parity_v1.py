#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

from nano_harness.baseline import sha256_file
from nano_harness.verified_tool_27b_parity import (
    load_config,
    load_source,
    select_shard,
)
from nano_harness.verified_tool_execution import build_cases


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/campaign/verified_tool_27b_parity_v1.json"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def merge() -> dict:
    config = load_config(CONFIG)
    parent, _ = load_source(config, ROOT)
    cases = build_cases(parent)
    output_root = ROOT / config["output_dir"]
    rows = []
    shards = []
    for shard_id in range(config["execution"]["num_shards"]):
        path = output_root / f"shard-{shard_id}.jsonl"
        result_path = output_root / f"shard-{shard_id}.result.json"
        shard_rows = read_jsonl(path)
        shard_result = json.loads(result_path.read_text(encoding="utf-8"))
        expected_ids = {
            case["case_id"]
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
        ):
            raise ValueError("verified-tool 27B parity shard differs")
        rows.extend(shard_rows)
        shards.append(
            {
                "shard_id": shard_id,
                "rows": len(shard_rows),
                "sha256": sha256_file(path),
                "service_sha256": shard_result["identity"]["service_sha256"],
            }
        )
    expected_ids = {case["case_id"] for case in cases}
    if (
        len(rows) != 256
        or len({row["case_id"] for row in rows}) != 256
        or {row["case_id"] for row in rows} != expected_ids
    ):
        raise ValueError("verified-tool 27B parity merged case set differs")
    rows.sort(key=lambda row: row["case_id"])
    merged = output_root / "cases.jsonl"
    with merged.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    result = {
        "schema_version": "nano_harness_verified_tool_27b_merged_v1",
        "experiment_id": config["experiment_id"],
        "identity": {
            "config_sha256": sha256_file(CONFIG),
            "raw_sha256": sha256_file(merged),
            "source_raw_sha256": config["source"]["raw_sha256"],
            "shards": shards,
        },
        "surface": {
            "cases": len(rows),
            "four_b_harness_reused": True,
            "four_b_generation_repeated": False,
            "nine_b_generation_repeated": False,
            "suite_changed": False,
        },
        "diagnostics": {
            "four_b_harness_correct": sum(
                row["four_b_harness"]["correct"] for row in rows
            ),
            "twenty_seven_b_correct": sum(
                row["twenty_seven_b_direct"]["correct"] for row in rows
            ),
            "twenty_seven_b_parseable": sum(
                row["twenty_seven_b_direct"]["parseable"] for row in rows
            ),
        },
    }
    (output_root / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    print(json.dumps(merge(), indent=2, sort_keys=True))
