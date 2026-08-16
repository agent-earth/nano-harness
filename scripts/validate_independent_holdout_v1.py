#!/usr/bin/env python3

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import yaml

from scripts.build_independent_holdout_v1 import (
    DATASET_ROOT,
    EXCLUDED_RESULT_PREFIX,
    MANIFEST,
    RESULTS,
    SELECTION,
    SPECS,
    canonical_sha256,
    sha256_file,
)


def current_seen_indices() -> dict[str, set[int]]:
    seen: dict[str, set[int]] = defaultdict(set)
    for path in sorted(RESULTS.rglob("*.jsonl")):
        relative = path.relative_to(RESULTS).as_posix()
        if EXCLUDED_RESULT_PREFIX in relative:
            continue
        for line in path.read_text(encoding="utf-8", errors="strict").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            benchmark = row.get("benchmark")
            source_index = row.get("source_index")
            if benchmark in SPECS and isinstance(source_index, int):
                seen[str(benchmark)].add(source_index)
    return seen


def main() -> None:
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    receipt = json.loads(SELECTION.read_text(encoding="utf-8"))
    if manifest["suite_id"] != receipt["suite_id"]:
        raise SystemExit("holdout manifest and receipt suite IDs differ")
    if receipt["policy"]["prompts_loaded_before_evaluation"] is not False:
        raise SystemExit("holdout prompt boundary is invalid")
    if receipt["policy"]["training_eligible"] is not False:
        raise SystemExit("holdout must be training-ineligible")

    seen = current_seen_indices()
    actual_counts = {}
    for dataset in manifest["datasets"]:
        benchmark = str(dataset["name"])
        expected = SPECS[benchmark]
        indices = [int(index) for index in dataset["indices"]]
        if (
            len(indices) != int(dataset["limit"])
            or len(indices) != len(set(indices))
            or any(index < 0 for index in indices)
        ):
            raise SystemExit(f"{benchmark} manifest indices are invalid")
        if set(indices) & seen[benchmark]:
            raise SystemExit(f"{benchmark} holdout overlaps current history")
        dataset_receipt = receipt["datasets"][benchmark]
        if (
            dataset_receipt["selected_indices"] != indices
            or dataset_receipt["selected_indices_sha256"]
            != canonical_sha256(indices)
            or dataset_receipt["history_overlap"] != 0
        ):
            raise SystemExit(f"{benchmark} selection receipt mismatch")
        path = (DATASET_ROOT / str(expected["path"])).resolve()
        if (
            sha256_file(path) != expected["sha256"]
            or pq.read_metadata(path).num_rows
            != dataset_receipt["dataset_rows"]
        ):
            raise SystemExit(f"{benchmark} dataset identity mismatch")
        actual_counts[benchmark] = len(indices)

    if actual_counts != receipt["summary"]["by_benchmark"]:
        raise SystemExit("holdout benchmark counts differ from receipt")
    if sum(actual_counts.values()) != receipt["summary"]["cases"]:
        raise SystemExit("holdout total differs from receipt")
    print(
        json.dumps(
            {
                "schema_version": (
                    "nano_harness_unseen_holdout_validation_v1"
                ),
                "suite_id": receipt["suite_id"],
                "cases": receipt["summary"]["cases"],
                "by_benchmark": actual_counts,
                "history_overlap": 0,
                "prompts_loaded": False,
                "references_loaded": False,
                "training_eligible": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
