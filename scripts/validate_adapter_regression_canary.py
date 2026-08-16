#!/usr/bin/env python3

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from nano_harness.baseline import (
    case_manifest,
    load_cases,
    load_manifest,
    sha256_file,
)


CANARY = Path(
    "configs/harness/qwen35_adapter_regression_canary_v1.yaml"
)
CANARY_CASES = Path(
    "configs/generated/qwen35_adapter_regression_canary_v1_cases.json"
)
FULL = Path("configs/harness/qwen35_three_task_replication_v1.yaml")
FOUR_B = Path(
    "results/harness/qwen35-three-task-replication-v1/4b/cases.jsonl"
)
V6 = Path(
    "results/harness/qwen35-v6-matched-adapter-v1/candidate/cases.jsonl"
)


def latest(path: Path) -> dict[str, dict]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            result[str(row["case_id"])] = row
    return result


def main() -> None:
    canary_manifest = load_manifest(CANARY)
    full_manifest = load_manifest(FULL)
    canary = load_cases(canary_manifest, Path("../../datasets"))
    full = load_cases(full_manifest, Path("../../datasets"))
    expected = json.loads(CANARY_CASES.read_text(encoding="utf-8"))
    if case_manifest(canary) != expected:
        raise SystemExit("canary cases differ from committed manifest")
    counts = Counter(case.benchmark for case in canary)
    expected_counts = {"gpqa_diamond": 8, "gsm8k": 16, "mmlu": 16}
    if dict(sorted(counts.items())) != expected_counts:
        raise SystemExit(f"unexpected canary counts: {counts}")
    if len(canary) != 40 or len({case.case_id for case in canary}) != 40:
        raise SystemExit("canary must contain 40 unique cases")
    full_by_benchmark = {
        benchmark: [
            case for case in full if case.benchmark == benchmark
        ]
        for benchmark in expected_counts
    }
    canary_by_benchmark = {
        benchmark: [
            case for case in canary if case.benchmark == benchmark
        ]
        for benchmark in expected_counts
    }
    for benchmark, limit in expected_counts.items():
        expected_prefix = [
            case.case_id for case in full_by_benchmark[benchmark][:limit]
        ]
        actual = [case.case_id for case in canary_by_benchmark[benchmark]]
        if actual != expected_prefix:
            raise SystemExit(
                f"{benchmark} canary is not the frozen full-suite prefix"
            )

    four_b = latest(FOUR_B)
    v6 = latest(V6)
    case_ids = {case.case_id for case in canary}
    if not case_ids <= set(four_b) or not case_ids <= set(v6):
        raise SystemExit("canary case set is missing from frozen results")
    scores = {}
    for label, records in (("base_four_b", four_b), ("v6_adapter", v6)):
        scores[label] = {
            benchmark: {
                "correct": int(
                    sum(
                        float(records[case.case_id]["score"])
                        for case in canary_by_benchmark[benchmark]
                    )
                ),
                "cases": expected_counts[benchmark],
                "parse_failures": sum(
                    records[case.case_id].get("prediction") is None
                    for case in canary_by_benchmark[benchmark]
                ),
                "api_errors": sum(
                    records[case.case_id].get("status") == "error"
                    for case in canary_by_benchmark[benchmark]
                ),
            }
            for benchmark in expected_counts
        }
    base_total = sum(
        row["correct"] for row in scores["base_four_b"].values()
    )
    v6_total = sum(
        row["correct"] for row in scores["v6_adapter"].values()
    )
    if base_total != 30 or v6_total != 28:
        raise SystemExit(
            f"unexpected calibration scores: base={base_total}, v6={v6_total}"
        )

    print(
        json.dumps(
            {
                "schema_version": (
                    "nano_harness_adapter_regression_canary_validation_v1"
                ),
                "suite_id": canary_manifest.suite_id,
                "cases": len(canary),
                "counts": expected_counts,
                "case_manifest_matches": True,
                "full_suite_prefix_matches": True,
                "policy": {
                    "source_split": "sealed_eval_canary",
                    "training_eligible": False,
                    "quality_claim_allowed": False,
                    "purpose": "future_adapter_regression_gate_only",
                    "post_v6_calibrated": True,
                },
                "calibration": {
                    "base_four_b_total": base_total,
                    "v6_adapter_total": v6_total,
                    "v6_rejection_reproduced": v6_total < base_total,
                    "by_arm": scores,
                },
                "artifacts": {
                    "canary_manifest_sha256": sha256_file(CANARY),
                    "canary_cases_sha256": sha256_file(CANARY_CASES),
                    "full_manifest_sha256": sha256_file(FULL),
                    "base_four_b_raw_sha256": sha256_file(FOUR_B),
                    "v6_raw_sha256": sha256_file(V6),
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
