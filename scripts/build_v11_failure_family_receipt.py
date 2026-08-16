#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs/results/v11_full_matched_adapter_v1.public.json"
OUTPUT = ROOT / "configs/feedback/v11_base_only_failure_families_v1.json"
EXPECTED_SOURCE_SET_SHA256 = (
    "7d12b24f2dcb5211a67a30bd73e0a5c9"
    "c5c880e5f4355018159f4aab7636261b"
)


def canonical_sha256(value: object) -> str:
    rendered = json.dumps(value, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    source_ids = sorted(
        report["comparisons"]["candidate_vs_four_b"]["overall_micro"][
            "baseline_only_cases"
        ]
    )
    source_set_sha256 = canonical_sha256(source_ids)
    if len(source_ids) != 4 or source_set_sha256 != EXPECTED_SOURCE_SET_SHA256:
        raise SystemExit(
            "v11 base-only source set differs from the reviewed diagnosis"
        )
    receipt = {
        "schema_version": "nano_harness_failure_family_receipt_v1",
        "receipt_id": "v11-base-only-failure-families-v1",
        "source": {
            "experiment_id": report["experiment_id"],
            "public_report_sha256": sha256_file(REPORT),
            "source_case_count": len(source_ids),
            "source_case_id_set_sha256": source_set_sha256,
        },
        "families": [
            {
                "family": "percentage_increase_total_composition",
                "count": 1,
                "task_kind": "numeric_reasoning",
            },
            {
                "family": "packing_efficiency_effective_volume",
                "count": 1,
                "task_kind": "numeric_reasoning",
            },
            {
                "family": "weighted_recurring_schedule_total",
                "count": 1,
                "task_kind": "numeric_reasoning",
            },
            {
                "family": "developmental_perception_experience_choice",
                "count": 1,
                "task_kind": "choice_reasoning",
            },
        ],
        "policy": {
            "source_split": "sealed_eval_feedback",
            "direct_training_allowed": False,
            "contains_case_ids": False,
            "contains_prompts": False,
            "contains_references": False,
            "contains_predictions": False,
            "contains_raw_outputs": False,
            "contains_reversible_payloads": False,
            "fresh_analog_generation_allowed": True,
            "benchmark_rows_training_eligible": False,
            "canary_rows_training_eligible": False,
        },
        "summary": {
            "families": 4,
            "source_cases": 4,
            "numeric_reasoning_families": 3,
            "choice_reasoning_families": 1,
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "receipt_id": receipt["receipt_id"],
                "families": receipt["summary"]["families"],
                "source_cases": receipt["summary"]["source_cases"],
                "source_case_id_set_sha256": source_set_sha256,
                "output": str(OUTPUT.relative_to(ROOT)),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
