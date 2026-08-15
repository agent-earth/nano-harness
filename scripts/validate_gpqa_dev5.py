#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

from nano_harness.baseline import case_manifest, load_cases, load_manifest


DIRECT = Path("configs/harness/qwen35_gpqa_dev5_direct_v1.yaml")
TREATMENT = Path("configs/harness/qwen35_gpqa_dev5_draft_verify_v1.yaml")
DIRECT_CASES = Path("configs/generated/qwen35_gpqa_dev5_direct_v1_cases.json")
TREATMENT_CASES = Path(
    "configs/generated/qwen35_gpqa_dev5_draft_verify_v1_cases.json"
)


def main() -> None:
    dataset_root = Path("../../datasets")
    selected: dict[str, list] = {}
    for label, manifest_path, case_path in (
        ("direct", DIRECT, DIRECT_CASES),
        ("treatment", TREATMENT, TREATMENT_CASES),
    ):
        manifest = load_manifest(manifest_path)
        cases = load_cases(manifest, dataset_root)
        expected = json.loads(case_path.read_text(encoding="utf-8"))
        if case_manifest(cases) != expected:
            raise SystemExit(f"{label} differs from committed case manifest")
        selected[label] = cases

    direct_ids = {case.case_id for case in selected["direct"]}
    treatment_ids = {case.case_id for case in selected["treatment"]}
    if direct_ids != treatment_ids or len(direct_ids) != 12:
        raise SystemExit("direct and treatment identities differ or are not unique")

    current = {DIRECT.resolve(), TREATMENT.resolve()}
    historical_ids: set[str] = set()
    checked = 0
    paths = sorted(Path("configs/baselines").glob("*.yaml"))
    paths += sorted(Path("configs/harness").glob("*.yaml"))
    for path in paths:
        if path.resolve() in current:
            continue
        manifest = load_manifest(path)
        historical_ids.update(
            case.case_id
            for case in load_cases(manifest, dataset_root)
            if case.benchmark == "gpqa_diamond"
        )
        checked += 1
    overlap = direct_ids & historical_ids
    if overlap:
        raise SystemExit(f"dev5 overlaps historical case ids: {sorted(overlap)}")

    print(
        json.dumps(
            {
                "schema_version": "nano_harness_gpqa_dev5_validation_v1",
                "cases": len(direct_ids),
                "case_manifests_match": True,
                "historical_manifests_checked": checked,
                "historical_overlap": 0,
                "direct_strategy": "direct",
                "treatment_strategy": "draft_verify",
                "draft_max_tokens": 384,
                "verifier_max_tokens": 32,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
