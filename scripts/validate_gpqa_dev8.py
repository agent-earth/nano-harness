#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

from nano_harness.baseline import case_manifest, load_cases, load_manifest


DIRECT = Path("configs/harness/qwen35_gpqa_dev8_direct_v1.yaml")
TREATMENT = Path("configs/harness/qwen35_gpqa_dev8_arbiter_v1.yaml")
DIRECT_CASES = Path("configs/generated/qwen35_gpqa_dev8_direct_v1_cases.json")
TREATMENT_CASES = Path(
    "configs/generated/qwen35_gpqa_dev8_arbiter_v1_cases.json"
)


def main() -> None:
    dataset_root = Path("../../datasets")
    selected = {}
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
    if direct_ids & historical_ids:
        raise SystemExit("dev8 overlaps historical case IDs")

    treatment = load_manifest(TREATMENT)
    if (
        treatment.strategy != "option_evidence_arbiter"
        or treatment.option_evidence_max_tokens != 96
        or treatment.verifier_max_tokens != 64
        or not treatment.normalize_bare_choice
    ):
        raise SystemExit("unexpected treatment contract")

    print(
        json.dumps(
            {
                "schema_version": "nano_harness_gpqa_dev8_validation_v1",
                "cases": len(direct_ids),
                "case_manifests_match": True,
                "historical_manifests_checked": checked,
                "historical_overlap": 0,
                "treatment_strategy": treatment.strategy,
                "option_evidence_max_tokens": treatment.option_evidence_max_tokens,
                "arbiter_max_tokens": treatment.verifier_max_tokens,
                "normalize_bare_choice": treatment.normalize_bare_choice,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
