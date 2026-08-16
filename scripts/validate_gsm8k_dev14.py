#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

from nano_harness.baseline import case_manifest, load_cases, load_manifest


DIRECT = Path("configs/harness/qwen35_gsm8k_dev14_direct_v1.yaml")
TREATMENT = Path(
    "configs/harness/qwen35_gsm8k_dev14_short_recovery_v1.yaml"
)
DIRECT_CASES = Path("configs/generated/qwen35_gsm8k_dev14_direct_v1_cases.json")
TREATMENT_CASES = Path(
    "configs/generated/qwen35_gsm8k_dev14_short_recovery_v1_cases.json"
)


def main() -> None:
    root = Path("../../datasets")
    selected = {}
    for label, manifest_path, cases_path in (
        ("direct", DIRECT, DIRECT_CASES),
        ("treatment", TREATMENT, TREATMENT_CASES),
    ):
        manifest = load_manifest(manifest_path)
        cases = load_cases(manifest, root)
        expected = json.loads(cases_path.read_text(encoding="utf-8"))
        if case_manifest(cases) != expected:
            raise SystemExit(f"{label} differs from committed case manifest")
        selected[label] = cases

    direct_ids = {case.case_id for case in selected["direct"]}
    treatment_ids = {case.case_id for case in selected["treatment"]}
    if direct_ids != treatment_ids or len(direct_ids) != 48:
        raise SystemExit("direct and treatment identities differ or are not unique")

    current = {DIRECT.resolve(), TREATMENT.resolve()}
    historical: set[str] = set()
    checked = 0
    paths = sorted(Path("configs/baselines").glob("*.yaml"))
    paths += sorted(Path("configs/harness").glob("*.yaml"))
    for path in paths:
        if path.resolve() in current:
            continue
        manifest = load_manifest(path)
        historical.update(
            case.case_id
            for case in load_cases(manifest, root)
            if case.benchmark == "gsm8k"
        )
        checked += 1
    if direct_ids & historical:
        raise SystemExit("dev14 overlaps historical or sealed case IDs")

    treatment = load_manifest(TREATMENT)
    if (
        treatment.strategy != "protected_math_short_recovery"
        or treatment.second_solve_max_tokens != 64
    ):
        raise SystemExit("unexpected treatment contract")

    print(
        json.dumps(
            {
                "schema_version": "nano_harness_gsm8k_dev14_validation_v1",
                "cases": len(direct_ids),
                "case_manifests_match": True,
                "historical_manifests_checked": checked,
                "historical_overlap": 0,
                "treatment_strategy": treatment.strategy,
                "recovery_max_tokens": treatment.second_solve_max_tokens,
                "conditional_recovery": True,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
