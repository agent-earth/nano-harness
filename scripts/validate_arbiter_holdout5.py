#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

from nano_harness.baseline import case_manifest, load_cases, load_manifest


DIRECT = Path("configs/harness/qwen35_arbiter_holdout5_direct_v1.yaml")
ROUTED = Path("configs/harness/qwen35_arbiter_holdout5_routed_v1.yaml")
DIRECT_CASES = Path(
    "configs/generated/qwen35_arbiter_holdout5_direct_v1_cases.json"
)
ROUTED_CASES = Path(
    "configs/generated/qwen35_arbiter_holdout5_routed_v1_cases.json"
)


def main() -> None:
    root = Path("../../datasets")
    selected = {}
    for label, manifest_path, cases_path in (
        ("direct", DIRECT, DIRECT_CASES),
        ("routed", ROUTED, ROUTED_CASES),
    ):
        manifest = load_manifest(manifest_path)
        cases = load_cases(manifest, root)
        expected = json.loads(cases_path.read_text(encoding="utf-8"))
        if case_manifest(cases) != expected:
            raise SystemExit(f"{label} differs from committed case manifest")
        selected[label] = cases

    direct_ids = {case.case_id for case in selected["direct"]}
    routed_ids = {case.case_id for case in selected["routed"]}
    if direct_ids != routed_ids or len(direct_ids) != 72:
        raise SystemExit("direct and routed identities differ or are not unique")

    current = {DIRECT.resolve(), ROUTED.resolve()}
    historical: set[str] = set()
    checked = 0
    paths = sorted(Path("configs/baselines").glob("*.yaml"))
    paths += sorted(Path("configs/harness").glob("*.yaml"))
    for path in paths:
        if path.resolve() in current:
            continue
        manifest = load_manifest(path)
        historical.update(case.case_id for case in load_cases(manifest, root))
        checked += 1
    if direct_ids & historical:
        raise SystemExit("holdout5 overlaps historical or sealed case IDs")

    routed = load_manifest(ROUTED)
    expected_routes = {
        "gsm8k": "direct",
        "mmlu": "direct",
        "gpqa_diamond": "option_evidence_arbiter",
    }
    if (
        routed.benchmark_routing != expected_routes
        or routed.option_evidence_max_tokens != 96
        or routed.verifier_max_tokens != 64
        or not routed.normalize_bare_choice
    ):
        raise SystemExit("unexpected routed contract")

    print(
        json.dumps(
            {
                "schema_version": "nano_harness_arbiter_holdout5_validation_v1",
                "cases": len(direct_ids),
                "case_manifests_match": True,
                "historical_manifests_checked": checked,
                "historical_overlap": 0,
                "benchmark_routing": expected_routes,
                "option_evidence_max_tokens": 96,
                "arbiter_max_tokens": 64,
                "normalize_bare_choice": True,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
