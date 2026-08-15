#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nano_harness.baseline import case_manifest, load_cases, load_manifest


EXPERIMENTS = {
    "dev_direct": (
        Path("configs/harness/qwen35_routing_dev4_direct_v1.yaml"),
        Path("configs/generated/qwen35_routing_dev4_direct_v1_cases.json"),
    ),
    "dev_routed": (
        Path("configs/harness/qwen35_routing_dev4_v1.yaml"),
        Path("configs/generated/qwen35_routing_dev4_v1_cases.json"),
    ),
    "holdout_direct": (
        Path("configs/harness/qwen35_routing_holdout4_direct_v1.yaml"),
        Path("configs/generated/qwen35_routing_holdout4_direct_v1_cases.json"),
    ),
    "holdout_routed": (
        Path("configs/harness/qwen35_routing_holdout4_v1.yaml"),
        Path("configs/generated/qwen35_routing_holdout4_v1_cases.json"),
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="../../datasets")
    args = parser.parse_args()
    dataset_root = Path(args.dataset_root)

    manifests = {}
    cases = {}
    for label, (manifest_path, case_path) in EXPERIMENTS.items():
        manifest = load_manifest(manifest_path)
        selected = load_cases(manifest, dataset_root)
        expected = json.loads(case_path.read_text(encoding="utf-8"))
        actual = case_manifest(selected)
        if actual != expected:
            raise SystemExit(f"{label} differs from committed case manifest")
        manifests[label] = manifest
        cases[label] = selected

    dev_ids = {case.case_id for case in cases["dev_direct"]}
    holdout_ids = {case.case_id for case in cases["holdout_direct"]}
    if dev_ids != {case.case_id for case in cases["dev_routed"]}:
        raise SystemExit("dev direct and routed case identities differ")
    if holdout_ids != {case.case_id for case in cases["holdout_routed"]}:
        raise SystemExit("holdout direct and routed case identities differ")
    if dev_ids & holdout_ids:
        raise SystemExit("dev and holdout case identities overlap")

    current_paths = {path.resolve() for path, _ in EXPERIMENTS.values()}
    prior_ids: set[str] = set()
    prior_manifests = []
    manifest_paths = sorted(Path("configs/baselines").glob("*.yaml"))
    manifest_paths += sorted(Path("configs/harness").glob("*.yaml"))
    for path in manifest_paths:
        if path.resolve() in current_paths:
            continue
        manifest = load_manifest(path)
        prior_ids.update(
            case.case_id for case in load_cases(manifest, dataset_root)
        )
        prior_manifests.append(str(path))
    if dev_ids & prior_ids:
        raise SystemExit("dev cases overlap a prior manifest")
    if holdout_ids & prior_ids:
        raise SystemExit("holdout cases overlap a prior manifest")

    expected_routes = {
        "gsm8k": "direct",
        "mmlu": "draft_verify",
        "gpqa_diamond": "draft_verify",
    }
    for label in ("dev_routed", "holdout_routed"):
        if manifests[label].benchmark_routing != expected_routes:
            raise SystemExit(f"{label} has unexpected routing")
    for label in ("dev_direct", "holdout_direct"):
        if manifests[label].strategy != "direct":
            raise SystemExit(f"{label} is not a direct control")

    print(
        json.dumps(
            {
                "schema_version": "nano_harness_routing_validation_v1",
                "dev_cases": len(dev_ids),
                "holdout_cases": len(holdout_ids),
                "prior_manifests_checked": len(prior_manifests),
                "historical_overlap": 0,
                "dev_holdout_overlap": 0,
                "benchmark_routing": expected_routes,
                "case_manifests_match": True,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
