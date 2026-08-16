#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from datasets import Dataset

from nano_harness.baseline import (
    NUMERIC_FINAL_REGEX,
    build_case,
    case_manifest,
    load_cases,
    load_manifest,
    resolve_dataset_path,
)


DIRECT = Path("configs/harness/qwen35_gsm8k_dev16_direct_v1.yaml")
TREATMENT = Path("configs/harness/qwen35_gsm8k_dev16_constrained_v1.yaml")
DIRECT_CASES = Path("configs/generated/qwen35_gsm8k_dev16_direct_v1_cases.json")
TREATMENT_CASES = Path(
    "configs/generated/qwen35_gsm8k_dev16_constrained_v1_cases.json"
)
FROZEN_DIRECT = Path("configs/harness/qwen35_gsm8k_dev15_direct_v1.yaml")
FROZEN_TREATMENT = Path(
    "configs/harness/qwen35_gsm8k_dev15_constrained_v1.yaml"
)
EXPECTED_REGEX = r"FINAL: [-+]?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)"


def _gsm8k_ids(path: Path) -> set[str]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(row["case_id"])
        for row in rows
        if isinstance(row, dict) and row.get("benchmark") == "gsm8k"
    }


def _all_sorted_cases(manifest_path: Path, root: Path):
    manifest = load_manifest(manifest_path)
    spec = manifest.datasets[0]
    path = resolve_dataset_path(root, spec.path)
    records = Dataset.from_parquet(str(path))
    cases = [
        build_case(
            spec.name,
            spec.scorer,
            index,
            record,
            answer_only=spec.answer_only,
            system_prompt=spec.system_prompt or manifest.system_prompt,
            max_tokens=spec.max_tokens or manifest.max_tokens,
        )
        for index, record in enumerate(records)
    ]
    cases.sort(
        key=lambda case: hashlib.sha256(
            f"{manifest.selection_seed}\0{case.benchmark}\0{case.case_id}".encode()
        ).hexdigest()
    )
    return cases


def _frozen_contract(manifest_path: Path) -> dict:
    manifest = load_manifest(manifest_path)
    spec = manifest.datasets[0]
    return {
        "selection_seed": manifest.selection_seed,
        "system_prompt": manifest.system_prompt,
        "max_tokens": manifest.max_tokens,
        "temperature": manifest.temperature,
        "chat_template_kwargs": manifest.chat_template_kwargs,
        "dataset_name": spec.name,
        "dataset_path": spec.path,
        "dataset_sha256": spec.sha256,
        "scorer": spec.scorer,
        "dataset_max_tokens": spec.max_tokens,
    }


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
    if direct_ids != treatment_ids or len(direct_ids) != 96:
        raise SystemExit("direct and treatment identities differ or are not unique")

    current_cases = {DIRECT_CASES.resolve(), TREATMENT_CASES.resolve()}
    historical: set[str] = set()
    historical_files = 0
    for path in sorted(Path("configs/generated").glob("*cases.json")):
        if path.resolve() in current_cases:
            continue
        ids = _gsm8k_ids(path)
        if ids:
            historical.update(ids)
            historical_files += 1
    if direct_ids & historical:
        raise SystemExit("dev16 overlaps historical or sealed case IDs")

    all_cases = _all_sorted_cases(DIRECT, root)
    limit = 96
    clean_starts = []
    for start in range(len(all_cases) - limit + 1):
        window_ids = {case.case_id for case in all_cases[start : start + limit]}
        if not window_ids & historical:
            clean_starts.append(start)
    if not clean_starts or clean_starts[0] != 576:
        raise SystemExit(f"unexpected first clean window: {clean_starts[:1]}")
    previous_ids = {
        case.case_id for case in all_cases[clean_starts[0] - 1 : clean_starts[0] - 1 + limit]
    }
    previous_overlap = previous_ids & historical
    if len(previous_overlap) != 1:
        raise SystemExit(
            f"expected exactly one overlap in prior window, got {len(previous_overlap)}"
        )

    if _frozen_contract(DIRECT) != _frozen_contract(FROZEN_DIRECT):
        raise SystemExit("dev16 direct contract differs from frozen dev15")
    if _frozen_contract(TREATMENT) != _frozen_contract(FROZEN_TREATMENT):
        raise SystemExit("dev16 treatment contract differs from frozen dev15")

    direct = load_manifest(DIRECT)
    treatment = load_manifest(TREATMENT)
    if direct.strategy != "direct":
        raise SystemExit("unexpected direct strategy")
    if (
        treatment.strategy != "protected_math_constrained_recovery"
        or treatment.second_solve_max_tokens != 32
        or NUMERIC_FINAL_REGEX != EXPECTED_REGEX
    ):
        raise SystemExit("unexpected constrained recovery contract")
    if direct.datasets[0].start != 576 or direct.datasets[0].limit != 96:
        raise SystemExit("unexpected direct window")
    if (
        treatment.datasets[0].start != 576
        or treatment.datasets[0].limit != 96
    ):
        raise SystemExit("unexpected treatment window")

    print(
        json.dumps(
            {
                "schema_version": "nano_harness_gsm8k_dev16_validation_v1",
                "cases": len(direct_ids),
                "case_manifests_match": True,
                "historical_manifest_files": historical_files,
                "historical_unique_ids": len(historical),
                "historical_overlap": 0,
                "first_clean_start": clean_starts[0],
                "previous_window_overlap": len(previous_overlap),
                "frozen_dev15_contract_match": True,
                "treatment_strategy": treatment.strategy,
                "recovery_max_tokens": treatment.second_solve_max_tokens,
                "conditional_recovery": True,
                "structured_outputs_regex": NUMERIC_FINAL_REGEX,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
