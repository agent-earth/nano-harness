#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

from datasets import Dataset

from nano_harness.baseline import (
    build_case,
    case_manifest,
    load_cases,
    load_manifest,
    resolve_dataset_path,
)


MANIFEST = Path("configs/harness/qwen35_large_confirmation_v1.yaml")
CASES = Path("configs/generated/qwen35_large_confirmation_v1_cases.json")
FROZEN = Path("configs/harness/qwen35_three_task_replication_v1.yaml")
HISTORICAL_REVISION = "3b17f9e3ad2d1b3dab68fa585a8fd0b2600cf3d4"
EXPECTED_HISTORICAL_FILES = 52


def _dataset_contracts(path: Path) -> dict:
    manifest = load_manifest(path)
    return {
        spec.name: {
            "path": spec.path,
            "sha256": spec.sha256,
            "scorer": spec.scorer,
            "max_source_chars": spec.max_source_chars,
            "answer_only": spec.answer_only,
            "max_tokens": spec.max_tokens,
            "system_prompt": spec.system_prompt,
        }
        for spec in manifest.datasets
        if spec.name in {"gsm8k", "mmlu"}
    }


def _global_contract(path: Path) -> dict:
    manifest = load_manifest(path)
    return {
        "selection_seed": manifest.selection_seed,
        "system_prompt": manifest.system_prompt,
        "max_tokens": manifest.max_tokens,
        "temperature": manifest.temperature,
        "chat_template_kwargs": manifest.chat_template_kwargs,
        "strategy": manifest.strategy,
    }


def _sorted_cases(spec, manifest, root: Path):
    records = Dataset.from_parquet(str(resolve_dataset_path(root, spec.path)))
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


def _historical_case_paths() -> list[Path]:
    output = subprocess.run(
        [
            "git",
            "ls-tree",
            "-r",
            "--name-only",
            HISTORICAL_REVISION,
            "configs/generated",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    paths = [
        Path(line)
        for line in output.splitlines()
        if line.endswith("cases.json")
    ]
    if len(paths) != EXPECTED_HISTORICAL_FILES:
        raise SystemExit(
            "frozen historical case-file count differs: "
            f"{len(paths)} != {EXPECTED_HISTORICAL_FILES}"
        )
    return paths


def main() -> None:
    root = Path("../../datasets")
    manifest = load_manifest(MANIFEST)
    cases = load_cases(manifest, root)
    expected = json.loads(CASES.read_text(encoding="utf-8"))
    if case_manifest(cases) != expected:
        raise SystemExit("selected cases differ from committed case manifest")

    ids = {case.case_id for case in cases}
    if len(ids) != 512:
        raise SystemExit("confirmation case IDs are not unique")
    counts = Counter(case.benchmark for case in cases)
    expected_counts = {"gsm8k": 256, "mmlu": 256}
    if dict(sorted(counts.items())) != expected_counts:
        raise SystemExit(f"unexpected benchmark counts: {counts}")

    historical: set[str] = set()
    historical_files = 0
    for path in _historical_case_paths():
        rows = json.loads(
            subprocess.run(
                ["git", "show", f"{HISTORICAL_REVISION}:{path.as_posix()}"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )
        historical.update(
            str(row["case_id"])
            for row in rows
            if isinstance(row, dict) and row.get("case_id")
        )
        historical_files += 1
    if ids & historical:
        raise SystemExit("confirmation overlaps historical or sealed case IDs")

    if (
        _global_contract(MANIFEST) != _global_contract(FROZEN)
        or _dataset_contracts(MANIFEST) != _dataset_contracts(FROZEN)
    ):
        raise SystemExit("confirmation contract differs from frozen replication")

    windows = {
        spec.name: {"start": spec.start, "limit": spec.limit}
        for spec in manifest.datasets
    }
    expected_windows = {
        "gsm8k": {"start": 768, "limit": 256},
        "mmlu": {"start": 2801, "limit": 256},
    }
    if windows != expected_windows:
        raise SystemExit(f"unexpected windows: {windows}")

    selection_audits = {}
    for spec in manifest.datasets:
        sorted_cases = _sorted_cases(spec, manifest, root)
        clean_starts = []
        for start in range(len(sorted_cases) - spec.limit + 1):
            window = sorted_cases[start : start + spec.limit]
            window_ids = {case.case_id for case in window}
            if len(window_ids) == spec.limit and not window_ids & historical:
                clean_starts.append(start)
        if not clean_starts or clean_starts[0] != spec.start:
            raise SystemExit(
                f"{spec.name} first clean start differs: {clean_starts[:1]}"
            )

        prior_window = sorted_cases[spec.start - 1 : spec.start - 1 + spec.limit]
        prior_ids = {case.case_id for case in prior_window}
        prior_overlap = len(prior_ids & historical)
        prior_duplicates = spec.limit - len(prior_ids)
        if prior_overlap < 1 and prior_duplicates < 1:
            raise SystemExit(f"{spec.name} preceding window is also clean")
        selection_audits[spec.name] = {
            "eligible_cases": len(sorted_cases),
            "remaining_fresh_cases": sum(
                case.case_id not in historical for case in sorted_cases
            ),
            "first_clean_start": clean_starts[0],
            "preceding_window_overlap": prior_overlap,
            "preceding_window_duplicates": prior_duplicates,
        }

    print(
        json.dumps(
            {
                "schema_version": "nano_harness_large_confirmation_validation_v1",
                "cases": len(ids),
                "counts": expected_counts,
                "case_manifest_matches": True,
                "historical_manifest_files": historical_files,
                "historical_overlap": 0,
                "frozen_replication_contract_match": True,
                "windows": windows,
                "selection_audits": selection_audits,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
