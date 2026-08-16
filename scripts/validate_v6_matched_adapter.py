#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from nano_harness.baseline import (
    case_manifest,
    load_cases,
    load_manifest,
    sha256_file,
)


MANIFEST = Path("configs/harness/qwen35_v6_matched_adapter_v1.yaml")
FROZEN = Path("configs/harness/qwen35_three_task_replication_v1.yaml")
CASES = Path("configs/generated/qwen35_three_task_replication_v1_cases.json")
FOUR_B = Path(
    "results/harness/qwen35-three-task-replication-v1/4b/cases.jsonl"
)
NINE_B = Path(
    "results/harness/qwen35-three-task-replication-v1/9b/cases.jsonl"
)
ADAPTER = Path(
    "../nano-train/artifacts/arithmetic-process-sft-smoke-v6/adapter"
)
V6_REPORT = Path(
    "../nano-train/docs/results/arithmetic_process_sft_smoke_v6.public.json"
)
EXPECTED = {
    "manifest_sha256": (
        "88f6e832d38e739c6b622a30633a27370"
        "77fc081037e6e1543cb5763b169a7b9"
    ),
    "case_manifest_sha256": (
        "eafbe4d42487a225322dd3b3bdc1d805"
        "c065fb15f0f8b968e65ccf747f96976f"
    ),
    "four_b_raw_sha256": (
        "c59383d3fd3d6087025d6e1ff649979d"
        "9d5a9e8dc73b5429a4f8e9fa41b6b8c7"
    ),
    "nine_b_raw_sha256": (
        "ffae93774d51b87a2e29258d170a84f8"
        "b165f996e2e78eedd102271dfc260044"
    ),
    "adapter_tree_sha256": (
        "49f08829e06aa75c1cf6e5f16891bf79"
        "378011b8fe874fde4e392f5fcb5aa083"
    ),
}


def sha256_tree(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(path.rglob("*")):
        if not item.is_file():
            continue
        relative = item.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(item).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def contract(path: Path) -> dict:
    manifest = load_manifest(path)
    return {
        "selection_seed": manifest.selection_seed,
        "system_prompt": manifest.system_prompt,
        "max_tokens": manifest.max_tokens,
        "temperature": manifest.temperature,
        "chat_template_kwargs": manifest.chat_template_kwargs,
        "strategy": manifest.strategy,
        "datasets": [
            {
                "name": spec.name,
                "path": spec.path,
                "sha256": spec.sha256,
                "scorer": spec.scorer,
                "start": spec.start,
                "limit": spec.limit,
                "max_source_chars": spec.max_source_chars,
                "answer_only": spec.answer_only,
                "max_tokens": spec.max_tokens,
                "system_prompt": spec.system_prompt,
            }
            for spec in manifest.datasets
        ],
    }


def latest(path: Path) -> dict[str, dict]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            result[str(row["case_id"])] = row
    return result


def main() -> None:
    actual = {
        "manifest_sha256": sha256_file(FROZEN),
        "case_manifest_sha256": sha256_file(CASES),
        "four_b_raw_sha256": sha256_file(FOUR_B),
        "nine_b_raw_sha256": sha256_file(NINE_B),
        "adapter_tree_sha256": sha256_tree(ADAPTER),
    }
    if actual != EXPECTED:
        raise SystemExit(
            f"v6 matched adapter identity drift: actual={actual}, "
            f"expected={EXPECTED}"
        )
    if contract(MANIFEST) != contract(FROZEN):
        raise SystemExit("v6 matched adapter contract differs from replication")
    manifest = load_manifest(MANIFEST)
    cases = load_cases(manifest, Path("../../datasets"))
    frozen_cases = json.loads(CASES.read_text(encoding="utf-8"))
    if case_manifest(cases) != frozen_cases:
        raise SystemExit("v6 matched adapter cases differ from frozen manifest")
    if len(cases) != 211 or len({case.case_id for case in cases}) != 211:
        raise SystemExit("v6 matched adapter requires 211 unique cases")

    case_ids = {case.case_id for case in cases}
    four_b = latest(FOUR_B)
    nine_b = latest(NINE_B)
    if set(four_b) != case_ids or set(nine_b) != case_ids:
        raise SystemExit("frozen result case sets differ from adapter cases")
    if any(row.get("status") != "completed" for row in four_b.values()):
        raise SystemExit("frozen 4B arm contains API errors")
    if any(row.get("status") != "completed" for row in nine_b.values()):
        raise SystemExit("frozen 9B arm contains API errors")

    v6 = json.loads(V6_REPORT.read_text(encoding="utf-8"))
    if (
        v6["passed"] is not True
        or v6["decision"]["matched_benchmark_evaluation_allowed"] is not True
        or v6["decision"]["rl_allowed"] is not False
        or v6["artifacts"]["adapter_sha256"]
        != EXPECTED["adapter_tree_sha256"]
    ):
        raise SystemExit("v6 report does not authorize matched evaluation")

    print(
        json.dumps(
            {
                "schema_version": (
                    "nano_harness_v6_matched_adapter_preflight_v1"
                ),
                "suite_id": manifest.suite_id,
                "cases": len(cases),
                "case_manifest_matches": True,
                "frozen_contract_matches": True,
                "frozen_result_case_sets_match": True,
                "frozen_four_b_correct": int(
                    sum(float(row["score"]) for row in four_b.values())
                ),
                "frozen_nine_b_correct": int(
                    sum(float(row["score"]) for row in nine_b.values())
                ),
                "identity": actual,
                "v6_local_gate_passed": True,
                "rl_allowed_before_evaluation": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
