#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from nano_harness.baseline import compare_baselines, summarize_baseline


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = (
    ROOT / "configs/generated/qwen35_complete_direct_v1_cases.public.json"
)
DEFAULT_FOUR_B = (
    ROOT / "results/full/qwen35-complete-direct-v1/4b/shard-0.jsonl"
)
DEFAULT_NINE_B = (
    ROOT / "results/full/qwen35-complete-direct-v1/9b/shard-0.jsonl"
)
DEFAULT_SERVICE = (
    ROOT
    / "results/full/qwen35-complete-direct-v1/services/startup.receipt.json"
)
DEFAULT_OUTPUT = (
    ROOT / "docs/results/qwen35_complete_direct_shard0_smoke_v1.public.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_latest(path: Path) -> dict[str, dict[str, Any]]:
    latest = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                latest[str(row["case_id"])] = row
    return latest


def expected_shard(
    case_contract: list[dict[str, Any]],
    *,
    num_shards: int,
    shard_id: int,
) -> dict[str, dict[str, Any]]:
    selected = {
        row["case_id"]: row
        for row in case_contract
        if int(hashlib.sha256(row["case_id"].encode()).hexdigest(), 16)
        % num_shards
        == shard_id
    }
    if len(selected) != 972:
        raise ValueError("shard 0 expected row count differs")
    return selected


def validate_arm(
    rows: dict[str, dict[str, Any]],
    expected: dict[str, dict[str, Any]],
    *,
    model: str,
    path: Path,
) -> dict[str, Any]:
    if set(rows) != set(expected):
        raise ValueError(f"{model} case ID set differs")
    failures = []
    for case_id, row in rows.items():
        contract = expected[case_id]
        if (
            row.get("benchmark") != contract["benchmark"]
            or row.get("source_index") != contract["source_index"]
            or row.get("max_tokens") != contract["max_tokens"]
            or row.get("prompt_sha256") != contract["prompt_sha256"]
            or row.get("system_prompt_sha256")
            != contract["system_prompt_sha256"]
            or row.get("model") != model
            or row.get("strategy") != "direct"
            or row.get("selected_strategy") != "direct"
            or row.get("status") != "completed"
        ):
            failures.append(case_id)
    if failures:
        raise ValueError(f"{model} row contract differs: {failures[:5]}")
    summary = summarize_baseline(path)
    if summary["total_cases"] != 972 or summary["error_cases"] != 0:
        raise ValueError(f"{model} summary differs")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-contract", default=str(DEFAULT_CASES))
    parser.add_argument("--four-b", default=str(DEFAULT_FOUR_B))
    parser.add_argument("--nine-b", default=str(DEFAULT_NINE_B))
    parser.add_argument("--service-receipt", default=str(DEFAULT_SERVICE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    case_path = Path(args.case_contract)
    four_path = Path(args.four_b)
    nine_path = Path(args.nine_b)
    service_path = Path(args.service_receipt)
    expected = expected_shard(
        json.loads(case_path.read_text(encoding="utf-8")),
        num_shards=16,
        shard_id=0,
    )
    four_rows = load_latest(four_path)
    nine_rows = load_latest(nine_path)
    four_summary = validate_arm(
        four_rows,
        expected,
        model="qwen3.5-4b",
        path=four_path,
    )
    nine_summary = validate_arm(
        nine_rows,
        expected,
        model="qwen3.5-9b",
        path=nine_path,
    )
    service = json.loads(service_path.read_text(encoding="utf-8"))
    if (
        service["models"]["qwen3.5-4b"]["served_model"] != "qwen3.5-4b"
        or service["models"]["qwen3.5-9b"]["served_model"] != "qwen3.5-9b"
    ):
        raise ValueError("service identity differs")
    comparison = compare_baselines(
        four_path,
        nine_path,
        bootstrap_samples=10_000,
        bootstrap_seed=20260820,
    )
    report = {
        "schema_version": (
            "nano_harness_complete_baseline_shard0_smoke_public_v1"
        ),
        "experiment_id": "qwen35-complete-direct-v1-shard0-smoke",
        "identity": {
            "case_contract_sha256": sha256_file(case_path),
            "service_receipt_sha256": sha256_file(service_path),
            "four_b_raw_sha256": sha256_file(four_path),
            "nine_b_raw_sha256": sha256_file(nine_path),
            "expected_case_ids_sha256": hashlib.sha256(
                "\n".join(sorted(expected)).encode("utf-8")
            ).hexdigest(),
        },
        "arms": {
            "qwen3.5-4b": four_summary,
            "qwen3.5-9b": nine_summary,
        },
        "comparison": comparison,
        "validation": {
            "expected_cases": 972,
            "both_case_sets_match": set(four_rows) == set(nine_rows),
            "prompt_and_system_hashes_match_contract": True,
            "models_match_service_receipt": True,
            "zero_api_errors": True,
            "raw_outputs_ignored": True,
        },
        "decision": {
            "expand_remaining_shards": True,
            "quality_claim_allowed": False,
            "training_allowed": False,
            "rl_allowed": False,
            "opd_allowed": False,
        },
        "claim_boundary": (
            "Shard 0 is a serving and execution smoke over a pre-registered "
            "subset. Its scores are preliminary and cannot establish full "
            "benchmark superiority."
        ),
    }
    if not all(report["validation"].values()):
        raise ValueError("shard 0 smoke validation differs")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
