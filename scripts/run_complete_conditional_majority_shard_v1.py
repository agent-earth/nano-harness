#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

from openai import OpenAI

from nano_harness.baseline import load_cases, load_manifest, sha256_file
from nano_harness.complete_conditional_majority import (
    generate_gsm8k_candidate,
    load_config,
    verify_four_b_service,
)
from nano_harness.orca_self_consistency import score_prediction
from nano_harness.v5_complete_treatment import jsonl_ids, jsonl_rows


ROOT = Path(__file__).resolve().parents[1]
EXECUTION_V1 = (
    ROOT
    / "configs/campaign/"
    "qwen35_complete_conditional_majority_v1.execution.json"
)
EXECUTION_V2 = (
    ROOT
    / "configs/campaign/"
    "qwen35_complete_conditional_majority_v1.execution_v2.json"
)
EXECUTION = EXECUTION_V2
EXECUTION_SHA256 = {
    EXECUTION_V1.name: (
        "35d737f3a82e584a48d1c53087338e065f95a1c4bd7c21c4130f974e30f74fa9"
    ),
    EXECUTION_V2.name: (
        "e7d705c36f8f975bfcdc8ba44143315156d38be3eae37484f18b395b05d14ff6"
    ),
}


def load_execution(path: str | Path = EXECUTION) -> dict:
    execution_path = Path(path)
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    if sha256_file(execution_path) != EXECUTION_SHA256.get(execution_path.name):
        raise ValueError("complete conditional majority execution SHA differs")
    schema = execution.get("schema_version")
    if (
        schema
        not in {
            "nano_harness_complete_conditional_majority_execution_v1",
            "nano_harness_complete_conditional_majority_execution_v2",
        }
        or execution.get("sharding", {}).get("assignment")
        != "global_sorted_case_index_mod_num_shards"
        or execution.get("sharding", {}).get(
            "preserve_global_case_index_for_seed"
        )
        is not True
        or execution.get("sharding", {}).get("preserve_completed_prefix")
        is not True
        or execution.get("sharding", {}).get(
            "merge_requires_exact_1319_case_set"
        )
        is not True
        or execution.get("observation_boundary", {}).get(
            "policy_tuned_from_prefix"
        )
        is not False
        or execution.get("observation_boundary", {}).get(
            "prefix_regenerated"
        )
        is not False
        or execution.get("observation_boundary", {}).get(
            "post_addendum_policy_change_allowed"
        )
        is not False
    ):
        raise ValueError("complete conditional majority execution differs")
    invariant = execution.get("invariance", {})
    required_false = {
        "case_set_changed",
        "prompt_changed",
        "parser_changed",
        "sampling_changed",
        "seed_changed",
        "vote_threshold_changed",
        "model_weights_changed",
        "scorer_changed",
    }
    if any(invariant.get(key) is not False for key in required_false):
        raise ValueError("complete conditional majority execution differs")
    if schema.endswith("_v1"):
        if (
            execution["sharding"]["num_shards"] != 2
            or invariant.get("only_execution_partition_changed") is not True
            or execution["completed_prefix"]["rows"] != 89
        ):
            raise ValueError("complete conditional majority execution differs")
    elif (
        execution["sharding"]["num_shards"] != 8
        or invariant.get(
            "only_execution_partition_and_batch_capacity_changed"
        )
        is not True
        or execution["completed_prefix"]["rows"] != 156
    ):
        raise ValueError("complete conditional majority execution differs")
    return execution


def select_shard(
    cases: list,
    *,
    prefix_ids: set[str],
    num_shards: int,
    shard_id: int,
) -> list[tuple[int, object]]:
    if num_shards <= 0 or shard_id not in range(num_shards):
        raise ValueError("complete conditional majority shard differs")
    return [
        (index, case)
        for index, case in enumerate(sorted(cases, key=lambda row: row.case_id))
        if case.case_id not in prefix_ids and index % num_shards == shard_id
    ]


def run_shard(
    shard_id: int,
    execution_path: str | Path = EXECUTION,
) -> dict:
    execution = load_execution(execution_path)
    config_path = ROOT / execution["parent_config_path"]
    preregister_path = ROOT / execution["parent_preregister_path"]
    if (
        sha256_file(config_path) != execution["parent_config_sha256"]
        or sha256_file(preregister_path)
        != execution["parent_preregister_sha256"]
    ):
        raise ValueError("complete conditional majority parent differs")
    config = load_config(config_path)
    service = next(
        row
        for row in execution["services"]
        if (
            row.get("shard_id") == shard_id
            or shard_id in row.get("shard_ids", [])
        )
    )
    service_config = {
        **config,
        "four_b": {
            **config["four_b"],
            "base_url": service["base_url"],
        },
    }
    verify_four_b_service(service_config)

    prefix_path = ROOT / execution["completed_prefix"]["path"]
    if (
        sha256_file(prefix_path) != execution["completed_prefix"]["sha256"]
        or len(jsonl_ids(prefix_path))
        != execution["completed_prefix"]["rows"]
    ):
        raise ValueError("complete conditional majority prefix differs")
    prefix_ids = set(jsonl_ids(prefix_path))
    manifest = load_manifest(ROOT / config["baseline"]["suite_manifest_path"])
    cases = [
        case
        for case in load_cases(
            manifest,
            (ROOT / "../../../datasets").resolve(),
        )
        if case.benchmark == "gsm8k"
    ]
    ordered = sorted(cases, key=lambda row: row.case_id)
    prefix_rows = execution["completed_prefix"]["rows"]
    if prefix_ids != {case.case_id for case in ordered[:prefix_rows]}:
        raise ValueError("complete conditional majority prefix order differs")
    selected = select_shard(
        cases,
        prefix_ids=prefix_ids,
        num_shards=execution["sharding"]["num_shards"],
        shard_id=shard_id,
    )
    output_path = ROOT / execution["output"]["shard_pattern"].replace(
        "<shard_id>", str(shard_id)
    )
    completed = set(jsonl_ids(output_path)) if output_path.exists() else set()
    selected_ids = {case.case_id for _, case in selected}
    if not completed.issubset(selected_ids):
        raise ValueError("complete conditional majority shard output differs")
    direct_by_id = {
        row["case_id"]: row
        for row in jsonl_rows(ROOT / config["baseline"]["four_b_raw_path"])
        if row["benchmark"] == "gsm8k"
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    client = OpenAI(
        api_key="local-vllm",
        base_url=service["base_url"],
        timeout=240,
        max_retries=0,
    )
    for case_index, case in selected:
        if case.case_id in completed:
            continue
        direct = {
            key: value
            for key, value in direct_by_id[case.case_id].items()
            if key not in {"expected", "score"}
        }
        candidate, receipt = generate_gsm8k_candidate(
            replace(case, expected="__SEALED_DURING_GENERATION__"),
            direct,
            config,
            client=client,
            case_index=case_index,
        )
        prediction = candidate["prediction"]
        candidate.update(
            {
                "schema_version": "nano_harness_baseline_case_v1",
                "suite_id": config["experiment_id"],
                "case_id": case.case_id,
                "benchmark": case.benchmark,
                "prediction": prediction,
                "score": float(score_prediction(prediction, case.expected)),
                "status": "completed",
                "expected": case.expected,
                "source_index": case.source_index,
                "treatment_receipt": receipt,
                "execution_shard_id": shard_id,
                "global_case_index": case_index,
            }
        )
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(candidate, sort_keys=True) + "\n")
    rows = jsonl_rows(output_path)
    if {row["case_id"] for row in rows} != selected_ids:
        raise ValueError("complete conditional majority shard incomplete")
    return {
        "schema_version": (
            "nano_harness_complete_conditional_majority_shard_v1"
        ),
        "experiment_id": config["experiment_id"],
        "shard_id": shard_id,
        "rows": len(rows),
        "case_ids_sha256": hashlib.sha256(
            "\n".join(sorted(selected_ids)).encode()
        ).hexdigest(),
        "output_sha256": sha256_file(output_path),
        "policy_changed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--execution", default=str(EXECUTION))
    args = parser.parse_args()
    print(
        json.dumps(
            run_shard(args.shard_id, args.execution),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
