from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import replace
from pathlib import Path
from typing import Any

from openai import OpenAI

from nano_harness.baseline import (
    BaselineCase,
    load_cases,
    load_manifest,
    sha256_file,
)
from nano_harness.orca_conditional_majority import conditional_consensus
from nano_harness.orca_recovered_self_consistency import (
    parse_recovered_final,
)
from nano_harness.orca_self_consistency import (
    _request,
    parse_final,
    score_prediction,
)
from nano_harness.v5_complete_treatment import jsonl_ids, jsonl_rows


CONFIG_SHA256 = (
    "4f8c138166ada6c03edddfd3205d2cf7b3bc8baf86bdab879fe067f38a2e5013"
)


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if sha256_file(config_path) != CONFIG_SHA256:
        raise ValueError("complete conditional majority config SHA differs")
    if (
        config.get("schema_version")
        != "nano_harness_complete_conditional_majority_v1"
        or config.get("experiment_id")
        != "qwen35-complete-conditional-majority-v1"
        or config.get("routes")
        != {
            "gsm8k": {
                "strategy": "conditional_majority_v4",
                "parser": {
                    "strict_final_first": True,
                    "fallback": "last_numeric_token_in_last_1500_chars",
                    "target_blind": True,
                },
                "replicas": 5,
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 384,
                "seed_base": 2026082600,
                "direct_strict_parse_failure_minimum_votes": 3,
                "direct_strict_parseable_minimum_votes": 5,
                "fallback": "frozen_four_b_recovered_direct",
            },
            "mmlu": {"strategy": "preserve_frozen_four_b_direct"},
            "gpqa_diamond": {
                "strategy": "reuse_frozen_v5_conservative_choice_consensus"
            },
        }
        or config.get("statistics")
        != {
            "bootstrap_samples": 10_000,
            "bootstrap_seed": "qwen35-complete-conditional-majority-v1",
            "exact_mcnemar": True,
            "nominal_alpha": 0.05,
            "complete_gsm8k_attempt_family_size": 2,
            "bonferroni_alpha": 0.025,
            "minimum_candidate_only_wins": 6,
            "four_b_preservation": {
                "point_delta_nonnegative": True,
                "bootstrap_ci_lower_nonnegative": True,
                "no_significant_regression": True,
                "strict_parseable_route_non_regression": True,
                "strict_parse_failure_route_non_regression": True,
            },
            "nine_b_superiority": {
                "point_delta_positive": True,
                "bootstrap_ci_lower_positive": True,
                "mcnemar_p_below_bonferroni_alpha": True,
                "minimum_candidate_only_wins": True,
                "candidate_only_exceeds_baseline_only": True,
            },
            "final_three_benchmark_family": {
                "procedure": "holm_bonferroni",
                "family_size": 3,
                "familywise_alpha": 0.05,
            },
        }
        or config.get("policy")
        != {
            "benchmark_rows_training_eligible": False,
            "benchmark_outputs_may_enter_training_reward_or_verifier": False,
            "expected_answer_used_during_generation": False,
            "case_correctness_used_during_generation": False,
            "case_id_allowlist_for_routing": False,
            "post_observation_search": False,
            "raw_outputs_committed": False,
            "prior_v5_gsm8k_outputs_reused": False,
            "mmlu_and_gpqa_model_requests_repeated": False,
        }
        or config.get("execution_boundary")
        != {
            "benchmark_generation_started": False,
            "benchmark_output_content_loaded_by_preregister": False,
            "full_candidate_composed": False,
            "this_commit_only_preregisters": True,
            "training_started": False,
            "rl_or_opd_started": False,
            "twenty_seven_b_accessed": False,
        }
    ):
        raise ValueError("complete conditional majority contract differs")
    return config


def verify_four_b_service(config: dict[str, Any]) -> dict[str, Any]:
    with urllib.request.urlopen(
        config["four_b"]["base_url"] + "/models",
        timeout=30,
    ) as response:
        health = json.loads(response.read().decode("utf-8"))
    rows = health.get("data", [])
    if len(rows) != 1:
        raise ValueError("complete conditional majority expects one model")
    row = rows[0]
    if (
        row.get("id") != config["four_b"]["model"]
        or row.get("root") != config["four_b"]["path"]
        or row.get("parent") is not None
        or row.get("max_model_len") != config["four_b"]["max_model_len"]
        or row.get("owned_by") != "vllm"
    ):
        raise ValueError("complete conditional majority service differs")
    return health


def generate_gsm8k_candidate(
    case: BaselineCase,
    direct: dict[str, Any],
    config: dict[str, Any],
    *,
    client: Any,
    case_index: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if case.benchmark != "gsm8k":
        raise ValueError("complete conditional majority only generates GSM8K")
    route = config["routes"]["gsm8k"]
    started = time.perf_counter()
    replicas = [
        _request(
            client,
            model=config["four_b"]["model"],
            messages=[
                {"role": "system", "content": case.system_prompt},
                {"role": "user", "content": case.prompt},
            ],
            temperature=route["temperature"],
            top_p=route["top_p"],
            max_tokens=route["max_tokens"],
            seed=route["seed_base"] + case_index * 10 + replica,
            parser=parse_recovered_final,
        )
        for replica in range(route["replicas"])
    ]
    direct_prediction = parse_recovered_final(direct["output"])
    strict_prediction = parse_final(direct["output"])
    candidate_prediction, receipt = conditional_consensus(
        [reply["prediction"] for reply in replicas],
        direct_prediction,
        direct_strict_parseable=strict_prediction is not None,
        parse_failure_minimum_votes=route[
            "direct_strict_parse_failure_minimum_votes"
        ],
        parseable_minimum_votes=route[
            "direct_strict_parseable_minimum_votes"
        ],
    )
    candidate_output = direct["output"]
    if not receipt["fallback"]:
        candidate_output = next(
            reply["output"]
            for reply in replicas
            if reply["prediction"] == candidate_prediction
        )
    candidate = {
        **direct,
        "model": "qwen3.5-4b-complete-conditional-majority",
        "output": candidate_output,
        "prediction": candidate_prediction,
        "treatment_route": "gsm8k_conditional_majority_v4",
        "latency_seconds": time.perf_counter() - started,
        "replica_usage": [reply["usage"] for reply in replicas],
    }
    return candidate, {
        **receipt,
        "route": "gsm8k_conditional_majority_v4",
        "model_calls": len(replicas),
        "api_errors": 0,
    }


def run(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    root = config_path.resolve().parents[2]
    config = load_config(config_path)
    for section in ("baseline", "predecessors"):
        values = config[section]
        for key, value in values.items():
            if not key.endswith("_path"):
                continue
            digest_key = key.removesuffix("_path") + "_sha256"
            path = root / value
            if not path.is_file() or sha256_file(path) != values[digest_key]:
                raise ValueError(
                    f"complete conditional majority {section}.{key} differs"
                )

    manifest = load_manifest(root / config["baseline"]["suite_manifest_path"])
    cases = [
        case
        for case in load_cases(
            manifest,
            (root / "../../../datasets").resolve(),
        )
        if case.benchmark == "gsm8k"
    ]
    case_by_id = {case.case_id: case for case in cases}
    direct_rows = [
        row
        for row in jsonl_rows(root / config["baseline"]["four_b_raw_path"])
        if row["benchmark"] == "gsm8k"
    ]
    direct_by_id = {str(row["case_id"]): row for row in direct_rows}
    if set(case_by_id) != set(direct_by_id) or len(cases) != 1_319:
        raise ValueError("complete conditional majority GSM8K set differs")

    service_health = verify_four_b_service(config)
    output_path = root / config["output"]["gsm8k_candidate_path"]
    receipt_path = root / config["output"]["gsm8k_receipts_path"]
    completed = set(jsonl_ids(output_path)) if output_path.exists() else set()
    if not completed.issubset(case_by_id):
        raise ValueError("complete conditional majority output IDs differ")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    client = OpenAI(
        api_key="local-vllm",
        base_url=config["four_b"]["base_url"],
        timeout=240,
        max_retries=0,
    )
    ordered_cases = sorted(cases, key=lambda case: case.case_id)
    for case_index, case in enumerate(ordered_cases):
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
        score = float(score_prediction(prediction, case.expected))
        candidate.update(
            {
                "schema_version": "nano_harness_baseline_case_v1",
                "suite_id": config["experiment_id"],
                "case_id": case.case_id,
                "benchmark": case.benchmark,
                "prediction": prediction,
                "score": score,
                "status": "completed",
                "expected": case.expected,
                "source_index": case.source_index,
                "treatment_receipt": receipt,
            }
        )
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(candidate, sort_keys=True) + "\n")

    candidate_ids = jsonl_ids(output_path)
    if set(candidate_ids) != set(case_by_id):
        raise ValueError("complete conditional majority generation incomplete")

    gsm8k_rows = jsonl_rows(output_path)
    if any("treatment_receipt" not in row for row in gsm8k_rows):
        raise ValueError("complete conditional majority receipt is missing")
    gsm8k_by_id = {row["case_id"]: row for row in gsm8k_rows}
    with receipt_path.open("w", encoding="utf-8") as handle:
        for row in sorted(gsm8k_rows, key=lambda item: item["case_id"]):
            handle.write(
                json.dumps(
                    {
                        "case_id": row["case_id"],
                        "receipt": row["treatment_receipt"],
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    prior_rows = jsonl_rows(
        root / config["predecessors"]["prior_complete_candidate_path"]
    )
    complete_rows = []
    for row in direct_rows + [
        item
        for item in jsonl_rows(
            root / config["baseline"]["four_b_raw_path"]
        )
        if item["benchmark"] == "mmlu"
    ]:
        if row["benchmark"] == "gsm8k":
            complete_rows.append(gsm8k_by_id[row["case_id"]])
        else:
            complete_rows.append(
                {
                    **row,
                    "suite_id": config["experiment_id"],
                    "model": "qwen3.5-4b-complete-conditional-majority",
                    "treatment_route": "mmlu_direct_preserve",
                }
            )
    complete_rows.extend(
        {
            **row,
            "suite_id": config["experiment_id"],
            "model": "qwen3.5-4b-complete-conditional-majority",
            "treatment_route": "gpqa_frozen_v5_reuse",
        }
        for row in prior_rows
        if row["benchmark"] == "gpqa_diamond"
    )
    complete_rows.sort(key=lambda row: row["case_id"])
    if (
        len(complete_rows) != 15_559
        or len({row["case_id"] for row in complete_rows}) != 15_559
    ):
        raise ValueError("complete conditional majority composition differs")
    complete_path = root / config["output"]["complete_candidate_path"]
    with complete_path.open("w", encoding="utf-8") as handle:
        for row in complete_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    result = {
        "schema_version": (
            "nano_harness_complete_conditional_majority_raw_v1"
        ),
        "experiment_id": config["experiment_id"],
        "identity": {
            "config_sha256": sha256_file(config_path),
            "gsm8k_candidate_sha256": sha256_file(output_path),
            "gsm8k_receipts_sha256": sha256_file(receipt_path),
            "complete_candidate_sha256": sha256_file(complete_path),
            "service_health_sha256": __import__("hashlib")
            .sha256(
                json.dumps(
                    service_health,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            )
            .hexdigest(),
        },
        "surface": {
            "generated_benchmark": "gsm8k",
            "generated_cases": len(candidate_ids),
            "mmlu_model_requests": 0,
            "gpqa_diamond_model_requests": 0,
        },
        "evaluation_boundary": {
            "benchmark_rows_training_eligible": False,
            "expected_answer_used_during_generation": False,
            "case_correctness_used_during_generation": False,
            "scoring_applied_after_generation": True,
            "raw_outputs_committed": False,
        },
    }
    result_path = root / config["output"]["result_path"]
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result
