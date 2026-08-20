from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from nano_harness.baseline import sha256_file


CONFIG_SHA256 = (
    "b083d320e7103cb0809b5c22e6f8ebbc9330a3eb96d3910fd17d34c8f9a52f10"
)


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if sha256_file(config_path) != CONFIG_SHA256:
        raise ValueError("V5 complete treatment config SHA differs")
    if config.get("schema_version") != "nano_harness_v5_complete_treatment_v1":
        raise ValueError("unsupported V5 complete treatment schema")
    if config.get("execution_boundary") != {
        "benchmark_generation_started": False,
        "benchmark_outputs_loaded_by_preregister": False,
        "canary_rerun": False,
        "holdout_accessed": False,
        "rl_started": False,
        "this_commit_only_preregisters": True,
        "training_started": False,
    }:
        raise ValueError("V5 complete treatment boundary differs")
    if config.get("policy") != {
        "benchmark_rows_training_eligible": False,
        "benchmark_outputs_may_enter_training_reward_or_verifier": False,
        "case_id_allowlist_for_routing": False,
        "expected_answer_used_by_routing_or_execution": False,
        "post_observation_search": False,
        "raw_outputs_committed": False,
    }:
        raise ValueError("V5 complete treatment policy differs")
    if config.get("routes") != {
        "gpqa_diamond": {
            "choice_regex": "FINAL: [A-D]",
            "confirmation_max_tokens": 64,
            "option_review_max_tokens": 96,
            "override_rule": (
                "two_independent_reviews_and_confirmation_agree_on_same_non_direct_choice"
            ),
            "otherwise": "preserve_frozen_4b_direct",
            "strategy": "conservative_choice_consensus",
        },
        "gsm8k": {
            "expression_regex": r"CALC: [0-9+\-*/(). ]+",
            "maximum_absolute_value": 10**15,
            "maximum_ast_nodes": 64,
            "maximum_expression_chars": 160,
            "otherwise": "preserve_frozen_4b_direct",
            "plan_max_tokens": 128,
            "plan_replicas": 3,
            "strategy": "grounded_expression_consensus",
            "verified_result_rule": (
                "at_least_two_executed_grounded_expressions_agree"
            ),
        },
        "mmlu": {"strategy": "preserve_frozen_4b_direct"},
    }:
        raise ValueError("V5 complete treatment routes differ")
    return config


def jsonl_ids(path: Path) -> list[str]:
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        match = re.search(r'"case_id"\s*:\s*"([^"]+)"', line)
        if match is None:
            raise ValueError("V5 complete treatment raw case ID is missing")
        values.append(match.group(1))
    if len(values) != len(set(values)):
        raise ValueError("V5 complete treatment raw IDs are duplicated")
    return values
