#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

from nano_harness.baseline import sha256_file
from nano_harness.orca_self_consistency_replication import (
    load_config,
    select_cases,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/campaign/orca_math_self_consistency_replication_v2.json"
)
OUTPUT = (
    ROOT
    / "docs/experiments/"
    "orca_math_self_consistency_replication_v2.preregister.json"
)


def build_receipt() -> dict:
    config = load_config(CONFIG)
    selection = select_cases(config)
    raw = config.raw
    return {
        "schema_version": (
            "nano_harness_orca_self_consistency_replication_preregister_v2"
        ),
        "experiment_id": raw["experiment_id"],
        "identity": {
            "config_sha256": sha256_file(CONFIG),
            "dataset_sha256": raw["dataset_sha256"],
            "four_b_model_config_sha256": raw["four_b"][
                "model_config_sha256"
            ],
            "nine_b_model_config_sha256": raw["nine_b"][
                "model_config_sha256"
            ],
            "prior_dpo_v1_preregister_sha256": raw[
                "prior_dpo_v1_preregister_sha256"
            ],
            "prior_dpo_v2_preregister_sha256": raw[
                "prior_dpo_v2_preregister_sha256"
            ],
            "prior_self_consistency_preregister_sha256": raw[
                "prior_self_consistency_preregister_sha256"
            ],
            "prior_self_consistency_result_sha256": raw[
                "prior_self_consistency_result_sha256"
            ],
        },
        "selection": {
            "cases": 160,
            "by_stratum": raw["cases_by_stratum"],
            "case_ids": selection["case_ids"],
            "case_ids_sha256": selection["case_ids_sha256"],
            "prior_ids_sha256": selection["prior_ids_sha256"],
            "covers_every_remaining_row": True,
        },
        "strategy": {
            "four_b_direct": raw["direct"],
            "candidate": raw["candidate"],
            "nine_b_direct": raw["direct"],
            "exactly_matches_v1": True,
        },
        "decision_rule": {
            "four_b_preservation": (
                "candidate delta >= 0, bootstrap lower >= 0, no stratum "
                "regression, and McNemar does not show significant regression"
            ),
            "nine_b_superiority": (
                "positive delta, bootstrap lower > 0, McNemar p < 0.05, "
                ">=6 wins, wins>losses, every stratum non-regressing"
            ),
            "complete_benchmark_allowed": (
                "four_b_preservation AND nine_b_superiority"
            ),
            "rerun_or_tuning_allowed": False,
        },
        "execution_boundary": {
            "generation_started": False,
            "expected_answer_used_during_generation": False,
            "benchmark_rows_used": False,
            "this_commit_only_preregisters": True,
        },
    }


def main() -> None:
    receipt = build_receipt()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
