#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

from nano_harness.baseline import sha256_file
from nano_harness.orca_self_consistency import load_config, select_cases


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/campaign/orca_math_self_consistency_v1.json"
OUTPUT = (
    ROOT / "docs/experiments/orca_math_self_consistency_v1.preregister.json"
)


def build_receipt() -> dict:
    config = load_config(CONFIG)
    selection = select_cases(config)
    raw = config.raw
    return {
        "schema_version": "nano_harness_orca_self_consistency_preregister_v1",
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
            "prior_dpo_v2_result_sha256": raw[
                "prior_dpo_v2_result_sha256"
            ],
        },
        "selection": {
            "cases": 96,
            "by_stratum": raw["cases_by_stratum"],
            "case_ids": selection["case_ids"],
            "case_ids_sha256": selection["case_ids_sha256"],
            "prior_ids_sha256": selection["prior_ids_sha256"],
        },
        "arms": {
            "four_b_direct": raw["direct"],
            "candidate": raw["candidate"],
            "nine_b_direct": raw["direct"],
        },
        "decision_rule": {
            "candidate_over_four": (
                "positive point delta, bootstrap lower > 0, McNemar p < "
                "0.05, >=6 wins, wins>losses, every stratum non-regressing"
            ),
            "candidate_over_nine": (
                "positive point delta, bootstrap lower > 0, McNemar p < "
                "0.05, >=6 wins, wins>losses, every stratum non-regressing"
            ),
            "complete_benchmark_allowed": (
                "both comparisons pass and independent raw identities match"
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
