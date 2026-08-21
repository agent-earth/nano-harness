#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

from nano_harness.baseline import sha256_file
from nano_harness.orca_conditional_majority import load_config, select_cases


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "configs/campaign/orca_math_conditional_majority_v4.json"
)
OUTPUT = (
    ROOT
    / "docs/experiments/orca_math_conditional_majority_v4.preregister.json"
)


def build_receipt() -> dict:
    config = load_config(CONFIG)
    selection = select_cases(config)
    raw = config.raw
    return {
        "schema_version": (
            "nano_harness_orca_conditional_majority_preregister_v4"
        ),
        "experiment_id": raw["experiment_id"],
        "identity": {
            "config_sha256": sha256_file(CONFIG),
            "source_dataset_sha256": raw["source_dataset_sha256"],
            "preference_dataset_sha256": raw[
                "preference_dataset_sha256"
            ],
            "prior_sft_preregister_sha256": raw[
                "prior_sft_preregister_sha256"
            ],
            "prior_v3_preregister_sha256": raw[
                "prior_v3_preregister_sha256"
            ],
            "prior_v3_result_sha256": raw["prior_v3_result_sha256"],
            "four_b_model_config_sha256": raw["four_b"][
                "model_config_sha256"
            ],
            "nine_b_model_config_sha256": raw["nine_b"][
                "model_config_sha256"
            ],
        },
        "selection": {
            "cases": 96,
            "by_stratum": raw["cases_by_stratum"],
            "case_ids": selection["case_ids"],
            "case_ids_sha256": selection["case_ids_sha256"],
            "excluded_source_ids_sha256": selection[
                "excluded_source_ids_sha256"
            ],
        },
        "parser": raw["parser"],
        "override_rule": raw["override_rule"],
        "strategy": {
            "four_b_direct": raw["direct"],
            "candidate": raw["candidate"],
            "nine_b_direct": raw["direct"],
        },
        "decision_rule": {
            "four_b_preservation": (
                "candidate delta >= 0, bootstrap lower >= 0, no stratum "
                "regression, and no significant regression"
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
