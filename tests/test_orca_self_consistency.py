from __future__ import annotations

import unittest
from pathlib import Path

from nano_harness.orca_self_consistency import (
    consensus_prediction,
    load_config,
    score_prediction,
    select_cases,
)
from scripts.preregister_orca_self_consistency_v1 import build_receipt


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/campaign/orca_math_self_consistency_v1.json"


class OrcaSelfConsistencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(CONFIG)
        cls.selection = select_cases(cls.config)

    def test_selection_is_fresh_and_deterministic(self):
        self.assertEqual(len(self.selection["cases"]), 96)
        self.assertEqual(
            self.selection["case_ids"],
            select_cases(self.config)["case_ids"],
        )

    def test_four_of_five_consensus_overrides(self):
        prediction, receipt = consensus_prediction(
            ["12", "12.0", "12", "12", "7"],
            "7",
            minimum_agreement=4,
        )
        self.assertTrue(score_prediction(prediction, "12"))
        self.assertTrue(receipt["override"])
        self.assertFalse(receipt["fallback"])

    def test_weak_consensus_falls_back(self):
        prediction, receipt = consensus_prediction(
            ["12", "12", "9", "9", "7"],
            "7",
            minimum_agreement=4,
        )
        self.assertEqual(prediction, "7")
        self.assertTrue(receipt["fallback"])
        self.assertFalse(receipt["override"])

    def test_preregister_is_deterministic_and_closed(self):
        first = build_receipt()
        second = build_receipt()
        self.assertEqual(first, second)
        self.assertFalse(
            first["decision_rule"]["rerun_or_tuning_allowed"]
        )
        self.assertTrue(
            first["execution_boundary"]["this_commit_only_preregisters"]
        )


if __name__ == "__main__":
    unittest.main()
