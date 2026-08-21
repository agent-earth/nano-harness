from __future__ import annotations

import unittest
from pathlib import Path

from nano_harness.orca_conditional_majority import (
    conditional_consensus,
    load_config,
    select_cases,
)
from scripts.preregister_orca_conditional_majority_v4 import build_receipt


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/campaign/orca_math_conditional_majority_v4.json"


class ConditionalMajorityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(CONFIG)
        cls.selection = select_cases(cls.config)

    def test_parse_failure_accepts_three_of_five(self):
        prediction, receipt = conditional_consensus(
            ["12", "12", "12", "9", "7"],
            "7",
            direct_strict_parseable=False,
            parse_failure_minimum_votes=3,
            parseable_minimum_votes=5,
        )
        self.assertEqual(prediction, "12")
        self.assertTrue(receipt["override"])
        self.assertEqual(receipt["minimum_votes"], 3)

    def test_parseable_direct_requires_five_of_five(self):
        prediction, receipt = conditional_consensus(
            ["12", "12", "12", "12", "7"],
            "7",
            direct_strict_parseable=True,
            parse_failure_minimum_votes=3,
            parseable_minimum_votes=5,
        )
        self.assertEqual(prediction, "7")
        self.assertTrue(receipt["fallback"])
        prediction, receipt = conditional_consensus(
            ["12", "12", "12", "12", "12"],
            "7",
            direct_strict_parseable=True,
            parse_failure_minimum_votes=3,
            parseable_minimum_votes=5,
        )
        self.assertEqual(prediction, "12")
        self.assertTrue(receipt["override"])

    def test_selection_is_fresh_and_deterministic(self):
        self.assertEqual(len(self.selection["cases"]), 96)
        self.assertEqual(
            self.selection["case_ids"],
            select_cases(self.config)["case_ids"],
        )

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
