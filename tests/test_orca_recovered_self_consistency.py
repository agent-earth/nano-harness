from __future__ import annotations

import unittest
from pathlib import Path

from nano_harness.orca_recovered_self_consistency import (
    load_config,
    parse_recovered_final,
    select_cases,
)
from scripts.preregister_orca_recovered_self_consistency_v3 import (
    build_receipt,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/campaign/orca_math_recovered_self_consistency_v3.json"
)


class RecoveredSelfConsistencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(CONFIG)
        cls.selection = select_cases(cls.config)

    def test_parser_prefers_strict_final(self):
        self.assertEqual(
            parse_recovered_final("WORK: 9\nFINAL: 12"),
            "12",
        )

    def test_parser_recovers_last_plain_or_latex_number(self):
        self.assertEqual(
            parse_recovered_final("First 4 then the answer is 15."),
            "15",
        )
        self.assertEqual(
            parse_recovered_final(r"Therefore the answer is \frac{3}{4}."),
            "3/4",
        )
        self.assertIsNone(parse_recovered_final("No numeric value."))

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
        self.assertTrue(first["parser"]["target_blind"])
        self.assertFalse(
            first["decision_rule"]["rerun_or_tuning_allowed"]
        )
        self.assertTrue(
            first["execution_boundary"]["this_commit_only_preregisters"]
        )


if __name__ == "__main__":
    unittest.main()
