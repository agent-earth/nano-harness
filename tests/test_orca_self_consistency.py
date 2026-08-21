from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from nano_harness.orca_self_consistency import (
    build_raw_result,
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

    def test_raw_result_accepts_generic_exclusion_identity(self):
        selection = {
            "cases": [{"sample_id": "a"}],
            "case_ids": ["a"],
            "case_ids_sha256": "case-hash",
            "excluded_source_ids_sha256": "excluded-hash",
        }
        rows = {
            "four_direct.jsonl": '{"case_id":"a"}\n',
            "candidate.jsonl": '{"case_id":"a"}\n',
            "nine_direct.jsonl": '{"case_id":"a"}\n',
            "receipts.json": '[{"case_id":"a"}]\n',
        }
        with self.subTest("alternate exclusion identity"):
            from tempfile import TemporaryDirectory

            with TemporaryDirectory() as directory:
                output = Path(directory)
                for name, value in rows.items():
                    (output / name).write_text(value, encoding="utf-8")
                config = mock.Mock()
                config.raw = {
                    "experiment_id": "test",
                    "output_dir": "ignored",
                }
                config.resolve.return_value = output
                result = build_raw_result(config, selection)
        self.assertEqual(
            result["selection"]["excluded_ids_sha256"],
            "excluded-hash",
        )


if __name__ == "__main__":
    unittest.main()
