from __future__ import annotations

import json
import unittest
from pathlib import Path

from nano_harness.mbpp_sanitized_test import (
    case_ids_sha256,
    load_config,
    load_test_cases,
    verify_unchanged_policy,
)
from scripts.preregister_mbpp_sanitized_test_v2 import build_receipt
from scripts.render_mbpp_sanitized_test_v2 import (
    build_report as build_test_report,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/campaign/mbpp_sanitized_test_v2.json"


class MbppSanitizedTestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(CONFIG)
        cls.cases = load_test_cases(cls.config, ROOT)

    def test_test_surface_is_exact(self):
        self.assertEqual(len(self.cases), 257)
        self.assertEqual(len({case.case_id for case in self.cases}), 257)
        self.assertEqual(
            case_ids_sha256(self.cases),
            case_ids_sha256(load_test_cases(self.config, ROOT)),
        )

    def test_policy_is_unchanged_and_preregistered(self):
        verify_unchanged_policy(self.config, ROOT)
        receipt = build_receipt()
        self.assertEqual(receipt, build_receipt())
        self.assertTrue(receipt["policy_identity"]["unchanged_from_v2"])
        self.assertEqual(receipt["surface"]["cases"], 257)
        self.assertEqual(
            receipt["surface"]["shard_counts"],
            [33, 32, 32, 32, 32, 32, 32, 32],
        )
        for key, value in receipt["surface"].items():
            if key.startswith("overlap_with_"):
                self.assertEqual(value, 0)
        self.assertFalse(receipt["surface"]["test_generation_started"])
        self.assertFalse(receipt["decision_rule"]["rerun_or_tuning_allowed"])

    def test_public_result_gate_when_available(self):
        path = ROOT / "docs/results/mbpp_sanitized_test_v2.public.json"
        if not path.exists():
            self.skipTest("sanitized-test result not generated yet")
        report = build_test_report()
        serialized = json.dumps(report).lower()
        self.assertNotIn('"output"', serialized)
        self.assertNotIn('"code"', serialized)
        self.assertNotIn('"test_list"', serialized)
        self.assertFalse(report["decision"]["rerun_or_tuning_allowed"])


if __name__ == "__main__":
    unittest.main()
