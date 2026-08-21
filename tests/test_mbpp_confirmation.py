from __future__ import annotations

import json
import unittest
from pathlib import Path

from nano_harness.mbpp_confirmation import (
    case_ids_sha256,
    load_config,
    load_confirmation_cases,
)
from nano_harness.mbpp_iterative_repair import load_train_cases
from nano_harness.mbpp_verified_selection import (
    load_cases as load_v1_validation_cases,
    load_config as load_v1_config,
)
from scripts.preregister_mbpp_full_validation_confirmation_v2 import (
    build_receipt,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/campaign/mbpp_full_validation_confirmation_v2.json"
V1_CONFIG = (
    ROOT
    / "configs/campaign/mbpp_sanitized_verified_selection_dev_v1.json"
)
V2_CONFIG = (
    ROOT
    / "configs/campaign/mbpp_sanitized_iterative_repair_train_v2.json"
)


class MbppConfirmationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(CONFIG)
        cls.cases = load_confirmation_cases(cls.config, ROOT)

    def test_confirmation_is_fresh_and_exact(self):
        self.assertEqual(len(self.cases), 47)
        self.assertEqual(len({case.case_id for case in self.cases}), 47)
        confirmation_task_ids = {case.task_id for case in self.cases}
        prior_validation_task_ids = {
            case.task_id
            for case in load_v1_validation_cases(
                load_v1_config(V1_CONFIG),
                ROOT,
            )
        }
        train_task_ids = {
            case.task_id
            for case in load_train_cases(
                __import__(
                    "nano_harness.mbpp_iterative_repair",
                    fromlist=["load_config"],
                ).load_config(V2_CONFIG),
                ROOT,
            )
        }
        self.assertFalse(confirmation_task_ids & prior_validation_task_ids)
        self.assertFalse(confirmation_task_ids & train_task_ids)
        self.assertEqual(
            case_ids_sha256(self.cases),
            case_ids_sha256(load_confirmation_cases(self.config, ROOT)),
        )

    def test_preregister_is_deterministic_and_test_closed(self):
        first = build_receipt()
        second = build_receipt()
        self.assertEqual(first, second)
        self.assertEqual(first["surface"]["cases"], 47)
        self.assertEqual(
            first["surface"]["shard_counts"],
            [12, 12, 12, 11],
        )
        self.assertEqual(first["surface"]["overlap_with_sanitized_train"], 0)
        self.assertEqual(
            first["surface"]["overlap_with_sanitized_validation"],
            0,
        )
        self.assertFalse(first["surface"]["test_generation_started"])
        self.assertFalse(first["decision_rule"]["rerun_or_tuning_allowed"])

    def test_public_result_gate_when_available(self):
        path = ROOT / "docs/results/mbpp_full_validation_confirmation_v2.public.json"
        if not path.exists():
            self.skipTest("confirmation result not generated yet")
        report = json.loads(path.read_text(encoding="utf-8"))
        serialized = json.dumps(report).lower()
        self.assertNotIn('"output"', serialized)
        self.assertNotIn('"code"', serialized)
        self.assertNotIn('"test_list"', serialized)


if __name__ == "__main__":
    unittest.main()
