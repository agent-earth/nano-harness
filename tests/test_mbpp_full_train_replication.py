from __future__ import annotations

import json
import unittest
from pathlib import Path

from nano_harness.mbpp_full_train_replication import (
    case_ids_sha256,
    load_config,
    load_replication_cases,
    verify_unchanged_policy,
)
from nano_harness.mbpp_iterative_repair import (
    load_config as load_v2_config,
    load_train_cases,
)
from scripts.preregister_mbpp_full_train_replication_v2 import build_receipt
from scripts.render_mbpp_full_train_replication_v2 import (
    build_report as build_replication_report,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/campaign/mbpp_full_train_replication_v2.json"
V2_CONFIG = (
    ROOT / "configs/campaign/mbpp_sanitized_iterative_repair_train_v2.json"
)


class MbppFullTrainReplicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(CONFIG)
        cls.cases = load_replication_cases(cls.config, ROOT)

    def test_replication_is_fresh_and_exact(self):
        self.assertEqual(len(self.cases), 254)
        self.assertEqual(len({case.case_id for case in self.cases}), 254)
        prior_train_ids = {
            case.task_id
            for case in load_train_cases(load_v2_config(V2_CONFIG), ROOT)
        }
        self.assertFalse(
            {case.task_id for case in self.cases} & prior_train_ids
        )
        self.assertEqual(
            case_ids_sha256(self.cases),
            "f3ad3ba8bfd7fd47e08e5db94c8a2de7e1ba5299e8d0396bb526725a79fae93a",
        )

    def test_policy_is_unchanged_and_test_closed(self):
        verify_unchanged_policy(self.config, ROOT)
        receipt = build_receipt()
        self.assertEqual(receipt, build_receipt())
        self.assertTrue(receipt["policy_identity"]["unchanged_from_v2"])
        self.assertEqual(receipt["surface"]["cases"], 254)
        self.assertEqual(
            receipt["surface"]["shard_counts"],
            [32, 32, 32, 32, 32, 32, 31, 31],
        )
        for key, value in receipt["surface"].items():
            if key.startswith("overlap_with_"):
                self.assertEqual(value, 0)
        self.assertFalse(receipt["surface"]["sanitized_test_generation_started"])
        self.assertFalse(receipt["decision_rule"]["rerun_or_tuning_allowed"])

    def test_public_result_gate_when_available(self):
        path = ROOT / "docs/results/mbpp_full_train_replication_v2.public.json"
        if not path.exists():
            self.skipTest("full-train replication result not generated yet")
        report = build_replication_report()
        serialized = json.dumps(report).lower()
        self.assertNotIn('"output"', serialized)
        self.assertNotIn('"code"', serialized)
        self.assertNotIn('"test_list"', serialized)
        self.assertEqual(
            report["decision"]["sanitized_test_preregistration_allowed"],
            report["decision"]["replication_admitted"],
        )


if __name__ == "__main__":
    unittest.main()
