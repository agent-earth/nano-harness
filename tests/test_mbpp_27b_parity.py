from __future__ import annotations

import json
import unittest
from pathlib import Path

from nano_harness.mbpp_27b_parity import (
    case_ids_sha256,
    load_config,
    load_policy_config,
)
from nano_harness.mbpp_iterative_repair import select_shard
from nano_harness.mbpp_sanitized_test import load_test_cases
from scripts.preregister_mbpp_27b_parity_v1 import build_receipt
from scripts.render_mbpp_27b_parity_v1 import build_report


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/campaign/mbpp_27b_parity_v1.json"


class MbppTwentySevenBParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(CONFIG)
        cls.policy = load_policy_config(cls.config, ROOT)
        cls.cases = load_test_cases(cls.config, ROOT)

    def test_complete_case_set_and_frozen_policy(self):
        self.assertEqual(len(self.cases), 257)
        self.assertEqual(
            self.config["direct"],
            self.policy["direct"],
        )
        self.assertEqual(
            case_ids_sha256([case.case_id for case in self.cases]),
            "4b7e1f74447041a3f9ad4c02a27c157fd1ef98beb2c770796f4bfd8e630863d8",
        )
        shards = [
            select_shard(self.cases, num_shards=2, shard_id=shard_id)
            for shard_id in range(2)
        ]
        self.assertEqual([len(shard) for shard in shards], [129, 128])

    def test_preregister_is_deterministic_and_one_shot(self):
        first = build_receipt()
        self.assertEqual(first, build_receipt())
        self.assertEqual(first["surface"]["cases"], 257)
        self.assertTrue(first["surface"]["complete_benchmark"])
        self.assertFalse(first["surface"]["four_b_generation_repeated"])
        self.assertFalse(first["surface"]["nine_b_generation_repeated"])
        self.assertEqual(
            first["comparison"]["noninferiority_margin"],
            0.02,
        )
        self.assertFalse(
            first["execution_boundary"]["parity_generation_started"]
        )
        self.assertFalse(
            first["policy"]["post_observation_tuning_or_rerun"]
        )

    def test_public_result_when_available(self):
        path = ROOT / "docs/results/mbpp_27b_parity_v1.public.json"
        if not path.exists():
            self.skipTest("MBPP 27B parity result not generated yet")
        report = build_report()
        self.assertFalse(report["decision"]["mbpp_complete_parity_with_27b"])
        self.assertEqual(
            (
                report["comparison"]["candidate_correct"],
                report["comparison"]["baseline_correct"],
            ),
            (219, 226),
        )
        self.assertLess(
            report["comparison"]["paired_bootstrap_95_ci"][0],
            -report["noninferiority"]["margin"],
        )
        self.assertFalse(
            report["noninferiority"]["gates"][
                "twenty_seven_b_parse_failures_zero"
            ]
        )
        serialized = json.dumps(report).lower()
        self.assertNotIn('"output"', serialized)
        self.assertNotIn('"code"', serialized)
        self.assertNotIn('"test_list"', serialized)
        self.assertFalse(report["decision"]["rerun_or_tuning_allowed"])


if __name__ == "__main__":
    unittest.main()
