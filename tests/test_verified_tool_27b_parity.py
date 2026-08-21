from __future__ import annotations

import json
import unittest
from pathlib import Path

from nano_harness.verified_tool_27b_parity import (
    load_config,
    load_source,
    select_shard,
)
from nano_harness.verified_tool_execution import build_cases
from scripts.preregister_verified_tool_27b_parity_v1 import build_receipt
from scripts.render_verified_tool_27b_parity_v1 import build_report


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/campaign/verified_tool_27b_parity_v1.json"


class VerifiedToolTwentySevenBParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(CONFIG)
        parent, cls.raw = load_source(cls.config, ROOT)
        cls.cases = build_cases(parent)

    def test_complete_frozen_source(self):
        self.assertEqual(len(self.cases), 256)
        self.assertEqual(len(self.raw["harness_rows"]), 256)
        self.assertEqual(
            sum(row["correct"] for row in self.raw["harness_rows"]),
            256,
        )
        shards = [
            select_shard(self.cases, num_shards=2, shard_id=shard_id)
            for shard_id in range(2)
        ]
        self.assertEqual([len(shard) for shard in shards], [128, 128])

    def test_preregister_is_deterministic_and_one_shot(self):
        first = build_receipt()
        self.assertEqual(first, build_receipt())
        self.assertEqual(first["surface"]["cases"], 256)
        self.assertEqual(
            first["surface"]["families"],
            {
                "box_total": 64,
                "labor_total": 64,
                "paired_average": 64,
                "remaining_stock": 64,
            },
        )
        self.assertFalse(first["surface"]["four_b_generation_repeated"])
        self.assertFalse(first["surface"]["nine_b_generation_repeated"])
        self.assertEqual(
            first["comparison"]["noninferiority_margin"],
            0.02,
        )
        self.assertFalse(
            first["execution_boundary"]["parity_generation_started"]
        )

    def test_public_result_when_available(self):
        path = ROOT / "docs/results/verified_tool_27b_parity_v1.public.json"
        if not path.exists():
            self.skipTest("verified-tool 27B parity result not generated yet")
        report = build_report()
        self.assertTrue(
            report["decision"]["complete_verified_tool_parity_with_27b"]
        )
        self.assertTrue(report["decision"]["four_b_harness_exceeds_27b"])
        self.assertEqual(
            (
                report["comparison"]["overall"]["candidate_accuracy"],
                report["comparison"]["overall"]["baseline_accuracy"],
            ),
            (1.0, 0.24609375),
        )
        self.assertTrue(all(report["noninferiority"]["gates"].values()))
        self.assertTrue(
            all(
                row["paired_bootstrap_95_ci"][0] >= -0.02
                for row in report["comparison"]["by_family"].values()
            )
        )
        serialized = json.dumps(report).lower()
        self.assertNotIn('"output"', serialized)
        self.assertFalse(report["decision"]["rerun_or_tuning_allowed"])
        self.assertEqual(
            report["noninferiority"]["parity_admitted"],
            report["decision"]["complete_verified_tool_parity_with_27b"],
        )


if __name__ == "__main__":
    unittest.main()
