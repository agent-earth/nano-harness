from __future__ import annotations

import json
import unittest

from scripts.render_orca_self_consistency_replication_v2 import (
    build_report,
    four_b_preservation_gates,
)


class RenderReplicationTests(unittest.TestCase):
    def test_four_b_preservation_accepts_zero_loss_directional_gain(self):
        comparison = {
            "delta": 0.01,
            "paired_bootstrap_95_ci": [0.0, 0.04],
            "mcnemar_exact_p": 0.25,
            "by_stratum": {
                "short": {"delta": 0.0},
                "medium": {"delta": 0.02},
                "long": {"delta": 0.01},
            },
        }
        self.assertTrue(
            all(four_b_preservation_gates(comparison).values())
        )

    def test_public_report_preserves_failed_long_gate(self):
        report = build_report()
        self.assertFalse(report["decision"]["replication_admitted"])
        self.assertFalse(
            report["decision"]["nine_b_superiority_gates"][
                "every_stratum_non_regression"
            ]
        )
        self.assertTrue(
            all(
                report["decision"]["four_b_preservation_gates"].values()
            )
        )
        self.assertFalse(
            report["boundary"]["pooled_result_overrides_replication_gate"]
        )
        serialized = json.dumps(report).lower()
        self.assertNotIn('"output"', serialized)


if __name__ == "__main__":
    unittest.main()
