from __future__ import annotations

import json
import unittest

from scripts.render_orca_recovered_self_consistency_v3 import build_report


class RenderRecoveredSelfConsistencyTests(unittest.TestCase):
    def test_public_report_keeps_strict_gate_and_finalize_boundary(self):
        report = build_report()
        self.assertFalse(report["decision"]["candidate_admitted"])
        self.assertTrue(
            report["decision"]["four_b_preservation_gates"][
                "point_delta_nonnegative"
            ]
        )
        self.assertTrue(
            report["decision"]["four_b_preservation_gates"][
                "every_stratum_non_regression"
            ]
        )
        self.assertFalse(
            report["decision"]["four_b_preservation_gates"][
                "bootstrap_ci_lower_nonnegative"
            ]
        )
        self.assertFalse(
            report["decision"]["nine_b_superiority_gates"][
                "mcnemar_below_alpha"
            ]
        )
        self.assertFalse(
            report["decision"]["nine_b_superiority_gates"][
                "every_stratum_non_regression"
            ]
        )
        self.assertTrue(report["diagnostics"]["metadata_finalize_required"])
        self.assertFalse(
            report["diagnostics"]["model_requests_rerun_for_finalize"]
        )
        serialized = json.dumps(report).lower()
        self.assertNotIn('"output"', serialized)


if __name__ == "__main__":
    unittest.main()
