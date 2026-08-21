from __future__ import annotations

import json
import unittest

from scripts.render_orca_conditional_majority_v4 import build_report


class RenderConditionalMajorityTests(unittest.TestCase):
    def test_public_report_admits_only_complete_benchmark_preregistration(self):
        report = build_report()
        self.assertTrue(report["decision"]["candidate_admitted"])
        self.assertTrue(report["decision"]["complete_benchmark_allowed"])
        self.assertTrue(
            all(
                report["decision"]["four_b_preservation_gates"].values()
            )
        )
        self.assertTrue(
            all(
                report["decision"]["nine_b_superiority_gates"].values()
            )
        )
        self.assertFalse(report["decision"]["rerun_or_tuning_allowed"])
        self.assertFalse(
            report["boundary"]["complete_benchmark_score_claimed"]
        )

    def test_report_recomputes_expected_metrics_and_diagnostics(self):
        report = build_report()
        four = report["comparisons"]["versus_four_b"]
        nine = report["comparisons"]["versus_nine_b"]
        self.assertEqual(
            (four["candidate_correct"], four["baseline_correct"]),
            (58, 54),
        )
        self.assertEqual(
            four["paired_counts"],
            {
                "candidate_only": 4,
                "baseline_only": 0,
                "both_correct": 54,
                "both_wrong": 38,
            },
        )
        self.assertEqual(
            (nine["candidate_correct"], nine["baseline_correct"]),
            (58, 46),
        )
        self.assertEqual(
            nine["paired_counts"],
            {
                "candidate_only": 14,
                "baseline_only": 2,
                "both_correct": 44,
                "both_wrong": 36,
            },
        )
        diagnostics = report["diagnostics"]
        self.assertEqual(diagnostics["overrides"], 6)
        self.assertEqual(diagnostics["fallbacks"], 42)
        self.assertEqual(diagnostics["direct_strict_parseable"], 55)
        self.assertEqual(diagnostics["direct_strict_parse_failure"], 41)
        self.assertEqual(
            diagnostics["minimum_vote_route_counts"],
            {"3": 41, "5": 55},
        )
        self.assertTrue(diagnostics["receipt_cases_match_raw"])
        self.assertFalse(diagnostics["model_requests_rerun_for_render"])

    def test_public_report_contains_no_raw_outputs(self):
        serialized = json.dumps(build_report()).lower()
        self.assertNotIn('"output"', serialized)
        self.assertNotIn('"expected"', serialized)
        self.assertNotIn('"case_id"', serialized)


if __name__ == "__main__":
    unittest.main()
