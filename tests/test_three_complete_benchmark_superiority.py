from __future__ import annotations

import json
import unittest

from scripts.render_three_complete_benchmark_superiority_v1 import build_report


class ThreeCompleteBenchmarkSuperiorityTests(unittest.TestCase):
    def test_three_complete_benchmarks_pass(self):
        report = build_report()
        self.assertTrue(
            report["decision"]["three_complete_benchmark_superiority"]
        )
        self.assertEqual(
            report["decision"]["complete_public_benchmarks_won"],
            3,
        )
        self.assertFalse(report["decision"]["all_selected_benchmarks_won"])
        self.assertTrue(report["benchmarks"]["mmlu"]["won"])
        self.assertTrue(report["benchmarks"]["gpqa_diamond"]["won"])
        self.assertTrue(report["benchmarks"]["mbpp"]["won"])
        self.assertFalse(report["benchmarks"]["gsm8k"]["won"])
        self.assertEqual(report["holm_bonferroni"]["family_size"], 4)
        self.assertTrue(report["holm_bonferroni"]["all_rejected"])

    def test_negative_and_27b_evidence_remain_explicit(self):
        report = build_report()
        self.assertFalse(
            report["preserved_negative_evidence"]["gsm8k"]["counted_as_win"]
        )
        self.assertFalse(
            report["preserved_negative_evidence"]["mbpp_27b"][
                "parity_admitted"
            ]
        )
        tool = report["twenty_seven_b"]["verified_tool_complete_suite"]
        self.assertTrue(tool["parity_admitted"])
        self.assertTrue(tool["four_b_harness_exceeds_27b"])
        self.assertFalse(
            report["claim_boundary"]["verified_tool_counted_as_public_benchmark"]
        )

    def test_public_report_excludes_raw_outputs(self):
        report = build_report()
        serialized = json.dumps(report).lower()
        self.assertNotIn('"output"', serialized)
        self.assertNotIn('"code"', serialized)
        self.assertNotIn('"test_list"', serialized)


if __name__ == "__main__":
    unittest.main()
