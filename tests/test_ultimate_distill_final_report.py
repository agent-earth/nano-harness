from __future__ import annotations

import json
import unittest

from scripts.render_ultimate_distill_final_report_v1 import (
    build_report,
    render_markdown,
)


class UltimateDistillFinalReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = build_report()

    def test_final_public_benchmark_claim(self):
        summary = self.report["executive_summary"]
        self.assertTrue(summary["goal_met"])
        self.assertEqual(
            summary["public_benchmark_wins"],
            ["mmlu", "gpqa_diamond", "mbpp"],
        )
        self.assertEqual(summary["public_benchmark_losses"], ["gsm8k"])
        self.assertEqual(
            summary["public_benchmarks_won_vs_matched_nine_b"],
            3,
        )
        self.assertEqual(
            self.report["public_benchmarks"]["decision"][
                "complete_public_benchmarks_won"
            ],
            3,
        )

    def test_training_is_not_credited_for_public_wins(self):
        summary = self.report["executive_summary"]
        boundary = self.report["claim_boundary"]
        ablation = self.report["ablation_contract"]
        self.assertEqual(
            summary["winning_layer"], "harness_routing_and_verification"
        )
        self.assertFalse(summary["training_quality_gain_established"])
        self.assertFalse(boundary["sft_quality_gain_established"])
        self.assertFalse(boundary["dpo_quality_gain_established"])
        self.assertFalse(boundary["rl_quality_gain_established"])
        self.assertFalse(boundary["opd_quality_gain_established"])
        self.assertFalse(boundary["sft_plus_rl_result_exists"])
        self.assertIn("not run", ablation["sft_plus_rl"])

    def test_twenty_seven_b_boundary(self):
        summary = self.report["executive_summary"]
        verified = self.report["twenty_seven_b"][
            "verified_tool_complete_local_suite"
        ]
        mbpp = self.report["twenty_seven_b"]["mbpp_complete"]
        self.assertFalse(summary["twenty_seven_b_public_benchmark_parity_met"])
        self.assertTrue(
            summary["twenty_seven_b_local_capability_suite_exceeded"]
        )
        self.assertEqual((verified["candidate_correct"], verified["baseline_correct"]), (256, 63))
        self.assertEqual((mbpp["candidate_correct"], mbpp["baseline_correct"]), (219, 226))
        self.assertFalse(
            self.report["claim_boundary"]["verified_tool_is_public_benchmark"]
        )

    def test_data_scale_is_separate_from_training_exposure(self):
        data = {item["id"]: item for item in self.report["data_pipeline"]}
        training = {
            item["id"]: item for item in self.report["training_ablation"]
        }
        skill_release = data["skill-sft-10k-10m-v2"]
        self.assertEqual(skill_release["train_rows"], 15888)
        self.assertEqual(skill_release["train_tokens"], 11425166)
        self.assertEqual(
            training["skill-release-long-sequence-sft-smoke-v1"][
                "train_rows_used"
            ],
            10,
        )
        self.assertEqual(
            training["skill-release-bounded-dose-sft-v2"]["train_rows_used"],
            80,
        )

    def test_mbpp_chain_preserves_nonsignificant_confirmation(self):
        stages = {
            item["stage"]: item for item in self.report["mbpp_evidence_chain"]
        }
        self.assertEqual(
            stages["fresh_confirmation"]["status"],
            "directional_but_not_significant",
        )
        self.assertEqual(
            stages["complete_sanitized_test"]["status"],
            "complete_benchmark_win",
        )
        self.assertFalse(stages["fresh_confirmation"]["benchmark_score"])
        self.assertTrue(stages["complete_sanitized_test"]["benchmark_score"])

    def test_agent_scans_are_not_scores(self):
        feasibility = self.report["agent_benchmark_feasibility"]
        self.assertEqual(feasibility["formal_scores_obtained"], 0)
        self.assertTrue(
            all(
                surface["passed"]
                for surface in feasibility["surfaces"].values()
            )
        )
        self.assertIn("not model-quality scores", feasibility["interpretation"])

    def test_public_report_excludes_raw_outputs_and_plain_language_is_present(self):
        serialized = json.dumps(self.report).lower()
        self.assertNotIn('"output"', serialized)
        self.assertNotIn('"test_list"', serialized)
        markdown = render_markdown(self.report)
        self.assertIn("同条件直接回答（matched direct）", markdown)
        self.assertIn("Verified", markdown)
        self.assertIn("生成了多少，实际训练了多少", markdown)
        self.assertIn("SFT+RL", markdown)


if __name__ == "__main__":
    unittest.main()
