from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from nano_harness.baseline import BaselineCase, sha256_file
from nano_harness.complete_conditional_majority import (
    CONFIG_SHA256,
    generate_gsm8k_candidate,
    load_config,
)
from scripts.preregister_complete_conditional_majority_v1 import (
    build_receipt,
)
from scripts.render_complete_conditional_majority_v1 import (
    _correct_rows,
    build_report as build_complete_report,
    holm_bonferroni,
)
from scripts.run_complete_conditional_majority_shard_v1 import (
    EXECUTION_V1,
    EXECUTION_V2,
    load_execution,
    select_shard,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/campaign/qwen35_complete_conditional_majority_v1.json"
)


class FakeCompletions:
    def __init__(self, outputs: list[str]):
        self.outputs = outputs
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        output = self.outputs.pop(0)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=output),
                    finish_reason="stop",
                )
            ],
            usage=None,
        )


class FakeOpenAI:
    def __init__(self, outputs: list[str]):
        self.chat = SimpleNamespace(completions=FakeCompletions(outputs))


class CompleteConditionalMajorityTests(unittest.TestCase):
    def make_case(self) -> BaselineCase:
        return BaselineCase(
            case_id="gsm8k-test",
            benchmark="gsm8k",
            prompt="Problem: one plus one",
            draft_prompt="Problem: one plus one",
            expected="__SEALED_DURING_GENERATION__",
            scorer="numeric_exact",
            source_index=0,
            source_chars=12,
            system_prompt="Solve and end with FINAL: <number>.",
            max_tokens=600,
            metadata={},
        )

    def test_config_freezes_policy_and_closed_boundary(self):
        config = load_config(CONFIG)
        self.assertEqual(
            config["routes"]["gsm8k"]["strategy"],
            "conditional_majority_v4",
        )
        self.assertEqual(
            config["routes"]["mmlu"]["strategy"],
            "preserve_frozen_four_b_direct",
        )
        self.assertEqual(
            config["routes"]["gpqa_diamond"]["strategy"],
            "reuse_frozen_v5_conservative_choice_consensus",
        )
        self.assertEqual(
            config["statistics"]["bonferroni_alpha"],
            0.025,
        )
        self.assertFalse(
            config["execution_boundary"]["benchmark_generation_started"]
        )
        self.assertFalse(
            config["execution_boundary"][
                "benchmark_output_content_loaded_by_preregister"
            ]
        )

    def test_config_rejects_any_mutation(self):
        raw = json.loads(CONFIG.read_text(encoding="utf-8"))
        altered = copy.deepcopy(raw)
        altered["routes"]["gsm8k"][
            "direct_strict_parseable_minimum_votes"
        ] = 4
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(altered), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "config SHA"):
                load_config(path)
        self.assertEqual(sha256_file(CONFIG), CONFIG_SHA256)

    def test_strict_failure_uses_three_votes(self):
        client = FakeOpenAI(
            [
                "FINAL: 12",
                "FINAL: 12",
                "FINAL: 12",
                "FINAL: 9",
                "FINAL: 7",
            ]
        )
        candidate, receipt = generate_gsm8k_candidate(
            self.make_case(),
            {"output": "Reasoning ended at 7", "prediction": None},
            load_config(CONFIG),
            client=client,
            case_index=3,
        )
        self.assertEqual(candidate["prediction"], "12")
        self.assertEqual(candidate["output"], "FINAL: 12")
        self.assertTrue(receipt["override"])
        self.assertEqual(receipt["minimum_votes"], 3)
        self.assertEqual(receipt["model_calls"], 5)

    def test_strict_direct_requires_unanimity(self):
        client = FakeOpenAI(
            [
                "FINAL: 12",
                "FINAL: 12",
                "FINAL: 12",
                "FINAL: 12",
                "FINAL: 7",
            ]
        )
        candidate, receipt = generate_gsm8k_candidate(
            self.make_case(),
            {"output": "FINAL: 7", "prediction": "7"},
            load_config(CONFIG),
            client=client,
            case_index=3,
        )
        self.assertEqual(candidate["prediction"], "7")
        self.assertEqual(candidate["output"], "FINAL: 7")
        self.assertTrue(receipt["fallback"])
        self.assertEqual(receipt["minimum_votes"], 5)

    def test_preregister_is_deterministic_and_generation_closed(self):
        first = build_receipt()
        second = build_receipt()
        self.assertEqual(first, second)
        self.assertEqual(first["surface"]["complete_cases"], 15_559)
        self.assertEqual(
            first["surface"]["new_model_generation"],
            {"gsm8k": 1_319, "mmlu": 0, "gpqa_diamond": 0},
        )
        self.assertTrue(
            first["surface"][
                "case_sets_match_frozen_direct_and_prior_candidate"
            ]
        )
        self.assertFalse(
            first["execution_boundary"]["benchmark_generation_started"]
        )
        self.assertFalse(
            first["decision_rule"]["rerun_or_tuning_allowed"]
        )

    def test_complete_gsm8k_scoring_recovers_only_target_blind_final(self):
        rows = [
            {
                "case_id": "gsm8k-a",
                "output": "The result is 12",
                "prediction": None,
                "expected": "12",
                "score": 0,
            }
        ]
        recovered = _correct_rows(rows, recovered_numeric=True)
        strict = _correct_rows(rows, recovered_numeric=False)
        self.assertTrue(recovered[0]["correct"])
        self.assertFalse(strict[0]["correct"])
        self.assertNotIn("expected", recovered[0])
        self.assertNotIn("output", recovered[0])

    def test_holm_bonferroni_requires_all_three_benchmarks(self):
        accepted = holm_bonferroni(
            {"gsm8k": 0.01, "mmlu": 1e-8, "gpqa_diamond": 0.04},
            alpha=0.05,
        )
        self.assertTrue(accepted["all_rejected"])
        rejected = holm_bonferroni(
            {"gsm8k": 0.03, "mmlu": 1e-8, "gpqa_diamond": 0.04},
            alpha=0.05,
        )
        self.assertFalse(rejected["all_rejected"])

    def test_complete_report_rejects_gsm8k_regression(self):
        report = build_complete_report()
        self.assertFalse(report["decision"]["complete_candidate_admitted"])
        self.assertTrue(
            report["decision"]["all_benchmarks_non_regressing_vs_four_b"]
        )
        self.assertEqual(
            report["decision"]["complete_benchmarks_significantly_won"],
            2,
        )
        gsm8k = report["comparisons"]["versus_nine_b"]["gsm8k"]
        self.assertEqual(
            (gsm8k["candidate_correct"], gsm8k["baseline_correct"]),
            (1220, 1243),
        )
        self.assertLess(gsm8k["paired_bootstrap_95_ci"][1], 0)
        holm = {
            row["benchmark"]: row for row in report["holm_bonferroni"][
                "ordered_tests"
            ]
        }
        self.assertTrue(holm["gsm8k"]["rejected"])
        self.assertFalse(holm["gsm8k"]["positive_direction"])
        self.assertFalse(holm["gsm8k"]["superiority"])

    def test_execution_shards_preserve_global_indices_and_are_disjoint(self):
        cases = [
            self.make_case().__class__(
                **{
                    **self.make_case().__dict__,
                    "case_id": f"gsm8k-{index:03d}",
                }
            )
            for index in range(8)
        ]
        prefix_ids = {"gsm8k-000", "gsm8k-001"}
        execution = load_execution(EXECUTION_V1)
        even = select_shard(
            cases,
            prefix_ids=prefix_ids,
            num_shards=execution["sharding"]["num_shards"],
            shard_id=0,
        )
        odd = select_shard(
            cases,
            prefix_ids=prefix_ids,
            num_shards=execution["sharding"]["num_shards"],
            shard_id=1,
        )
        self.assertEqual(
            [(index, case.case_id) for index, case in even],
            [(2, "gsm8k-002"), (4, "gsm8k-004"), (6, "gsm8k-006")],
        )
        self.assertEqual(
            [(index, case.case_id) for index, case in odd],
            [(3, "gsm8k-003"), (5, "gsm8k-005"), (7, "gsm8k-007")],
        )
        self.assertFalse(
            {case.case_id for _, case in even}
            & {case.case_id for _, case in odd}
        )
        accelerated = load_execution(EXECUTION_V2)
        shards = [
            select_shard(
                cases,
                prefix_ids=prefix_ids,
                num_shards=accelerated["sharding"]["num_shards"],
                shard_id=shard_id,
            )
            for shard_id in range(8)
        ]
        assigned = [
            case.case_id for shard in shards for _, case in shard
        ]
        self.assertEqual(set(assigned), {f"gsm8k-{i:03d}" for i in range(2, 8)})
        self.assertEqual(len(assigned), len(set(assigned)))


if __name__ == "__main__":
    unittest.main()
