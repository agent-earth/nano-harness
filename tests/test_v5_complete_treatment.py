from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from nano_harness.baseline import sha256_file
from nano_harness.baseline import BaselineCase
from nano_harness.client import ScriptedClient
from nano_harness.types import ModelReply
from nano_harness.v5_complete_treatment import (
    CONFIG_SHA256,
    generate_candidate,
    jsonl_ids,
    load_config,
)
from scripts.preregister_v5_complete_treatment import build_receipt


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/campaign/qwen35_v5_complete_treatment_v1.json"


class V5CompleteTreatmentTests(unittest.TestCase):
    def make_case(
        self,
        *,
        benchmark: str,
        prompt: str,
        scorer: str,
    ) -> BaselineCase:
        return BaselineCase(
            case_id=f"{benchmark}-test",
            benchmark=benchmark,
            prompt=prompt,
            draft_prompt=prompt,
            expected="__SEALED_DURING_GENERATION__",
            scorer=scorer,
            source_index=0,
            source_chars=len(prompt),
            system_prompt="system",
            max_tokens=32,
            metadata={},
        )

    def test_config_freezes_three_task_routes_and_closed_boundary(self):
        config = load_config(CONFIG)
        self.assertEqual(
            config["routes"]["mmlu"],
            {"strategy": "preserve_frozen_4b_direct"},
        )
        self.assertEqual(
            config["routes"]["gsm8k"]["plan_replicas"],
            3,
        )
        self.assertEqual(
            config["routes"]["gpqa_diamond"]["override_rule"],
            "two_independent_reviews_and_confirmation_agree_on_same_non_direct_choice",
        )
        self.assertFalse(
            config["execution_boundary"]["benchmark_generation_started"]
        )
        self.assertFalse(
            config["execution_boundary"][
                "benchmark_outputs_loaded_by_preregister"
            ]
        )

    def test_preregister_is_deterministic_and_matches_complete_surface(self):
        first = build_receipt()
        second = build_receipt()
        self.assertEqual(first, second)
        self.assertEqual(first["surface"]["cases"], 15_559)
        self.assertEqual(
            first["surface"]["by_benchmark"],
            {
                "gpqa_diamond": 198,
                "gsm8k": 1_319,
                "mmlu": 14_042,
            },
        )
        self.assertTrue(first["surface"]["case_set_matches_both_direct_arms"])
        self.assertFalse(
            first["execution_boundary"]["benchmark_generation_started"]
        )
        self.assertFalse(
            first["execution_boundary"][
                "benchmark_outputs_loaded_by_preregister"
            ]
        )
        self.assertFalse(
            first["surface"]["prompts_or_outputs_published"]
        )

    def test_raw_identity_reader_extracts_only_case_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_text(
                "\n".join(
                    [
                        '{"case_id":"a","output":"secret-a","prediction":1}',
                        '{"case_id":"b","output":"secret-b","prediction":2}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual(jsonl_ids(path), ["a", "b"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.jsonl"
            path.write_text(
                '{"case_id":"a"}\n{"case_id":"a"}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicated"):
                jsonl_ids(path)

    def test_config_rejects_any_mutation(self):
        raw = json.loads(CONFIG.read_text(encoding="utf-8"))
        altered = copy.deepcopy(raw)
        altered["routes"]["gsm8k"]["plan_replicas"] = 2
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(altered), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "config SHA"):
                load_config(path)
        self.assertEqual(sha256_file(CONFIG), CONFIG_SHA256)

    def test_candidate_generation_preserves_mmlu_without_calls(self):
        config = load_config(CONFIG)
        case = self.make_case(
            benchmark="mmlu",
            prompt="Question and choices",
            scorer="choice_exact",
        )
        empty = ScriptedClient([])
        candidate, receipt = generate_candidate(
            case,
            {
                "case_id": case.case_id,
                "output": "FINAL: B",
                "prediction": "B",
                "usage": {},
            },
            config,
            calculator_client=empty,
            choice_review_client=empty,
            choice_confirmation_client=empty,
        )
        self.assertEqual(candidate["output"], "FINAL: B")
        self.assertEqual(receipt["model_calls"], 0)
        self.assertFalse(receipt["override"])
        self.assertEqual(empty.calls, [])

    def test_gsm8k_requires_two_grounded_results(self):
        config = load_config(CONFIG)
        case = self.make_case(
            benchmark="gsm8k",
            prompt="Problem: There are 3 groups with 4 items each.",
            scorer="numeric_exact",
        )
        calculator = ScriptedClient(
            [
                ModelReply(content="CALC: 3 * 4"),
                ModelReply(content="CALC: 3 * 4"),
                ModelReply(content="CALC: 3 + 4"),
            ]
        )
        candidate, receipt = generate_candidate(
            case,
            {
                "case_id": case.case_id,
                "output": "FINAL: 7",
                "prediction": "7",
                "usage": {},
            },
            config,
            calculator_client=calculator,
            choice_review_client=ScriptedClient([]),
            choice_confirmation_client=ScriptedClient([]),
        )
        self.assertEqual(candidate["output"], "FINAL: 12")
        self.assertTrue(receipt["override"])
        self.assertEqual(receipt["consensus_result"], 12)
        self.assertEqual(receipt["api_errors"], 0)

    def test_gpqa_requires_two_reviews_and_confirmation(self):
        config = load_config(CONFIG)
        case = self.make_case(
            benchmark="gpqa_diamond",
            prompt="Question\nA. one\nB. two\nC. three\nD. four",
            scorer="choice_exact",
        )
        candidate, receipt = generate_candidate(
            case,
            {
                "case_id": case.case_id,
                "output": "FINAL: A",
                "prediction": "A",
                "usage": {},
            },
            config,
            calculator_client=ScriptedClient([]),
            choice_review_client=ScriptedClient(
                [
                    ModelReply(content="FINAL: C"),
                    ModelReply(content="FINAL: C"),
                ]
            ),
            choice_confirmation_client=ScriptedClient(
                [ModelReply(content="FINAL: C")]
            ),
        )
        self.assertEqual(candidate["output"], "FINAL: C")
        self.assertTrue(receipt["override"])
        self.assertEqual(receipt["confirmation"], "C")


if __name__ == "__main__":
    unittest.main()
