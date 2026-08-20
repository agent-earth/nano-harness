from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from nano_harness.baseline import sha256_file
from nano_harness.v5_complete_treatment import (
    CONFIG_SHA256,
    jsonl_ids,
    load_config,
)
from scripts.preregister_v5_complete_treatment import build_receipt


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/campaign/qwen35_v5_complete_treatment_v1.json"


class V5CompleteTreatmentTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
