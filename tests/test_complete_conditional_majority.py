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
    holm_bonferroni,
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


if __name__ == "__main__":
    unittest.main()
