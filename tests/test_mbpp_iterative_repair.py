from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace

from nano_harness.mbpp_iterative_repair import (
    extract_code,
    generate_case,
    load_config,
    load_few_shots,
    load_train_cases,
    select_shard,
)
from nano_harness.mbpp_verified_selection import MbppCase
from scripts.preregister_mbpp_iterative_repair_train_v2 import build_receipt


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/campaign/mbpp_sanitized_iterative_repair_train_v2.json"
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


class MbppIterativeRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(CONFIG)
        cls.case = MbppCase(
            case_id="mbpp-sanitized-train-test",
            task_id=1,
            prompt="Write add(a, b).",
            test_imports=(),
            test_list=(
                "assert add(1, 2) == 3",
                "assert add(-1, 1) == 0",
            ),
        )
        cls.few_shots = []

    def test_parser_accepts_fence_begin_done_and_plain_python(self):
        expected = "def add(a, b): return a + b"
        self.assertEqual(
            extract_code(f"Here is the code:\n```python\n{expected}\n```"),
            expected,
        )
        self.assertEqual(extract_code(f"[BEGIN]\n{expected}\n[DONE]"), expected)
        self.assertEqual(extract_code(expected), expected)
        self.assertIsNone(extract_code("the answer is obvious"))

    def test_five_candidates_and_iterative_repair_find_passing_code(self):
        wrong = "```python\ndef add(a, b): return a - b\n```"
        four = FakeOpenAI(
            [
                wrong,
                wrong,
                wrong,
                wrong,
                wrong,
                wrong,
                wrong,
                "```python\ndef add(a, b): return a + b\n```",
            ]
        )
        nine = FakeOpenAI([wrong])
        result = generate_case(
            self.config,
            self.case,
            self.few_shots,
            four_client=four,
            nine_client=nine,
            case_index=0,
        )
        self.assertTrue(result["candidate"]["test_result"]["full_pass"])
        self.assertEqual(result["receipt"]["replicas_generated"], 5)
        self.assertEqual(result["receipt"]["repair_rounds_generated"], 2)
        self.assertEqual(result["receipt"]["selected_source"], "repair_1")
        repair_prompt = four.chat.completions.calls[-1]["messages"][-1][
            "content"
        ]
        self.assertIn('"failed_public_test_indices"', repair_prompt)
        self.assertEqual(repair_prompt.count("assert add"), 2)

    def test_real_train_and_few_shot_identities_are_disjoint(self):
        cases = load_train_cases(self.config, ROOT)
        few_shots = load_few_shots(self.config, ROOT)
        self.assertEqual(len(cases), 120)
        self.assertEqual([row.task_id for row in few_shots], [2, 3, 4])
        self.assertFalse(
            {case.task_id for case in cases}
            & {example.task_id for example in few_shots}
        )
        shards = [
            select_shard(cases, num_shards=4, shard_id=shard_id)
            for shard_id in range(4)
        ]
        ids = [case.case_id for shard in shards for _, case in shard]
        self.assertEqual(len(ids), 120)
        self.assertEqual(len(set(ids)), 120)

    def test_preregister_is_deterministic_and_validation_closed(self):
        first = build_receipt()
        second = build_receipt()
        self.assertEqual(first, second)
        self.assertEqual(first["surface"]["cases"], 120)
        self.assertFalse(first["surface"]["validation_v1_rerun"])
        self.assertFalse(first["surface"]["validation_rows_loaded_by_v2"])
        self.assertFalse(first["surface"]["test_generation_started"])
        self.assertFalse(first["decision_rule"]["rerun_or_tuning_allowed"])


if __name__ == "__main__":
    unittest.main()
