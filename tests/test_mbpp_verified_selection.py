from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from nano_harness.mbpp_verified_selection import (
    CODE_BLOCK,
    MbppCase,
    extract_code,
    generate_case,
    load_cases,
    load_config,
    run_public_tests,
    select_best,
    select_shard,
    validate_code,
)
from scripts.preregister_mbpp_verified_selection_dev_v1 import build_receipt


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/campaign/mbpp_sanitized_verified_selection_dev_v1.json"
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


class MbppVerifiedSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(CONFIG)
        cls.case = MbppCase(
            case_id="mbpp-sanitized-validation-test",
            task_id=1,
            prompt="Write add(a, b).",
            test_imports=(),
            test_list=(
                "assert add(1, 2) == 3",
                "assert add(-1, 1) == 0",
            ),
        )

    def test_parser_requires_exact_python_block(self):
        self.assertEqual(
            extract_code("```python\ndef add(a, b): return a + b\n```"),
            "def add(a, b): return a + b",
        )
        self.assertIsNone(extract_code("def add(a, b): return a + b"))
        self.assertIsNone(
            extract_code(
                "text\n```python\ndef add(a, b): return a + b\n```"
            )
        )
        self.assertIsNotNone(CODE_BLOCK)

    def test_static_gate_rejects_dangerous_code(self):
        allowed = set(self.config["sandbox"]["allowed_imports"])
        self.assertEqual(
            validate_code("import os\ndef add(a,b): return a+b", allowed),
            "forbidden_import",
        )
        self.assertEqual(
            validate_code("def add(a,b): return eval('a+b')", allowed),
            "forbidden_builtin",
        )

    def test_sandbox_executes_public_tests_without_returning_sources(self):
        result = run_public_tests(
            "def add(a, b): return a + b",
            self.case,
            self.config["sandbox"],
        )
        self.assertEqual(result["passed"], 2)
        self.assertTrue(result["full_pass"])
        self.assertNotIn("tests", json.dumps(result))
        wrong = run_public_tests(
            "def add(a, b): return a - b",
            self.case,
            self.config["sandbox"],
        )
        self.assertEqual(wrong["passed"], 0)
        self.assertEqual(wrong["failure_classes"], {"assertion_failure": 2})

    def test_selector_uses_pass_count_then_length_then_index(self):
        selected = select_best(
            [
                {
                    "code": "def x(): return 1",
                    "replica_index": 1,
                    "test_result": {"passed": 1},
                },
                {
                    "code": "def x():return 1",
                    "replica_index": 2,
                    "test_result": {"passed": 1},
                },
                {
                    "code": "def x():return 0",
                    "replica_index": 0,
                    "test_result": {"passed": 0},
                },
            ]
        )
        self.assertEqual(selected["replica_index"], 2)

    def test_execution_shards_are_disjoint_and_complete(self):
        cases = load_cases(self.config, ROOT)
        shards = [
            select_shard(cases, num_shards=4, shard_id=shard_id)
            for shard_id in range(4)
        ]
        identities = [
            case.case_id for shard in shards for _, case in shard
        ]
        self.assertEqual(len(identities), 43)
        self.assertEqual(len(set(identities)), 43)
        for shard_id, shard in enumerate(shards):
            self.assertTrue(all(index % 4 == shard_id for index, _ in shard))

    def test_passing_direct_is_preserved_without_extra_four_b_calls(self):
        four = FakeOpenAI(["```python\ndef add(a, b): return a + b\n```"])
        nine = FakeOpenAI(["```python\ndef add(a, b): return a - b\n```"])
        result = generate_case(
            self.config,
            self.case,
            four_client=four,
            nine_client=nine,
            case_index=0,
        )
        self.assertTrue(
            result["four_b_direct"]["test_result"]["full_pass"]
        )
        self.assertEqual(result["receipt"]["selected_source"], "four_b_direct")
        self.assertEqual(len(four.chat.completions.calls), 1)
        self.assertEqual(len(nine.chat.completions.calls), 1)

    def test_failure_uses_replicas_then_aggregate_feedback_repair(self):
        four = FakeOpenAI(
            [
                "```python\ndef add(a, b): return a - b\n```",
                "```python\ndef add(a, b): return a * b\n```",
                "```python\ndef add(a, b): return a - b\n```",
                "```python\ndef add(a, b): return 0\n```",
                "```python\ndef add(a, b): return a + b\n```",
            ]
        )
        nine = FakeOpenAI(["```python\ndef add(a, b): return a - b\n```"])
        result = generate_case(
            self.config,
            self.case,
            four_client=four,
            nine_client=nine,
            case_index=0,
        )
        self.assertTrue(result["candidate"]["test_result"]["full_pass"])
        self.assertEqual(result["receipt"]["selected_source"], "repair")
        repair_prompt = four.chat.completions.calls[-1]["messages"][-1][
            "content"
        ]
        self.assertIn('"failure_classes"', repair_prompt)
        self.assertEqual(repair_prompt.count("assert add"), 2)
        self.assertNotIn("__sealed", repair_prompt.lower())

    def test_validation_identity_and_preregister_are_deterministic(self):
        cases = load_cases(self.config, ROOT)
        self.assertEqual(len(cases), 43)
        self.assertEqual(len({case.case_id for case in cases}), 43)
        first = build_receipt()
        second = build_receipt()
        self.assertEqual(first, second)
        self.assertEqual(first["surface"]["test_feasibility_probe_rows"], 1)
        self.assertFalse(
            first["surface"]["test_content_used_for_policy_design"]
        )
        self.assertFalse(first["surface"]["test_generation_allowed"])
        self.assertTrue(all(first["sandbox"]["probe"].values()))


if __name__ == "__main__":
    unittest.main()
