import json
import tempfile
import unittest
from pathlib import Path

from nano_harness.adapters.swebench import extract_patch
from nano_harness.client import ScriptedClient
from nano_harness.coding_tools import CodingToolExecutor
from nano_harness.config import HarnessConfig, load_run_config
from nano_harness.harness import AgentHarness
from nano_harness.runner import completed_task_ids, merge_paths, run_config
from nano_harness.state import StateLedger, compact_messages
from nano_harness.types import ModelReply, Task


class FakeToolExecutor:
    def __init__(self):
        self.calls = []

    def execute(self, name, arguments):
        self.calls.append((name, arguments))
        return json.dumps({"status": "active"})


class CoreTests(unittest.TestCase):
    def test_model_alias_and_relative_output_resolution(self):
        config = load_run_config("configs/benchmarks/synthetic_base.yaml")
        self.assertEqual(config.model.name, "nvidia/nemotron-nano-9b-v2:free")
        self.assertTrue(config.output_dir.is_absolute())

    def test_extract_swebench_patch(self):
        patch = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x\n+y"
        self.assertEqual(extract_patch(f"<<PATCH>>\n{patch}\n<<END_PATCH>>"), patch)

    def test_optimized_harness_executes_one_tool_then_finishes(self):
        client = ScriptedClient(
            [
                ModelReply(
                    content="",
                    tool_calls=[
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "lookup_account",
                                "arguments": '{"account_id":"A-17"}',
                            },
                        }
                    ],
                ),
                ModelReply(content="The verified status is active."),
            ]
        )
        task = Task(
            task_id="t1",
            benchmark="taubench",
            messages=[{"role": "user", "content": "Verify account A-17."}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "lookup_account",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        )
        executor = FakeToolExecutor()
        result = AgentHarness(
            client,
            "test-model",
            HarnessConfig(strategy="optimized", max_steps=3),
        ).run(task, executor)
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.tool_calls, 1)
        self.assertEqual(executor.calls[0][1], {"account_id": "A-17"})
        second_request = client.calls[1]["messages"]
        self.assertTrue(
            any(
                "<state_ledger>" in str(message.get("content", ""))
                for message in second_request
            )
        )
        self.assertEqual(len(client.calls), 2)

    def test_audit_pass_can_repair_output_type(self):
        client = ScriptedClient(
            [
                ModelReply(content='{"result":12,"evidence":{"value":"7+5"}}'),
                ModelReply(content='{"result":12,"evidence":"BETA = 7 + 5"}'),
            ]
        )
        task = Task(
            task_id="audit",
            benchmark="clbench",
            messages=[
                {
                    "role": "user",
                    "content": "Return result and a string evidence field.",
                }
            ],
            metadata={
                "constraints": ["The evidence field must be a string."],
                "audit_policy": "contract_failure",
                "output_contract": {
                    "format": "json",
                    "type": "object",
                    "required": ["result", "evidence"],
                    "property_types": {"result": "number", "evidence": "string"},
                },
            },
        )
        result = AgentHarness(
            client,
            "test-model",
            HarnessConfig(strategy="optimized", audit_passes=1),
        ).run(task)
        self.assertEqual(
            json.loads(result.output),
            {"result": 12, "evidence": "BETA = 7 + 5"},
        )
        self.assertEqual(result.trajectory[-1]["kind"], "audit")

    def test_honest_tool_limitation_is_not_false_early_stop(self):
        client = ScriptedClient(
            [ModelReply(content="I cannot confirm verification; observed status is active.")]
        )
        task = Task(
            task_id="honest",
            benchmark="taubench",
            messages=[{"role": "user", "content": "Report observed status."}],
            metadata={"audit_policy": "never"},
        )
        result = AgentHarness(
            client,
            "test-model",
            HarnessConfig(strategy="optimized"),
        ).run(task)
        self.assertEqual(result.status, "completed")
        self.assertIsNone(result.failure_type)

    def test_swebench_rejects_claimed_patch_without_mutation_and_validation(self):
        client = ScriptedClient(
            [
                ModelReply(content="<<PATCH>>fake patch<<END_PATCH>>"),
                ModelReply(
                    content="",
                    tool_calls=[
                        {
                            "id": "apply",
                            "type": "function",
                            "function": {
                                "name": "apply_patch",
                                "arguments": '{"patch":"diff"}',
                            },
                        }
                    ],
                ),
                ModelReply(
                    content="",
                    tool_calls=[
                        {
                            "id": "test",
                            "type": "function",
                            "function": {
                                "name": "run_command",
                                "arguments": '{"argv":["pytest"]}',
                            },
                        }
                    ],
                ),
                ModelReply(content="Validated patch complete."),
            ]
        )

        class GateExecutor:
            def execute(self, name, arguments):
                if name == "apply_patch":
                    return "one file changed"
                return '{"exit_code":0,"output":"1 passed"}'

        task = Task(
            task_id="swe",
            benchmark="swebench",
            messages=[{"role": "user", "content": "Fix the bug."}],
            tools=[{"type": "function", "function": {"name": "apply_patch"}}],
            metadata={"audit_policy": "never"},
        )
        result = AgentHarness(
            client,
            "test-model",
            HarnessConfig(strategy="optimized", max_steps=6),
        ).run(task, GateExecutor())
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.tool_calls, 2)
        self.assertEqual(result.trajectory[1]["kind"], "completion_gate")
        self.assertIn("No patch was applied", result.trajectory[1]["errors"][0])

    def test_context_compaction_preserves_ledger_and_latest_messages(self):
        messages = [{"role": "system", "content": "system"}] + [
            {"role": "user", "content": str(index) * 500} for index in range(10)
        ]
        compacted = compact_messages(
            messages,
            StateLedger(objective="keep objective", facts=["verified fact"]),
            max_chars=3000,
            reserve_chars=1000,
            scratchpad_chars=1000,
        )
        self.assertIn("<state_ledger>", compacted[1]["content"])
        self.assertEqual(compacted[-1]["content"], "9" * 500)
        self.assertLess(len(compacted), len(messages) + 1)

    def test_resume_skips_completed_tasks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "run.yaml"
            config_path.write_text(
                """
model:
  name: nano-9b
harness:
  strategy: base
benchmark:
  name: synthetic
  source: builtin
  limit: 1
output_dir: results
run_id: resume
""".strip(),
                encoding="utf-8",
            )
            first_client = ScriptedClient(
                [ModelReply(content='{"result":12,"evidence":"BETA = ALPHA + 5 = 7 + 5"}')]
            )
            first = run_config(load_run_config(config_path), first_client)
            self.assertEqual(first["written_this_invocation"], 1)
            second = run_config(load_run_config(config_path), ScriptedClient([]))
            self.assertEqual(second["written_this_invocation"], 0)
            output = root / "results/resume/shard-000-of-001.jsonl"
            self.assertEqual(completed_task_ids(output), {"constraint-audit"})

    def test_merge_deduplicates_by_task_id(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a = root / "a.jsonl"
            b = root / "b.jsonl"
            a.write_text('{"task_id":"1","score":0}\n', encoding="utf-8")
            b.write_text(
                '{"task_id":"1","score":1}\n{"task_id":"2","score":1}\n',
                encoding="utf-8",
            )
            output = root / "merged.jsonl"
            summary = merge_paths([a, b], output)
            self.assertEqual(summary["total"], 2)
            self.assertEqual(summary["score"], 1.0)

    def test_coding_tools_enforce_repository_boundary_and_command_allowlist(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "hello.txt").write_text("hello\nworld\n", encoding="utf-8")
            executor = CodingToolExecutor(root)
            self.assertIn("1: hello", executor.execute("read_file", {"path": "hello.txt"}))
            with self.assertRaises(ValueError):
                executor.execute("read_file", {"path": "../outside.txt"})
            with self.assertRaises(ValueError):
                executor.execute("run_command", {"argv": ["bash", "-c", "echo bad"]})


if __name__ == "__main__":
    unittest.main()
