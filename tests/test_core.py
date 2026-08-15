import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from nano_harness.baseline import (
    DatasetSpec,
    SuiteManifest,
    _run_draft_critique_verify_case,
    _run_draft_verify_case,
    _run_dual_solve_verify_case,
    build_case,
    compare_baselines,
    extract_prediction,
    load_cases,
    load_manifest,
    score_output,
    summarize_baseline,
)
from nano_harness.adapters.clbench import CLBenchAdapter
from nano_harness.adapters.swebench import extract_patch
from nano_harness.client import OpenRouterClient, ProviderQuotaError, ScriptedClient
from nano_harness.coding_tools import CodingToolExecutor
from nano_harness.config import (
    BenchmarkConfig,
    HarnessConfig,
    ModelConfig,
    load_run_config,
)
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
    def test_single_task_suite_requires_explicit_min_task_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "suite.yaml"
            base = {
                "schema_version": "nano_harness_baseline_suite_v1",
                "suite_id": "single-task",
                "selection_seed": "fixed",
                "system_prompt": "answer",
                "max_tokens": 32,
                "temperature": 0.0,
                "chat_template_kwargs": {"enable_thinking": False},
                "datasets": [
                    {
                        "name": "gsm8k",
                        "path": "gsm8k.parquet",
                        "sha256": "0" * 64,
                        "scorer": "numeric_exact",
                        "limit": 1,
                    }
                ],
            }
            import yaml

            path.write_text(yaml.safe_dump(base), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "at least 3"):
                load_manifest(path)
            base["min_task_groups"] = 1
            path.write_text(yaml.safe_dump(base), encoding="utf-8")
            self.assertEqual(load_manifest(path).min_task_groups, 1)

    def test_baseline_numeric_and_choice_scorers_require_final_line(self):
        self.assertEqual(extract_prediction("work\nFINAL: $1,234.50", "numeric_exact"), "1234.5")
        self.assertEqual(score_output("FINAL: -0", "0", "numeric_exact"), (1.0, "0"))
        self.assertEqual(score_output("reason\nFINAL: c", "C", "choice_exact"), (1.0, "C"))
        self.assertEqual(score_output("The answer is C.", "C", "choice_exact"), (0.0, None))

    def test_baseline_case_ids_are_content_stable(self):
        record = {
            "question": "What is 7 plus 5?",
            "answer": "Compute 7 + 5 = 12.\n#### 12",
        }
        first = build_case("gsm8k", "numeric_exact", 3, record)
        second = build_case("gsm8k", "numeric_exact", 99, record)
        self.assertEqual(first.case_id, second.case_id)
        self.assertEqual(first.expected, "12")
        self.assertNotEqual(first.source_index, second.source_index)

    def test_baseline_answer_only_keeps_case_identity_and_changes_contract(self):
        record = {
            "question": "What is 7 plus 5?",
            "answer": "Compute 7 + 5 = 12.\n#### 12",
        }
        reasoning = build_case("gsm8k", "numeric_exact", 3, record)
        answer_only = build_case(
            "gsm8k",
            "numeric_exact",
            3,
            record,
            answer_only=True,
        )
        self.assertEqual(reasoning.case_id, answer_only.case_id)
        self.assertIn("Show concise reasoning", reasoning.prompt)
        self.assertIn("Do not show reasoning", answer_only.prompt)
        self.assertNotEqual(reasoning.prompt, answer_only.prompt)
        self.assertNotEqual(
            reasoning.system_prompt,
            build_case(
                "gsm8k",
                "numeric_exact",
                3,
                record,
                system_prompt="answer only",
            ).system_prompt,
        )

    def test_draft_verify_uses_candidate_then_strict_verifier(self):
        case = build_case(
            "gsm8k",
            "numeric_exact",
            0,
            {
                "question": "What is 7 plus 5?",
                "answer": "Compute 7 + 5 = 12.\n#### 12",
            },
            system_prompt="direct",
            max_tokens=600,
        )
        draft_client = ScriptedClient(
            [
                ModelReply(
                    content="Candidate reasoning: 7 + 5 = 12.",
                    usage={
                        "prompt_tokens": 20,
                        "completion_tokens": 10,
                        "total_tokens": 30,
                    },
                    raw={"choices": [{"finish_reason": "stop"}]},
                )
            ]
        )
        verifier_client = ScriptedClient(
            [
                ModelReply(
                    content="FINAL: 12",
                    usage={
                        "prompt_tokens": 40,
                        "completion_tokens": 4,
                        "total_tokens": 44,
                    },
                    raw={"choices": [{"finish_reason": "stop"}]},
                )
            ]
        )
        manifest = SuiteManifest(
            schema_version="nano_harness_baseline_suite_v1",
            suite_id="draft-test",
            selection_seed="fixed",
            system_prompt="direct",
            max_tokens=600,
            temperature=0.0,
            chat_template_kwargs={"enable_thinking": False},
            strategy="draft_verify",
            draft_max_tokens=384,
            critique_max_tokens=192,
            second_solve_max_tokens=384,
            verifier_max_tokens=32,
            min_task_groups=1,
            datasets=(),
        )
        reply, stages = _run_draft_verify_case(
            case,
            manifest,
            ModelConfig(name="test"),
            {384: draft_client, 32: verifier_client},
        )
        self.assertEqual(reply.content, "FINAL: 12")
        self.assertEqual(
            reply.usage,
            {"prompt_tokens": 60, "completion_tokens": 14, "total_tokens": 74},
        )
        self.assertEqual(stages["draft"]["max_tokens"], 384)
        self.assertEqual(stages["verifier"]["max_tokens"], 32)
        verifier_prompt = verifier_client.calls[0]["messages"][-1]["content"]
        self.assertIn("Candidate reasoning: 7 + 5 = 12.", verifier_prompt)
        self.assertIn(case.prompt, verifier_prompt)
        self.assertIn(case.draft_prompt, draft_client.calls[0]["messages"][-1]["content"])

    def test_draft_critique_verify_records_three_stage_evidence(self):
        case = build_case(
            "gpqa_diamond",
            "choice_exact",
            0,
            {"question": "A. alpha\nB. beta", "answer": "B"},
            answer_only=True,
            system_prompt="answer",
            max_tokens=32,
        )
        draft_client = ScriptedClient(
            [
                ModelReply(
                    content="Candidate A",
                    usage={"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
                    raw={"choices": [{"finish_reason": "stop"}]},
                )
            ]
        )
        critique_client = ScriptedClient(
            [
                ModelReply(
                    content="Candidate A is wrong; corrected answer is B.",
                    usage={"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28},
                    raw={"choices": [{"finish_reason": "stop"}]},
                )
            ]
        )
        verifier_client = ScriptedClient(
            [
                ModelReply(
                    content="FINAL: B",
                    usage={"prompt_tokens": 30, "completion_tokens": 4, "total_tokens": 34},
                    raw={"choices": [{"finish_reason": "stop"}]},
                )
            ]
        )
        manifest = SuiteManifest(
            schema_version="nano_harness_baseline_suite_v1",
            suite_id="critique-test",
            selection_seed="fixed",
            system_prompt="answer",
            max_tokens=32,
            temperature=0.0,
            chat_template_kwargs={"enable_thinking": False},
            strategy="draft_critique_verify",
            draft_max_tokens=256,
            critique_max_tokens=192,
            second_solve_max_tokens=384,
            verifier_max_tokens=32,
            min_task_groups=1,
            datasets=(),
        )
        reply, stages = _run_draft_critique_verify_case(
            case,
            manifest,
            ModelConfig(name="test"),
            {256: draft_client, 192: critique_client, 32: verifier_client},
        )
        self.assertEqual(reply.content, "FINAL: B")
        self.assertEqual(
            reply.usage,
            {"prompt_tokens": 60, "completion_tokens": 16, "total_tokens": 76},
        )
        self.assertIn("Candidate A", critique_client.calls[0]["messages"][-1]["content"])
        final_prompt = verifier_client.calls[0]["messages"][-1]["content"]
        self.assertIn("corrected answer is B", final_prompt)
        self.assertEqual(stages["critique"]["max_tokens"], 192)
        self.assertEqual(stages["verifier"]["finish_reason"], "stop")

    def test_dual_solve_verifier_keeps_solutions_independent(self):
        case = build_case(
            "gsm8k",
            "numeric_exact",
            0,
            {
                "question": "What is 7 plus 5?",
                "answer": "Compute 7 + 5 = 12.\n#### 12",
            },
            system_prompt="answer",
            max_tokens=600,
        )
        draft_client = ScriptedClient(
            [
                ModelReply(
                    content="Solution A says 11.",
                    usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                    raw={"choices": [{"finish_reason": "stop"}]},
                )
            ]
        )
        second_client = ScriptedClient(
            [
                ModelReply(
                    content="Solution B: 7 + 5 = 12.",
                    usage={"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
                    raw={"choices": [{"finish_reason": "stop"}]},
                )
            ]
        )
        verifier_client = ScriptedClient(
            [
                ModelReply(
                    content="FINAL: 12",
                    usage={"prompt_tokens": 30, "completion_tokens": 4, "total_tokens": 34},
                    raw={"choices": [{"finish_reason": "stop"}]},
                )
            ]
        )
        manifest = SuiteManifest(
            schema_version="nano_harness_baseline_suite_v1",
            suite_id="dual-test",
            selection_seed="fixed",
            system_prompt="answer",
            max_tokens=600,
            temperature=0.0,
            chat_template_kwargs={"enable_thinking": False},
            strategy="dual_solve_verify",
            draft_max_tokens=256,
            critique_max_tokens=192,
            second_solve_max_tokens=384,
            verifier_max_tokens=32,
            min_task_groups=1,
            datasets=(),
        )
        reply, stages = _run_dual_solve_verify_case(
            case,
            manifest,
            ModelConfig(name="test"),
            {256: draft_client, 384: second_client, 32: verifier_client},
        )
        self.assertEqual(reply.content, "FINAL: 12")
        self.assertEqual(
            reply.usage,
            {"prompt_tokens": 51, "completion_tokens": 16, "total_tokens": 67},
        )
        self.assertNotIn(
            "Solution A says 11.",
            second_client.calls[0]["messages"][-1]["content"],
        )
        final_prompt = verifier_client.calls[0]["messages"][-1]["content"]
        self.assertIn("Solution A says 11.", final_prompt)
        self.assertIn("Solution B: 7 + 5 = 12.", final_prompt)
        self.assertEqual(stages["second_solve"]["max_tokens"], 384)
        self.assertEqual(stages["draft"]["output"], "Solution A says 11.")
        self.assertEqual(
            stages["second_solve"]["output"],
            "Solution B: 7 + 5 = 12.",
        )

    def test_baseline_manifest_filters_long_prompts_before_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            from datasets import Dataset

            root = Path(directory)
            path = root / "gpqa.parquet"
            Dataset.from_list(
                [
                    {"question": "short\n\nA. a\nB. b", "answer": "A"},
                    {"question": "x" * 1000, "answer": "B"},
                ]
            ).to_parquet(str(path))
            from nano_harness.baseline import sha256_file

            manifest = SuiteManifest(
                schema_version="nano_harness_baseline_suite_v1",
                suite_id="filter-test",
                selection_seed="fixed",
                system_prompt="answer",
                max_tokens=8,
                temperature=0.0,
                chat_template_kwargs={"enable_thinking": False},
                strategy="direct",
                draft_max_tokens=384,
                critique_max_tokens=192,
                second_solve_max_tokens=384,
                verifier_max_tokens=32,
                min_task_groups=1,
                datasets=(
                    DatasetSpec(
                        name="gpqa_diamond",
                        path="gpqa.parquet",
                        sha256=sha256_file(path),
                        scorer="choice_exact",
                        limit=1,
                        max_source_chars=200,
                    ),
                ),
            )
            cases = load_cases(manifest, root)
            self.assertEqual(len(cases), 1)
            self.assertEqual(cases[0].source_index, 0)

    def test_openai_client_forwards_chat_template_kwargs(self):
        response = MagicMock()
        message = MagicMock()
        message.content = "FINAL: A"
        message.tool_calls = []
        message.model_dump.return_value = {"content": "FINAL: A"}
        response.choices = [SimpleNamespace(message=message)]
        response.usage = None
        response.model_dump.return_value = {}

        with (
            patch.dict("os.environ", {"TEST_API_KEY": "local"}),
            patch("nano_harness.client.OpenAI") as openai,
        ):
            openai.return_value.chat.completions.create.return_value = response
            client = OpenRouterClient(
                ModelConfig(
                    name="test-model",
                    api_key_env="TEST_API_KEY",
                    max_retries=1,
                    chat_template_kwargs={"enable_thinking": False},
                )
            )
            reply = client.complete([{"role": "user", "content": "answer"}])

        self.assertEqual(reply.content, "FINAL: A")
        kwargs = openai.return_value.chat.completions.create.call_args.kwargs
        self.assertEqual(
            kwargs["extra_body"],
            {"chat_template_kwargs": {"enable_thinking": False}},
        )

    def test_baseline_summary_keeps_per_benchmark_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.jsonl"
            records = [
                {
                    "case_id": "gsm8k-a",
                    "benchmark": "gsm8k",
                    "score": 1.0,
                    "prediction": "12",
                    "latency_seconds": 1.0,
                    "usage": {"prompt_tokens": 10, "completion_tokens": 4},
                },
                {
                    "case_id": "mmlu-a",
                    "benchmark": "mmlu",
                    "score": 0.0,
                    "prediction": None,
                    "latency_seconds": 3.0,
                    "usage": {"prompt_tokens": 20, "completion_tokens": 6},
                },
            ]
            path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            summary = summarize_baseline(path)
            self.assertEqual(summary["total_cases"], 2)
            self.assertEqual(summary["total_attempts"], 2)
            self.assertEqual(summary["macro_accuracy"], 0.5)
            self.assertEqual(summary["benchmarks"]["gsm8k"]["accuracy"], 1.0)
            self.assertEqual(summary["benchmarks"]["mmlu"]["parse_failures"], 1)

    def test_baseline_summary_retains_attempts_but_scores_latest_case_result(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "retry.jsonl"
            records = [
                {
                    "case_id": "gsm8k-a",
                    "benchmark": "gsm8k",
                    "status": "error",
                    "score": 0.0,
                    "prediction": None,
                    "latency_seconds": 1.0,
                    "usage": {},
                },
                {
                    "case_id": "gsm8k-a",
                    "benchmark": "gsm8k",
                    "status": "completed",
                    "score": 1.0,
                    "prediction": "12",
                    "latency_seconds": 2.0,
                    "usage": {"prompt_tokens": 10, "completion_tokens": 4},
                },
            ]
            path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            summary = summarize_baseline(path)
            self.assertEqual(summary["total_attempts"], 2)
            self.assertEqual(summary["total_cases"], 1)
            self.assertEqual(summary["completed_cases"], 1)
            self.assertEqual(summary["macro_accuracy"], 1.0)

    def test_baseline_comparison_is_paired_and_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate_path = root / "candidate.jsonl"
            baseline_path = root / "baseline.jsonl"
            candidate_rows = [
                {
                    "case_id": "gsm8k-a",
                    "benchmark": "gsm8k",
                    "model": "4b",
                    "score": 1.0,
                    "prediction": "12",
                },
                {
                    "case_id": "gsm8k-b",
                    "benchmark": "gsm8k",
                    "model": "4b",
                    "score": 0.0,
                    "prediction": None,
                },
                {
                    "case_id": "mmlu-a",
                    "benchmark": "mmlu",
                    "model": "4b",
                    "score": 1.0,
                    "prediction": "A",
                },
                {
                    "case_id": "mmlu-b",
                    "benchmark": "mmlu",
                    "model": "4b",
                    "score": 0.0,
                    "prediction": "B",
                },
            ]
            baseline_rows = [
                {
                    "case_id": "gsm8k-a",
                    "benchmark": "gsm8k",
                    "model": "9b",
                    "score": 0.0,
                    "prediction": "10",
                },
                {
                    "case_id": "gsm8k-b",
                    "benchmark": "gsm8k",
                    "model": "9b",
                    "score": 1.0,
                    "prediction": "9",
                },
                {
                    "case_id": "mmlu-a",
                    "benchmark": "mmlu",
                    "model": "9b",
                    "score": 1.0,
                    "prediction": "A",
                },
                {
                    "case_id": "mmlu-b",
                    "benchmark": "mmlu",
                    "model": "9b",
                    "score": 0.0,
                    "prediction": "C",
                },
            ]
            for path, rows in (
                (candidate_path, candidate_rows),
                (baseline_path, baseline_rows),
            ):
                path.write_text(
                    "".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8",
                )

            first = compare_baselines(
                candidate_path,
                baseline_path,
                bootstrap_samples=500,
                bootstrap_seed=7,
            )
            second = compare_baselines(
                candidate_path,
                baseline_path,
                bootstrap_samples=500,
                bootstrap_seed=7,
            )
            self.assertEqual(first, second)
            self.assertEqual(first["candidate_model"], "4b")
            self.assertEqual(first["baseline_model"], "9b")
            self.assertEqual(first["macro_delta"], 0.0)
            self.assertEqual(
                first["overall_micro"]["paired_counts"],
                {
                    "both_correct": 1,
                    "candidate_only": 1,
                    "baseline_only": 1,
                    "both_wrong": 1,
                },
            )
            self.assertEqual(first["overall_micro"]["mcnemar_exact_p"], 1.0)
            self.assertEqual(
                first["benchmarks"]["gsm8k"]["candidate_parse_failures"],
                ["gsm8k-b"],
            )

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
        messages = [
            {"role": "system", "content": "harness"},
            {"role": "system", "content": "task context" * 300},
            {"role": "user", "content": "original task"},
            {"role": "assistant", "content": "old action" * 300},
            {"role": "tool", "content": "old observation" * 300},
            {"role": "assistant", "content": "latest action"},
            {"role": "tool", "content": "latest observation"},
        ]
        compacted = compact_messages(
            messages,
            StateLedger(objective="keep objective", facts=["verified fact"]),
            max_chars=5000,
            reserve_chars=1000,
            scratchpad_chars=1000,
        )
        self.assertEqual(compacted[1]["content"], "task context" * 300)
        self.assertEqual(compacted[2]["content"], "original task")
        self.assertTrue(
            any("<state_ledger>" in item.get("content", "") for item in compacted)
        )
        self.assertEqual(compacted[-1]["content"], "latest observation")
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

    def test_resume_retries_model_api_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "errors.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "task_id": "retry-me",
                        "status": "error",
                        "failure_type": "model_api_error",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual(completed_task_ids(path), set())

    def test_provider_quota_is_structured_and_retryable(self):
        class QuotaClient:
            def complete(self, messages, tools=None):
                raise ProviderQuotaError("daily quota exhausted", "1785974400000")

        task = Task(
            task_id="quota",
            benchmark="clbench",
            messages=[{"role": "user", "content": "answer"}],
        )
        result = AgentHarness(
            QuotaClient(),
            "test-model",
            HarnessConfig(strategy="base"),
        ).run(task)
        self.assertEqual(result.failure_type, "provider_daily_quota")
        self.assertEqual(
            result.metadata["provider_quota_reset_at"], "1785974400000"
        )

    def test_runner_stops_shard_after_provider_quota(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "quota.yaml"
            config_path.write_text(
                f"""
model:
  name: nano-9b
harness:
  strategy: base
benchmark:
  name: synthetic
  source: builtin
output_dir: {root / "results"}
run_id: quota
""".strip(),
                encoding="utf-8",
            )

            class QuotaClient:
                def complete(self, messages, tools=None):
                    raise ProviderQuotaError(
                        "daily quota exhausted", "1785974400000"
                    )

            summary = run_config(load_run_config(config_path), QuotaClient())
            self.assertEqual(summary["attempted_this_invocation"], 1)
            self.assertEqual(summary["written_this_invocation"], 1)
            self.assertEqual(
                summary["failure_types"], {"provider_daily_quota": 1}
            )

    def test_swe_runner_does_not_mask_provider_quota_with_empty_diff(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = root / "repo"
            repository.mkdir()
            (repository / "file.py").write_text("value = 1\n", encoding="utf-8")
            import subprocess

            subprocess.run(["git", "init", "-b", "main"], cwd=repository, check=True)
            subprocess.run(
                ["git", "config", "user.name", "Test"], cwd=repository, check=True
            )
            subprocess.run(
                ["git", "config", "user.email", "test@example.com"],
                cwd=repository,
                check=True,
            )
            subprocess.run(["git", "add", "."], cwd=repository, check=True)
            subprocess.run(
                ["git", "commit", "-m", "baseline"], cwd=repository, check=True
            )
            result = AgentHarness(
                type(
                    "QuotaClient",
                    (),
                    {
                        "complete": lambda self, messages, tools=None: (
                            (_ for _ in ()).throw(
                                ProviderQuotaError(
                                    "daily quota exhausted", "1785974400000"
                                )
                            )
                        )
                    },
                )(),
                "test-model",
                HarnessConfig(strategy="optimized"),
            ).run(
                Task(
                    task_id="swe-quota",
                    benchmark="swebench",
                    messages=[{"role": "user", "content": "fix"}],
                    tools=[{"type": "function", "function": {"name": "read_file"}}],
                ),
                CodingToolExecutor(repository),
            )
            self.assertEqual(result.failure_type, "provider_daily_quota")
            self.assertEqual(result.status, "error")

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

    def test_clbench_preserves_historical_assistant_turns(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cl.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "messages": [
                            {"role": "system", "content": "context"},
                            {"role": "user", "content": "first"},
                            {"role": "assistant", "content": "prior answer"},
                            {"role": "user", "content": "follow-up"},
                        ],
                        "rubrics": ["use prior answer"],
                        "metadata": {"task_id": "multi-turn"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            task = next(
                CLBenchAdapter().load(
                    BenchmarkConfig(name="clbench", source=str(path))
                )
            )
            self.assertEqual(
                [message["role"] for message in task.messages],
                ["system", "user", "assistant", "user"],
            )

    def test_taubench_response_is_sent_back_to_environment(self):
        client = ScriptedClient(
            [
                ModelReply(content="Could you provide the order id?"),
                ModelReply(content="Your order is already shipped."),
            ]
        )

        class TauExecutor:
            def __init__(self):
                self.done = False
                self.responses = []

            def respond(self, content):
                self.responses.append(content)
                if len(self.responses) == 1:
                    return "The order id is O-17."
                self.done = True
                return "###STOP###"

        executor = TauExecutor()
        task = Task(
            task_id="tau",
            benchmark="taubench",
            messages=[{"role": "user", "content": "Where is my order?"}],
            metadata={"audit_policy": "never"},
        )
        result = AgentHarness(
            client,
            "test-model",
            HarnessConfig(strategy="optimized", max_steps=4),
        ).run(task, executor)
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.steps, 2)
        self.assertEqual(executor.responses[0], "Could you provide the order id?")
        self.assertTrue(
            any(
                item.get("kind") == "user_environment"
                and item.get("observation") == "The order id is O-17."
                for item in result.trajectory
            )
        )


if __name__ == "__main__":
    unittest.main()
