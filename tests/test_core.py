import json
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from nano_harness.analog_contract import (
    load_config as load_analog_contract_config,
    run_choice_calculation_selector,
    summarize_rows as summarize_analog_rows,
)
from nano_harness.baseline import (
    DatasetSpec,
    SuiteManifest,
    _run_draft_critique_verify_case,
    _run_direct_case,
    _run_draft_verify_case,
    _run_dual_solve_verify_case,
    _run_option_evidence_arbiter_case,
    _run_option_evidence_verify_case,
    _run_protected_math_arbiter_case,
    _run_protected_math_gate_case,
    _run_protected_math_majority_case,
    _run_protected_math_recovery_case,
    _run_protected_math_short_recovery_case,
    _run_protected_math_constrained_recovery_case,
    _strategy_for_case,
    _mcnemar_exact_p,
    build_case,
    compare_baselines,
    extract_prediction,
    load_cases,
    load_manifest,
    merge_baseline_shards,
    public_case_contract,
    select_case_shard,
    score_output,
    summarize_baseline,
)
from nano_harness.adapters.clbench import CLBenchAdapter
from nano_harness.adapters.swebench import extract_patch
from nano_harness.client import OpenRouterClient, ProviderQuotaError, ScriptedClient
from nano_harness.coding_tools import CodingToolExecutor
from nano_harness.choice_matrix_eval import (
    load_config as load_choice_matrix_eval_config,
)
from nano_harness.choice_matrix_eval_v2 import (
    load_config as load_choice_matrix_eval_v2_config,
)
from nano_harness.choice_verifier_matrix_eval_v2 import (
    load_config as load_choice_verifier_matrix_eval_config,
)
from nano_harness.choice_exact_replication_eval_v3 import (
    load_config as load_choice_exact_replication_eval_config,
)
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
from nano_harness.verified_choice import (
    load_config as load_verified_choice_config,
    verify_explicit_average_choice,
)
from nano_harness.verified_choice_canary import (
    load_config as load_verified_choice_canary_config,
)
from nano_harness.verified_choice_full import (
    load_config as load_verified_choice_full_config,
)
from nano_harness.verified_choice_v2 import verify_choice_v2


class FakeToolExecutor:
    def __init__(self):
        self.calls = []

    def execute(self, name, arguments):
        self.calls.append((name, arguments))
        return json.dumps({"status": "active"})


class CoreTests(unittest.TestCase):
    def test_choice_verifier_matrix_config_is_frozen(self):
        source = Path(
            "configs/harness/generic_choice_verifier_matrix_eval_v2.json"
        )
        raw = json.loads(source.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            config = load_choice_verifier_matrix_eval_config(path)
            self.assertEqual(
                config.parser_version,
                "host_count_and_verbal_average_v2",
            )
            raw["parser_version"] = "other"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "parser_version"):
                load_choice_verifier_matrix_eval_config(path)

    def test_choice_exact_replication_config_is_frozen(self):
        source = Path(
            "configs/harness/generic_choice_exact_replication_eval_v3.json"
        )
        raw = json.loads(source.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            config = load_choice_exact_replication_eval_config(path)
            self.assertEqual(config.scored_cases, 32)
            self.assertEqual(config.minimum_executor_wins_over_nine_b, 6)
            self.assertEqual(config.maximum_executor_losses_over_nine_b, 0)
            self.assertEqual(config.significance_alpha, 0.05)
            raw["minimum_executor_wins_over_nine_b"] = 5
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError, "minimum_executor_wins_over_nine_b"
            ):
                load_choice_exact_replication_eval_config(path)

    def test_verified_choice_v2_host_exact_and_ambiguity(self):
        exact = verify_choice_v2(
            "A coordinator attends a summit, registers 23 delegates, "
            "and every delegate brings 3 guests. Including the coordinator, "
            "how many people attend?\nA. 91\nB. 93\nC. 95\nD. 97"
        )
        self.assertTrue(exact["override"])
        self.assertEqual(exact["selected_letter"], "B")
        self.assertEqual(exact["expression"], "1 + 23 + 23 * 3")
        no_exact = verify_choice_v2(
            "A coordinator attends a summit, registers 23 delegates, "
            "and every delegate brings 3 guests. Including the coordinator, "
            "how many people attend?\nA. 90\nB. 92\nC. 94\nD. 96"
        )
        self.assertFalse(no_exact["override"])
        self.assertEqual(no_exact["reason"], "no_unique_exact_option_match")
        duplicate = verify_choice_v2(
            "A coordinator attends a summit, registers 23 delegates, "
            "and every delegate brings 3 guests. Including the coordinator, "
            "how many people attend?\nA. 91\nB. 93\nC. 93\nD. 97"
        )
        self.assertFalse(duplicate["override"])
        self.assertEqual(duplicate["reason"], "option_values_not_unique")

    def test_verified_choice_v2_verbal_average_exact_and_ambiguity(self):
        exact = verify_choice_v2(
            "A north depot processed 100 parcels and a south depot processed "
            "140 parcels. What is the average number of parcels processed by "
            "the two depots?\nA. 110\nB. 115\nC. 120\nD. 125"
        )
        self.assertTrue(exact["override"])
        self.assertEqual(exact["selected_letter"], "C")
        self.assertEqual(exact["result"], "120")
        no_exact = verify_choice_v2(
            "A north depot processed 101 parcels and a south depot processed "
            "140 parcels. What is the average number of parcels processed by "
            "the two depots?\nA. 118\nB. 119\nC. 120\nD. 121"
        )
        self.assertFalse(no_exact["override"])
        self.assertEqual(no_exact["result"], "241/2")
        duplicate = verify_choice_v2(
            "A north depot processed 100 parcels and a south depot processed "
            "140 parcels. What is the average number of parcels processed by "
            "the two depots?\nA. 110\nB. 120\nC. 120\nD. 125"
        )
        self.assertFalse(duplicate["override"])
        self.assertEqual(duplicate["reason"], "option_values_not_unique")

    def test_choice_matrix_eval_v2_config_is_frozen(self):
        source = Path(
            "configs/harness/generic_choice_capability_matrix_eval_v2.json"
        )
        raw = json.loads(source.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            config = load_choice_matrix_eval_v2_config(path)
            self.assertEqual(config.structured_output_regex, r"FINAL: [A-D]")
            raw["structured_output_regex"] = r"[A-D]"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "structured_output_regex"):
                load_choice_matrix_eval_v2_config(path)

    def test_choice_matrix_eval_config_is_frozen(self):
        source = Path(
            "configs/harness/generic_choice_capability_matrix_eval_v1.json"
        )
        raw = json.loads(source.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            config = load_choice_matrix_eval_config(path)
            self.assertEqual(config.max_tokens, 32)
            raw["temperature"] = 0.2
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "temperature"):
                load_choice_matrix_eval_config(path)

    def test_verified_choice_full_config_is_frozen(self):
        source = Path(
            "configs/harness/anchored_v1_verified_choice_full_v1.json"
        )
        raw = json.loads(source.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            config = load_verified_choice_full_config(path)
            self.assertTrue(config.exact_option_match_required)
            raw["ambiguous_fallback"] = "nearest_option"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "ambiguous_fallback"):
                load_verified_choice_full_config(path)

    def test_verified_choice_canary_config_is_frozen(self):
        source = Path(
            "configs/harness/anchored_v1_verified_choice_canary_v1.json"
        )
        raw = json.loads(source.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            config = load_verified_choice_canary_config(path)
            self.assertTrue(config.exact_option_match_required)
            raw["parser_version"] = "another-parser"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "parser_version"):
                load_verified_choice_canary_config(path)

    def test_verified_choice_config_is_frozen(self):
        source = Path(
            "configs/harness/anchored_v1_verified_choice_executor_v1.json"
        )
        raw = json.loads(source.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            config = load_verified_choice_config(path)
            self.assertTrue(config.exact_option_match_required)
            raw["ambiguous_fallback"] = "round_to_nearest"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "ambiguous_fallback"):
                load_verified_choice_config(path)

    def test_verified_choice_selects_unique_exact_average(self):
        prompt = (
            "Two players have total scores 27 * 4 and 26 * 7. "
            "What is the average of the two player totals?\n"
            "A. 133\nB. 145\nC. 151\nD. 159\n"
            "Return only one standalone line: FINAL: <letter>."
        )
        receipt = verify_explicit_average_choice(prompt)
        self.assertTrue(receipt["eligible"])
        self.assertTrue(receipt["override"])
        self.assertEqual(receipt["selected_letter"], "B")
        self.assertEqual(receipt["expression_values"], ["108", "182"])
        self.assertEqual(receipt["result"], "145")

    def test_verified_choice_rejects_fraction_without_exact_option(self):
        prompt = (
            "Two players have total scores 51 * 7 and 50 * 6. "
            "What is the average of the two player totals?\n"
            "A. 316\nB. 328\nC. 334\nD. 342\n"
            "Return only one standalone line: FINAL: <letter>."
        )
        receipt = verify_explicit_average_choice(prompt)
        self.assertTrue(receipt["eligible"])
        self.assertFalse(receipt["override"])
        self.assertEqual(receipt["result"], "657/2")
        self.assertEqual(receipt["exact_matching_options"], [])
        self.assertEqual(receipt["reason"], "no_unique_exact_option_match")

    def test_verified_choice_rejects_unsupported_intent(self):
        receipt = verify_explicit_average_choice(
            "What is 7 * 8?\nA. 54\nB. 56\nC. 58\nD. 60"
        )
        self.assertFalse(receipt["eligible"])
        self.assertFalse(receipt["override"])
        self.assertEqual(receipt["reason"], "unsupported_intent")

    def test_verified_choice_rejects_decimal_expression(self):
        prompt = (
            "Two players have total scores 7.5 * 4 and 6 * 5. "
            "What is the average of the two player totals?\n"
            "A. 28\nB. 30\nC. 32\nD. 34"
        )
        receipt = verify_explicit_average_choice(prompt)
        self.assertFalse(receipt["eligible"])
        self.assertFalse(receipt["override"])
        self.assertEqual(receipt["reason"], "expression_count_not_two")

    def test_analog_contract_config_is_frozen(self):
        source = Path(
            "configs/harness/anchored_v1_choice_calculation_selector_v1.json"
        )
        raw = json.loads(source.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            config = load_analog_contract_config(path)
            self.assertEqual(config.selector_regex, r"FINAL: [A-D]")
            raw["selector_max_tokens"] = 16
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "selector_max_tokens"):
                load_analog_contract_config(path)

    def test_choice_calculation_selector_forwards_frozen_regex(self):
        config = load_analog_contract_config(
            "configs/harness/anchored_v1_choice_calculation_selector_v1.json"
        )
        sample = {
            "sample_id": "choice-1",
            "format_family": "final_choice",
            "messages": [
                {"role": "system", "content": "answer only"},
                {
                    "role": "user",
                    "content": "What is 2 + 2?\nA. 3\nB. 4\nFINAL only.",
                },
                {"role": "assistant", "content": "FINAL: B"},
            ],
        }
        calculation = ScriptedClient(
            [
                ModelReply(
                    content="2 + 2 = 4. Option B matches. CANDIDATE: B",
                    usage={"prompt_tokens": 20, "completion_tokens": 12},
                )
            ]
        )
        selector = ScriptedClient(
            [
                ModelReply(
                    content="FINAL: B",
                    usage={"prompt_tokens": 40, "completion_tokens": 4},
                )
            ]
        )
        output, usage, stages = run_choice_calculation_selector(
            sample,
            config,
            calculation,
            selector,
        )
        self.assertEqual(output, "FINAL: B")
        self.assertEqual(
            selector.calls[0]["extra_body"],
            {"structured_outputs": {"regex": r"FINAL: [A-D]"}},
        )
        self.assertIn(
            "2 + 2 = 4",
            selector.calls[0]["messages"][-1]["content"],
        )
        self.assertEqual(usage["prompt_tokens"], 60.0)
        self.assertEqual(stages["calculation"]["max_tokens"], 128)
        self.assertEqual(stages["selector"]["max_tokens"], 8)

    def test_analog_contract_summary_keeps_family_metrics(self):
        rows = [
            {
                "sample_id": "choice-1",
                "task_family": "choice",
                "exact": True,
                "semantic_valid": True,
            },
            {
                "sample_id": "choice-2",
                "task_family": "choice",
                "exact": False,
                "semantic_valid": False,
            },
            {
                "sample_id": "numeric-1",
                "task_family": "numeric",
                "exact": False,
                "semantic_valid": True,
            },
        ]
        summary = summarize_analog_rows(rows)
        self.assertEqual(summary["exact"], 1)
        self.assertEqual(summary["semantic_exact"], 2)
        self.assertEqual(summary["by_family"]["choice"]["semantic_exact"], 1)
        self.assertEqual(
            summary["by_family"]["choice"]["failure_sample_ids"],
            ["choice-2"],
        )

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

    def test_row_stable_case_ids_preserve_duplicate_rows(self):
        record = {
            "question": "What is 7 plus 5?",
            "answer": "Compute 7 + 5 = 12.\n#### 12",
        }
        first = build_case(
            "gsm8k",
            "numeric_exact",
            3,
            record,
            case_id_policy="row_stable_v2",
        )
        second = build_case(
            "gsm8k",
            "numeric_exact",
            99,
            record,
            case_id_policy="row_stable_v2",
        )
        self.assertNotEqual(first.case_id, second.case_id)
        self.assertEqual(first.expected, second.expected)

    def test_case_shards_are_deterministic_disjoint_and_complete(self):
        cases = [
            build_case(
                "gsm8k",
                "numeric_exact",
                index,
                {
                    "question": f"What is {index} plus 1?",
                    "answer": f"#### {index + 1}",
                },
                case_id_policy="row_stable_v2",
            )
            for index in range(37)
        ]
        first = [
            select_case_shard(cases, num_shards=5, shard_id=index)
            for index in range(5)
        ]
        second = [
            select_case_shard(cases, num_shards=5, shard_id=index)
            for index in range(5)
        ]
        self.assertEqual(
            [[case.case_id for case in shard] for shard in first],
            [[case.case_id for case in shard] for shard in second],
        )
        flattened = [case.case_id for shard in first for case in shard]
        self.assertEqual(len(flattened), len(cases))
        self.assertEqual(set(flattened), {case.case_id for case in cases})

    def test_baseline_merge_requires_exact_disjoint_case_set(self):
        records = [
            {
                "schema_version": "nano_harness_baseline_case_v1",
                "case_id": f"case-{index}",
                "benchmark": "gsm8k",
                "model": "model",
                "status": "completed",
                "score": 1.0,
                "prediction": "1",
                "expected": "1",
                "latency_seconds": 0.1,
                "usage": {"total_tokens": 10},
                "finish_reason": "stop",
            }
            for index in range(4)
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.jsonl"
            second = root / "second.jsonl"
            output = root / "merged.jsonl"
            first.write_text(
                "\n".join(json.dumps(row) for row in records[:2]) + "\n",
                encoding="utf-8",
            )
            second.write_text(
                "\n".join(json.dumps(row) for row in records[2:]) + "\n",
                encoding="utf-8",
            )
            receipt = merge_baseline_shards(
                [first, second],
                output,
                expected_case_ids={row["case_id"] for row in records},
            )
            self.assertEqual(receipt["case_count"], 4)
            self.assertEqual(summarize_baseline(output)["total_cases"], 4)
            with self.assertRaisesRegex(ValueError, "merged case IDs differ"):
                merge_baseline_shards(
                    [first],
                    output,
                    expected_case_ids={row["case_id"] for row in records},
                )
            second.write_text(
                json.dumps(records[1]) + "\n" + json.dumps(records[2]) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "multiple shards"):
                merge_baseline_shards(
                    [first, second],
                    output,
                    expected_case_ids={row["case_id"] for row in records[:3]},
                )

    def test_public_case_contract_excludes_prompts_and_answers(self):
        case = build_case(
            "gsm8k",
            "numeric_exact",
            0,
            {
                "question": "Private benchmark question",
                "answer": "Hidden work\n#### 12",
            },
            case_id_policy="row_stable_v2",
        )
        public = public_case_contract([case])[0]
        self.assertNotIn("prompt", public)
        self.assertNotIn("expected", public)
        self.assertNotIn("answer", json.dumps(public))
        self.assertIn("prompt_sha256", public)

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
            benchmark_routing={},
            draft_max_tokens=384,
            critique_max_tokens=192,
            second_solve_max_tokens=384,
            option_evidence_max_tokens=96,
            verifier_max_tokens=32,
            normalize_bare_choice=False,
            fallback_to_protected_on_parse_failure=False,
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

    def test_direct_and_draft_stages_use_their_declared_prompt_contracts(self):
        case = build_case(
            "mmlu",
            "choice_exact",
            0,
            {
                "question": "Which option is correct?",
                "choices": ["first", "second"],
                "answer": 1,
                "subject": "test",
            },
            answer_only=True,
            system_prompt="answer only",
            max_tokens=32,
        )
        direct_client = ScriptedClient(
            [
                ModelReply(
                    content="FINAL: B",
                    usage={"prompt_tokens": 10, "completion_tokens": 4},
                    raw={"choices": [{"finish_reason": "stop"}]},
                )
            ]
        )
        _run_direct_case(
            case,
            ModelConfig(name="test"),
            {32: direct_client},
        )
        self.assertIn(
            "Do not show reasoning",
            direct_client.calls[0]["messages"][-1]["content"],
        )
        _, direct_stages = _run_direct_case(
            case,
            ModelConfig(name="test"),
            {
                32: ScriptedClient(
                    [
                        ModelReply(
                            content="FINAL: B",
                            usage={"prompt_tokens": 10, "completion_tokens": 4},
                            raw={"choices": [{"finish_reason": "stop"}]},
                        )
                    ]
                )
            },
        )
        import hashlib

        self.assertEqual(
            direct_stages["direct"]["input_sha256"],
            hashlib.sha256(case.prompt.encode()).hexdigest(),
        )

        draft_client = ScriptedClient(
            [
                ModelReply(
                    content="The second option is correct.",
                    usage={"prompt_tokens": 10, "completion_tokens": 6},
                    raw={"choices": [{"finish_reason": "stop"}]},
                )
            ]
        )
        verifier_client = ScriptedClient(
            [
                ModelReply(
                    content="FINAL: B",
                    usage={"prompt_tokens": 20, "completion_tokens": 4},
                    raw={"choices": [{"finish_reason": "stop"}]},
                )
            ]
        )
        manifest = SuiteManifest(
            schema_version="nano_harness_baseline_suite_v1",
            suite_id="draft-contract-test",
            selection_seed="fixed",
            system_prompt="answer only",
            max_tokens=32,
            temperature=0.0,
            chat_template_kwargs={"enable_thinking": False},
            strategy="draft_verify",
            benchmark_routing={},
            draft_max_tokens=256,
            critique_max_tokens=192,
            second_solve_max_tokens=384,
            option_evidence_max_tokens=96,
            verifier_max_tokens=32,
            normalize_bare_choice=False,
            fallback_to_protected_on_parse_failure=False,
            min_task_groups=1,
            datasets=(),
        )
        _, draft_stages = _run_draft_verify_case(
            case,
            manifest,
            ModelConfig(name="test"),
            {256: draft_client, 32: verifier_client},
        )
        self.assertIn(
            "Reason concisely",
            draft_client.calls[0]["messages"][-1]["content"],
        )
        self.assertIn(
            "Do not show reasoning",
            verifier_client.calls[0]["messages"][-1]["content"],
        )
        self.assertEqual(
            draft_stages["draft"]["input_sha256"],
            hashlib.sha256(case.draft_prompt.encode()).hexdigest(),
        )

    def test_benchmark_routing_requires_complete_supported_routes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "suite.yaml"
            base = {
                "schema_version": "nano_harness_baseline_suite_v1",
                "suite_id": "routing-test",
                "selection_seed": "fixed",
                "system_prompt": "answer",
                "max_tokens": 32,
                "temperature": 0.0,
                "chat_template_kwargs": {"enable_thinking": False},
                "strategy": "benchmark_routing",
                "min_task_groups": 1,
                "datasets": [
                    {
                        "name": "mmlu",
                        "path": "mmlu.parquet",
                        "sha256": "0" * 64,
                        "scorer": "choice_exact",
                        "limit": 1,
                    }
                ],
            }
            import yaml

            path.write_text(yaml.safe_dump(base), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "cover every dataset"):
                load_manifest(path)
            base["benchmark_routing"] = {"mmlu": "unsupported"}
            path.write_text(yaml.safe_dump(base), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unsupported benchmark routes"):
                load_manifest(path)
            base["benchmark_routing"] = {"mmlu": "draft_verify"}
            path.write_text(yaml.safe_dump(base), encoding="utf-8")
            manifest = load_manifest(path)
            case = build_case(
                "mmlu",
                "choice_exact",
                0,
                {
                    "question": "Question?",
                    "choices": ["A", "B"],
                    "answer": 0,
                    "subject": "test",
                },
            )
            self.assertEqual(_strategy_for_case(manifest, case), "draft_verify")

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
            benchmark_routing={},
            draft_max_tokens=256,
            critique_max_tokens=192,
            second_solve_max_tokens=384,
            option_evidence_max_tokens=96,
            verifier_max_tokens=32,
            normalize_bare_choice=False,
            fallback_to_protected_on_parse_failure=False,
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
            benchmark_routing={},
            draft_max_tokens=256,
            critique_max_tokens=192,
            second_solve_max_tokens=384,
            option_evidence_max_tokens=96,
            verifier_max_tokens=32,
            normalize_bare_choice=False,
            fallback_to_protected_on_parse_failure=False,
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

    def test_option_evidence_keeps_evaluators_independent_then_selects(self):
        case = build_case(
            "gpqa_diamond",
            "choice_exact",
            0,
            {
                "question": "Question\n\nA. alpha\nB. beta\nC. gamma\nD. delta",
                "answer": "C",
            },
            answer_only=True,
            system_prompt="answer",
            max_tokens=32,
        )
        option_outputs = [
            "Evidence A. VERDICT A: REJECT",
            "Evidence B. VERDICT B: REJECT",
            "Evidence C. VERDICT C: SUPPORT",
            "Evidence D. VERDICT D: REJECT",
        ]
        option_client = ScriptedClient(
            [
                ModelReply(
                    content=content,
                    usage={"prompt_tokens": 20, "completion_tokens": 8},
                    raw={"choices": [{"finish_reason": "stop"}]},
                )
                for content in option_outputs
            ]
        )
        selector_client = ScriptedClient(
            [
                ModelReply(
                    content="FINAL: C",
                    usage={"prompt_tokens": 80, "completion_tokens": 4},
                    raw={"choices": [{"finish_reason": "stop"}]},
                )
            ]
        )
        manifest = SuiteManifest(
            schema_version="nano_harness_baseline_suite_v1",
            suite_id="option-test",
            selection_seed="fixed",
            system_prompt="answer",
            max_tokens=32,
            temperature=0.0,
            chat_template_kwargs={"enable_thinking": False},
            strategy="option_evidence_verify",
            benchmark_routing={},
            draft_max_tokens=256,
            critique_max_tokens=192,
            second_solve_max_tokens=384,
            option_evidence_max_tokens=96,
            verifier_max_tokens=64,
            normalize_bare_choice=False,
            fallback_to_protected_on_parse_failure=False,
            min_task_groups=1,
            datasets=(),
        )
        reply, stages = _run_option_evidence_verify_case(
            case,
            manifest,
            ModelConfig(name="test"),
            {96: option_client, 64: selector_client},
        )
        self.assertEqual(reply.content, "FINAL: C")
        self.assertEqual(
            reply.usage,
            {"prompt_tokens": 160, "completion_tokens": 36},
        )
        self.assertEqual(set(stages["option_evidence"]), {"A", "B", "C", "D"})
        for index, call in enumerate(option_client.calls):
            prompt = call["messages"][-1]["content"]
            self.assertIn(f"Evaluate option {'ABCD'[index]}", prompt)
            for prior_output in option_outputs:
                self.assertNotIn(prior_output, prompt)
        selector_prompt = selector_client.calls[0]["messages"][-1]["content"]
        for output in option_outputs:
            self.assertIn(output, selector_prompt)
        self.assertEqual(stages["selector"]["max_tokens"], 64)
        self.assertFalse(stages["selector"]["normalized_bare_choice"])
        self.assertIn("input_sha256", stages["option_evidence"]["A"])

    def test_option_evidence_normalizes_only_an_exact_bare_choice(self):
        case = build_case(
            "gpqa_diamond",
            "choice_exact",
            0,
            {
                "question": "Question\n\nA. alpha\nB. beta\nC. gamma\nD. delta",
                "answer": "D",
            },
            answer_only=True,
            system_prompt="answer",
            max_tokens=32,
        )
        option_client = ScriptedClient(
            [
                ModelReply(
                    content=f"Evidence {letter}",
                    usage={"prompt_tokens": 10, "completion_tokens": 2},
                    raw={"choices": [{"finish_reason": "stop"}]},
                )
                for letter in "ABCD"
            ]
        )
        selector_client = ScriptedClient(
            [
                ModelReply(
                    content=" d \n",
                    usage={"prompt_tokens": 40, "completion_tokens": 1},
                    raw={"choices": [{"finish_reason": "stop"}]},
                )
            ]
        )
        manifest = SuiteManifest(
            schema_version="nano_harness_baseline_suite_v1",
            suite_id="normalizer-test",
            selection_seed="fixed",
            system_prompt="answer",
            max_tokens=32,
            temperature=0.0,
            chat_template_kwargs={"enable_thinking": False},
            strategy="option_evidence_verify",
            benchmark_routing={},
            draft_max_tokens=256,
            critique_max_tokens=192,
            second_solve_max_tokens=384,
            option_evidence_max_tokens=96,
            verifier_max_tokens=64,
            normalize_bare_choice=True,
            fallback_to_protected_on_parse_failure=False,
            min_task_groups=1,
            datasets=(),
        )
        reply, stages = _run_option_evidence_verify_case(
            case,
            manifest,
            ModelConfig(name="test"),
            {96: option_client, 64: selector_client},
        )
        self.assertEqual(reply.content, "FINAL: D")
        self.assertTrue(stages["selector"]["normalized_bare_choice"])
        self.assertIn("raw_output_sha256", stages["selector"])

        self.assertIsNone(
            re.fullmatch(r"[A-Da-d]", "The answer is D.".strip())
        )

    def test_option_arbiter_protects_direct_candidate_then_compares_evidence(self):
        case = build_case(
            "gpqa_diamond",
            "choice_exact",
            0,
            {
                "question": "Question\n\nA. alpha\nB. beta\nC. gamma\nD. delta",
                "answer": "B",
            },
            answer_only=True,
            system_prompt="answer",
            max_tokens=32,
        )
        direct_client = ScriptedClient(
            [
                ModelReply(
                    content="FINAL: A",
                    usage={"prompt_tokens": 10, "completion_tokens": 4},
                    raw={"choices": [{"finish_reason": "stop"}]},
                )
            ]
        )
        option_outputs = [
            "A has a contradiction.",
            "B has strong support.",
            "C is rejected.",
            "D is rejected.",
        ]
        option_client = ScriptedClient(
            [
                ModelReply(
                    content=content,
                    usage={"prompt_tokens": 20, "completion_tokens": 8},
                    raw={"choices": [{"finish_reason": "stop"}]},
                )
                for content in option_outputs
            ]
        )
        arbiter_client = ScriptedClient(
            [
                ModelReply(
                    content="FINAL: B",
                    usage={"prompt_tokens": 90, "completion_tokens": 4},
                    raw={"choices": [{"finish_reason": "stop"}]},
                )
            ]
        )
        manifest = SuiteManifest(
            schema_version="nano_harness_baseline_suite_v1",
            suite_id="arbiter-test",
            selection_seed="fixed",
            system_prompt="answer",
            max_tokens=64,
            temperature=0.0,
            chat_template_kwargs={"enable_thinking": False},
            strategy="option_evidence_arbiter",
            benchmark_routing={},
            draft_max_tokens=256,
            critique_max_tokens=192,
            second_solve_max_tokens=384,
            option_evidence_max_tokens=96,
            verifier_max_tokens=64,
            normalize_bare_choice=True,
            fallback_to_protected_on_parse_failure=False,
            min_task_groups=1,
            datasets=(),
        )
        reply, stages = _run_option_evidence_arbiter_case(
            case,
            manifest,
            ModelConfig(name="test"),
            {32: direct_client, 96: option_client, 64: arbiter_client},
        )
        self.assertEqual(reply.content, "FINAL: B")
        self.assertEqual(
            reply.usage,
            {"prompt_tokens": 180, "completion_tokens": 40},
        )
        arbiter_prompt = arbiter_client.calls[0]["messages"][-1]["content"]
        self.assertIn("<protected_direct_candidate>\nFINAL: A", arbiter_prompt)
        for output in option_outputs:
            self.assertIn(output, arbiter_prompt)
        self.assertEqual(stages["protected_direct"]["output"], "FINAL: A")
        self.assertEqual(set(stages["option_evidence"]), {"A", "B", "C", "D"})
        self.assertFalse(stages["arbiter"]["normalized_bare_choice"])
        self.assertIn("input_sha256", stages["arbiter"])

    def test_protected_math_arbiter_keeps_resolve_independent(self):
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
        direct_client = ScriptedClient(
            [
                ModelReply(
                    content="Bad direct reasoning.\nFINAL: 11",
                    usage={"prompt_tokens": 10, "completion_tokens": 8},
                    raw={"choices": [{"finish_reason": "stop"}]},
                )
            ]
        )
        resolve_client = ScriptedClient(
            [
                ModelReply(
                    content="7 + 5 = 12.\nFINAL: 12",
                    usage={"prompt_tokens": 20, "completion_tokens": 10},
                    raw={"choices": [{"finish_reason": "stop"}]},
                )
            ]
        )
        arbiter_client = ScriptedClient(
            [
                ModelReply(
                    content="FINAL: 12",
                    usage={"prompt_tokens": 30, "completion_tokens": 4},
                    raw={"choices": [{"finish_reason": "stop"}]},
                )
            ]
        )
        manifest = SuiteManifest(
            schema_version="nano_harness_baseline_suite_v1",
            suite_id="math-arbiter-test",
            selection_seed="fixed",
            system_prompt="answer",
            max_tokens=600,
            temperature=0.0,
            chat_template_kwargs={"enable_thinking": False},
            strategy="protected_math_arbiter",
            benchmark_routing={},
            draft_max_tokens=256,
            critique_max_tokens=192,
            second_solve_max_tokens=384,
            option_evidence_max_tokens=96,
            verifier_max_tokens=64,
            normalize_bare_choice=False,
            fallback_to_protected_on_parse_failure=False,
            min_task_groups=1,
            datasets=(),
        )
        reply, stages = _run_protected_math_arbiter_case(
            case,
            manifest,
            ModelConfig(name="test"),
            {600: direct_client, 384: resolve_client, 64: arbiter_client},
        )
        self.assertEqual(reply.content, "FINAL: 12")
        self.assertEqual(
            reply.usage,
            {"prompt_tokens": 60, "completion_tokens": 22},
        )
        resolve_prompt = resolve_client.calls[0]["messages"][-1]["content"]
        self.assertNotIn("Bad direct reasoning", resolve_prompt)
        arbiter_prompt = arbiter_client.calls[0]["messages"][-1]["content"]
        self.assertIn("<protected_direct_answer>11", arbiter_prompt)
        self.assertNotIn("Bad direct reasoning", arbiter_prompt)
        self.assertIn("7 + 5 = 12", arbiter_prompt)
        self.assertEqual(stages["protected_direct"]["prediction"], "11")
        self.assertEqual(stages["independent_resolve"]["prediction"], "12")
        self.assertIn("input_sha256", stages["arbiter"])

    def test_math_arbiter_falls_back_only_when_final_is_unparseable(self):
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
        clients = {
            600: ScriptedClient(
                [
                    ModelReply(
                        content="FINAL: 12",
                        usage={"prompt_tokens": 10, "completion_tokens": 4},
                        raw={"choices": [{"finish_reason": "stop"}]},
                    )
                ]
            ),
            384: ScriptedClient(
                [
                    ModelReply(
                        content="FINAL: 12",
                        usage={"prompt_tokens": 20, "completion_tokens": 4},
                        raw={"choices": [{"finish_reason": "stop"}]},
                    )
                ]
            ),
            64: ScriptedClient(
                [
                    ModelReply(
                        content="The protected answer is correct but",
                        usage={"prompt_tokens": 30, "completion_tokens": 6},
                        raw={"choices": [{"finish_reason": "length"}]},
                    )
                ]
            ),
        }
        manifest = SuiteManifest(
            schema_version="nano_harness_baseline_suite_v1",
            suite_id="math-fallback-test",
            selection_seed="fixed",
            system_prompt="answer",
            max_tokens=600,
            temperature=0.0,
            chat_template_kwargs={"enable_thinking": False},
            strategy="protected_math_arbiter",
            benchmark_routing={},
            draft_max_tokens=256,
            critique_max_tokens=192,
            second_solve_max_tokens=384,
            option_evidence_max_tokens=96,
            verifier_max_tokens=64,
            normalize_bare_choice=False,
            fallback_to_protected_on_parse_failure=True,
            min_task_groups=1,
            datasets=(),
        )
        reply, stages = _run_protected_math_arbiter_case(
            case,
            manifest,
            ModelConfig(name="test"),
            clients,
        )
        self.assertEqual(reply.content, "FINAL: 12")
        self.assertTrue(stages["arbiter"]["fallback_to_protected_applied"])
        self.assertEqual(
            stages["arbiter"]["raw_output"],
            "The protected answer is correct but",
        )

    def test_math_gate_separates_decision_from_numeric_formatting(self):
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
        clients = {
            600: ScriptedClient(
                [
                    ModelReply(
                        content="FINAL: 11",
                        usage={"prompt_tokens": 10, "completion_tokens": 4},
                        raw={"choices": [{"finish_reason": "stop"}]},
                    )
                ]
            ),
            384: ScriptedClient(
                [
                    ModelReply(
                        content="7 + 5 = 12.\nFINAL: 12",
                        usage={"prompt_tokens": 20, "completion_tokens": 8},
                        raw={"choices": [{"finish_reason": "stop"}]},
                    )
                ]
            ),
            8: ScriptedClient(
                [
                    ModelReply(
                        content="USE_RESOLVE",
                        usage={"prompt_tokens": 30, "completion_tokens": 3},
                        raw={"choices": [{"finish_reason": "stop"}]},
                    )
                ]
            ),
        }
        manifest = SuiteManifest(
            schema_version="nano_harness_baseline_suite_v1",
            suite_id="math-gate-test",
            selection_seed="fixed",
            system_prompt="answer",
            max_tokens=600,
            temperature=0.0,
            chat_template_kwargs={"enable_thinking": False},
            strategy="protected_math_gate",
            benchmark_routing={},
            draft_max_tokens=256,
            critique_max_tokens=192,
            second_solve_max_tokens=384,
            option_evidence_max_tokens=96,
            verifier_max_tokens=8,
            normalize_bare_choice=False,
            fallback_to_protected_on_parse_failure=False,
            min_task_groups=1,
            datasets=(),
        )
        reply, stages = _run_protected_math_gate_case(
            case,
            manifest,
            ModelConfig(name="test"),
            clients,
        )
        self.assertEqual(reply.content, "FINAL: 12")
        self.assertEqual(stages["decision_gate"]["decision"], "USE_RESOLVE")
        self.assertEqual(stages["decision_gate"]["selected_prediction"], "12")
        self.assertEqual(
            reply.usage,
            {"prompt_tokens": 60, "completion_tokens": 15},
        )

    def test_math_gate_defaults_to_keep_for_non_exact_decision(self):
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
        clients = {
            600: ScriptedClient(
                [
                    ModelReply(
                        content="FINAL: 12",
                        usage={"prompt_tokens": 10, "completion_tokens": 4},
                        raw={"choices": [{"finish_reason": "stop"}]},
                    )
                ]
            ),
            384: ScriptedClient(
                [
                    ModelReply(
                        content="FINAL: 13",
                        usage={"prompt_tokens": 20, "completion_tokens": 4},
                        raw={"choices": [{"finish_reason": "stop"}]},
                    )
                ]
            ),
            8: ScriptedClient(
                [
                    ModelReply(
                        content="USE_RESOLVE because",
                        usage={"prompt_tokens": 30, "completion_tokens": 3},
                        raw={"choices": [{"finish_reason": "length"}]},
                    )
                ]
            ),
        }
        manifest = SuiteManifest(
            schema_version="nano_harness_baseline_suite_v1",
            suite_id="math-gate-default-test",
            selection_seed="fixed",
            system_prompt="answer",
            max_tokens=600,
            temperature=0.0,
            chat_template_kwargs={"enable_thinking": False},
            strategy="protected_math_gate",
            benchmark_routing={},
            draft_max_tokens=256,
            critique_max_tokens=192,
            second_solve_max_tokens=384,
            option_evidence_max_tokens=96,
            verifier_max_tokens=8,
            normalize_bare_choice=False,
            fallback_to_protected_on_parse_failure=False,
            min_task_groups=1,
            datasets=(),
        )
        reply, stages = _run_protected_math_gate_case(
            case,
            manifest,
            ModelConfig(name="test"),
            clients,
        )
        self.assertEqual(reply.content, "FINAL: 12")
        self.assertEqual(stages["decision_gate"]["decision"], "KEEP")

    def test_math_majority_selects_two_matching_resolves(self):
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
        direct_client = ScriptedClient(
            [
                ModelReply(
                    content="FINAL: 11",
                    usage={"prompt_tokens": 10, "completion_tokens": 4},
                    raw={"choices": [{"finish_reason": "stop"}]},
                )
            ]
        )
        resolve_client = ScriptedClient(
            [
                ModelReply(
                    content="FINAL: 12",
                    usage={"prompt_tokens": 20, "completion_tokens": 4},
                    raw={"choices": [{"finish_reason": "stop"}]},
                ),
                ModelReply(
                    content="Check: 12.\nFINAL: 12",
                    usage={"prompt_tokens": 21, "completion_tokens": 6},
                    raw={"choices": [{"finish_reason": "stop"}]},
                ),
            ]
        )
        manifest = SuiteManifest(
            schema_version="nano_harness_baseline_suite_v1",
            suite_id="majority-test",
            selection_seed="fixed",
            system_prompt="answer",
            max_tokens=600,
            temperature=0.0,
            chat_template_kwargs={"enable_thinking": False},
            strategy="protected_math_majority",
            benchmark_routing={},
            draft_max_tokens=256,
            critique_max_tokens=192,
            second_solve_max_tokens=384,
            option_evidence_max_tokens=96,
            verifier_max_tokens=8,
            normalize_bare_choice=False,
            fallback_to_protected_on_parse_failure=False,
            min_task_groups=1,
            datasets=(),
        )
        reply, stages = _run_protected_math_majority_case(
            case,
            manifest,
            ModelConfig(name="test"),
            {600: direct_client, 384: resolve_client},
        )
        self.assertEqual(reply.content, "FINAL: 12")
        self.assertEqual(
            stages["deterministic_vote"]["selection_reason"],
            "numeric_majority",
        )
        self.assertEqual(
            stages["deterministic_vote"]["predictions"],
            ["11", "12", "12"],
        )
        self.assertEqual(reply.usage, {"prompt_tokens": 51, "completion_tokens": 14})
        self.assertNotIn(
            "FINAL: 11",
            resolve_client.calls[0]["messages"][-1]["content"],
        )
        self.assertNotEqual(
            resolve_client.calls[0]["messages"][0]["content"],
            resolve_client.calls[1]["messages"][0]["content"],
        )

    def test_math_majority_keeps_direct_without_consensus(self):
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
        clients = {
            600: ScriptedClient(
                [
                    ModelReply(
                        content="FINAL: 12",
                        usage={"prompt_tokens": 10, "completion_tokens": 4},
                        raw={"choices": [{"finish_reason": "stop"}]},
                    )
                ]
            ),
            384: ScriptedClient(
                [
                    ModelReply(
                        content="FINAL: 11",
                        usage={"prompt_tokens": 20, "completion_tokens": 4},
                        raw={"choices": [{"finish_reason": "stop"}]},
                    ),
                    ModelReply(
                        content="FINAL: 13",
                        usage={"prompt_tokens": 20, "completion_tokens": 4},
                        raw={"choices": [{"finish_reason": "stop"}]},
                    ),
                ]
            ),
        }
        manifest = SuiteManifest(
            schema_version="nano_harness_baseline_suite_v1",
            suite_id="no-majority-test",
            selection_seed="fixed",
            system_prompt="answer",
            max_tokens=600,
            temperature=0.0,
            chat_template_kwargs={"enable_thinking": False},
            strategy="protected_math_majority",
            benchmark_routing={},
            draft_max_tokens=256,
            critique_max_tokens=192,
            second_solve_max_tokens=384,
            option_evidence_max_tokens=96,
            verifier_max_tokens=8,
            normalize_bare_choice=False,
            fallback_to_protected_on_parse_failure=False,
            min_task_groups=1,
            datasets=(),
        )
        reply, stages = _run_protected_math_majority_case(
            case,
            manifest,
            ModelConfig(name="test"),
            clients,
        )
        self.assertEqual(reply.content, "FINAL: 12")
        self.assertEqual(
            stages["deterministic_vote"]["selection_reason"],
            "no_majority_keep_direct",
        )

    def test_math_recovery_skips_second_call_for_parseable_direct(self):
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
        direct_client = ScriptedClient(
            [
                ModelReply(
                    content="FINAL: 12",
                    usage={"prompt_tokens": 10, "completion_tokens": 4},
                    raw={"choices": [{"finish_reason": "stop"}]},
                )
            ]
        )
        recovery_client = ScriptedClient([])
        manifest = SuiteManifest(
            schema_version="nano_harness_baseline_suite_v1",
            suite_id="recovery-skip-test",
            selection_seed="fixed",
            system_prompt="answer",
            max_tokens=600,
            temperature=0.0,
            chat_template_kwargs={"enable_thinking": False},
            strategy="protected_math_recovery",
            benchmark_routing={},
            draft_max_tokens=256,
            critique_max_tokens=192,
            second_solve_max_tokens=384,
            option_evidence_max_tokens=96,
            verifier_max_tokens=8,
            normalize_bare_choice=False,
            fallback_to_protected_on_parse_failure=False,
            min_task_groups=1,
            datasets=(),
        )
        reply, stages = _run_protected_math_recovery_case(
            case,
            manifest,
            ModelConfig(name="test"),
            {600: direct_client, 384: recovery_client},
        )
        self.assertEqual(reply.content, "FINAL: 12")
        self.assertEqual(recovery_client.calls, [])
        self.assertIsNone(stages["conditional_recovery"])
        self.assertFalse(stages["deterministic_selection"]["recovery_triggered"])

    def test_math_recovery_runs_only_for_unparseable_direct(self):
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
        direct_client = ScriptedClient(
            [
                ModelReply(
                    content="Reasoning without final",
                    usage={"prompt_tokens": 10, "completion_tokens": 5},
                    raw={"choices": [{"finish_reason": "length"}]},
                )
            ]
        )
        recovery_client = ScriptedClient(
            [
                ModelReply(
                    content="7 + 5 = 12.\nFINAL: 12",
                    usage={"prompt_tokens": 20, "completion_tokens": 8},
                    raw={"choices": [{"finish_reason": "stop"}]},
                )
            ]
        )
        manifest = SuiteManifest(
            schema_version="nano_harness_baseline_suite_v1",
            suite_id="recovery-run-test",
            selection_seed="fixed",
            system_prompt="answer",
            max_tokens=600,
            temperature=0.0,
            chat_template_kwargs={"enable_thinking": False},
            strategy="protected_math_recovery",
            benchmark_routing={},
            draft_max_tokens=256,
            critique_max_tokens=192,
            second_solve_max_tokens=384,
            option_evidence_max_tokens=96,
            verifier_max_tokens=8,
            normalize_bare_choice=False,
            fallback_to_protected_on_parse_failure=False,
            min_task_groups=1,
            datasets=(),
        )
        reply, stages = _run_protected_math_recovery_case(
            case,
            manifest,
            ModelConfig(name="test"),
            {600: direct_client, 384: recovery_client},
        )
        self.assertEqual(reply.content, "FINAL: 12")
        self.assertEqual(len(recovery_client.calls), 1)
        self.assertEqual(stages["conditional_recovery"]["prediction"], "12")
        self.assertTrue(stages["deterministic_selection"]["recovery_triggered"])
        self.assertEqual(
            stages["deterministic_selection"]["selection_reason"],
            "recovery_prediction",
        )
        self.assertEqual(
            reply.usage,
            {"prompt_tokens": 30, "completion_tokens": 13},
        )

    def test_short_math_recovery_uses_answer_only_budget(self):
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
        direct_client = ScriptedClient(
            [
                ModelReply(
                    content="Reasoning without final",
                    usage={"prompt_tokens": 10, "completion_tokens": 5},
                    raw={"choices": [{"finish_reason": "length"}]},
                )
            ]
        )
        recovery_client = ScriptedClient(
            [
                ModelReply(
                    content="FINAL: 12",
                    usage={"prompt_tokens": 20, "completion_tokens": 4},
                    raw={"choices": [{"finish_reason": "stop"}]},
                )
            ]
        )
        manifest = SuiteManifest(
            schema_version="nano_harness_baseline_suite_v1",
            suite_id="short-recovery-test",
            selection_seed="fixed",
            system_prompt="answer",
            max_tokens=600,
            temperature=0.0,
            chat_template_kwargs={"enable_thinking": False},
            strategy="protected_math_short_recovery",
            benchmark_routing={},
            draft_max_tokens=256,
            critique_max_tokens=192,
            second_solve_max_tokens=64,
            option_evidence_max_tokens=96,
            verifier_max_tokens=8,
            normalize_bare_choice=False,
            fallback_to_protected_on_parse_failure=False,
            min_task_groups=1,
            datasets=(),
        )
        reply, stages = _run_protected_math_short_recovery_case(
            case,
            manifest,
            ModelConfig(name="test"),
            {600: direct_client, 64: recovery_client},
        )
        self.assertEqual(reply.content, "FINAL: 12")
        self.assertEqual(stages["conditional_recovery"]["max_tokens"], 64)
        prompt = recovery_client.calls[0]["messages"][-1]["content"]
        self.assertIn("Do not show reasoning", prompt)
        self.assertEqual(
            stages["deterministic_selection"]["selection_reason"],
            "short_recovery_prediction",
        )

    def test_constrained_math_recovery_forwards_numeric_regex(self):
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
        direct_client = ScriptedClient(
            [
                ModelReply(
                    content="Reasoning without final",
                    usage={"prompt_tokens": 10, "completion_tokens": 5},
                    raw={"choices": [{"finish_reason": "length"}]},
                )
            ]
        )
        recovery_client = ScriptedClient(
            [
                ModelReply(
                    content="FINAL: 12",
                    usage={"prompt_tokens": 20, "completion_tokens": 4},
                    raw={"choices": [{"finish_reason": "stop"}]},
                )
            ]
        )
        manifest = SuiteManifest(
            schema_version="nano_harness_baseline_suite_v1",
            suite_id="constrained-recovery-test",
            selection_seed="fixed",
            system_prompt="answer",
            max_tokens=600,
            temperature=0.0,
            chat_template_kwargs={"enable_thinking": False},
            strategy="protected_math_constrained_recovery",
            benchmark_routing={},
            draft_max_tokens=256,
            critique_max_tokens=192,
            second_solve_max_tokens=32,
            option_evidence_max_tokens=96,
            verifier_max_tokens=8,
            normalize_bare_choice=False,
            fallback_to_protected_on_parse_failure=False,
            min_task_groups=1,
            datasets=(),
        )
        reply, stages = _run_protected_math_constrained_recovery_case(
            case,
            manifest,
            ModelConfig(name="test"),
            {600: direct_client, 32: recovery_client},
        )
        self.assertEqual(reply.content, "FINAL: 12")
        extra = recovery_client.calls[0]["extra_body"]
        self.assertEqual(
            extra,
            {
                "structured_outputs": {
                    "regex": r"FINAL: [-+]?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)"
                }
            },
        )
        self.assertEqual(
            stages["conditional_recovery"]["structured_outputs"],
            extra["structured_outputs"],
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
                benchmark_routing={},
                draft_max_tokens=384,
                critique_max_tokens=192,
                second_solve_max_tokens=384,
                option_evidence_max_tokens=96,
                verifier_max_tokens=32,
                normalize_bare_choice=False,
                fallback_to_protected_on_parse_failure=False,
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

    def test_baseline_manifest_selects_explicit_source_indices(self):
        with tempfile.TemporaryDirectory() as directory:
            from datasets import Dataset

            root = Path(directory)
            path = root / "gsm8k.parquet"
            Dataset.from_list(
                [
                    {
                        "question": f"What is {index} plus one?",
                        "answer": f"#### {index + 1}",
                    }
                    for index in range(5)
                ]
            ).to_parquet(str(path))
            from nano_harness.baseline import sha256_file

            manifest = SuiteManifest(
                schema_version="nano_harness_baseline_suite_v1",
                suite_id="indices-test",
                selection_seed="fixed",
                system_prompt="answer",
                max_tokens=8,
                temperature=0.0,
                chat_template_kwargs={"enable_thinking": False},
                strategy="direct",
                benchmark_routing={},
                draft_max_tokens=384,
                critique_max_tokens=192,
                second_solve_max_tokens=384,
                option_evidence_max_tokens=96,
                verifier_max_tokens=32,
                normalize_bare_choice=False,
                fallback_to_protected_on_parse_failure=False,
                min_task_groups=1,
                datasets=(
                    DatasetSpec(
                        name="gsm8k",
                        path="gsm8k.parquet",
                        sha256=sha256_file(path),
                        scorer="numeric_exact",
                        limit=3,
                        indices=(4, 1, 3),
                    ),
                ),
            )
            cases = load_cases(manifest, root)
            self.assertEqual(
                [case.source_index for case in cases],
                [4, 1, 3],
            )

    def test_baseline_manifest_rejects_invalid_explicit_indices(self):
        with tempfile.TemporaryDirectory() as directory:
            import yaml

            path = Path(directory) / "suite.yaml"
            raw = {
                "schema_version": "nano_harness_baseline_suite_v1",
                "suite_id": "indices-test",
                "selection_seed": "fixed",
                "system_prompt": "answer",
                "max_tokens": 8,
                "temperature": 0.0,
                "chat_template_kwargs": {"enable_thinking": False},
                "min_task_groups": 1,
                "datasets": [
                    {
                        "name": "gsm8k",
                        "path": "gsm8k.parquet",
                        "sha256": "0" * 64,
                        "scorer": "numeric_exact",
                        "limit": 2,
                        "indices": [1, 1],
                    }
                ],
            }
            path.write_text(yaml.safe_dump(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unique"):
                load_manifest(path)

    def test_openai_client_merges_extra_body_with_chat_template_kwargs(self):
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
            reply = client.complete(
                [{"role": "user", "content": "answer"}],
                extra_body={
                    "structured_outputs": {"regex": r"FINAL: [A-D]"}
                },
            )

        self.assertEqual(reply.content, "FINAL: A")
        kwargs = openai.return_value.chat.completions.create.call_args.kwargs
        self.assertEqual(
            kwargs["extra_body"],
            {
                "chat_template_kwargs": {"enable_thinking": False},
                "structured_outputs": {"regex": r"FINAL: [A-D]"},
            },
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

    def test_mcnemar_exact_is_stable_for_large_discordant_counts(self):
        self.assertAlmostEqual(_mcnemar_exact_p(4, 0), 0.125)
        self.assertAlmostEqual(_mcnemar_exact_p(5, 1), 0.21875)
        self.assertAlmostEqual(
            _mcnemar_exact_p(2076, 869),
            1.2417186326741173e-112,
            delta=5e-124,
        )
        self.assertAlmostEqual(
            _mcnemar_exact_p(2128, 951),
            3.299261613346318e-102,
            delta=5e-113,
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
