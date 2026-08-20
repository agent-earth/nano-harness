from __future__ import annotations

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from nano_harness.client import ScriptedClient
from nano_harness.types import ModelReply
from nano_harness.verified_tool_execution import (
    DIRECT_REGEX,
    FAMILIES,
    PLAN_REGEX,
    _harness_row,
    build_cases,
    contamination_audit,
    execute_verified_tool,
    load_config,
    parse_and_execute_plan,
    public_case_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/harness/qwen35_verified_tool_execution_v1.json"
RENDER_PATH = ROOT / "scripts/render_verified_tool_execution_v1.py"
RENDER_SPEC = importlib.util.spec_from_file_location(
    "render_verified_tool_execution_v1",
    RENDER_PATH,
)
RENDER_MODULE = importlib.util.module_from_spec(RENDER_SPEC)
assert RENDER_SPEC.loader is not None
RENDER_SPEC.loader.exec_module(RENDER_MODULE)


class VerifiedToolExecutionTests(unittest.TestCase):
    def test_config_and_cases_are_frozen_balanced_and_deterministic(self):
        config = load_config(CONFIG)
        self.assertEqual(config.direct_structured_output_regex, DIRECT_REGEX)
        self.assertEqual(config.plan_structured_output_regex, PLAN_REGEX)
        first = build_cases(config)
        second = build_cases(config)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 256)
        counts = {}
        for row in first:
            counts[row["family"]] = counts.get(row["family"], 0) + 1
            self.assertEqual(
                execute_verified_tool(row["family"], row["source_facts"]),
                row["expected"],
            )
        self.assertEqual(counts, {family: 64 for family in FAMILIES})
        self.assertEqual(len({row["case_id"] for row in first}), 256)

    def test_public_contract_excludes_prompt_facts_and_expected(self):
        contract = public_case_contract(build_cases(load_config(CONFIG)))
        self.assertEqual(contract["case_count"], 256)
        self.assertEqual(
            set(contract["cases"][0]),
            {
                "case_id",
                "family",
                "prompt_sha256",
                "source_facts_sha256",
            },
        )

    def test_fresh_surface_has_zero_prior_and_benchmark_prompt_overlap(self):
        config = load_config(CONFIG)
        audit = contamination_audit(config, build_cases(config))
        self.assertTrue(audit["passed"])
        self.assertEqual(
            audit["prior_surface_prompt_overlap"],
            {
                "choice_capability_matrix_v1": 0,
                "choice_verifier_matrix_v2": 0,
                "choice_exact_replication_v3": 0,
            },
        )
        self.assertEqual(
            audit["benchmark_prompt_overlap"],
            {"gsm8k": 0, "mmlu": 0, "gpqa_diamond": 0},
        )
        self.assertEqual(
            audit["benchmark_rows_hashed"],
            {"gsm8k": 1319, "mmlu": 14042, "gpqa_diamond": 198},
        )

    def test_typed_tools_execute_exactly(self):
        examples = {
            "box_total": (
                {"boxes": 7, "items_per_box": 11, "loose_items": 5},
                82,
            ),
            "remaining_stock": (
                {
                    "starting_units": 100,
                    "batches_used": 4,
                    "units_per_batch": 13,
                },
                48,
            ),
            "paired_average": (
                {"first_total": 37, "second_total": 45},
                41,
            ),
            "labor_total": (
                {"hourly_rate": 23, "regular_hours": 8, "bonus": 17},
                201,
            ),
        }
        for name, (arguments, expected) in examples.items():
            with self.subTest(name=name):
                self.assertEqual(
                    execute_verified_tool(name, arguments),
                    expected,
                )

    def test_plan_requires_exact_tool_fields_types_and_source_facts(self):
        facts = {"boxes": 7, "items_per_box": 11, "loose_items": 5}
        valid = parse_and_execute_plan(
            'TOOL: box_total {"boxes":7,"items_per_box":11,"loose_items":5}',
            expected_tool="box_total",
            source_facts=facts,
        )
        self.assertTrue(valid["executed"])
        self.assertEqual(valid["result"], 82)
        for text, reason in (
            (
                'TOOL: labor_total {"boxes":7,"items_per_box":11,'
                '"loose_items":5}',
                "tool_name_mismatch",
            ),
            (
                'TOOL: box_total {"boxes":7,"items_per_box":11}',
                "argument_fields_mismatch",
            ),
            (
                'TOOL: box_total {"boxes":8,"items_per_box":11,'
                '"loose_items":5}',
                "source_facts_mismatch",
            ),
            (
                'TOOL: box_total {"boxes":7.0,"items_per_box":11,'
                '"loose_items":5}',
                "source_facts_mismatch",
            ),
        ):
            with self.subTest(reason=reason):
                receipt = parse_and_execute_plan(
                    text,
                    expected_tool="box_total",
                    source_facts=facts,
                )
                self.assertFalse(receipt["executed"])
                self.assertEqual(receipt["reason"], reason)
                self.assertFalse(receipt["executor_uses_expected_answer"])
                self.assertFalse(receipt["executor_uses_case_correctness"])

    def test_harness_retries_then_feeds_verified_result(self):
        config = load_config(CONFIG)
        case = next(
            row for row in build_cases(config) if row["family"] == "box_total"
        )
        facts = case["source_facts"]
        valid_plan = (
            "TOOL: box_total "
            + json.dumps(facts, separators=(",", ":"))
        )
        plan_client = ScriptedClient(
            [
                ModelReply(
                    content=(
                        'TOOL: box_total {"boxes":0,"items_per_box":0,'
                        '"loose_items":0}'
                    )
                ),
                ModelReply(content=valid_plan),
            ]
        )
        final_client = ScriptedClient(
            [ModelReply(content=f"FINAL: {case['expected']}")]
        )
        direct = {
            "case_id": case["case_id"],
            "family": case["family"],
            "model": config.four_b_model,
            "route": "direct",
            "output": "FINAL: 0",
            "prediction": 0,
            "parseable": True,
            "correct": False,
            "usage": {},
            "latency_seconds": 0.0,
        }
        row, receipt = _harness_row(
            case,
            direct,
            plan_client,
            final_client,
            config,
        )
        self.assertTrue(row["correct"])
        self.assertEqual(row["route"], "verified_tool_feedback")
        self.assertEqual(len(receipt["plan_attempts"]), 2)
        self.assertTrue(receipt["receipt"]["executed"])
        feedback = final_client.calls[0]["messages"][1]["content"]
        self.assertIn(f"result={case['expected']}", feedback)
        self.assertNotIn("expected=", feedback)

    def test_harness_falls_back_to_direct_after_invalid_retry(self):
        config = load_config(CONFIG)
        case = next(
            row for row in build_cases(config) if row["family"] == "labor_total"
        )
        plan_client = ScriptedClient(
            [
                ModelReply(content="TOOL: invalid {}"),
                ModelReply(content="TOOL: invalid {}"),
            ]
        )
        final_client = ScriptedClient([])
        direct = {
            "case_id": case["case_id"],
            "family": case["family"],
            "model": config.four_b_model,
            "route": "direct",
            "output": f"FINAL: {case['expected']}",
            "prediction": case["expected"],
            "parseable": True,
            "correct": True,
            "usage": {},
            "latency_seconds": 0.0,
        }
        row, receipt = _harness_row(
            case,
            direct,
            plan_client,
            final_client,
            config,
        )
        self.assertTrue(row["correct"])
        self.assertEqual(
            row["route"],
            "direct_fallback_after_invalid_plan",
        )
        self.assertTrue(receipt["fallback_used"])
        self.assertFalse(receipt["final_feedback_sent"])
        self.assertEqual(len(final_client.calls), 0)

    def test_config_rejects_posthoc_changes(self):
        raw = json.loads(CONFIG.read_text(encoding="utf-8"))
        for key, value, error in (
            ("cases_per_family", 32, "cases_per_family"),
            ("value_offset", 81000, "value_offset"),
            ("plan_retry_limit", 2, "plan_retry_limit"),
            ("plan_max_tokens", 128, "plan_max_tokens"),
            ("minimum_harness_wins", 6, "minimum_harness_wins"),
            ("maximum_harness_losses", 1, "maximum_harness_losses"),
        ):
            with self.subTest(key=key):
                altered = copy.deepcopy(raw)
                altered[key] = value
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "config.json"
                    path.write_text(json.dumps(altered), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, error):
                        load_config(path)

    def test_gate_rejects_executor_contract_failures_despite_quality_gain(self):
        by_family = {
            family: {"cases": 64, "correct": 0, "parseable": 64}
            for family in FAMILIES
        }
        raw = {
            "arms": {
                "four_b_direct": {
                    "cases": 256,
                    "correct": 21,
                    "accuracy": 21 / 256,
                    "parseable": 256,
                    "by_family": copy.deepcopy(by_family),
                },
                "nine_b_direct": {
                    "cases": 256,
                    "correct": 13,
                    "accuracy": 13 / 256,
                    "parseable": 256,
                    "by_family": copy.deepcopy(by_family),
                },
                "four_b_verified_tool": {
                    "cases": 256,
                    "correct": 192,
                    "accuracy": 192 / 256,
                    "parseable": 256,
                    "by_family": {
                        family: {
                            "cases": 64,
                            "correct": 64 if family != "labor_total" else 0,
                            "parseable": 64,
                        }
                        for family in FAMILIES
                    },
                },
            },
            "routing": {"verified_executions": 192},
        }
        raw["arms"]["four_b_direct"]["by_family"]["paired_average"][
            "correct"
        ] = 21
        raw["arms"]["nine_b_direct"]["by_family"]["paired_average"][
            "correct"
        ] = 13
        comparison = {
            "candidate_accuracy": 0.75,
            "baseline_accuracy": 0.08,
            "paired_bootstrap_95_ci": [0.5, 0.8],
            "mcnemar_exact_p": 1e-20,
            "paired_counts": {
                "candidate_only": 171,
                "baseline_only": 0,
            },
        }
        gates = RENDER_MODULE.admission_gates(
            raw,
            comparison,
            comparison,
            contract_failures=64,
        )
        self.assertTrue(
            gates["harness_vs_four_b_bootstrap_ci_lower_gt_zero"]
        )
        self.assertTrue(
            gates["harness_vs_nine_b_bootstrap_ci_lower_gt_zero"]
        )
        self.assertFalse(gates["executor_contract_failures_zero"])
        self.assertFalse(all(gates.values()))


if __name__ == "__main__":
    unittest.main()
