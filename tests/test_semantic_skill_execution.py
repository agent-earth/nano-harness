from __future__ import annotations

import copy
import json
import re
import tempfile
import unittest
from pathlib import Path

from nano_harness.client import ScriptedClient
from nano_harness.semantic_skill_execution import (
    FAMILIES,
    ROUTE_MARKERS,
    SKILL_PROMPTS,
    TOOL_REGEX_BY_FAMILY,
    _harness_row,
    build_cases,
    execute_semantic_tool,
    load_config,
    parent_config,
    parse_and_execute_plan,
    route_prompt,
)
from nano_harness.types import ModelReply
from scripts.preregister_semantic_skill_execution_v1 import (
    build_receipt,
    render_markdown,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/harness/qwen35_semantic_skill_execution_v1.json"
PUBLIC_RESULT = (
    ROOT / "docs/results/qwen35_semantic_skill_execution_v1.public.json"
)


class SemanticSkillExecutionTests(unittest.TestCase):
    def test_config_freezes_fresh_surface_and_closed_boundaries(self):
        config = load_config(CONFIG)
        parent = parent_config(config)
        cases = build_cases(config)
        self.assertEqual(len(cases), 256)
        self.assertEqual(config.cases_per_family, 128)
        self.assertEqual(parent.value_offset, 120000)
        self.assertEqual(
            {family: sum(row["family"] == family for row in cases)
             for family in FAMILIES},
            {family: 128 for family in FAMILIES},
        )
        self.assertFalse(config.execution_boundary["model_generation_started"])
        self.assertFalse(config.execution_boundary["evaluation_started"])
        self.assertFalse(config.execution_boundary["canary_accessed"])
        self.assertFalse(config.execution_boundary["benchmark_accessed"])
        self.assertFalse(config.execution_boundary["holdout_accessed"])

    def test_semantic_tools_encode_implicit_and_strict_boundaries(self):
        self.assertEqual(
            execute_semantic_tool(
                "implicit_scale_total",
                {
                    "rows": 6,
                    "columns": 6,
                    "extra": 20,
                    "scale_word": "double",
                },
            ),
            92,
        )
        self.assertEqual(
            execute_semantic_tool(
                "implicit_scale_total",
                {
                    "rows": 6,
                    "columns": 6,
                    "extra": 20,
                    "scale_word": "triple",
                },
            ),
            128,
        )
        self.assertEqual(
            execute_semantic_tool(
                "first_strict_profit_period",
                {
                    "setup_cost": 90,
                    "units_per_period": 7,
                    "price_per_unit": 15,
                    "recurring_cost": 30,
                },
            ),
            2,
        )
        self.assertEqual(
            execute_semantic_tool(
                "first_strict_profit_period",
                {
                    "setup_cost": 90,
                    "units_per_period": 7,
                    "price_per_unit": 15,
                    "recurring_cost": 98,
                },
            ),
            13,
        )

    def test_router_uses_prompt_markers_not_case_metadata(self):
        prompts = {
            "implicit_scale_total": (
                "A planner requests extra=20 more than double the number of "
                "slots in rows=6 by columns=6."
            ),
            "first_strict_profit_period": (
                "setup_cost=90, units_per_period=7, price_per_unit=15, "
                "recurring_cost=98. Find the first whole period when "
                "cumulative profit is strictly positive."
            ),
        }
        for family, prompt in prompts.items():
            with self.subTest(family=family):
                route = route_prompt(prompt)
                self.assertTrue(route["routed"])
                self.assertEqual(route["family"], family)
                self.assertFalse(route["router_uses_case_metadata"])
        missing = route_prompt("Compute an unrelated value.")
        self.assertFalse(missing["routed"])
        self.assertEqual(missing["reason"], "route_missing")
        ambiguous = route_prompt(
            "extra more than double the number of slots; find the first whole "
            "period when cumulative profit is strictly positive."
        )
        self.assertFalse(ambiguous["routed"])
        self.assertEqual(ambiguous["reason"], "route_ambiguous")

    def test_plan_regexes_and_prompts_expose_one_semantic_skill(self):
        self.assertEqual(set(TOOL_REGEX_BY_FAMILY), set(FAMILIES))
        self.assertEqual(set(SKILL_PROMPTS), set(FAMILIES))
        self.assertEqual(set(ROUTE_MARKERS), set(FAMILIES))
        examples = {
            "implicit_scale_total": (
                'TOOL: implicit_scale_total {"rows":6,"columns":6,'
                '"extra":20,"scale_word":"double"}'
            ),
            "first_strict_profit_period": (
                'TOOL: first_strict_profit_period {"setup_cost":90,'
                '"units_per_period":7,"price_per_unit":15,'
                '"recurring_cost":98}'
            ),
        }
        for family, text in examples.items():
            with self.subTest(family=family):
                self.assertIsNotNone(
                    re.fullmatch(TOOL_REGEX_BY_FAMILY[family], text)
                )
                self.assertIn(family.replace("_", "-"), SKILL_PROMPTS[family])
                self.assertNotIn("expected", SKILL_PROMPTS[family])

    def test_plan_execution_requires_prompt_route_and_exact_source_facts(self):
        source = {
            "rows": 6,
            "columns": 6,
            "extra": 20,
            "scale_word": "double",
        }
        route = route_prompt(
            "rows=6, columns=6, extra=20 more than double the number of slots."
        )
        text = (
            'TOOL: implicit_scale_total {"rows":6,"columns":6,'
            '"extra":20,"scale_word":"double"}'
        )
        receipt = parse_and_execute_plan(
            text,
            route=route,
            source_facts=source,
        )
        self.assertTrue(receipt["executed"])
        self.assertEqual(receipt["result"], 92)
        self.assertFalse(receipt["router_uses_case_metadata"])
        self.assertFalse(receipt["executor_uses_expected_answer"])

        altered = text.replace('"double"', '"triple"')
        rejected = parse_and_execute_plan(
            altered,
            route=route,
            source_facts=source,
        )
        self.assertFalse(rejected["executed"])
        self.assertEqual(rejected["reason"], "source_facts_mismatch")

    def test_harness_executes_each_family_with_verified_feedback(self):
        config = load_config(CONFIG)
        parent = parent_config(config)
        cases = build_cases(config)
        for family in FAMILIES:
            with self.subTest(family=family):
                case = next(row for row in cases if row["family"] == family)
                plan = (
                    f"TOOL: {family} "
                    + json.dumps(case["source_facts"], separators=(",", ":"))
                )
                plan_client = ScriptedClient([ModelReply(content=plan)])
                final_client = ScriptedClient(
                    [ModelReply(content=f"FINAL: {case['expected']}")]
                )
                direct = {
                    "case_id": case["case_id"],
                    "family": case["family"],
                    "model": parent.four_b_model,
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
                    parent,
                )
                self.assertTrue(row["correct"])
                self.assertEqual(
                    row["route"],
                    "prompt_routed_verified_semantic_feedback",
                )
                self.assertEqual(receipt["exposed_tools"], [family])
                self.assertTrue(receipt["receipt"]["executed"])
                self.assertTrue(receipt["feedback_result_match"])
                self.assertEqual(
                    plan_client.calls[0]["extra_body"]["structured_outputs"][
                        "regex"
                    ],
                    TOOL_REGEX_BY_FAMILY[family],
                )

    def test_feedback_mismatch_falls_back_to_direct(self):
        config = load_config(CONFIG)
        parent = parent_config(config)
        case = next(
            row
            for row in build_cases(config)
            if row["family"] == "implicit_scale_total"
        )
        plan = (
            "TOOL: implicit_scale_total "
            + json.dumps(case["source_facts"], separators=(",", ":"))
        )
        plan_client = ScriptedClient([ModelReply(content=plan)])
        final_client = ScriptedClient([ModelReply(content="FINAL: 0")])
        direct = {
            "case_id": case["case_id"],
            "family": case["family"],
            "model": parent.four_b_model,
            "route": "direct",
            "output": "FINAL: 1",
            "prediction": 1,
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
            parent,
        )
        self.assertEqual(row["output"], direct["output"])
        self.assertEqual(
            row["route"],
            "direct_fallback_after_feedback_mismatch",
        )
        self.assertTrue(receipt["fallback_used"])
        self.assertFalse(receipt["feedback_result_match"])

    def test_config_rejects_route_schema_budget_or_boundary_change(self):
        raw = json.loads(CONFIG.read_text(encoding="utf-8"))
        mutations = (
            ("value_offset", 90000, "value_offset"),
            ("skill_router", "case_metadata", "skill_router"),
            ("plan_retry_limit", 2, "plan_retry_limit"),
            ("canary_rejection_sha256", "0" * 64, "canary_rejection_sha256"),
        )
        for key, value, error in mutations:
            with self.subTest(key=key):
                altered = copy.deepcopy(raw)
                altered[key] = value
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "config.json"
                    path.write_text(json.dumps(altered), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, error):
                        load_config(path)

    def test_preregister_is_deterministic_and_generation_closed(self):
        first = build_receipt()
        second = build_receipt()
        self.assertEqual(first, second)
        self.assertEqual(first["case_contract"]["case_count"], 256)
        self.assertTrue(
            all(
                value == 0
                for value in first["freshness"][
                    "benchmark_prompt_overlap"
                ].values()
            )
        )
        self.assertFalse(
            first["execution_boundary"]["model_generation_started"]
        )
        self.assertFalse(
            first["execution_boundary"]["evaluation_started"]
        )
        self.assertFalse(first["execution_boundary"]["canary_accessed"])
        self.assertFalse(first["execution_boundary"]["benchmark_accessed"])
        markdown = render_markdown(first)
        self.assertIn("Typed Semantic Skill", markdown)
        self.assertIn("prompt marker", markdown)
        self.assertIn("canary accessed：false", markdown)

    def test_public_result_admits_only_fresh_local_replication(self):
        report = json.loads(PUBLIC_RESULT.read_text(encoding="utf-8"))
        self.assertEqual(
            report["arms"]["four_b_semantic_skills"]["correct"],
            256,
        )
        self.assertEqual(report["arms"]["four_b_direct"]["correct"], 0)
        self.assertEqual(report["arms"]["nine_b_direct"]["correct"], 0)
        self.assertEqual(
            report["routing"],
            {
                "prompt_routes": 256,
                "single_tool_exposures": 256,
                "verified_executions": 256,
                "plan_retries": 0,
                "fallbacks": 0,
                "final_feedback_calls": 256,
                "feedback_result_matches": 256,
            },
        )
        for name in ("harness_vs_four_b", "harness_vs_nine_b"):
            comparison = report["comparisons"][name]
            self.assertEqual(comparison["delta"], 1.0)
            self.assertEqual(
                comparison["paired_bootstrap_95_ci"],
                [1.0, 1.0],
            )
            self.assertEqual(
                comparison["paired_counts"]["candidate_only"],
                256,
            )
            self.assertEqual(
                comparison["paired_counts"]["baseline_only"],
                0,
            )
        decision = report["decision"]
        self.assertTrue(decision["local_semantic_skill_admitted"])
        self.assertTrue(
            decision["fresh_local_replication_preregistration_allowed"]
        )
        self.assertFalse(
            decision["fresh_local_replication_generation_allowed"]
        )
        self.assertFalse(decision["canary_allowed"])
        self.assertFalse(decision["benchmark_allowed"])
        self.assertFalse(decision["independent_holdout_allowed"])
        self.assertFalse(decision["training_allowed"])
        self.assertFalse(
            decision["further_tuning_or_rerun_on_observed_cases_allowed"]
        )
        self.assertEqual(
            report["data"]["benchmark_canary_holdout_rows_or_outputs"],
            0,
        )
        self.assertEqual(report["data"]["training_eligible_cases"], 0)
        self.assertEqual(len(report["interrupted_preflights"]), 2)
        self.assertTrue(
            all(
                row["model_generation_started"] is False
                and row["result_artifact_created"] is False
                for row in report["interrupted_preflights"]
            )
        )


if __name__ == "__main__":
    unittest.main()
