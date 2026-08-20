from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from nano_harness.client import ScriptedClient
from nano_harness.semantic_model_router import (
    ALL_FAMILIES,
    NEGATIVE_FAMILIES,
    POSITIVE_FAMILIES,
    ROUTE_REGEX,
    _candidate_row,
    build_cases,
    load_config,
    parent_config,
    parse_route,
)
from nano_harness.semantic_skill_execution import (
    load_config as load_mechanism_config,
)
from nano_harness.types import ModelReply
from scripts.preregister_semantic_model_router_v1 import (
    build_receipt,
    render_markdown,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/harness/qwen35_semantic_model_router_v1.json"
PUBLIC_RESULT = (
    ROOT / "docs/results/qwen35_semantic_model_router_v1.public.json"
)


class SemanticModelRouterTests(unittest.TestCase):
    def test_config_freezes_balanced_surface_and_closed_boundaries(self):
        config = load_config(CONFIG)
        cases = build_cases(config)
        self.assertEqual(len(cases), 256)
        self.assertEqual(
            {
                family: sum(case["family"] == family for case in cases)
                for family in ALL_FAMILIES
            },
            {family: 64 for family in ALL_FAMILIES},
        )
        self.assertEqual(sum(case["positive"] for case in cases), 128)
        self.assertEqual(sum(not case["positive"] for case in cases), 128)
        self.assertEqual(config.router_structured_output_regex, ROUTE_REGEX)
        self.assertEqual(config.router_max_tokens, 16)
        self.assertFalse(config.execution_boundary["model_generation_started"])
        self.assertFalse(config.execution_boundary["evaluation_started"])
        self.assertFalse(config.execution_boundary["benchmark_accessed"])

    def test_positive_prompts_do_not_contain_old_exact_markers(self):
        config = load_config(CONFIG)
        mechanism = load_mechanism_config(config.mechanism_config_path)
        markers = [
            marker
            for family_markers in mechanism.route_markers.values()
            for marker in family_markers
        ]
        for case in build_cases(config):
            if not case["positive"]:
                continue
            with self.subTest(case_id=case["case_id"]):
                self.assertFalse(any(marker in case["prompt"] for marker in markers))

    def test_route_parser_is_enum_constrained(self):
        self.assertEqual(parse_route("ROUTE: implicit_scale_total"), "implicit_scale_total")
        self.assertEqual(
            parse_route("ROUTE: first_strict_profit_period"),
            "first_strict_profit_period",
        )
        self.assertEqual(parse_route("ROUTE: NONE"), "NONE")
        self.assertIsNone(parse_route("implicit_scale_total"))
        self.assertIsNone(parse_route("ROUTE: box_total"))

    def test_positive_route_executes_untouched_original_prompt(self):
        config = load_config(CONFIG)
        mechanism = load_mechanism_config(config.mechanism_config_path)
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
        router = ScriptedClient(
            [ModelReply(content="ROUTE: implicit_scale_total")]
        )
        planner = ScriptedClient([ModelReply(content=plan)])
        final = ScriptedClient(
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
        candidate, receipt = _candidate_row(
            case,
            direct,
            router,
            planner,
            final,
            config,
            mechanism,
            parent,
        )
        self.assertTrue(candidate["correct"])
        self.assertTrue(receipt["router"]["correct"])
        self.assertEqual(receipt["exposed_tools"], ["implicit_scale_total"])
        self.assertEqual(
            planner.calls[0]["messages"][1]["content"],
            case["prompt"],
        )

    def test_negative_none_preserves_direct_without_tool_calls(self):
        config = load_config(CONFIG)
        mechanism = load_mechanism_config(config.mechanism_config_path)
        parent = parent_config(config)
        case = next(
            row
            for row in build_cases(config)
            if row["family"] in NEGATIVE_FAMILIES
        )
        router = ScriptedClient([ModelReply(content="ROUTE: NONE")])
        planner = ScriptedClient([])
        final = ScriptedClient([])
        direct = {
            "case_id": case["case_id"],
            "family": case["family"],
            "model": parent.four_b_model,
            "route": "direct",
            "output": "FINAL: 42",
            "prediction": 42,
            "parseable": True,
            "correct": False,
            "usage": {},
            "latency_seconds": 0.0,
        }
        candidate, receipt = _candidate_row(
            case,
            direct,
            router,
            planner,
            final,
            config,
            mechanism,
            parent,
        )
        self.assertEqual(candidate["output"], direct["output"])
        self.assertEqual(candidate["prediction"], direct["prediction"])
        self.assertEqual(candidate["correct"], direct["correct"])
        self.assertEqual(candidate["route"], "direct_preserve_after_none")
        self.assertEqual(planner.calls, [])
        self.assertEqual(final.calls, [])
        self.assertTrue(receipt["router"]["correct"])

    def test_negative_false_route_fails_closed_on_source_schema(self):
        config = load_config(CONFIG)
        mechanism = load_mechanism_config(config.mechanism_config_path)
        parent = parent_config(config)
        case = next(
            row
            for row in build_cases(config)
            if row["family"] == "box_total"
        )
        router = ScriptedClient(
            [ModelReply(content="ROUTE: implicit_scale_total")]
        )
        invalid_plan = (
            'TOOL: implicit_scale_total {"rows":1,"columns":2,'
            '"extra":3,"scale_word":"double"}'
        )
        planner = ScriptedClient(
            [ModelReply(content=invalid_plan), ModelReply(content=invalid_plan)]
        )
        final = ScriptedClient([])
        direct = {
            "case_id": case["case_id"],
            "family": case["family"],
            "model": parent.four_b_model,
            "route": "direct",
            "output": "FINAL: 42",
            "prediction": 42,
            "parseable": True,
            "correct": False,
            "usage": {},
            "latency_seconds": 0.0,
        }
        candidate, receipt = _candidate_row(
            case,
            direct,
            router,
            planner,
            final,
            config,
            mechanism,
            parent,
        )
        self.assertEqual(candidate["output"], direct["output"])
        self.assertEqual(candidate["route"], "direct_fallback_after_invalid_plan")
        self.assertTrue(receipt["fallback_used"])
        self.assertFalse(receipt["router"]["correct"])
        self.assertEqual(final.calls, [])

    def test_config_rejects_router_or_gate_changes(self):
        raw = json.loads(CONFIG.read_text(encoding="utf-8"))
        mutations = (
            ("router_max_tokens", 32, "router_max_tokens"),
            ("router_structured_output_regex", ".*", "router_structured_output_regex"),
            ("negative_cases_per_family", 32, "negative_cases_per_family"),
            ("applicability_report_sha256", "0" * 64, "applicability_report_sha256"),
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
        self.assertEqual(first["freshness"]["exact_marker_occurrences"], 0)
        self.assertEqual(first["surface"]["positive_cases"], 128)
        self.assertEqual(first["surface"]["negative_cases"], 128)
        self.assertFalse(
            first["execution_boundary"]["model_generation_started"]
        )
        self.assertFalse(first["execution_boundary"]["benchmark_accessed"])
        markdown = render_markdown(first)
        self.assertIn("128 unsupported", markdown)
        self.assertIn("false positive 0", markdown)
        self.assertIn("question-only real router scan", markdown)

    def test_public_result_rejects_recall_failure(self):
        report = json.loads(PUBLIC_RESULT.read_text(encoding="utf-8"))
        self.assertEqual(
            report["routing"],
            {
                "cases": 256,
                "router_correct": 192,
                "positive_cases": 128,
                "positive_route_correct": 64,
                "negative_cases": 128,
                "negative_none_correct": 128,
                "negative_false_positive_routes": 0,
                "verified_executions": 64,
                "fallbacks": 0,
            },
        )
        self.assertEqual(
            report["confusion"],
            [
                {
                    "expected_route": "NONE",
                    "selected_route": "NONE",
                    "cases": 128,
                },
                {
                    "expected_route": "first_strict_profit_period",
                    "selected_route": "first_strict_profit_period",
                    "cases": 64,
                },
                {
                    "expected_route": "implicit_scale_total",
                    "selected_route": "NONE",
                    "cases": 64,
                },
            ],
        )
        decision = report["decision"]
        self.assertFalse(decision["router_admitted"])
        self.assertTrue(decision["router_precision_direction_supported"])
        self.assertFalse(decision["router_recall_supported"])
        self.assertFalse(
            decision["real_question_model_scan_preregistration_allowed"]
        )
        self.assertFalse(decision["benchmark_generation_allowed"])
        self.assertFalse(decision["canary_rerun_allowed"])
        self.assertFalse(decision["independent_holdout_allowed"])
        self.assertFalse(decision["training_allowed"])
        self.assertFalse(
            decision["further_tuning_or_rerun_on_observed_cases_allowed"]
        )
        self.assertFalse(decision["gates"]["positive_route_recall_128"])
        self.assertTrue(decision["gates"]["negative_none_correct_128"])
        self.assertTrue(
            decision["gates"]["negative_false_positive_routes_zero"]
        )


if __name__ == "__main__":
    unittest.main()
