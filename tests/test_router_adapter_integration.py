from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nano_harness.baseline import sha256_file
from nano_harness.client import ScriptedClient
from nano_harness.router_adapter_integration import (
    ROUTER_SYSTEM,
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
from scripts.preregister_router_adapter_integration_v1 import (
    build_receipt,
    render_markdown,
)
from scripts.render_router_adapter_integration_v1 import admission_gates
from scripts.render_router_adapter_service_v1 import build_receipt as build_service


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/harness/qwen35_router_adapter_integration_v1.json"
PUBLIC_RESULT = (
    ROOT / "docs/results/qwen35_router_adapter_integration_v1.public.json"
)


class RouterAdapterIntegrationTests(unittest.TestCase):
    def test_config_freezes_adapter_surface_and_closed_boundaries(self):
        config = load_config(CONFIG)
        cases = build_cases(config)
        self.assertEqual(len(cases), 128)
        self.assertEqual(
            {
                family: sum(case["family"] == family for case in cases)
                for family in (
                    "implicit_scale_total",
                    "first_strict_profit_period",
                    "box_total",
                    "remaining_stock",
                )
            },
            {
                "implicit_scale_total": 32,
                "first_strict_profit_period": 32,
                "box_total": 32,
                "remaining_stock": 32,
            },
        )
        self.assertEqual(config.route_structured_output_regex, r"FINAL: [A-C]")
        self.assertEqual(config.route_max_tokens, 8)
        self.assertEqual(
            config.adapter_tree_sha256,
            "48d72666cf269f64825791801c71aad5ae1d9e97cbc70574c216b12974e46e63",
        )
        self.assertFalse(
            config.execution_boundary["adapter_service_started"]
        )
        self.assertFalse(
            config.execution_boundary["model_generation_started"]
        )
        self.assertFalse(config.execution_boundary["benchmark_accessed"])

    def test_training_prompt_contract_is_unchanged(self):
        self.assertEqual(
            ROUTER_SYSTEM,
            "Classify the task for a semantic tool router. Return exactly one "
            "line: FINAL: A for implicit rectangular scale totals, FINAL: B "
            "for first strictly profitable whole periods, or FINAL: C for "
            "every unsupported task.",
        )

    def test_route_parser_is_strict(self):
        self.assertEqual(parse_route("FINAL: A"), "A")
        self.assertEqual(parse_route("FINAL: B"), "B")
        self.assertEqual(parse_route("FINAL: C"), "C")
        self.assertIsNone(parse_route("A"))
        self.assertIsNone(parse_route("FINAL: NONE"))

    def test_positive_route_uses_adapter_then_base_typed_executor(self):
        config = load_config(CONFIG)
        mechanism = load_mechanism_config(config.mechanism_config_path)
        _, parent = parent_config(config)
        case = next(
            case
            for case in build_cases(config)
            if case["family"] == "implicit_scale_total"
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
        plan = (
            "TOOL: implicit_scale_total "
            + json.dumps(case["source_facts"], separators=(",", ":"))
        )
        router = ScriptedClient([ModelReply(content="FINAL: A")])
        planner = ScriptedClient([ModelReply(content=plan)])
        final = ScriptedClient(
            [ModelReply(content=f"FINAL: {case['expected']}")]
        )
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
        self.assertEqual(
            candidate["model"], f"{parent.four_b_model}+router-adapter-v1"
        )
        self.assertTrue(receipt["router"]["correct"])
        self.assertEqual(receipt["router"]["model"], config.served_adapter_name)
        self.assertEqual(receipt["exposed_tools"], ["implicit_scale_total"])
        self.assertEqual(
            planner.calls[0]["messages"][1]["content"], case["prompt"]
        )

    def test_unsupported_c_preserves_direct_without_tool_calls(self):
        config = load_config(CONFIG)
        mechanism = load_mechanism_config(config.mechanism_config_path)
        _, parent = parent_config(config)
        case = next(
            case
            for case in build_cases(config)
            if case["family"] == "box_total"
        )
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
        planner = ScriptedClient([])
        final = ScriptedClient([])
        candidate, receipt = _candidate_row(
            case,
            direct,
            ScriptedClient([ModelReply(content="FINAL: C")]),
            planner,
            final,
            config,
            mechanism,
            parent,
        )
        for field in ("output", "prediction", "parseable", "correct"):
            self.assertEqual(candidate[field], direct[field])
        self.assertEqual(candidate["route"], "direct_preserve_after_router_c")
        self.assertEqual(planner.calls, [])
        self.assertEqual(final.calls, [])
        self.assertTrue(receipt["router"]["correct"])

    def test_config_rejects_adapter_route_or_source_changes(self):
        raw = json.loads(CONFIG.read_text(encoding="utf-8"))
        mutations = (
            ("route_max_tokens", 16, "route_max_tokens"),
            ("route_structured_output_regex", ".*", "route_structured_output_regex"),
            ("case_seed", 1, "case_seed"),
            ("adapter_tree_sha256", "0" * 64, "adapter_tree_sha256"),
            (
                "router_training_data_sha256",
                "0" * 64,
                "router_training_data_sha256",
            ),
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

    def test_preregister_is_deterministic_history_disjoint_and_closed(self):
        first = build_receipt()
        second = build_receipt()
        self.assertEqual(first, second)
        self.assertEqual(first["case_contract"]["case_count"], 128)
        self.assertEqual(
            first["freshness"]["history_disjoint_prompt_overlap"],
            {
                "router_training_prompts": 0,
                "prior_multiclass_prompts": 0,
                "prior_binary_prompts": 0,
            },
        )
        self.assertTrue(
            first["router"]["system_prompt_matches_training_contract"]
        )
        self.assertFalse(
            first["execution_boundary"]["adapter_service_started"]
        )
        self.assertFalse(
            first["execution_boundary"]["model_generation_started"]
        )
        markdown = render_markdown(first)
        self.assertIn("128 cases", markdown)
        self.assertIn("all zero", markdown)
        self.assertIn("question-only scan", markdown)

    def test_service_receipt_requires_base_and_adapter_health(self):
        config = load_config(CONFIG)
        fake = {
            "data": [
                {
                    "id": "qwen3.5-4b",
                    "owned_by": "vllm",
                    "root": "../../../models/Qwen3.5-4B",
                    "parent": None,
                    "max_model_len": 4096,
                },
                {
                    "id": config.served_adapter_name,
                    "owned_by": "vllm",
                    "root": config.adapter_path,
                    "parent": "qwen3.5-4b",
                    "max_model_len": 4096,
                },
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            health = Path(directory) / "models.json"
            health.write_text(json.dumps(fake), encoding="utf-8")
            with patch(
                "scripts.render_router_adapter_service_v1.RAW_HEALTH",
                health,
            ), patch(
                "scripts.render_router_adapter_service_v1."
                "committed_preregister_sha256",
                return_value=sha256_file(
                    ROOT
                    / "docs/experiments/"
                    "qwen35_router_adapter_integration_v1.preregister.json"
                ),
            ):
                receipt = build_service()
        self.assertTrue(receipt["healthy"])
        self.assertFalse(receipt["generation_started"])
        self.assertEqual(
            set(receipt["models"]),
            {"qwen3.5-4b", config.served_adapter_name},
        )

    def test_admission_gates_require_zero_loss_and_all_routes(self):
        families = (
            "implicit_scale_total",
            "first_strict_profit_period",
            "box_total",
            "remaining_stock",
        )
        arms = {
            name: {
                "cases": 128,
                "parseable": 128,
                "by_family": {
                    family: {"correct": 32} for family in families
                },
            }
            for name in (
                "four_b_direct",
                "nine_b_direct",
                "four_b_router_adapter",
            )
        }
        receipts = {}
        for index in range(128):
            label = "A" if index < 32 else "B" if index < 64 else "C"
            receipts[str(index)] = {
                "router": {
                    "label": label,
                    "expected_label": label,
                },
                "feedback_result_match": index < 64,
            }
        raw = {
            "arms": arms,
            "routing": {
                "negative_c_correct": 64,
                "negative_false_positive_routes": 0,
                "verified_executions": 64,
                "fallbacks": 0,
            },
            "receipts": receipts,
            "candidate_rows": [],
            "four_b_rows": [],
        }
        comparison = {
            "candidate_accuracy": 1.0,
            "baseline_accuracy": 0.5,
            "paired_bootstrap_95_ci": [0.25, 0.75],
            "mcnemar_exact_p": 0.001,
            "paired_counts": {"candidate_only": 64, "baseline_only": 0},
        }
        with patch(
            "scripts.render_router_adapter_integration_v1.build_cases",
            return_value=[],
        ):
            gates = admission_gates(raw, comparison, comparison)
        self.assertTrue(all(gates.values()))
        losing = copy.deepcopy(comparison)
        losing["paired_counts"]["baseline_only"] = 1
        with patch(
            "scripts.render_router_adapter_integration_v1.build_cases",
            return_value=[],
        ):
            gates = admission_gates(raw, losing, comparison)
        self.assertFalse(gates["candidate_vs_four_b_maximum_losses"])

    def test_public_result_rejects_unsupported_route_collapse(self):
        report = json.loads(PUBLIC_RESULT.read_text(encoding="utf-8"))
        self.assertEqual(
            report["routing"],
            {
                "cases": 128,
                "correct": 64,
                "positive_cases": 64,
                "positive_correct": 64,
                "negative_cases": 64,
                "negative_c_correct": 0,
                "negative_false_positive_routes": 64,
                "verified_executions": 64,
                "fallbacks": 64,
            },
        )
        self.assertEqual(
            report["confusion"],
            [
                {
                    "expected_label": "A",
                    "selected_label": "A",
                    "cases": 32,
                },
                {
                    "expected_label": "B",
                    "selected_label": "B",
                    "cases": 32,
                },
                {
                    "expected_label": "C",
                    "selected_label": "A",
                    "cases": 64,
                },
            ],
        )
        decision = report["decision"]
        self.assertFalse(decision["adapter_integration_admitted"])
        self.assertFalse(
            decision["question_only_scan_preregistration_allowed"]
        )
        self.assertFalse(decision["question_only_scan_generation_allowed"])
        self.assertFalse(decision["benchmark_generation_allowed"])
        self.assertFalse(decision["training_or_rl_allowed"])
        self.assertTrue(decision["gates"]["router_a_recall_32"])
        self.assertTrue(decision["gates"]["router_b_recall_32"])
        self.assertFalse(decision["gates"]["router_c_precision_64"])
        self.assertFalse(
            decision["gates"]["negative_false_positive_routes_zero"]
        )
        self.assertFalse(decision["gates"]["fallbacks_zero"])
        self.assertTrue(
            decision["gates"]["candidate_vs_nine_b_significant"]
        )


if __name__ == "__main__":
    unittest.main()
