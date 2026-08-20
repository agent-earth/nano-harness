from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nano_harness.baseline import sha256_file
from nano_harness.client import ScriptedClient
from nano_harness.router_adapter_integration import ROUTER_SYSTEM
from nano_harness.router_adapter_integration_v2 import (
    build_cases,
    load_config,
    parent_config,
)
from nano_harness.semantic_skill_execution import (
    load_config as load_mechanism_config,
)
from nano_harness.types import ModelReply
from scripts.preregister_router_adapter_integration_v2 import (
    build_receipt,
    render_markdown,
)
from scripts.render_router_adapter_integration_v2 import admission_gates
from scripts.render_router_adapter_integration_v2_service import (
    build_receipt as build_service,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/harness/qwen35_router_adapter_integration_v2.json"
PUBLIC_RESULT = (
    ROOT / "docs/results/qwen35_router_adapter_integration_v2.public.json"
)


class RouterAdapterIntegrationV2Tests(unittest.TestCase):
    def test_config_freezes_new_surface_remap_and_no_v1_rerun(self):
        config = load_config(CONFIG)
        self.assertEqual(config.case_seed, 20260826)
        self.assertEqual(config.value_offset, 25000)
        self.assertEqual(config.cases_per_family, 32)
        self.assertEqual(
            config.adapter_tree_sha256,
            "fbaa39dcb3fcf34e9aab280308cb5a5416094c1968e4ac3a69cd739a806ecc49",
        )
        self.assertEqual(
            config.served_adapter_name, "qwen3.5-router-remapped-v1"
        )
        self.assertFalse(config.policy["integration_v1_outputs_loaded"])
        self.assertFalse(config.policy["integration_v1_rerun_allowed"])
        self.assertFalse(config.execution_boundary["integration_v1_rerun"])
        self.assertFalse(
            config.execution_boundary["model_generation_started"]
        )

    def test_cases_are_balanced_answer_tasks_without_classification_leak(self):
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
        self.assertEqual(
            {
                label: sum(case["expected_label"] == label for case in cases)
                for label in ("A", "B", "C")
            },
            {"A": 32, "B": 32, "C": 64},
        )
        for case in cases:
            with self.subTest(case_id=case["case_id"]):
                prompt = case["prompt"].casefold()
                self.assertFalse(
                    any(
                        term in prompt
                        for term in (
                            "route",
                            "router",
                            "classify",
                            "classification",
                        )
                    )
                )

    def test_positive_case_uses_remapped_router_and_base_executor(self):
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
        from nano_harness.router_adapter_integration import _candidate_row

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
            receipt["router"]["model"], "qwen3.5-router-remapped-v1"
        )
        self.assertEqual(
            receipt["router"]["adapter_sha256"],
            config.adapter_tree_sha256,
        )
        self.assertEqual(
            router.calls[0]["messages"][0]["content"], ROUTER_SYSTEM
        )
        self.assertEqual(
            planner.calls[0]["messages"][1]["content"], case["prompt"]
        )

    def test_preregister_is_deterministic_and_history_disjoint(self):
        first = build_receipt()
        second = build_receipt()
        self.assertEqual(first, second)
        self.assertEqual(first["surface"]["cases"], 128)
        self.assertEqual(
            first["surface"]["classification_instruction_occurrences"], 0
        )
        self.assertTrue(
            not any(
                first["freshness"]["history_disjoint_prompt_overlap"].values()
            )
        )
        self.assertTrue(
            not any(first["freshness"]["benchmark_prompt_overlap"].values())
        )
        self.assertTrue(
            not any(
                first["freshness"]["prior_surface_prompt_overlap"].values()
            )
        )
        self.assertFalse(first["freshness"]["integration_v1_outputs_loaded"])
        self.assertFalse(
            first["acceptance"]["integration_v1_rerun_allowed_after_pass"]
        )
        self.assertFalse(
            first["execution_boundary"]["model_generation_started"]
        )
        markdown = render_markdown(first)
        self.assertIn("does not rerun V1", markdown)
        self.assertIn("all zero", markdown)
        self.assertIn("cannot be rerun", markdown)

    def test_config_rejects_seed_adapter_or_predecessor_changes(self):
        raw = json.loads(CONFIG.read_text(encoding="utf-8"))
        mutations = (
            ("case_seed", 1, "case_seed"),
            ("value_offset", 11000, "value_offset"),
            ("adapter_tree_sha256", "0" * 64, "adapter_tree_sha256"),
            ("parity_report_sha256", "0" * 64, "parity_report_sha256"),
            (
                "integration_v1_report_sha256",
                "0" * 64,
                "integration_v1_report_sha256",
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

    def test_service_receipt_reuses_parity_service_without_v1_rerun(self):
        config = load_config(CONFIG)
        fake = {
            "data": [
                {
                    "id": config.service_models["base"],
                    "owned_by": "vllm",
                    "root": "../../../models/Qwen3.5-4B",
                    "parent": None,
                    "max_model_len": 4096,
                },
                {
                    "id": config.service_models["original_unused"],
                    "owned_by": "vllm",
                    "root": "original",
                    "parent": config.service_models["base"],
                    "max_model_len": None,
                },
                {
                    "id": config.service_models["remapped_router"],
                    "owned_by": "vllm",
                    "root": config.adapter_path,
                    "parent": config.service_models["base"],
                    "max_model_len": None,
                },
            ]
        }
        prereg = (
            ROOT
            / "docs/experiments/"
            "qwen35_router_adapter_integration_v2.preregister.json"
        )
        with tempfile.TemporaryDirectory() as directory:
            health = Path(directory) / "models.json"
            health.write_text(json.dumps(fake), encoding="utf-8")
            with patch(
                "scripts.render_router_adapter_integration_v2_service."
                "RAW_HEALTH",
                health,
            ), patch(
                "scripts.render_router_adapter_integration_v2_service."
                "committed_preregister_sha256",
                return_value=sha256_file(prereg),
            ):
                receipt = build_service()
        self.assertTrue(receipt["healthy"])
        self.assertFalse(receipt["v2_generation_started"])
        self.assertFalse(receipt["integration_v1_rerun"])
        self.assertEqual(
            receipt["remapped_adapter_sha256"],
            config.adapter_tree_sha256,
        )

    def test_admission_gates_require_all_routes_and_zero_loss(self):
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
                "four_b_router_adapter_v2",
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
            "scripts.render_router_adapter_integration_v2.build_cases",
            return_value=[],
        ):
            decision = admission_gates(raw, comparison, comparison)
        self.assertTrue(all(decision.values()))
        raw["routing"]["negative_false_positive_routes"] = 1
        with patch(
            "scripts.render_router_adapter_integration_v2.build_cases",
            return_value=[],
        ):
            decision = admission_gates(raw, comparison, comparison)
        self.assertFalse(decision["negative_false_positive_routes_zero"])

    def test_public_result_rejects_box_total_subtype_collapse(self):
        report = json.loads(PUBLIC_RESULT.read_text(encoding="utf-8"))
        self.assertEqual(
            report["routing"],
            {
                "cases": 128,
                "correct": 96,
                "positive_cases": 64,
                "positive_correct": 64,
                "negative_cases": 64,
                "negative_c_correct": 32,
                "negative_false_positive_routes": 32,
                "verified_executions": 64,
                "fallbacks": 32,
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
                    "cases": 32,
                },
                {
                    "expected_label": "C",
                    "selected_label": "C",
                    "cases": 32,
                },
            ],
        )
        self.assertEqual(
            report["arms"]["four_b_router_adapter_v2"]["by_family"],
            {
                "implicit_scale_total": {
                    "cases": 32,
                    "correct": 32,
                    "parseable": 32,
                },
                "first_strict_profit_period": {
                    "cases": 32,
                    "correct": 32,
                    "parseable": 32,
                },
                "box_total": {
                    "cases": 32,
                    "correct": 0,
                    "parseable": 32,
                },
                "remaining_stock": {
                    "cases": 32,
                    "correct": 0,
                    "parseable": 32,
                },
            },
        )
        self.assertEqual(
            report["routing_by_family"],
            {
                "implicit_scale_total": {
                    "cases": 32,
                    "route_correct": 32,
                    "selected_labels": {"A": 32},
                    "verified_executions": 32,
                    "fallbacks": 0,
                },
                "first_strict_profit_period": {
                    "cases": 32,
                    "route_correct": 32,
                    "selected_labels": {"B": 32},
                    "verified_executions": 32,
                    "fallbacks": 0,
                },
                "box_total": {
                    "cases": 32,
                    "route_correct": 0,
                    "selected_labels": {"A": 32},
                    "verified_executions": 0,
                    "fallbacks": 32,
                },
                "remaining_stock": {
                    "cases": 32,
                    "route_correct": 32,
                    "selected_labels": {"C": 32},
                    "verified_executions": 0,
                    "fallbacks": 0,
                },
            },
        )
        decision = report["decision"]
        self.assertFalse(decision["adapter_integration_v2_admitted"])
        self.assertFalse(
            decision["question_only_scan_preregistration_allowed"]
        )
        self.assertFalse(decision["integration_v1_rerun_allowed"])
        self.assertFalse(decision["integration_v2_rerun_allowed"])
        self.assertFalse(decision["benchmark_generation_allowed"])
        self.assertFalse(decision["training_or_rl_allowed"])
        self.assertTrue(decision["gates"]["router_a_recall_32"])
        self.assertTrue(decision["gates"]["router_b_recall_32"])
        self.assertFalse(decision["gates"]["router_c_precision_64"])
        self.assertFalse(
            decision["gates"]["negative_false_positive_routes_zero"]
        )
        self.assertFalse(decision["gates"]["fallbacks_zero"])
        self.assertFalse(
            report["data"]["integration_v1_rows_or_outputs_loaded"]
        )


if __name__ == "__main__":
    unittest.main()
