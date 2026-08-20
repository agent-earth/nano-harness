from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nano_harness.baseline import sha256_file
from nano_harness.client import ScriptedClient
from nano_harness.router_adapter_integration import ROUTER_SYSTEM, _candidate_row
from nano_harness.router_adapter_integration_v3 import (
    CONFIG_SHA256,
    POSITIVE_FAMILIES,
    build_cases,
    load_config,
    parent_config,
)
from nano_harness.semantic_skill_execution import (
    load_config as load_mechanism_config,
)
from nano_harness.types import ModelReply
from scripts.preregister_router_adapter_integration_v3 import (
    build_receipt,
    render_markdown,
)
from scripts.render_router_adapter_integration_v3 import admission_gates
from scripts.render_router_adapter_integration_v3_service import (
    build_receipt as build_service,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/harness/qwen35_router_adapter_integration_v3.json"


class RouterAdapterIntegrationV3Tests(unittest.TestCase):
    def test_config_freezes_new_adapter_surface_and_no_prior_rerun(self):
        config = load_config(CONFIG)
        self.assertEqual(config.case_seed, 20260828)
        self.assertEqual(config.value_offset, 8_000_000)
        self.assertEqual(config.positive_cases_per_family, 16)
        self.assertEqual(config.negative_cases_per_subtype, 16)
        self.assertEqual(len(config.negative_subtypes), 8)
        self.assertEqual(
            config.adapter_tree_sha256,
            "cea357d281ed100437268e213564fc5a5c00e6024b0c7a4be207cc686453e3f9",
        )
        self.assertEqual(
            config.served_adapter_name,
            "qwen3.5-router-negative-diversity-v2-remapped",
        )
        self.assertFalse(
            config.policy["integration_v1_or_v2_outputs_loaded"]
        )
        self.assertFalse(
            config.policy["integration_v1_or_v2_rerun_allowed"]
        )
        self.assertFalse(
            config.execution_boundary["integration_v1_or_v2_rerun"]
        )
        self.assertFalse(
            config.execution_boundary["model_generation_started"]
        )

    def test_cases_cover_a_b_and_all_eight_c_subtypes(self):
        config = load_config(CONFIG)
        cases = build_cases(config)
        self.assertEqual(len(cases), 160)
        expected_families = (*POSITIVE_FAMILIES, *config.negative_subtypes)
        self.assertEqual(
            {
                family: sum(case["family"] == family for case in cases)
                for family in expected_families
            },
            {family: 16 for family in expected_families},
        )
        self.assertEqual(
            {
                label: sum(case["expected_label"] == label for case in cases)
                for label in ("A", "B", "C")
            },
            {"A": 16, "B": 16, "C": 128},
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
            receipt["router"]["model"],
            "qwen3.5-router-negative-diversity-v2-remapped",
        )
        self.assertEqual(
            receipt["router"]["adapter_sha256"],
            config.adapter_tree_sha256,
        )
        self.assertEqual(
            router.calls[0]["messages"][0]["content"],
            ROUTER_SYSTEM,
        )
        self.assertEqual(
            planner.calls[0]["messages"][1]["content"],
            case["prompt"],
        )

    def test_preregister_is_deterministic_and_history_disjoint(self):
        first = build_receipt()
        second = build_receipt()
        self.assertEqual(first, second)
        self.assertEqual(first["surface"]["cases"], 160)
        self.assertEqual(
            first["surface"]["expected_label_counts"],
            {"A": 16, "B": 16, "C": 128},
        )
        self.assertEqual(
            set(first["surface"]["family_counts"].values()),
            {16},
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
        self.assertFalse(
            first["freshness"]["integration_v1_or_v2_outputs_loaded"]
        )
        self.assertFalse(
            first["acceptance"][
                "integration_v1_or_v2_rerun_allowed_after_pass"
            ]
        )
        self.assertFalse(
            first["execution_boundary"]["model_generation_started"]
        )
        markdown = render_markdown(first)
        self.assertIn("all eight C", markdown)
        self.assertIn("all zero", markdown)
        self.assertIn("cannot be rerun", markdown)

    def test_config_rejects_any_mutation(self):
        raw = json.loads(CONFIG.read_text(encoding="utf-8"))
        altered = copy.deepcopy(raw)
        altered["case_seed"] = 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(altered), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "config SHA"):
                load_config(path)
        self.assertEqual(sha256_file(CONFIG), CONFIG_SHA256)

    def test_service_receipt_requires_4b_9b_and_remapped_router(self):
        config = load_config(CONFIG)
        four = {
            "data": [
                {
                    "id": config.service_models["four_b_base"],
                    "owned_by": "vllm",
                    "root": "../../../models/Qwen3.5-4B",
                    "parent": None,
                    "max_model_len": 4096,
                },
                {
                    "id": config.service_models["remapped_router"],
                    "owned_by": "vllm",
                    "root": config.adapter_path,
                    "parent": config.service_models["four_b_base"],
                    "max_model_len": None,
                },
            ]
        }
        nine = {
            "data": [
                {
                    "id": config.service_models["nine_b_base"],
                    "owned_by": "vllm",
                    "root": "../../../models/Qwen3.5-9B",
                    "parent": None,
                    "max_model_len": 4096,
                }
            ]
        }
        prereg = (
            ROOT
            / "docs/experiments/"
            "qwen35_router_adapter_integration_v3.preregister.json"
        )
        with tempfile.TemporaryDirectory() as directory:
            four_health = Path(directory) / "four.json"
            nine_health = Path(directory) / "nine.json"
            four_health.write_text(json.dumps(four), encoding="utf-8")
            nine_health.write_text(json.dumps(nine), encoding="utf-8")
            with patch(
                "scripts.render_router_adapter_integration_v3_service."
                "RAW_FOUR_HEALTH",
                four_health,
            ), patch(
                "scripts.render_router_adapter_integration_v3_service."
                "RAW_NINE_HEALTH",
                nine_health,
            ), patch(
                "scripts.render_router_adapter_integration_v3_service."
                "committed_preregister_sha256",
                return_value=sha256_file(prereg),
            ):
                receipt = build_service()
        self.assertTrue(receipt["healthy"])
        self.assertFalse(receipt["v3_generation_started"])
        self.assertFalse(receipt["integration_v1_or_v2_rerun"])
        self.assertEqual(receipt["models"], config.service_models)

    def test_admission_gates_require_every_c_subtype_and_zero_loss(self):
        config = load_config(CONFIG)
        cases = build_cases(config)
        all_families = (*POSITIVE_FAMILIES, *config.negative_subtypes)
        arms = {
            name: {
                "cases": 160,
                "parseable": 160,
                "by_family": {
                    family: {"correct": 16} for family in all_families
                },
            }
            for name in (
                "four_b_direct",
                "nine_b_direct",
                "four_b_router_adapter_v3",
            )
        }
        receipts = {
            case["case_id"]: {
                "router": {
                    "label": case["expected_label"],
                    "expected_label": case["expected_label"],
                },
                "feedback_result_match": case["positive"],
            }
            for case in cases
        }
        rows = [
            {
                "output": "FINAL: 1",
                "prediction": 1,
                "parseable": True,
                "correct": True,
            }
            for _ in cases
        ]
        raw = {
            "arms": arms,
            "routing": {
                "negative_c_correct": 128,
                "negative_false_positive_routes": 0,
                "verified_executions": 32,
                "fallbacks": 0,
            },
            "receipts": receipts,
            "candidate_rows": copy.deepcopy(rows),
            "four_b_rows": copy.deepcopy(rows),
        }
        comparison = {
            "candidate_accuracy": 1.0,
            "baseline_accuracy": 0.5,
            "paired_bootstrap_95_ci": [0.1, 0.3],
            "mcnemar_exact_p": 0.001,
            "paired_counts": {"candidate_only": 32, "baseline_only": 0},
        }
        decision = admission_gates(raw, comparison, comparison)
        self.assertTrue(all(decision.values()))
        first_negative = next(case for case in cases if not case["positive"])
        raw["receipts"][first_negative["case_id"]]["router"]["label"] = "A"
        raw["routing"]["negative_c_correct"] = 127
        raw["routing"]["negative_false_positive_routes"] = 1
        decision = admission_gates(raw, comparison, comparison)
        self.assertFalse(decision["each_c_subtype_recall_16"])
        self.assertFalse(decision["negative_false_positive_routes_zero"])


if __name__ == "__main__":
    unittest.main()
