from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nano_harness.baseline import sha256_file
from nano_harness.client import ScriptedClient
from nano_harness.router_skill_fallback_v4 import (
    CONFIG_SHA256,
    C_FAMILIES,
    FAMILY_TO_TOOL,
    POSITIVE_FAMILIES,
    _c_skill_candidate,
    build_cases,
    execute_c_skill,
    load_config,
    parent_config,
    parse_and_execute_c_plan,
)
from nano_harness.types import ModelReply
from scripts.preregister_router_skill_fallback_v4 import (
    build_receipt,
    render_markdown,
)
from scripts.render_router_skill_fallback_v4 import admission_gates
from scripts.render_router_skill_fallback_v4_service import (
    build_receipt as build_service,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/harness/qwen35_router_skill_fallback_v4.json"


class RouterSkillFallbackV4Tests(unittest.TestCase):
    def test_config_freezes_fresh_c_skill_policy(self):
        config = load_config(CONFIG)
        self.assertEqual(config.case_seed, 20260829)
        self.assertEqual(config.value_offset, 12_000_000)
        self.assertEqual(config.cases_per_family, 16)
        self.assertEqual(
            config.adapter_tree_sha256,
            "cea357d281ed100437268e213564fc5a5c00e6024b0c7a4be207cc686453e3f9",
        )
        self.assertFalse(
            config.policy["integration_v1_v2_v3_outputs_loaded"]
        )
        self.assertFalse(
            config.policy["integration_v1_v2_v3_rerun_allowed"]
        )
        self.assertFalse(
            config.execution_boundary["model_generation_started"]
        )

    def test_cases_cover_ten_families_and_deterministic_targets(self):
        config = load_config(CONFIG)
        cases = build_cases(config)
        families = (*POSITIVE_FAMILIES, *C_FAMILIES)
        self.assertEqual(len(cases), 160)
        self.assertEqual(
            {
                family: sum(case["family"] == family for case in cases)
                for family in families
            },
            {family: 16 for family in families},
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
                if case["family"] in C_FAMILIES:
                    self.assertEqual(
                        execute_c_skill(
                            FAMILY_TO_TOOL[case["family"]],
                            case["source_facts"],
                        ),
                        case["expected"],
                    )

    def test_all_eight_typed_plans_execute_and_tamper_fails(self):
        config = load_config(CONFIG)
        cases = build_cases(config)
        for family in C_FAMILIES:
            case = next(case for case in cases if case["family"] == family)
            tool = FAMILY_TO_TOOL[family]
            plan = (
                f"TOOL: {tool} "
                + json.dumps(case["source_facts"], separators=(",", ":"))
            )
            with self.subTest(family=family):
                receipt = parse_and_execute_c_plan(
                    plan,
                    source_facts=case["source_facts"],
                )
                self.assertTrue(receipt["executed"])
                self.assertEqual(receipt["result"], case["expected"])
                tampered = copy.deepcopy(case["source_facts"])
                first = next(
                    key
                    for key, value in tampered.items()
                    if isinstance(value, int)
                )
                tampered[first] += 1
                bad_plan = (
                    f"TOOL: {tool} "
                    + json.dumps(tampered, separators=(",", ":"))
                )
                rejected = parse_and_execute_c_plan(
                    bad_plan,
                    source_facts=case["source_facts"],
                )
                self.assertFalse(rejected["executed"])
                self.assertEqual(rejected["reason"], "source_facts_mismatch")

    def test_c_skill_candidate_emits_deterministic_result_without_rewrite(self):
        config = load_config(CONFIG)
        _, parent = parent_config(config)
        case = next(
            case
            for case in build_cases(config)
            if case["family"] == "weighted_total"
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
            "TOOL: weighted_total "
            + json.dumps(case["source_facts"], separators=(",", ":"))
        )
        planner = ScriptedClient([ModelReply(content=plan)])
        candidate, receipt = _c_skill_candidate(
            case,
            direct,
            planner,
            {},
            config,
            parent,
            {
                "label": "C",
                "correct": True,
            },
        )
        self.assertTrue(candidate["correct"])
        self.assertEqual(candidate["prediction"], case["expected"])
        self.assertEqual(candidate["output"], f"FINAL: {case['expected']}")
        self.assertTrue(receipt["c_skill_receipt"]["executed"])
        self.assertFalse(receipt["fallback_used"])
        self.assertEqual(len(planner.calls), 1)

    def test_preregister_is_deterministic_and_history_disjoint(self):
        first = build_receipt()
        second = build_receipt()
        self.assertEqual(first, second)
        self.assertEqual(first["surface"]["cases"], 160)
        self.assertEqual(
            first["surface"]["expected_label_counts"],
            {"A": 16, "B": 16, "C": 128},
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
            first["freshness"]["integration_v1_v2_v3_outputs_loaded"]
        )
        self.assertFalse(
            first["execution_boundary"]["model_generation_started"]
        )
        markdown = render_markdown(first)
        self.assertIn("eight typed skills", markdown)
        self.assertIn("all zero", markdown)
        self.assertIn("cannot be rerun", markdown)

    def test_config_rejects_any_mutation(self):
        raw = json.loads(CONFIG.read_text(encoding="utf-8"))
        altered = copy.deepcopy(raw)
        altered["plan_max_tokens"] = 256
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(altered), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "config SHA"):
                load_config(path)
        self.assertEqual(sha256_file(CONFIG), CONFIG_SHA256)

    def test_service_receipt_requires_4b_9b_and_router(self):
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
            "qwen35_router_skill_fallback_v4.preregister.json"
        )
        with tempfile.TemporaryDirectory() as directory:
            four_path = Path(directory) / "four.json"
            nine_path = Path(directory) / "nine.json"
            four_path.write_text(json.dumps(four), encoding="utf-8")
            nine_path.write_text(json.dumps(nine), encoding="utf-8")
            with patch(
                "scripts.render_router_skill_fallback_v4_service."
                "RAW_FOUR_HEALTH",
                four_path,
            ), patch(
                "scripts.render_router_skill_fallback_v4_service."
                "RAW_NINE_HEALTH",
                nine_path,
            ), patch(
                "scripts.render_router_skill_fallback_v4_service."
                "committed_preregister_sha256",
                return_value=sha256_file(prereg),
            ):
                receipt = build_service()
        self.assertTrue(receipt["healthy"])
        self.assertFalse(receipt["v4_generation_started"])
        self.assertEqual(receipt["models"], config.service_models)

    def test_admission_gates_require_zero_loss_and_all_skills(self):
        config = load_config(CONFIG)
        cases = build_cases(config)
        rows = [
            {
                "case_id": case["case_id"],
                "family": case["family"],
                "output": f"FINAL: {case['expected']}",
                "prediction": case["expected"],
                "parseable": True,
                "correct": True,
            }
            for case in cases
        ]
        receipts = {
            case["case_id"]: {
                "router": {
                    "output": f"FINAL: {case['expected_label']}",
                    "expected_label": case["expected_label"],
                },
            }
            for case in cases
        }
        comparison = {
            "candidate_accuracy": 1.0,
            "baseline_accuracy": 0.5,
            "paired_bootstrap_95_ci": [0.2, 0.4],
            "mcnemar_exact_p": 0.001,
            "paired_counts": {"candidate_only": 80, "baseline_only": 0},
        }
        raw = {
            "four_b_rows": copy.deepcopy(rows),
            "nine_b_rows": copy.deepcopy(rows),
            "candidate_rows": copy.deepcopy(rows),
            "receipts": receipts,
            "routing": {
                "ab_verified_executions": 32,
                "c_skill_executions": 128,
                "fallbacks": 0,
            },
        }
        gates = admission_gates(raw, comparison, comparison)
        self.assertTrue(all(gates.values()))
        raw["routing"]["c_skill_executions"] = 127
        gates = admission_gates(raw, comparison, comparison)
        self.assertFalse(gates["c_skill_verified_executions_128"])


if __name__ == "__main__":
    unittest.main()
