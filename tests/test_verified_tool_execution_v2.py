from __future__ import annotations

import copy
import importlib.util
import json
import re
import tempfile
import unittest
from pathlib import Path

from nano_harness.client import ScriptedClient
from nano_harness.types import ModelReply
from nano_harness.verified_tool_execution import FAMILIES, build_cases
from nano_harness.verified_tool_execution_v2 import (
    SKILL_PROMPTS,
    TOOL_REGEX_BY_FAMILY,
    _harness_row,
    load_config,
    parent_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/harness/qwen35_verified_tool_execution_v2.json"
RENDER_PATH = ROOT / "scripts/render_verified_tool_execution_v2.py"
RENDER_SPEC = importlib.util.spec_from_file_location(
    "render_verified_tool_execution_v2",
    RENDER_PATH,
)
RENDER_MODULE = importlib.util.module_from_spec(RENDER_SPEC)
assert RENDER_SPEC.loader is not None
RENDER_SPEC.loader.exec_module(RENDER_MODULE)


class VerifiedToolExecutionV2Tests(unittest.TestCase):
    def test_v2_freezes_parent_and_fresh_surface(self):
        config = load_config(CONFIG)
        parent = parent_config(config)
        self.assertEqual(config.value_offset, 90000)
        self.assertEqual(parent.value_offset, 90000)
        self.assertEqual(parent.cases_per_family, 64)
        self.assertEqual(parent.plan_retry_limit, 1)
        cases = build_cases(parent)
        self.assertEqual(len(cases), 256)
        self.assertTrue(
            all(row["case_id"].startswith("verified-tool-") for row in cases)
        )

    def test_each_family_exposes_one_matching_regex_and_skill(self):
        self.assertEqual(set(TOOL_REGEX_BY_FAMILY), set(FAMILIES))
        self.assertEqual(set(SKILL_PROMPTS), set(FAMILIES))
        examples = {
            "box_total": (
                'TOOL: box_total {"boxes":7,"items_per_box":11,'
                '"loose_items":5}'
            ),
            "remaining_stock": (
                'TOOL: remaining_stock {"starting_units":100,'
                '"batches_used":4,"units_per_batch":13}'
            ),
            "paired_average": (
                'TOOL: paired_average {"first_total":37,"second_total":45}'
            ),
            "labor_total": (
                'TOOL: labor_total {"hourly_rate":23,"regular_hours":8,'
                '"bonus":17}'
            ),
        }
        for family, text in examples.items():
            with self.subTest(family=family):
                self.assertIsNotNone(
                    re.fullmatch(TOOL_REGEX_BY_FAMILY[family], text)
                )
                self.assertIn(family, SKILL_PROMPTS[family])
                self.assertNotIn("expected", SKILL_PROMPTS[family])

    def test_skill_routed_harness_executes_labor_tool(self):
        config = load_config(CONFIG)
        parent = parent_config(config)
        case = next(
            row
            for row in build_cases(parent)
            if row["family"] == "labor_total"
        )
        plan = (
            "TOOL: labor_total "
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
            case, direct, plan_client, final_client, config, parent
        )
        self.assertTrue(row["correct"])
        self.assertEqual(row["route"], "skill_routed_verified_tool_feedback")
        self.assertEqual(receipt["skill_id"], "labor_total")
        self.assertEqual(receipt["exposed_tools"], ["labor_total"])
        self.assertTrue(receipt["receipt"]["executed"])
        structured = plan_client.calls[0]["extra_body"]["structured_outputs"]
        self.assertEqual(
            structured["regex"], TOOL_REGEX_BY_FAMILY["labor_total"]
        )

    def test_config_rejects_any_parent_or_mechanism_change(self):
        raw = json.loads(CONFIG.read_text(encoding="utf-8"))
        for key, value, error in (
            ("value_offset", 80000, "value_offset"),
            ("skill_router", "all_tools", "skill_router"),
            ("parent_config_sha256", "0" * 64, "parent_config_sha256"),
            ("service_receipt_sha256", "0" * 64, "service_receipt_sha256"),
        ):
            with self.subTest(key=key):
                altered = copy.deepcopy(raw)
                altered[key] = value
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "config.json"
                    path.write_text(json.dumps(altered), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, error):
                        load_config(path)

    def test_gate_admits_complete_zero_loss_skill_routing(self):
        by_family = {
            family: {"cases": 64, "correct": 0, "parseable": 64}
            for family in FAMILIES
        }
        raw = {
            "arms": {
                "four_b_direct": {
                    "cases": 256,
                    "correct": 30,
                    "accuracy": 30 / 256,
                    "parseable": 256,
                    "by_family": copy.deepcopy(by_family),
                },
                "nine_b_direct": {
                    "cases": 256,
                    "correct": 19,
                    "accuracy": 19 / 256,
                    "parseable": 256,
                    "by_family": copy.deepcopy(by_family),
                },
                "four_b_skill_verified_tool": {
                    "cases": 256,
                    "correct": 256,
                    "accuracy": 1.0,
                    "parseable": 256,
                    "by_family": {
                        family: {
                            "cases": 64,
                            "correct": 64,
                            "parseable": 64,
                        }
                        for family in FAMILIES
                    },
                },
            },
            "routing": {
                "skill_routes": 256,
                "single_tool_exposures": 256,
                "verified_executions": 256,
                "plan_retries": 0,
                "fallbacks": 0,
                "final_feedback_calls": 256,
            },
        }
        comparison = {
            "candidate_accuracy": 1.0,
            "baseline_accuracy": 0.1,
            "paired_bootstrap_95_ci": [0.8, 0.95],
            "mcnemar_exact_p": 1e-30,
            "paired_counts": {
                "candidate_only": 226,
                "baseline_only": 0,
            },
        }
        gates = RENDER_MODULE.admission_gates(
            raw, comparison, comparison
        )
        self.assertTrue(all(gates.values()))


if __name__ == "__main__":
    unittest.main()
