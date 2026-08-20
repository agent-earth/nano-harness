from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from nano_harness.baseline import sha256_file
from nano_harness.router_skill_fallback_v4 import C_FAMILIES, POSITIVE_FAMILIES
from nano_harness.router_skill_registry_v5 import (
    CONFIG_SHA256,
    FAMILY_TO_TOOL,
    SKILL_PROMPTS,
    applicable_c_skills,
    build_cases,
    execute_c_skill,
    load_config,
    parse_and_execute_c_plan,
)
from scripts.preregister_router_skill_registry_v5 import build_receipt


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/harness/qwen35_router_skill_registry_v5.json"


class RouterSkillRegistryV5Tests(unittest.TestCase):
    def test_config_freezes_registry_policy(self):
        config = load_config(CONFIG)
        self.assertEqual(config.case_seed, 20260830)
        self.assertEqual(config.value_offset, 16_000_000)
        self.assertEqual(
            config.skill_registry_policy,
            "target_blind_applicability_then_single_schema_v1",
        )
        self.assertFalse(config.execution_boundary["model_generation_started"])
        self.assertFalse(config.policy["v1_v2_v3_v4_outputs_loaded"])

    def test_registry_uniquely_matches_c_and_never_ab(self):
        config = load_config(CONFIG)
        cases = build_cases(config)
        self.assertEqual(len(cases), 160)
        for case in cases:
            with self.subTest(case_id=case["case_id"]):
                expected = [case["family"]] if case["family"] in C_FAMILIES else []
                self.assertEqual(applicable_c_skills(case["prompt"]), expected)

    def test_single_schema_plans_execute_all_c_skills(self):
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
                    plan, source_facts=case["source_facts"]
                )
                self.assertTrue(receipt["executed"])
                self.assertEqual(receipt["result"], case["expected"])
                self.assertEqual(
                    execute_c_skill(tool, case["source_facts"]),
                    case["expected"],
                )
                self.assertNotIn("available schemas", SKILL_PROMPTS[family])

    def test_tampered_single_schema_plan_fails_closed(self):
        config = load_config(CONFIG)
        case = next(
            case
            for case in build_cases(config)
            if case["family"] == "quotient_remainder"
        )
        tampered = copy.deepcopy(case["source_facts"])
        tampered["dividend"] += 1
        plan = "TOOL: quotient " + json.dumps(tampered, separators=(",", ":"))
        receipt = parse_and_execute_c_plan(
            plan, source_facts=case["source_facts"]
        )
        self.assertFalse(receipt["executed"])
        self.assertEqual(receipt["reason"], "source_facts_mismatch")

    def test_preregister_is_deterministic_and_zero_overlap(self):
        first = build_receipt()
        second = build_receipt()
        self.assertEqual(first, second)
        self.assertEqual(first["registry"]["c_unique_matches"], 128)
        self.assertEqual(first["registry"]["ab_false_matches"], 0)
        self.assertFalse(any(first["freshness"]["prompt_overlap"].values()))
        self.assertFalse(
            any(first["freshness"]["benchmark_prompt_overlap"].values())
        )
        self.assertFalse(
            any(first["freshness"]["prior_surface_prompt_overlap"].values())
        )
        self.assertFalse(
            first["execution_boundary"]["model_generation_started"]
        )

    def test_config_rejects_any_mutation(self):
        raw = json.loads(CONFIG.read_text(encoding="utf-8"))
        altered = copy.deepcopy(raw)
        altered["plan_retry_limit"] = 2
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(altered), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "config SHA"):
                load_config(path)
        self.assertEqual(sha256_file(CONFIG), CONFIG_SHA256)


if __name__ == "__main__":
    unittest.main()
