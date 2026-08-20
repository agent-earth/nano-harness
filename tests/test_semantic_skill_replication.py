from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from nano_harness.client import ScriptedClient
from nano_harness.semantic_skill_execution import (
    FAMILIES,
    _harness_row,
    build_cases as build_parent_cases,
    load_config as load_parent_config,
    route_prompt,
)
from nano_harness.semantic_skill_replication import (
    build_cases,
    load_config,
    parent_config,
)
from nano_harness.types import ModelReply
from scripts.preregister_semantic_skill_replication_v1 import (
    build_receipt,
    render_markdown,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "configs/harness/qwen35_semantic_skill_replication_v1.json"
)
PUBLIC_RESULT = (
    ROOT / "docs/results/qwen35_semantic_skill_replication_v1.public.json"
)


def normalized_hash(text: str) -> str:
    return hashlib.sha256(
        " ".join(text.casefold().split()).encode()
    ).hexdigest()


class SemanticSkillReplicationTests(unittest.TestCase):
    def test_config_freezes_parent_and_generation_boundary(self):
        config = load_config(CONFIG)
        parent = parent_config(config)
        self.assertEqual(config.case_seed, 20260821)
        self.assertEqual(config.cases_per_family, 128)
        self.assertEqual(config.value_regime, "small_integer_cross_product_v1")
        self.assertEqual(
            config.prompt_regime,
            "unseen_context_paraphrase_v1",
        )
        self.assertEqual(parent.direct_max_tokens, 32)
        self.assertEqual(parent.plan_max_tokens, 96)
        self.assertEqual(parent.final_max_tokens, 32)
        self.assertEqual(parent.plan_retry_limit, 1)
        self.assertFalse(config.execution_boundary["model_generation_started"])
        self.assertFalse(config.execution_boundary["evaluation_started"])
        self.assertFalse(config.execution_boundary["canary_accessed"])
        self.assertFalse(config.execution_boundary["benchmark_accessed"])
        self.assertFalse(config.execution_boundary["holdout_accessed"])

    def test_replication_cases_are_fresh_and_small_regime(self):
        config = load_config(CONFIG)
        mechanism = load_parent_config(config.parent_config_path)
        cases = build_cases(config)
        parent_cases = build_parent_cases(mechanism)
        self.assertEqual(len(cases), 256)
        self.assertEqual(
            {
                family: sum(row["family"] == family for row in cases)
                for family in FAMILIES
            },
            {family: 128 for family in FAMILIES},
        )
        self.assertFalse(
            {row["case_id"] for row in cases}
            & {row["case_id"] for row in parent_cases}
        )
        self.assertFalse(
            {normalized_hash(row["prompt"]) for row in cases}
            & {normalized_hash(row["prompt"]) for row in parent_cases}
        )
        self.assertTrue(
            all(
                row["source_facts"]["rows"] < 25
                and row["source_facts"]["columns"] < 25
                and row["source_facts"]["extra"] < 55
                for row in cases
                if row["family"] == "implicit_scale_total"
            )
        )
        self.assertTrue(
            all(
                row["source_facts"]["units_per_period"] < 25
                and row["source_facts"]["price_per_unit"] < 35
                for row in cases
                if row["family"] == "first_strict_profit_period"
            )
        )

    def test_parent_prompt_router_still_routes_every_replication_case(self):
        config = load_config(CONFIG)
        for case in build_cases(config):
            with self.subTest(case_id=case["case_id"]):
                route = route_prompt(case["prompt"])
                self.assertTrue(route["routed"])
                self.assertEqual(route["family"], case["family"])
                self.assertFalse(route["router_uses_case_metadata"])

    def test_parent_harness_executes_replication_without_mechanism_change(self):
        config = load_config(CONFIG)
        mechanism = load_parent_config(config.parent_config_path)
        parent = parent_config(config)
        for family in FAMILIES:
            with self.subTest(family=family):
                case = next(
                    row
                    for row in build_cases(config)
                    if row["family"] == family
                )
                plan = (
                    f"TOOL: {family} "
                    + json.dumps(
                        case["source_facts"],
                        separators=(",", ":"),
                    )
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
                    mechanism,
                    parent,
                )
                self.assertTrue(row["correct"])
                self.assertTrue(receipt["receipt"]["executed"])
                self.assertEqual(receipt["exposed_tools"], [family])
                self.assertTrue(receipt["feedback_result_match"])

    def test_feedback_mismatch_still_falls_back(self):
        config = load_config(CONFIG)
        mechanism = load_parent_config(config.parent_config_path)
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
            ScriptedClient([ModelReply(content=plan)]),
            ScriptedClient([ModelReply(content="FINAL: 0")]),
            mechanism,
            parent,
        )
        self.assertEqual(row["output"], direct["output"])
        self.assertEqual(
            row["route"],
            "direct_fallback_after_feedback_mismatch",
        )
        self.assertTrue(receipt["fallback_used"])
        self.assertFalse(receipt["feedback_result_match"])

    def test_config_rejects_mechanism_or_evidence_change(self):
        raw = json.loads(CONFIG.read_text(encoding="utf-8"))
        mutations = (
            ("case_seed", 20260820, "case_seed"),
            ("plan_max_tokens", 128, "plan_max_tokens"),
            ("plan_retry_limit", 2, "plan_retry_limit"),
            ("parent_report_sha256", "0" * 64, "parent_report_sha256"),
            ("value_regime", "large", "value_regime"),
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

    def test_preregister_is_deterministic_and_real_tasks_closed(self):
        first = build_receipt()
        second = build_receipt()
        self.assertEqual(first, second)
        self.assertEqual(first["case_contract"]["case_count"], 256)
        self.assertEqual(first["freshness"]["parent_case_id_overlap"], 0)
        self.assertEqual(first["freshness"]["parent_prompt_overlap"], 0)
        self.assertEqual(first["freshness"]["parent_source_fact_overlap"], 0)
        self.assertFalse(
            first["execution_boundary"]["model_generation_started"]
        )
        self.assertFalse(first["execution_boundary"]["canary_accessed"])
        self.assertFalse(first["execution_boundary"]["benchmark_accessed"])
        self.assertFalse(first["execution_boundary"]["holdout_accessed"])
        self.assertFalse(
            first["acceptance"]["canary_rerun_allowed_after_pass"]
        )
        self.assertFalse(
            first["acceptance"]["benchmark_generation_allowed_after_pass"]
        )
        markdown = render_markdown(first)
        self.assertIn("中小整数", markdown)
        self.assertIn("parent prompt overlap：0", markdown)
        self.assertIn("不允许重跑已观察 canary", markdown)

    def test_public_result_admits_preregistration_not_generation(self):
        report = json.loads(PUBLIC_RESULT.read_text(encoding="utf-8"))
        self.assertEqual(
            report["arms"]["four_b_semantic_skills"]["correct"],
            256,
        )
        self.assertEqual(report["arms"]["four_b_direct"]["correct"], 5)
        self.assertEqual(report["arms"]["nine_b_direct"]["correct"], 4)
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
        versus_four = report["comparisons"]["harness_vs_four_b"]
        versus_nine = report["comparisons"]["harness_vs_nine_b"]
        self.assertEqual(versus_four["paired_counts"]["candidate_only"], 251)
        self.assertEqual(versus_four["paired_counts"]["baseline_only"], 0)
        self.assertGreater(versus_four["paired_bootstrap_95_ci"][0], 0)
        self.assertEqual(versus_nine["paired_counts"]["candidate_only"], 252)
        self.assertEqual(versus_nine["paired_counts"]["baseline_only"], 0)
        self.assertGreater(versus_nine["paired_bootstrap_95_ci"][0], 0)
        decision = report["decision"]
        self.assertTrue(decision["replication_admitted"])
        self.assertTrue(
            decision["real_task_transfer_preregistration_allowed"]
        )
        self.assertFalse(decision["real_task_generation_allowed"])
        self.assertFalse(decision["canary_rerun_allowed"])
        self.assertFalse(decision["benchmark_generation_allowed"])
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


if __name__ == "__main__":
    unittest.main()
