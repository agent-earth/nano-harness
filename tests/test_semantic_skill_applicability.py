from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from nano_harness.semantic_skill_applicability import (
    extract_source_facts,
    load_config,
)
from nano_harness.semantic_skill_execution import route_prompt
from scripts.preregister_semantic_skill_applicability_v1 import (
    build_receipt,
    render_markdown,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "configs/harness/qwen35_semantic_skill_applicability_v1.json"
)


class SemanticSkillApplicabilityTests(unittest.TestCase):
    def test_config_freezes_question_only_scan_and_generation_closed(self):
        config = load_config(CONFIG)
        self.assertEqual(
            config.expected_cases_by_benchmark,
            {"gsm8k": 1319, "mmlu": 14042, "gpqa_diamond": 198},
        )
        self.assertEqual(
            config.source_fact_extractor,
            "exact_labeled_integer_fields_v1",
        )
        self.assertEqual(config.minimum_eligible_rows_for_transfer, 1)
        self.assertTrue(config.policy["loads_question_column_only"])
        self.assertFalse(config.policy["loads_answer_columns"])
        self.assertFalse(config.policy["loads_choices_column"])
        self.assertFalse(config.policy["loads_model_outputs"])
        self.assertFalse(config.execution_boundary["scan_started"])
        self.assertFalse(
            config.execution_boundary["model_generation_started"]
        )
        self.assertFalse(
            config.execution_boundary["benchmark_generation_started"]
        )
        self.assertFalse(config.execution_boundary["canary_rerun_started"])

    def test_extractor_requires_route_and_all_exact_labeled_fields(self):
        implicit_prompt = (
            "rows=6 columns=7 extra=20 more than double the number of slots"
        )
        route = route_prompt(implicit_prompt)
        extracted = extract_source_facts(implicit_prompt, route)
        self.assertTrue(extracted["extracted"])
        self.assertEqual(
            extracted["source_facts"],
            {
                "rows": 6,
                "columns": 7,
                "extra": 20,
                "scale_word": "double",
            },
        )

        profit_prompt = (
            "setup_cost=90 units_per_period=7 price_per_unit=15 "
            "recurring_cost=98. first whole period when cumulative profit "
            "is strictly positive"
        )
        route = route_prompt(profit_prompt)
        extracted = extract_source_facts(profit_prompt, route)
        self.assertTrue(extracted["extracted"])
        self.assertEqual(
            extracted["source_facts"],
            {
                "setup_cost": 90,
                "units_per_period": 7,
                "price_per_unit": 15,
                "recurring_cost": 98,
            },
        )

        missing = extract_source_facts(
            "rows=6 extra=20 more than double the number of slots",
            route_prompt(
                "rows=6 extra=20 more than double the number of slots"
            ),
        )
        self.assertFalse(missing["extracted"])
        self.assertEqual(missing["reason"], "labeled_source_facts_missing")

        unrouted = extract_source_facts(
            "rows=6 columns=7 extra=20",
            route_prompt("rows=6 columns=7 extra=20"),
        )
        self.assertFalse(unrouted["extracted"])
        self.assertEqual(unrouted["reason"], "route_missing")

    def test_config_rejects_columns_threshold_or_boundary_changes(self):
        raw = json.loads(CONFIG.read_text(encoding="utf-8"))
        mutations = (
            (
                "question_column_by_benchmark",
                {**raw["question_column_by_benchmark"], "gsm8k": "answer"},
                "question_column_by_benchmark",
            ),
            (
                "minimum_eligible_rows_for_transfer",
                0,
                "minimum_eligible_rows_for_transfer",
            ),
            (
                "source_fact_extractor",
                "fuzzy",
                "source_fact_extractor",
            ),
            (
                "replication_report_sha256",
                "0" * 64,
                "replication_report_sha256",
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

    def test_preregister_is_deterministic_without_row_scan(self):
        first = build_receipt()
        second = build_receipt()
        self.assertEqual(first, second)
        self.assertEqual(first["surface"]["cases"], 15559)
        self.assertFalse(
            first["surface"]["case_manifest_has_expected_or_answer"]
        )
        for audit in first["surface"]["schema_audit"].values():
            self.assertTrue(audit["answer_column_not_requested"])
            self.assertTrue(audit["choices_column_not_requested"])
        self.assertFalse(first["execution_boundary"]["scan_started"])
        self.assertFalse(
            first["execution_boundary"]["model_generation_started"]
        )
        markdown = render_markdown(first)
        self.assertIn("不读 answer", markdown)
        self.assertIn("eligible rows = 0", markdown)
        self.assertIn("scan started：false", markdown)


if __name__ == "__main__":
    unittest.main()
