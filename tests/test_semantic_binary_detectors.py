from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from nano_harness.client import ScriptedClient
from nano_harness.semantic_binary_detectors import (
    ALL_FAMILIES,
    _candidate_row,
    build_cases,
    load_config,
    parent_config,
    parse_detection,
)
from nano_harness.semantic_skill_execution import (
    load_config as load_mechanism_config,
)
from nano_harness.types import ModelReply
from scripts.preregister_semantic_binary_detectors_v1 import (
    build_receipt,
    render_markdown,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "configs/harness/qwen35_semantic_binary_detectors_v1.json"
)


class SemanticBinaryDetectorsTests(unittest.TestCase):
    def test_config_freezes_small_balanced_surface_and_boundaries(self):
        config = load_config(CONFIG)
        cases = build_cases(config)
        self.assertEqual(len(cases), 128)
        self.assertEqual(
            {
                family: sum(case["family"] == family for case in cases)
                for family in ALL_FAMILIES
            },
            {family: 32 for family in ALL_FAMILIES},
        )
        self.assertEqual(config.detector_max_tokens, 8)
        self.assertEqual(config.cases_per_family, 32)
        self.assertFalse(config.execution_boundary["model_generation_started"])
        self.assertFalse(config.execution_boundary["benchmark_accessed"])

    def test_detection_parser_is_binary(self):
        self.assertTrue(parse_detection("DETECT: YES"))
        self.assertFalse(parse_detection("DETECT: NO"))
        self.assertIsNone(parse_detection("YES"))
        self.assertIsNone(parse_detection("DETECT: MAYBE"))

    def test_exactly_one_yes_executes_selected_skill(self):
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
        detectors = {
            "implicit_scale_total": ScriptedClient(
                [ModelReply(content="DETECT: YES")]
            ),
            "first_strict_profit_period": ScriptedClient(
                [ModelReply(content="DETECT: NO")]
            ),
        }
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
            detectors,
            ScriptedClient([ModelReply(content=plan)]),
            ScriptedClient(
                [ModelReply(content=f"FINAL: {case['expected']}")]
            ),
            config,
            mechanism,
            parent,
        )
        self.assertTrue(candidate["correct"])
        self.assertTrue(receipt["detector_correct"])
        self.assertEqual(receipt["selected_route"], "implicit_scale_total")
        self.assertFalse(receipt["conflict"])

    def test_dual_no_preserves_direct(self):
        config = load_config(CONFIG)
        mechanism = load_mechanism_config(config.mechanism_config_path)
        parent = parent_config(config)
        case = next(
            row for row in build_cases(config) if row["family"] == "box_total"
        )
        detectors = {
            family: ScriptedClient([ModelReply(content="DETECT: NO")])
            for family in ("implicit_scale_total", "first_strict_profit_period")
        }
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
            detectors,
            ScriptedClient([]),
            ScriptedClient([]),
            config,
            mechanism,
            parent,
        )
        self.assertEqual(candidate["output"], direct["output"])
        self.assertEqual(candidate["route"], "direct_preserve_after_detector_none")
        self.assertTrue(receipt["detector_correct"])
        self.assertFalse(receipt["conflict"])

    def test_dual_yes_conflict_preserves_direct(self):
        config = load_config(CONFIG)
        mechanism = load_mechanism_config(config.mechanism_config_path)
        parent = parent_config(config)
        case = next(
            row
            for row in build_cases(config)
            if row["family"] == "implicit_scale_total"
        )
        detectors = {
            family: ScriptedClient([ModelReply(content="DETECT: YES")])
            for family in ("implicit_scale_total", "first_strict_profit_period")
        }
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
            detectors,
            ScriptedClient([]),
            ScriptedClient([]),
            config,
            mechanism,
            parent,
        )
        self.assertEqual(candidate["output"], direct["output"])
        self.assertEqual(candidate["route"], "direct_preserve_after_detector_none")
        self.assertFalse(receipt["detector_correct"])
        self.assertTrue(receipt["conflict"])

    def test_config_rejects_detector_or_composition_changes(self):
        raw = json.loads(CONFIG.read_text(encoding="utf-8"))
        mutations = (
            ("detector_max_tokens", 16, "detector_max_tokens"),
            ("cases_per_family", 64, "cases_per_family"),
            ("detector_structured_output_regex", ".*", "detector_structured_output_regex"),
            ("multiclass_report_sha256", "0" * 64, "multiclass_report_sha256"),
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
        self.assertEqual(first["case_contract"]["case_count"], 128)
        self.assertEqual(first["surface"]["positive_cases"], 64)
        self.assertEqual(first["surface"]["negative_cases"], 64)
        self.assertFalse(
            first["execution_boundary"]["model_generation_started"]
        )
        self.assertFalse(first["execution_boundary"]["benchmark_accessed"])
        markdown = render_markdown(first)
        self.assertIn("双 NO 或双 YES", markdown)
        self.assertIn("positive recall 64/64", markdown)
        self.assertIn("question-only real detector scan", markdown)


if __name__ == "__main__":
    unittest.main()
