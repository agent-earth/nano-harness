from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nano_harness.baseline import sha256_file
from nano_harness.router_serving_parity_v2 import (
    CONFIG_SHA256,
    case_contract,
    load_config,
    summarize,
    validation_cases,
)
from scripts.preregister_router_serving_parity_v2 import (
    build_receipt,
    render_markdown,
)
from scripts.render_router_serving_parity_service_v2 import (
    build_receipt as build_service,
)
from scripts.render_router_serving_parity_v2 import gates


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/harness/qwen35_router_serving_parity_v2.json"


class RouterServingParityV2Tests(unittest.TestCase):
    def test_config_freezes_single_remapped_arm_and_closed_boundary(self):
        config = load_config(CONFIG)
        self.assertEqual(config.validation_rows, 1536)
        self.assertEqual(
            config.served_models,
            {
                "base": "qwen3.5-4b",
                "remapped": (
                    "qwen3.5-router-negative-diversity-v2-remapped"
                ),
            },
        )
        self.assertEqual(config.service_launch["max_loras"], 1)
        self.assertEqual(config.service_launch["max_model_len"], 1024)
        self.assertFalse(
            config.execution_boundary["parity_service_started"]
        )
        self.assertFalse(
            config.execution_boundary["model_generation_started"]
        )
        self.assertFalse(config.execution_boundary["benchmark_accessed"])

    def test_validation_surface_matches_frozen_hf_reference(self):
        config = load_config(CONFIG)
        cases = validation_cases(config)
        self.assertEqual(len(cases), 1536)
        self.assertEqual(
            {
                label: sum(case["label"] == label for case in cases)
                for label in ("A", "B", "C")
            },
            {"A": 512, "B": 512, "C": 512},
        )
        subtype_counts = {
            subtype: sum(
                case["negative_subtype"] == subtype for case in cases
            )
            for subtype in {
                str(case["negative_subtype"])
                for case in cases
                if case["label"] == "C"
            }
        }
        self.assertEqual(len(subtype_counts), 8)
        self.assertEqual(set(subtype_counts.values()), {64})
        self.assertEqual(
            case_contract(cases)["case_contract_sha256"],
            "9b295456d56c7cb7d6acf2ecf2666ab50ad1fe564852080288ab31e9a36422d0",
        )
        hf = json.loads(
            Path(config.hf_generations_path).read_text(encoding="utf-8")
        )["post_sft"]
        self.assertEqual(
            [case["sample_id"] for case in cases],
            [row["sample_id"] for row in hf],
        )
        self.assertTrue(all(row["exact"] for row in hf))

    def test_preregister_is_deterministic_and_starts_no_generation(self):
        first = build_receipt()
        second = build_receipt()
        self.assertEqual(first, second)
        self.assertEqual(first["surface"]["rows"], 1536)
        self.assertEqual(
            first["surface"]["label_counts"],
            {"A": 512, "B": 512, "C": 512},
        )
        self.assertEqual(set(first["surface"]["c_subtype_counts"].values()), {64})
        self.assertTrue(
            first["namespace_audit"]["tensor_content_hashes_match"]
        )
        self.assertTrue(
            first["tokenizer_audit"]["semantic_equivalence_passed"]
        )
        self.assertTrue(all(first["tokenizer_audit"]["checks"].values()))
        self.assertFalse(
            first["execution_boundary"]["model_generation_started"]
        )
        markdown = render_markdown(first)
        self.assertIn("1,536/1,536", markdown)
        self.assertIn("history-disjoint", markdown)
        self.assertIn("does not permit rerunning integration v1/v2", markdown)

    def test_config_rejects_any_config_mutation(self):
        raw = json.loads(CONFIG.read_text(encoding="utf-8"))
        altered = copy.deepcopy(raw)
        altered["generation_max_tokens"] = 16
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(altered), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "config SHA"):
                load_config(path)
        self.assertEqual(sha256_file(CONFIG), CONFIG_SHA256)

    def test_service_receipt_requires_base_and_remapped(self):
        config = load_config(CONFIG)
        fake = {
            "data": [
                {
                    "id": config.served_models["base"],
                    "owned_by": "vllm",
                    "root": "../../../models/Qwen3.5-4B",
                    "parent": None,
                    "max_model_len": 1024,
                },
                {
                    "id": config.served_models["remapped"],
                    "owned_by": "vllm",
                    "root": config.remapped_adapter_path,
                    "parent": config.served_models["base"],
                    "max_model_len": None,
                },
            ]
        }
        prereg = (
            ROOT
            / "docs/experiments/"
            "qwen35_router_serving_parity_v2.preregister.json"
        )
        with tempfile.TemporaryDirectory() as directory:
            health = Path(directory) / "models.json"
            health.write_text(json.dumps(fake), encoding="utf-8")
            with patch(
                "scripts.render_router_serving_parity_service_v2.RAW_HEALTH",
                health,
            ), patch(
                "scripts.render_router_serving_parity_service_v2."
                "committed_preregister_sha256",
                return_value=sha256_file(prereg),
            ):
                receipt = build_service()
        self.assertTrue(receipt["healthy"])
        self.assertFalse(receipt["generation_started"])
        self.assertEqual(receipt["models"], config.served_models)

    def test_result_gates_require_exact_label_and_subtype_parity(self):
        config = load_config(CONFIG)
        cases = validation_cases(config)
        rows = [
            {
                "sample_id": case["sample_id"],
                "task_family": case["task_family"],
                "negative_subtype": case["negative_subtype"],
                "target": case["target"],
                "output": case["target"],
                "exact": True,
                "hf_output": case["target"],
                "hf_output_match": True,
            }
            for case in cases
        ]
        raw = {"rows": rows, "summary": summarize(rows)}
        self.assertTrue(all(gates(raw).values()))
        raw["rows"][0]["hf_output_match"] = False
        raw["summary"] = summarize(raw["rows"])
        self.assertFalse(gates(raw)["remapped_hf_output_match_1536"])


if __name__ == "__main__":
    unittest.main()
