from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nano_harness.baseline import sha256_file
from nano_harness.router_serving_parity import (
    case_contract,
    load_config,
    summarize,
    validation_cases,
)
from scripts.preregister_router_serving_parity_v1 import (
    build_receipt,
    render_markdown,
)
from scripts.render_router_serving_parity_service_v1 import (
    build_receipt as build_service,
)
from scripts.render_router_serving_parity_v1 import gates


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/harness/qwen35_router_serving_parity_v1.json"


class RouterServingParityTests(unittest.TestCase):
    def test_config_freezes_three_arm_diagnostic_and_closed_boundary(self):
        config = load_config(CONFIG)
        self.assertEqual(config.validation_rows, 192)
        self.assertEqual(
            config.served_models,
            {
                "base": "qwen3.5-4b",
                "original": "qwen3.5-router-original-v1",
                "remapped": "qwen3.5-router-remapped-v1",
            },
        )
        self.assertEqual(config.service_launch["max_loras"], 2)
        self.assertFalse(
            config.execution_boundary["parity_service_started"]
        )
        self.assertFalse(
            config.execution_boundary["model_generation_started"]
        )
        self.assertFalse(config.execution_boundary["benchmark_accessed"])

    def test_validation_surface_is_balanced_and_matches_hf_reference(self):
        config = load_config(CONFIG)
        cases = validation_cases(config)
        self.assertEqual(len(cases), 192)
        self.assertEqual(
            {
                label: sum(case["label"] == label for case in cases)
                for label in ("A", "B", "C")
            },
            {"A": 64, "B": 64, "C": 64},
        )
        self.assertEqual(
            case_contract(cases)["case_contract_sha256"],
            "b87e3fd102b3cfb765e79add32124529fce68a5e0fa5e33680cd32534b5eb2c2",
        )
        hf = json.loads(
            Path(config.hf_generations_path).read_text(encoding="utf-8")
        )["post_sft"]
        self.assertEqual(
            [case["sample_id"] for case in cases],
            [row["sample_id"] for row in hf],
        )
        self.assertTrue(all(row["exact"] for row in hf))

    def test_preregister_proves_namespace_not_tokenizer_difference(self):
        first = build_receipt()
        second = build_receipt()
        self.assertEqual(first, second)
        self.assertEqual(first["surface"]["rows"], 192)
        self.assertEqual(
            first["surface"]["label_counts"], {"A": 64, "B": 64, "C": 64}
        )
        self.assertEqual(
            first["namespace_audit"]["original_parsed_module"],
            "model.layers.0.mlp.down_proj",
        )
        self.assertEqual(
            first["namespace_audit"]["remapped_parsed_module"],
            "language_model.model.layers.0.mlp.down_proj",
        )
        self.assertTrue(
            first["namespace_audit"]["tensor_content_hashes_match"]
        )
        self.assertTrue(
            first["tokenizer_audit"]["semantic_equivalence_passed"]
        )
        self.assertTrue(
            all(first["tokenizer_audit"]["checks"].values())
        )
        self.assertFalse(
            first["execution_boundary"]["model_generation_started"]
        )
        markdown = render_markdown(first)
        self.assertIn("content-identical", markdown)
        self.assertIn("history-disjoint integration v2", markdown)
        self.assertIn("does not permit rerunning observed integration v1", markdown)

    def test_config_rejects_adapter_converter_or_vllm_source_changes(self):
        raw = json.loads(CONFIG.read_text(encoding="utf-8"))
        source = next(iter(raw["vllm_source_files"]))
        mutations = (
            ("generation_max_tokens", 16, "generation_max_tokens"),
            (
                "original_adapter_tree_sha256",
                "0" * 64,
                "original_adapter_tree_sha256",
            ),
            (
                "remapped_adapter_tree_sha256",
                "0" * 64,
                "remapped_adapter_tree_sha256",
            ),
            (
                "remap_converter_sha256",
                "0" * 64,
                "remap_converter_sha256",
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
        altered = copy.deepcopy(raw)
        altered["vllm_source_files"][source] = "0" * 64
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(altered), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "vllm_source_files"):
                load_config(path)

    def test_service_receipt_requires_base_original_and_remapped(self):
        config = load_config(CONFIG)
        fake = {
            "data": [
                {
                    "id": config.served_models["base"],
                    "owned_by": "vllm",
                    "root": "../../../models/Qwen3.5-4B",
                    "parent": None,
                    "max_model_len": 4096,
                },
                {
                    "id": config.served_models["original"],
                    "owned_by": "vllm",
                    "root": config.original_adapter_path,
                    "parent": config.served_models["base"],
                    "max_model_len": None,
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
            "qwen35_router_serving_parity_v1.preregister.json"
        )
        with tempfile.TemporaryDirectory() as directory:
            health = Path(directory) / "models.json"
            health.write_text(json.dumps(fake), encoding="utf-8")
            with patch(
                "scripts.render_router_serving_parity_service_v1.RAW_HEALTH",
                health,
            ), patch(
                "scripts.render_router_serving_parity_service_v1."
                "committed_preregister_sha256",
                return_value=sha256_file(prereg),
            ):
                receipt = build_service()
        self.assertTrue(receipt["healthy"])
        self.assertFalse(receipt["generation_started"])
        self.assertEqual(receipt["models"], config.served_models)

    def test_result_gates_require_exact_hf_parity_and_namespace_gain(self):
        families = ("router_a", "router_b", "router_c")

        def rows(exact: int) -> list[dict]:
            values = []
            for index in range(192):
                label = ("A", "B", "C")[index // 64]
                values.append(
                    {
                        "sample_id": str(index),
                        "task_family": families[index // 64],
                        "target": f"FINAL: {label}",
                        "output": f"FINAL: {label}" if index < exact else "FINAL: A",
                        "exact": index < exact,
                    }
                )
            return values

        raw = {
            "arms": {
                "base": rows(112),
                "original": rows(112),
                "remapped": rows(192),
            },
            "summaries": {
                "base": summarize(rows(112)),
                "original": summarize(rows(112)),
                "remapped": summarize(rows(192)),
            },
            "hf_output_matches": {
                "base": 112,
                "original": 112,
                "remapped": 192,
            },
        }
        self.assertTrue(all(gates(raw).values()))
        raw["hf_output_matches"]["remapped"] = 191
        self.assertFalse(gates(raw)["remapped_hf_output_match_192"])


if __name__ == "__main__":
    unittest.main()
