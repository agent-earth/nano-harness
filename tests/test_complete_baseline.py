from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from nano_harness.complete_baseline import build_receipt, load_config
from nano_harness.fullstack_campaign import sha256_file
from scripts.preregister_complete_baseline_v1 import render_markdown


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "configs/campaign/qwen35_complete_direct_v1.preregister.json"
)


class StableWordTokenizer:
    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking,
    ):
        if tokenize or not add_generation_prompt or enable_thinking:
            raise AssertionError("unexpected template options")
        return "\n".join(
            f"{row['role']}: {row['content']}" for row in messages
        ) + "\nassistant:"

    def encode(self, text):
        return text.replace("\n", " \n ").split()


class CompleteBaselineTests(unittest.TestCase):
    def test_complete_config_is_frozen(self):
        config = load_config(CONFIG)
        self.assertEqual(config["execution"]["num_shards"], 16)
        self.assertEqual(config["serving"]["max_model_len"], 4096)
        self.assertEqual(config["serving"]["vllm_version"], "0.19.1")
        self.assertEqual(
            config["uncertainty"],
            {
                "alpha": 0.05,
                "bootstrap_samples": 10000,
                "bootstrap_seed": 20260820,
                "exact_mcnemar": True,
            },
        )
        self.assertTrue(
            all(value is False for value in config["policy"].values())
        )
        self.assertFalse(config["execution_boundary"]["model_generation_started"])

    def test_complete_config_rejects_gate_mutation(self):
        raw = json.loads(CONFIG.read_text(encoding="utf-8"))
        for path, value, message in (
            (("execution", "num_shards"), 8, "16 shards"),
            (("serving", "max_model_len"), 8192, "serving contract"),
            (("uncertainty", "alpha"), 0.1, "uncertainty contract"),
            (
                ("policy", "benchmark_rows_training_eligible"),
                True,
                "policy differs",
            ),
        ):
            with self.subTest(path=path):
                altered = copy.deepcopy(raw)
                altered[path[0]][path[1]] = value
                with tempfile.TemporaryDirectory() as directory:
                    config = Path(directory) / "config.json"
                    config.write_text(json.dumps(altered), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, message):
                        load_config(config)

    def test_receipt_is_deterministic_and_all_rows_are_preserved(self):
        config = load_config(CONFIG)
        expected_hashes = {
            config["suite_manifest_path"]: config["suite_manifest_sha256"],
        }
        for source in config["historical_cost_sources"]:
            expected_hashes[source["path"]] = source["sha256"]
        for model in config["model_contracts"]:
            expected_hashes[
                str(Path(model["path"]) / "config.json")
            ] = model["config_sha256"]
            expected_hashes[
                str(Path(model["path"]) / "model.safetensors.index.json")
            ] = model["index_sha256"]
        real_sha256 = sha256_file

        def cached_sha256(path: Path) -> str:
            for relative, digest in expected_hashes.items():
                if path.resolve() == (ROOT / relative).resolve():
                    return digest
            return real_sha256(path)

        with mock.patch(
            "nano_harness.complete_baseline.sha256_file",
            side_effect=cached_sha256,
        ):
            first, first_cases = build_receipt(
                CONFIG,
                tokenizer=StableWordTokenizer(),
            )
            second, second_cases = build_receipt(
                CONFIG,
                tokenizer=StableWordTokenizer(),
            )
        self.assertEqual(first, second)
        self.assertEqual(first_cases, second_cases)
        self.assertEqual(first["cases"]["total"], 15559)
        self.assertEqual(
            first["cases"]["by_benchmark"],
            {"gsm8k": 1319, "mmlu": 14042, "gpqa_diamond": 198},
        )
        self.assertEqual(
            len({row["case_id"] for row in first_cases}),
            15559,
        )
        self.assertEqual(sum(first["sharding"]["counts"]), 15559)
        self.assertTrue(all(first["checks"].values()))
        self.assertTrue(
            all(
                "prompt" not in row
                and "expected" not in row
                and "answer" not in row
                for row in first_cases
            )
        )
        self.assertFalse(first["execution_boundary"]["model_generation_started"])
        self.assertFalse(first["policy"]["benchmark_rows_training_eligible"])

    def test_markdown_explains_duplicate_and_context_fixes(self):
        receipt = {
            "experiment_id": "test",
            "identity": {
                "config_sha256": "0" * 64,
                "suite_manifest_sha256": "1" * 64,
                "case_contract_sha256": "2" * 64,
                "case_ids_sha256": "3" * 64,
            },
            "cases": {
                "total": 15559,
                "by_benchmark": {
                    "gsm8k": 1319,
                    "mmlu": 14042,
                    "gpqa_diamond": 198,
                },
            },
            "context": {
                "by_benchmark": {
                    "gsm8k": {
                        "input_max": 264,
                        "input_p99": 203,
                        "input_plus_budget_max": 864,
                    },
                    "mmlu": {
                        "input_max": 1054,
                        "input_p99": 528,
                        "input_plus_budget_max": 1086,
                    },
                    "gpqa_diamond": {
                        "input_max": 2798,
                        "input_p99": 916,
                        "input_plus_budget_max": 2830,
                    },
                }
            },
            "sharding": {"minimum": 900, "maximum": 1000},
            "historical_cost_projection": {
                "total_per_model": {
                    "four_b": {
                        "projected_wall_seconds_min": 3600,
                        "projected_wall_seconds_max": 7200,
                    },
                    "nine_b": {
                        "projected_wall_seconds_min": 3600,
                        "projected_wall_seconds_max": 7200,
                    },
                }
            },
            "execution_boundary": {"model_generation_started": False},
        }
        markdown = render_markdown(receipt)
        self.assertIn("少算 174 行", markdown)
        self.assertIn("row_stable_v2", markdown)
        self.assertIn("15,559", markdown)
        self.assertIn("context 冻结为 4096", markdown)
        self.assertIn("没有启动", markdown)


if __name__ == "__main__":
    unittest.main()
