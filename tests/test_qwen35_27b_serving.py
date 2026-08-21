from __future__ import annotations

import json
import unittest
from pathlib import Path

from scripts.validate_qwen35_27b_bf16_tp2_v1 import (
    PUBLIC,
    load_config,
)


ROOT = Path(__file__).resolve().parents[1]


class QwenThirtyFiveTwentySevenBServingTests(unittest.TestCase):
    def test_config_is_smoke_only_and_gptq_is_rejected(self):
        config = load_config()
        self.assertEqual(config["download"]["bucket"], "ai-infra")
        self.assertTrue(config["download"]["turbo"])
        self.assertEqual(config["service"]["tensor_parallel_size"], 2)
        self.assertEqual(config["service"]["max_model_len"], 1024)
        self.assertFalse(config["policy"]["benchmark_generation_started"])
        self.assertFalse(config["rejected_gptq"]["parity_eligible"])

    def test_public_result_when_available(self):
        if not PUBLIC.exists():
            self.skipTest("27B serving result not generated yet")
        report = json.loads(PUBLIC.read_text(encoding="utf-8"))
        self.assertTrue(report["decision"]["bf16_tp2_service_ready"])
        self.assertTrue(report["decision"]["gptq_service_rejected"])
        self.assertTrue(report["decision"]["parity_preregistration_allowed"])
        self.assertFalse(report["decision"]["benchmark_score_established"])
        self.assertEqual(
            report["smoke"]["passed_probes"],
            report["smoke"]["total_probes"],
        )
        self.assertNotIn("api_key", json.dumps(report).lower())

    def test_4096_service_result_when_available(self):
        config_path = (
            ROOT / "configs/serving/qwen35_27b_bf16_tp2_4096_v2.json"
        )
        public_path = (
            ROOT / "docs/results/qwen35_27b_bf16_tp2_4096_v2.public.json"
        )
        config = load_config(config_path)
        self.assertEqual(config["service"]["max_model_len"], 4096)
        self.assertEqual(config["service"]["kv_cache_memory_bytes"], 1073741824)
        self.assertEqual(config["service"]["max_num_seqs"], 2)
        if not public_path.exists():
            self.skipTest("27B 4096 serving result not generated yet")
        report = json.loads(public_path.read_text(encoding="utf-8"))
        self.assertTrue(report["decision"]["bf16_tp2_service_ready"])
        self.assertEqual(report["smoke"]["passed_probes"], 3)


if __name__ == "__main__":
    unittest.main()
