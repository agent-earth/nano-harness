from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from nano_harness.fullstack_campaign import (
    build_campaign_receipt,
    load_campaign,
    sha256_file,
)
from scripts.preregister_fullstack_campaign_v1 import render_markdown


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/campaign/ultimate_distill_fullstack_v1.json"


class FullstackCampaignTests(unittest.TestCase):
    def test_campaign_schema_and_execution_boundary_are_frozen(self):
        campaign = load_campaign(CONFIG)
        self.assertEqual(
            [
                row["name"] for row in campaign["complete_benchmarks"]
            ],
            ["gsm8k", "mmlu", "gpqa_diamond"],
        )
        self.assertEqual(
            campaign["acceptance"]["minimum_complete_benchmarks_won"],
            3,
        )
        self.assertEqual(
            campaign["acceptance"]["twenty_seven_b_parity"],
            {
                "benchmarks": ["gsm8k", "mmlu"],
                "minimum_benchmarks_at_parity": 2,
                "noninferiority_margin": 0.02,
                "paired_bootstrap_ci_lower_gte_negative_margin": True,
            },
        )
        self.assertEqual(
            campaign["execution_boundary"],
            {
                "benchmark_scoring_started": False,
                "model_generation_started": False,
                "opd_started": False,
                "rl_started": False,
                "this_commit_only_audits_and_preregisters": True,
                "training_started": False,
            },
        )

    def test_campaign_rejects_significance_and_ladder_mutation(self):
        raw = json.loads(CONFIG.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "campaign.json"
            altered = copy.deepcopy(raw)
            altered["acceptance"]["alpha"] = 0.1
            path.write_text(json.dumps(altered), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError,
                "significance contract",
            ):
                load_campaign(path)

            altered = copy.deepcopy(raw)
            altered["candidate_ladder"][0]["depends_on"] = [
                "twenty-seven-b-parity"
            ]
            path.write_text(json.dumps(altered), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError,
                "prior stages",
            ):
                load_campaign(path)

    def test_campaign_rejects_formal_scan_as_score(self):
        raw = json.loads(CONFIG.read_text(encoding="utf-8"))
        raw["formal_agent_benchmarks"][0]["quality_score_claimed"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "campaign.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not scores"):
                load_campaign(path)

    def test_campaign_rejects_rl_or_opd_without_evidence(self):
        raw = json.loads(CONFIG.read_text(encoding="utf-8"))
        for capability_name in ("rl", "opd"):
            with self.subTest(capability=capability_name):
                altered = copy.deepcopy(raw)
                capability = next(
                    row
                    for row in altered["capabilities"]
                    if row["name"] == capability_name
                )
                capability["status"] = "implemented"
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "campaign.json"
                    path.write_text(
                        json.dumps(altered),
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(
                        ValueError,
                        "needs evidence",
                    ):
                        load_campaign(path)

    def test_receipt_reproduces_inventory_and_fail_closed_gates(self):
        expected_hashes = {
            row["path"]: row["sha256"]
            for row in load_campaign(CONFIG)["artifacts"]
        }
        for model in load_campaign(CONFIG)["models"]:
            if model["status"] != "ready":
                continue
            expected_hashes[
                str(Path(model["path"]) / "config.json")
            ] = model["config_sha256"]
            expected_hashes[
                str(Path(model["path"]) / "model.safetensors.index.json")
            ] = model["index_sha256"]
            for shard in model["shards"]:
                expected_hashes[
                    str(Path(model["path"]) / shard["name"])
                ] = shard["sha256"]
        for benchmark in load_campaign(CONFIG)["complete_benchmarks"]:
            expected_hashes[benchmark["path"]] = benchmark["sha256"]

        real_sha256_file = sha256_file

        def cached_sha256(path: Path) -> str:
            relative = None
            try:
                relative = path.resolve().relative_to(ROOT.resolve()).as_posix()
            except ValueError:
                pass
            if relative in expected_hashes:
                return expected_hashes[relative]
            for configured, digest in expected_hashes.items():
                if path.resolve() == (ROOT / configured).resolve():
                    return digest
            return real_sha256_file(path)

        with mock.patch(
            "nano_harness.fullstack_campaign.sha256_file",
            side_effect=cached_sha256,
        ):
            first = build_campaign_receipt(CONFIG)
            second = build_campaign_receipt(CONFIG)

        self.assertEqual(first, second)
        self.assertTrue(all(first["checks"].values()))
        self.assertTrue(
            first["readiness"]["matched_4b_9b_complete_benchmarks"]["ready"]
        )
        self.assertFalse(
            first["readiness"]["twenty_seven_b_parity"]["ready"]
        )
        self.assertFalse(first["readiness"]["rl"]["ready"])
        self.assertFalse(first["readiness"]["opd"]["ready"])
        self.assertFalse(
            first["readiness"]["formal_agent_benchmarks"]["ready"]
        )
        self.assertEqual(
            first["skill_evolution"]["parent"]["passed"],
            4,
        )
        self.assertEqual(
            first["skill_evolution"]["candidate"]["passed"],
            6,
        )
        self.assertEqual(
            first["next_executable_slice"]["stage_id"],
            "matched-direct-complete-baselines",
        )
        self.assertFalse(
            first["next_executable_slice"]["training_allowed"]
        )

    def test_receipt_rejects_artifact_tamper(self):
        raw = json.loads(CONFIG.read_text(encoding="utf-8"))
        raw["artifacts"][0]["sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "campaign.json"
            path.write_text(
                json.dumps(raw),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError,
                "artifact identity mismatch",
            ):
                build_campaign_receipt(
                    path,
                    repository_root=ROOT,
                )

    def test_markdown_explains_concrete_scope(self):
        receipt = {
            "campaign_id": "test",
            "identity": {
                "base_revision": "abcdef0",
                "config_sha256": "0" * 64,
                "artifacts": {"a": "1" * 64},
            },
            "inventory": {
                "models": [
                    {"model_id": "qwen3.5-4b"},
                    {"model_id": "qwen3.5-9b"},
                    {"model_id": "qwen3.5-27b"},
                ],
                "complete_benchmarks": [
                    {"name": "gsm8k", "rows": 1319, "scorer": "numeric_exact"},
                    {"name": "mmlu", "rows": 14042, "scorer": "choice_exact"},
                    {"name": "gpqa_diamond", "rows": 198, "scorer": "choice_exact"},
                ],
                "capabilities": [],
            },
            "skill_evolution": {
                "parent": {"passed": 4, "total": 6},
                "candidate": {"passed": 6, "total": 6},
                "promotion": {"promoted": True},
            },
            "prior_evidence": [
                {
                    "evidence_id": "prior",
                    "claim_boundary": "not a benchmark claim",
                }
            ],
            "candidate_ladder": [
                {
                    "stage_id": "direct",
                    "treatment": "run direct",
                    "stop_rule": "stop on mismatch",
                }
            ],
            "checks": {"all": True},
            "execution_boundary": {"training_started": False},
            "next_executable_slice": {"action": "run baseline"},
        }
        markdown = render_markdown(receipt)
        self.assertIn("1,319", markdown)
        self.assertIn("14,042", markdown)
        self.assertIn("RL 和 OPD", markdown)
        self.assertIn("scan", markdown)
        self.assertIn("没有启动", markdown)


if __name__ == "__main__":
    unittest.main()
