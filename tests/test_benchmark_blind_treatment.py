from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from nano_harness.benchmark_blind_treatment import (
    _evaluate_peer_result,
    build_treatment_receipt,
    load_treatment,
)
from scripts.preregister_benchmark_blind_treatment_v1 import render_markdown


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/campaign/qwen35_benchmark_blind_treatment_v1.json"


class BenchmarkBlindTreatmentTests(unittest.TestCase):
    def test_treatment_contract_freezes_arms_and_gates(self):
        treatment = load_treatment(CONFIG)
        self.assertEqual(
            [row["arm_id"] for row in treatment["arms"]],
            ["adapter_only", "arbiter_only", "adapter_plus_arbiter"],
        )
        self.assertEqual(
            treatment["admission_gates"]["canary"][
                "benchmark_minimum_correct"
            ],
            {"gsm8k": 90, "mmlu": 67, "gpqa_diamond": 6},
        )
        self.assertEqual(
            treatment["admission_gates"]["complete"][
                "minimum_benchmarks_significantly_won"
            ],
            3,
        )
        self.assertFalse(
            treatment["execution_boundary"]["canary_generation_allowed"]
        )
        self.assertFalse(
            treatment["execution_boundary"][
                "complete_treatment_generation_allowed"
            ]
        )

    def test_receipt_is_deterministic_and_blocked_without_peer_result(self):
        first = build_treatment_receipt(CONFIG)
        second = build_treatment_receipt(CONFIG)
        self.assertEqual(first, second)
        self.assertTrue(all(first["checks"].values()))
        self.assertFalse(first["peer_result"]["exists"])
        self.assertFalse(first["peer_result"]["admitted"])
        self.assertFalse(
            first["readiness"]["canary_generation_allowed"]
        )
        self.assertFalse(
            first["readiness"]["complete_treatment_generation_allowed"]
        )
        for surface in first["evaluation_surfaces"].values():
            self.assertEqual(
                surface["manifest_invariance"],
                {
                    "case_selection_equal": True,
                    "dataset_identity_equal": True,
                    "gsm8k_direct_unchanged": True,
                    "mmlu_direct_unchanged": True,
                    "gpqa_only_arbiter": True,
                    "option_evidence_max_tokens": 96,
                    "arbiter_max_tokens": 64,
                },
            )

    def test_config_rejects_weaker_canary_or_missing_search_guard(self):
        raw = json.loads(CONFIG.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "treatment.json"
            altered = copy.deepcopy(raw)
            altered["admission_gates"]["canary"]["minimum_overall_correct"] = (
                163
            )
            path.write_text(json.dumps(altered), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError,
                "canary admission gates differ",
            ):
                load_treatment(path)

            altered = copy.deepcopy(raw)
            altered["decision_policy"][
                "forbidden_after_any_treatment_observation"
            ].remove("parser_change")
            path.write_text(json.dumps(altered), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError,
                "no-post-hoc-search policy is incomplete",
            ):
                load_treatment(path)

    def test_peer_result_requires_every_frozen_gate(self):
        dependency = load_treatment(CONFIG)["dependencies"][0]
        accepted = {
            "schema_version": (
                "nano_train_paired_consistency_replication_public_v1"
            ),
            "experiment_id": "paired-consistency-replication-v1",
            "identity": {
                "config_sha256": (
                    "534c9d7794643a0a4a4c242995bfe931c9834844bbd20acb6bfe"
                    "97f42e9a6c8f"
                ),
                "preregister_sha256": (
                    "f02d710beef452a6bcb2c8d56d5e4e81efb347bde048b34f7ebe"
                    "9aba9713b06c"
                ),
                "release_sha256": (
                    "db649d2e603a7a24305d01ed108f2fa4f640f1c315a4e07cf427"
                    "8c5a7534856a"
                ),
                "preregister_revision": (
                    "6b6ac50f706ab29edfce267a4d286ed408628012"
                ),
                "adapter_sha256": "a" * 64,
            },
            "stability": {
                "all_adapter_tensors_finite": True,
                "independent_reload_exact": True,
                "failure_receipt_exists": False,
            },
            "comparisons": {
                name: {
                    "paired_bootstrap_95_ci": [0.01, 0.08],
                    "mcnemar_exact_p": 0.03125,
                    **(
                        {
                            "candidate_only_wins": 6,
                            "baseline_only_wins": 0,
                        }
                        if name == "final"
                        else {}
                    ),
                }
                for name in ("aggregate", "final", "pair")
            },
            "json_families": {
                name: {
                    "baseline_correct": 28,
                    "post_correct": 28,
                }
                for name in (
                    "coding-and-validation",
                    "planning-and-state",
                    "skill-routing-and-reflection",
                    "tool-use-and-recovery",
                )
            },
            "decision": {"accepted": True},
            "claim_boundary": "local synthetic replication only",
        }
        receipt = _evaluate_peer_result(accepted, dependency)
        self.assertTrue(receipt["admitted"])
        self.assertTrue(all(receipt["checks"].values()))

        for mutation in (
            ("comparisons", "aggregate", "paired_bootstrap_95_ci", 0, 0.0),
            ("comparisons", "final", "candidate_only_wins", None, 5),
            ("comparisons", "final", "baseline_only_wins", None, 1),
            ("stability", "independent_reload_exact", None, None, False),
            (
                "json_families",
                "coding-and-validation",
                "post_correct",
                None,
                27,
            ),
        ):
            with self.subTest(mutation=mutation):
                altered = copy.deepcopy(accepted)
                first, second, third, index, value = mutation
                if third is None:
                    altered[first][second] = value
                elif index is None:
                    target = altered[first][second]
                    target[third] = value
                else:
                    target = altered[first][second]
                    target[third][index] = value
                rejected = _evaluate_peer_result(altered, dependency)
                self.assertFalse(rejected["admitted"])

    def test_markdown_explains_arbiter_is_not_promoted(self):
        markdown = render_markdown(build_treatment_receipt(CONFIG))
        self.assertIn("不是已晋级的 harness", markdown)
        self.assertIn("1 赢 1 输", markdown)
        self.assertIn("至少 `164/211`", markdown)
        self.assertIn("strict score 是唯一正式分数", markdown)
        self.assertIn("独立 holdout 继续密封", markdown)


if __name__ == "__main__":
    unittest.main()
