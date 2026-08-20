from __future__ import annotations

import unittest

from scripts.render_orca_self_consistency_v1 import (
    comparison_gates,
    paired_metrics,
)


class RenderOrcaSelfConsistencyTests(unittest.TestCase):
    def test_paired_metrics_counts_wins_and_losses(self):
        candidate = [
            {"case_id": "a", "correct": True, "prediction": "1"},
            {"case_id": "b", "correct": False, "prediction": "2"},
            {"case_id": "c", "correct": True, "prediction": "3"},
        ]
        baseline = [
            {"case_id": "a", "correct": False, "prediction": "2"},
            {"case_id": "b", "correct": True, "prediction": "1"},
            {"case_id": "c", "correct": True, "prediction": "3"},
        ]
        metrics = paired_metrics(
            candidate,
            baseline,
            bootstrap_samples=1_000,
            bootstrap_seed="test",
        )
        self.assertEqual(metrics["paired_counts"]["candidate_only"], 1)
        self.assertEqual(metrics["paired_counts"]["baseline_only"], 1)
        self.assertEqual(metrics["delta"], 0)
        self.assertEqual(metrics["mcnemar_exact_p"], 1.0)

    def test_gates_require_significance_and_stratum_non_regression(self):
        comparison = {
            "delta": 0.1,
            "paired_bootstrap_95_ci": [0.01, 0.2],
            "mcnemar_exact_p": 0.03,
            "paired_counts": {
                "candidate_only": 8,
                "baseline_only": 1,
            },
            "by_stratum": {
                "short": {"delta": 0.0},
                "medium": {"delta": 0.1},
                "long": {"delta": 0.2},
            },
        }
        gates = comparison_gates(
            comparison,
            alpha=0.05,
            minimum_candidate_only_wins=6,
        )
        self.assertTrue(all(gates.values()))
        comparison["by_stratum"]["short"]["delta"] = -0.01
        gates = comparison_gates(
            comparison,
            alpha=0.05,
            minimum_candidate_only_wins=6,
        )
        self.assertFalse(gates["every_stratum_non_regression"])


if __name__ == "__main__":
    unittest.main()
