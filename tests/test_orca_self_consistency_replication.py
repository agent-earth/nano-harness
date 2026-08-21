from __future__ import annotations

import json
import unittest
from pathlib import Path

from nano_harness.orca_self_consistency_replication import (
    load_config,
    select_cases,
)
from scripts.preregister_orca_self_consistency_replication_v2 import (
    build_receipt,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/campaign/orca_math_self_consistency_replication_v2.json"
)


class OrcaSelfConsistencyReplicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(CONFIG)
        cls.selection = select_cases(cls.config)

    def test_selection_covers_every_remaining_row(self):
        self.assertEqual(len(self.selection["cases"]), 160)
        counts = {
            stratum: sum(
                row["stratum"] == stratum
                for row in self.selection["cases"]
            )
            for stratum in ("short", "medium", "long")
        }
        self.assertEqual(
            counts,
            {"short": 40, "medium": 80, "long": 40},
        )

    def test_selection_excludes_all_prior_evidence(self):
        raw = self.config.raw
        prior_ids = set()
        for key in (
            "prior_dpo_v1_preregister_path",
            "prior_dpo_v2_preregister_path",
            "prior_self_consistency_preregister_path",
        ):
            receipt = json.loads(
                self.config.resolve(raw[key]).read_text(encoding="utf-8")
            )
            selection = receipt["selection"]
            if "train_ids" in selection:
                prior_ids.update(selection["train_ids"])
                prior_ids.update(selection["dev_ids"])
            else:
                prior_ids.update(selection["case_ids"])
        self.assertFalse(
            prior_ids & set(self.selection["case_ids"])
        )

    def test_preregister_is_deterministic_and_closed(self):
        first = build_receipt()
        second = build_receipt()
        self.assertEqual(first, second)
        self.assertTrue(
            first["selection"]["covers_every_remaining_row"]
        )
        self.assertTrue(first["strategy"]["exactly_matches_v1"])
        self.assertFalse(
            first["decision_rule"]["rerun_or_tuning_allowed"]
        )


if __name__ == "__main__":
    unittest.main()
