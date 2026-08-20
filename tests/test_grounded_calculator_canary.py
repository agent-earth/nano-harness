from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from nano_harness.baseline import load_cases, load_manifest
from nano_harness.client import ScriptedClient
from nano_harness.grounded_calculator_canary import (
    _is_recovery_eligible,
    load_config,
    parse_and_execute_grounded_expression,
    recover_case,
    verify_frozen_inputs,
)
from nano_harness.types import ModelReply
from scripts.preregister_grounded_calculator_canary_v1 import (
    build_receipt,
    render_markdown,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/harness/qwen35_grounded_calculator_canary_v1.json"
)


class GroundedCalculatorCanaryTests(unittest.TestCase):
    def test_config_and_frozen_inputs_are_exact(self):
        config = load_config(CONFIG)
        frozen = verify_frozen_inputs(config, verify_service=False)
        self.assertEqual(len(frozen["cases"]), 211)
        self.assertEqual(len(frozen["eligible_case_ids"]), 2)
        self.assertEqual(
            config.route_by_benchmark,
            {
                "gsm8k": "grounded_calculator_on_direct_parse_failure",
                "mmlu": "direct_preserve",
                "gpqa_diamond": "direct_preserve",
            },
        )
        self.assertFalse(
            config.execution_boundary["canary_generation_started"]
        )
        self.assertTrue(
            config.execution_boundary["prior_direct_outputs_loaded"]
        )

    def test_grounded_expression_executes_exact_rational_math(self):
        receipt = parse_and_execute_grounded_expression(
            "CALC: 20 + (6 * 6) + (6 * 6)",
            prompt=(
                "Wendy wants 20 more than double the books in a shelving "
                "system with 6 rows and 6 columns."
            ),
            maximum_expression_chars=160,
            maximum_ast_nodes=64,
            maximum_absolute_value=10**15,
        )
        self.assertTrue(receipt["executed"])
        self.assertEqual(receipt["result"], 92)
        self.assertFalse(receipt["executor_uses_expected_answer"])
        self.assertFalse(receipt["executor_uses_case_correctness"])
        self.assertFalse(receipt["case_id_allowlist_used"])

        decimal = parse_and_execute_grounded_expression(
            "CALC: 90 / (7 * 1.5 - 3) + 3 / 3",
            prompt=(
                "It costs $90. Each year there are 7 lemons sold for $1.5 "
                "and costs are $3 a year."
            ),
            maximum_expression_chars=160,
            maximum_ast_nodes=64,
            maximum_absolute_value=10**15,
        )
        self.assertTrue(decimal["executed"])
        self.assertEqual(decimal["result"], 13)

    def test_parser_rejects_target_injection_and_unsafe_syntax(self):
        cases = (
            ("CALC: 20 + 72", "Wendy has 20, 6, and 6.", "ungrounded_literal"),
            ("CALC: 6 ** 2", "There are 6 rows.", "unsupported_binary_operator"),
            ("CALC: abs(6)", "There are 6 rows.", "plan_parse_failure"),
            ("CALC: 6 / (6 - 6)", "There are 6 rows.", "division_by_zero"),
            ("FINAL: 92", "There are 20, 6, and 6.", "plan_parse_failure"),
        )
        for text, prompt, reason in cases:
            with self.subTest(text=text):
                receipt = parse_and_execute_grounded_expression(
                    text,
                    prompt=prompt,
                    maximum_expression_chars=160,
                    maximum_ast_nodes=64,
                    maximum_absolute_value=10**15,
                )
                self.assertFalse(receipt["executed"])
                self.assertEqual(receipt["reason"], reason)

    def test_only_dynamic_gsm8k_parse_failures_are_eligible(self):
        manifest = load_manifest(
            ROOT / "configs/harness/qwen35_three_task_replication_v1.yaml"
        )
        cases = load_cases(
            manifest,
            ROOT.parent.parent.parent / "datasets",
        )
        gsm8k = next(case for case in cases if case.benchmark == "gsm8k")
        mmlu = next(case for case in cases if case.benchmark == "mmlu")
        base = {"status": "completed", "prediction": None}
        self.assertTrue(_is_recovery_eligible(gsm8k, base))
        self.assertFalse(_is_recovery_eligible(mmlu, base))
        self.assertFalse(
            _is_recovery_eligible(
                gsm8k,
                {"status": "completed", "prediction": "1"},
            )
        )
        self.assertFalse(
            _is_recovery_eligible(
                gsm8k,
                {"status": "error", "prediction": None},
            )
        )

    def test_noneligible_row_is_preserved_without_model_calls(self):
        config = load_config(CONFIG)
        frozen = verify_frozen_inputs(config, verify_service=False)
        cases = {case.case_id: case for case in frozen["cases"]}
        direct = next(
            row
            for row in frozen["four_b_rows"]
            if row["prediction"] is not None
        )
        plan_client = ScriptedClient([])
        final_client = ScriptedClient([])
        candidate, receipt = recover_case(
            cases[direct["case_id"]],
            direct,
            plan_client,
            final_client,
            config,
        )
        self.assertEqual(candidate["output"], direct["output"])
        self.assertEqual(candidate["prediction"], direct["prediction"])
        self.assertEqual(candidate["score"], direct["score"])
        self.assertEqual(candidate["canary_route"], "direct_preserve")
        self.assertEqual(plan_client.calls, [])
        self.assertEqual(final_client.calls, [])
        self.assertFalse(receipt["eligible"])

    def test_eligible_row_executes_plan_and_verified_feedback(self):
        config = load_config(CONFIG)
        frozen = verify_frozen_inputs(config, verify_service=False)
        cases = {case.case_id: case for case in frozen["cases"]}
        direct = next(
            row
            for row in frozen["four_b_rows"]
            if row["case_id"] == "gsm8k-b83fb1e0d547a470"
        )
        plan_client = ScriptedClient(
            [ModelReply(content="CALC: 20 + (6 * 6) + (6 * 6)")]
        )
        final_client = ScriptedClient(
            [ModelReply(content="FINAL: 92")]
        )
        candidate, receipt = recover_case(
            cases[direct["case_id"]],
            direct,
            plan_client,
            final_client,
            config,
        )
        self.assertEqual(candidate["prediction"], "92")
        self.assertEqual(candidate["score"], 1.0)
        self.assertEqual(
            candidate["canary_route"],
            "grounded_calculator_verified_feedback",
        )
        self.assertTrue(receipt["receipt"]["executed"])
        self.assertEqual(receipt["receipt"]["result"], 92)
        self.assertEqual(
            plan_client.calls[0]["extra_body"]["structured_outputs"]["regex"],
            config.plan_structured_output_regex,
        )

    def test_invalid_plan_retries_then_falls_back_to_direct(self):
        config = load_config(CONFIG)
        frozen = verify_frozen_inputs(config, verify_service=False)
        cases = {case.case_id: case for case in frozen["cases"]}
        direct = next(
            row
            for row in frozen["four_b_rows"]
            if row["case_id"] == "gsm8k-b83fb1e0d547a470"
        )
        plan_client = ScriptedClient(
            [
                ModelReply(content="CALC: 92"),
                ModelReply(content="CALC: 20 + 72"),
            ]
        )
        final_client = ScriptedClient([])
        candidate, receipt = recover_case(
            cases[direct["case_id"]],
            direct,
            plan_client,
            final_client,
            config,
        )
        self.assertEqual(candidate["output"], direct["output"])
        self.assertIsNone(candidate["prediction"])
        self.assertEqual(
            candidate["canary_route"],
            "direct_fallback_after_invalid_grounded_plan",
        )
        self.assertEqual(len(plan_client.calls), 2)
        self.assertEqual(final_client.calls, [])
        self.assertTrue(receipt["fallback_used"])

    def test_feedback_cannot_change_verified_result(self):
        config = load_config(CONFIG)
        frozen = verify_frozen_inputs(config, verify_service=False)
        cases = {case.case_id: case for case in frozen["cases"]}
        direct = next(
            row
            for row in frozen["four_b_rows"]
            if row["case_id"] == "gsm8k-b83fb1e0d547a470"
        )
        plan_client = ScriptedClient(
            [ModelReply(content="CALC: 20 + (6 * 6) + (6 * 6)")]
        )
        final_client = ScriptedClient(
            [ModelReply(content="FINAL: 20")]
        )
        candidate, receipt = recover_case(
            cases[direct["case_id"]],
            direct,
            plan_client,
            final_client,
            config,
        )
        self.assertEqual(candidate["output"], direct["output"])
        self.assertIsNone(candidate["prediction"])
        self.assertEqual(
            candidate["canary_route"],
            "direct_fallback_after_feedback_result_mismatch",
        )
        self.assertEqual(receipt["reason"], "feedback_result_mismatch")
        self.assertTrue(receipt["fallback_used"])

    def test_config_rejects_route_budget_or_boundary_change(self):
        raw = json.loads(CONFIG.read_text(encoding="utf-8"))
        mutations = (
            ("plan_max_tokens", 128, "plan_max_tokens"),
            ("plan_retry_limit", 2, "plan_retry_limit"),
            (
                "route_by_benchmark",
                {
                    **raw["route_by_benchmark"],
                    "mmlu": "grounded_calculator_on_direct_parse_failure",
                },
                "route_by_benchmark",
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

    def test_preregister_is_deterministic_and_generation_closed(self):
        first = build_receipt()
        second = build_receipt()
        self.assertEqual(first, second)
        self.assertEqual(
            first["surface"]["prior_direct_parse_failure_eligible_rows"],
            2,
        )
        self.assertFalse(
            first["execution_boundary"]["canary_generation_started"]
        )
        self.assertFalse(
            first["execution_boundary"]["model_generation_started"]
        )
        self.assertFalse(first["surface"]["case_id_allowlist_used"])
        markdown = render_markdown(first)
        self.assertIn("只预注册，不生成新的 canary output", markdown)
        self.assertIn("没有 case-ID", markdown)
        self.assertIn("Fraction", markdown)


if __name__ == "__main__":
    unittest.main()
