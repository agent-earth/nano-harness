from __future__ import annotations

import ast
import hashlib
import json
import re
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from nano_harness.baseline import (
    BaselineCase,
    case_manifest,
    compare_baselines,
    load_cases,
    load_manifest,
    score_output,
    sha256_file,
    summarize_baseline,
)
from nano_harness.verified_tool_execution import (
    _client,
    _sum_usage,
    verify_inputs as verify_service_inputs,
)
from nano_harness.verified_tool_execution_v2 import (
    load_config as load_v2_config,
    parent_config,
)


CONFIG_SCHEMA = "nano_harness_grounded_calculator_canary_v1"
RESULT_SCHEMA = "nano_harness_grounded_calculator_canary_result_v1"
PLAN_REGEX = r"CALC: [0-9+\-*/(). ]+"
PLAN_PATTERN = re.compile(r"^CALC: ([0-9+\-*/(). ]+)$")
SOURCE_NUMBER_PATTERN = re.compile(
    r"(?<![\w.])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
    r"(?!\w|\.\d)"
    r"|(?<![\w.])\.\d+(?!\w)"
)
ALLOWED_BINARY_OPERATORS = {
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
}
ALLOWED_UNARY_OPERATORS = {
    ast.UAdd,
    ast.USub,
}


@dataclass(frozen=True)
class GroundedCalculatorCanaryConfig:
    schema_version: str
    experiment_id: str
    manifest_path: str
    manifest_sha256: str
    case_manifest_path: str
    case_manifest_sha256: str
    dataset_root: str
    four_b_raw_path: str
    four_b_raw_sha256: str
    nine_b_raw_path: str
    nine_b_raw_sha256: str
    baseline_report_path: str
    baseline_report_sha256: str
    v2_config_path: str
    v2_config_sha256: str
    v2_preregister_path: str
    v2_preregister_sha256: str
    v2_report_path: str
    v2_report_sha256: str
    service_receipt_sha256: str
    output_path: str
    candidate_output_path: str
    route_by_benchmark: dict[str, str]
    recovery_eligibility: dict[str, Any]
    plan_structured_output_regex: str
    plan_max_tokens: int
    final_max_tokens: int
    plan_retry_limit: int
    maximum_expression_chars: int
    maximum_ast_nodes: int
    maximum_absolute_value: int
    admission_gates: dict[str, Any]
    policy: dict[str, bool]
    execution_boundary: dict[str, bool]


def load_config(path: str | Path) -> GroundedCalculatorCanaryConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(raw) != set(GroundedCalculatorCanaryConfig.__dataclass_fields__):
        raise ValueError("grounded calculator canary config fields differ")
    config = GroundedCalculatorCanaryConfig(**raw)
    expected = {
        "schema_version": CONFIG_SCHEMA,
        "experiment_id": "qwen35-grounded-calculator-canary-v1",
        "manifest_path": (
            "configs/harness/qwen35_three_task_replication_v1.yaml"
        ),
        "manifest_sha256": (
            "88f6e832d38e739c6b622a30633a2737077fc081037e6e1543cb5763b169a7b9"
        ),
        "case_manifest_path": (
            "configs/generated/qwen35_three_task_replication_v1_cases.json"
        ),
        "case_manifest_sha256": (
            "eafbe4d42487a225322dd3b3bdc1d805c065fb15f0f8b968e65ccf747f96976f"
        ),
        "dataset_root": "../../../datasets",
        "four_b_raw_path": (
            "../../nano-harness/results/harness/"
            "qwen35-three-task-replication-v1/4b/cases.jsonl"
        ),
        "four_b_raw_sha256": (
            "c59383d3fd3d6087025d6e1ff649979d9d5a9e8dc73b5429a4f8e9fa41b6b8c7"
        ),
        "nine_b_raw_path": (
            "../../nano-harness/results/harness/"
            "qwen35-three-task-replication-v1/9b/cases.jsonl"
        ),
        "nine_b_raw_sha256": (
            "ffae93774d51b87a2e29258d170a84f8b165f996e2e78eedd102271dfc260044"
        ),
        "baseline_report_path": (
            "docs/results/three_task_replication_v1.public.json"
        ),
        "baseline_report_sha256": (
            "e7aad4abf515de23289359a04a589e0f87d29493cf4b172a8213e369a0cdbd4f"
        ),
        "v2_config_path": (
            "configs/harness/qwen35_verified_tool_execution_v2.json"
        ),
        "v2_config_sha256": (
            "ae6740e37da66b393f0732e7d86b785148e9d6fc663cbbdea0c8554d68f5ae0f"
        ),
        "v2_preregister_path": (
            "docs/experiments/"
            "qwen35_verified_tool_execution_v2.preregister.json"
        ),
        "v2_preregister_sha256": (
            "1c312a575b4c4bc4000e64495f3157ceff529361d5c0dc43112b318faf1c0797"
        ),
        "v2_report_path": (
            "docs/results/qwen35_verified_tool_execution_v2.public.json"
        ),
        "v2_report_sha256": (
            "cd20bd3f6abccf3e8b70f8ec6504150dc30665fbe477364608ec8b34366ab0cc"
        ),
        "service_receipt_sha256": (
            "895bd40d886d870f99230441baecdf1feb926e2992b6a151e54d8165145f1c0d"
        ),
        "output_path": (
            "results/harness/"
            "qwen35-grounded-calculator-canary-v1/result.json"
        ),
        "candidate_output_path": (
            "results/harness/"
            "qwen35-grounded-calculator-canary-v1/candidate.jsonl"
        ),
        "route_by_benchmark": {
            "gsm8k": "grounded_calculator_on_direct_parse_failure",
            "mmlu": "direct_preserve",
            "gpqa_diamond": "direct_preserve",
        },
        "recovery_eligibility": {
            "benchmark": "gsm8k",
            "direct_status": "completed",
            "direct_prediction_is_none": True,
            "case_id_allowlist_used": False,
            "expected_answer_used": False,
            "case_correctness_used": False,
        },
        "plan_structured_output_regex": PLAN_REGEX,
        "plan_max_tokens": 96,
        "final_max_tokens": 32,
        "plan_retry_limit": 1,
        "maximum_expression_chars": 160,
        "maximum_ast_nodes": 64,
        "maximum_absolute_value": 10**15,
        "admission_gates": {
            "cases": 211,
            "minimum_overall_correct": 164,
            "benchmark_minimum_correct": {
                "gsm8k": 90,
                "mmlu": 67,
                "gpqa_diamond": 6,
            },
            "maximum_parse_failures": 2,
            "maximum_api_errors": 0,
            "paired_candidate_only_wins_gt_base_only_wins": True,
            "require_exact_case_identity": True,
            "require_direct_preservation_outside_eligible_rows": True,
            "require_zero_unsafe_executions": True,
        },
        "policy": {
            "evaluation_only": True,
            "training_eligible": False,
            "contains_canary_rows": True,
            "contains_prior_direct_canary_outputs": True,
            "contains_candidate_canary_outputs_before_run": False,
            "contains_holdout_rows": False,
            "uses_expected_answer_for_routing": False,
            "executor_uses_expected_answer": False,
            "executor_uses_case_correctness": False,
            "case_id_allowlist_used": False,
            "post_observation_prompt_parser_budget_search": False,
        },
        "execution_boundary": {
            "service_reused": True,
            "canary_inputs_audited": True,
            "prior_direct_outputs_loaded": True,
            "model_generation_started": False,
            "canary_generation_started": False,
            "complete_benchmark_started": False,
            "independent_holdout_accessed": False,
            "training_started": False,
            "this_commit_only_preregisters": True,
        },
    }
    for field, expected_value in expected.items():
        if getattr(config, field) != expected_value:
            raise ValueError(
                f"grounded calculator canary freezes {field}={expected_value}"
            )
    return config


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _rows_by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output = {str(row["case_id"]): row for row in rows}
    if len(output) != len(rows):
        raise ValueError("grounded calculator raw rows contain duplicate cases")
    return output


def verify_frozen_inputs(
    config: GroundedCalculatorCanaryConfig,
    *,
    verify_service: bool,
) -> dict[str, Any]:
    paths = {
        "manifest": Path(config.manifest_path),
        "case_manifest": Path(config.case_manifest_path),
        "four_b_raw": Path(config.four_b_raw_path),
        "nine_b_raw": Path(config.nine_b_raw_path),
        "baseline_report": Path(config.baseline_report_path),
        "v2_config": Path(config.v2_config_path),
        "v2_preregister": Path(config.v2_preregister_path),
        "v2_report": Path(config.v2_report_path),
    }
    expected_hashes = {
        "manifest": config.manifest_sha256,
        "case_manifest": config.case_manifest_sha256,
        "four_b_raw": config.four_b_raw_sha256,
        "nine_b_raw": config.nine_b_raw_sha256,
        "baseline_report": config.baseline_report_sha256,
        "v2_config": config.v2_config_sha256,
        "v2_preregister": config.v2_preregister_sha256,
        "v2_report": config.v2_report_sha256,
    }
    for name, path in paths.items():
        if not path.is_file() or sha256_file(path) != expected_hashes[name]:
            raise ValueError(f"grounded calculator {name} identity differs")

    manifest = load_manifest(paths["manifest"])
    cases = load_cases(manifest, Path(config.dataset_root))
    committed_cases = json.loads(
        paths["case_manifest"].read_text(encoding="utf-8")
    )
    if case_manifest(cases) != committed_cases or len(cases) != 211:
        raise ValueError("grounded calculator case contract differs")

    case_by_id = {case.case_id: case for case in cases}
    committed_by_id = {
        str(row["case_id"]): row for row in committed_cases
    }
    raw = {
        "four_b": _jsonl(paths["four_b_raw"]),
        "nine_b": _jsonl(paths["nine_b_raw"]),
    }
    for arm, rows in raw.items():
        by_id = _rows_by_id(rows)
        if set(by_id) != set(case_by_id) or len(rows) != 211:
            raise ValueError(f"grounded calculator {arm} case identity differs")
        expected_model = "qwen3.5-4b" if arm == "four_b" else "qwen3.5-9b"
        for case_id, case in case_by_id.items():
            row = by_id[case_id]
            committed = committed_by_id[case_id]
            if (
                row.get("model") != expected_model
                or row.get("suite_id") != manifest.suite_id
                or row.get("benchmark") != case.benchmark
                or row.get("source_index") != case.source_index
                or row.get("max_tokens") != case.max_tokens
                or row.get("expected") != case.expected
                or row.get("prompt_sha256") != committed["prompt_sha256"]
                or row.get("system_prompt_sha256")
                != committed["system_prompt_sha256"]
                or row.get("selected_strategy") != "direct"
                or row.get("status") != "completed"
            ):
                raise ValueError(
                    f"grounded calculator {arm} row parity differs: {case_id}"
                )

    baseline_report = json.loads(
        paths["baseline_report"].read_text(encoding="utf-8")
    )
    if (
        baseline_report.get("comparison", {}).get("cases") != 211
        or baseline_report.get("artifacts", {}).get("four_b_raw_sha256")
        != config.four_b_raw_sha256
        or baseline_report.get("artifacts", {}).get("nine_b_raw_sha256")
        != config.nine_b_raw_sha256
    ):
        raise ValueError("grounded calculator baseline report differs")
    v2_report = json.loads(paths["v2_report"].read_text(encoding="utf-8"))
    if (
        v2_report.get("decision", {}).get("local_harness_admitted") is not True
        or v2_report.get("decision", {}).get(
            "canary_preregistration_allowed"
        )
        is not True
        or v2_report.get("decision", {}).get("canary_generation_allowed")
        is not False
    ):
        raise ValueError("grounded calculator v2 admission differs")

    v2 = load_v2_config(paths["v2_config"])
    parent = parent_config(v2)
    if config.service_receipt_sha256 != v2.service_receipt_sha256:
        raise ValueError("grounded calculator service identity differs")
    service = verify_service_inputs(parent) if verify_service else None
    four_by_id = _rows_by_id(raw["four_b"])
    eligible = [
        case.case_id
        for case in cases
        if _is_recovery_eligible(case, four_by_id[case.case_id])
    ]
    return {
        "manifest": manifest,
        "cases": cases,
        "four_b_rows": raw["four_b"],
        "nine_b_rows": raw["nine_b"],
        "eligible_case_ids": eligible,
        "eligible_case_ids_sha256": _sha256_lines(eligible),
        "service_receipt": service,
        "parent_config": parent,
    }


def _sha256_lines(values: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode()).hexdigest()


def _is_recovery_eligible(
    case: BaselineCase,
    direct: dict[str, Any],
) -> bool:
    return (
        case.benchmark == "gsm8k"
        and direct.get("status") == "completed"
        and direct.get("prediction") is None
    )


def _source_numbers(prompt: str) -> set[Fraction]:
    return {
        Fraction(match.group(0).replace(",", ""))
        for match in SOURCE_NUMBER_PATTERN.finditer(prompt)
    }


def parse_and_execute_grounded_expression(
    text: str,
    *,
    prompt: str,
    maximum_expression_chars: int,
    maximum_ast_nodes: int,
    maximum_absolute_value: int,
) -> dict[str, Any]:
    base = {
        "schema_version": "nano_harness_grounded_calculator_receipt_v1",
        "eligible": True,
        "executed": False,
        "reason": "",
        "executor_uses_expected_answer": False,
        "executor_uses_case_correctness": False,
        "case_id_allowlist_used": False,
    }
    match = PLAN_PATTERN.fullmatch(text.strip())
    if not match:
        return {**base, "reason": "plan_parse_failure"}
    expression = match.group(1).strip()
    if not expression or len(expression) > maximum_expression_chars:
        return {**base, "reason": "expression_length_failure"}
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return {**base, "reason": "expression_syntax_failure"}
    nodes = list(ast.walk(tree))
    if len(nodes) > maximum_ast_nodes:
        return {**base, "reason": "expression_node_limit"}
    grounded_numbers = _source_numbers(prompt)

    def evaluate(node: ast.AST) -> Fraction:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(
                node.value, (int, float)
            ):
                raise ValueError("unsupported_literal")
            source = ast.get_source_segment(expression, node)
            if source is None:
                raise ValueError("literal_source_missing")
            value = Fraction(source)
            if value not in grounded_numbers:
                raise ValueError("ungrounded_literal")
            return value
        if isinstance(node, ast.UnaryOp):
            if type(node.op) not in ALLOWED_UNARY_OPERATORS:
                raise ValueError("unsupported_unary_operator")
            value = evaluate(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp):
            if type(node.op) not in ALLOWED_BINARY_OPERATORS:
                raise ValueError("unsupported_binary_operator")
            left = evaluate(node.left)
            right = evaluate(node.right)
            if isinstance(node.op, ast.Add):
                value = left + right
            elif isinstance(node.op, ast.Sub):
                value = left - right
            elif isinstance(node.op, ast.Mult):
                value = left * right
            else:
                if right == 0:
                    raise ValueError("division_by_zero")
                value = left / right
            if (
                abs(value.numerator) > maximum_absolute_value
                or abs(value.denominator) > maximum_absolute_value
            ):
                raise ValueError("expression_magnitude_limit")
            return value
        raise ValueError("unsupported_ast_node")

    try:
        value = evaluate(tree)
    except (ValueError, ZeroDivisionError) as exc:
        return {**base, "reason": str(exc)}
    if value.denominator != 1:
        return {**base, "reason": "non_integral_result"}
    return {
        **base,
        "executed": True,
        "reason": "verified_grounded_execution",
        "expression": expression,
        "expression_sha256": hashlib.sha256(expression.encode()).hexdigest(),
        "result": value.numerator,
        "grounded_source_value_count": len(grounded_numbers),
    }


def _preserved_row(
    direct: dict[str, Any],
    *,
    route: str,
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        **direct,
        "model": "qwen3.5-4b+grounded-calculator-recovery-v1",
        "canary_route": route,
        "usage": usage if usage is not None else direct.get("usage", {}),
    }


def recover_case(
    case: BaselineCase,
    direct: dict[str, Any],
    plan_client: Any,
    final_client: Any,
    config: GroundedCalculatorCanaryConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not _is_recovery_eligible(case, direct):
        return (
            _preserved_row(direct, route="direct_preserve"),
            {
                "eligible": False,
                "reason": "direct_preserve",
                "plan_calls": 0,
                "final_feedback_calls": 0,
                "api_errors": 0,
                "fallback_used": False,
            },
        )

    started = time.perf_counter()
    plan_messages = [
        {
            "role": "system",
            "content": (
                "The GSM8K calculator-recovery skill was selected because the "
                "existing direct response lacked a parseable final line. "
                "Translate the original problem into one arithmetic expression. "
                "Every numeric literal must copy a numeric value from the "
                "original task; a source value may be reused. Use only +, -, *, "
                "/, and parentheses. Do not answer, estimate, or explain. Return "
                "only CALC: <expression>."
            ),
        },
        {"role": "user", "content": case.prompt},
    ]
    attempts = []
    usages = []
    receipt = None
    api_errors = 0
    for attempt in range(config.plan_retry_limit + 1):
        try:
            reply = plan_client.complete(
                plan_messages,
                extra_body={
                    "structured_outputs": {
                        "regex": config.plan_structured_output_regex
                    }
                },
            )
        except Exception as exc:
            api_errors += 1
            attempts.append(
                {
                    "attempt": attempt + 1,
                    "reason": "model_api_error",
                    "error_type": type(exc).__name__,
                    "executed": False,
                }
            )
            receipt = {
                "eligible": True,
                "executed": False,
                "reason": "model_api_error",
                "executor_uses_expected_answer": False,
                "executor_uses_case_correctness": False,
            }
            break
        usages.append(reply.usage)
        receipt = parse_and_execute_grounded_expression(
            reply.content,
            prompt=case.prompt,
            maximum_expression_chars=config.maximum_expression_chars,
            maximum_ast_nodes=config.maximum_ast_nodes,
            maximum_absolute_value=config.maximum_absolute_value,
        )
        attempts.append(
            {
                "attempt": attempt + 1,
                "output": reply.content,
                "output_sha256": hashlib.sha256(
                    reply.content.encode()
                ).hexdigest(),
                "reason": receipt["reason"],
                "executed": receipt["executed"],
            }
        )
        if receipt["executed"]:
            break
        if attempt < config.plan_retry_limit:
            plan_messages.extend(
                [
                    {"role": "assistant", "content": reply.content},
                    {
                        "role": "user",
                        "content": (
                            "The strict grounded calculator rejected this plan "
                            f"with reason={receipt['reason']}. Retry using only "
                            "numeric literals copied from the original task and "
                            "only +, -, *, /, and parentheses."
                        ),
                    },
                ]
            )
    assert receipt is not None
    if not receipt["executed"]:
        return (
            _preserved_row(
                direct,
                route="direct_fallback_after_invalid_grounded_plan",
                usage=_sum_usage(direct.get("usage", {}), *usages),
            ),
            {
                "eligible": True,
                "reason": receipt["reason"],
                "plan_attempts": attempts,
                "receipt": receipt,
                "plan_calls": len(attempts),
                "final_feedback_calls": 0,
                "api_errors": api_errors,
                "fallback_used": True,
                "latency_seconds": time.perf_counter() - started,
            },
        )

    feedback = (
        f"<original_task>\n{case.prompt}\n</original_task>\n\n"
        f"<verified_calculator>\nexpression={receipt['expression']}\n"
        f"result={receipt['result']}\n</verified_calculator>\n\n"
        "Use the verified result as authoritative. Return only "
        "FINAL: <number>."
    )
    try:
        final = final_client.complete(
            [
                {
                    "role": "system",
                    "content": (
                        "Return the verified calculator result without changing "
                        "it. Return only one FINAL: <number> line."
                    ),
                },
                {"role": "user", "content": feedback},
            ],
            extra_body={
                "structured_outputs": {
                    "regex": r"FINAL: -?[0-9]+"
                }
            },
        )
    except Exception as exc:
        api_errors += 1
        return (
            _preserved_row(
                direct,
                route="direct_fallback_after_feedback_api_error",
                usage=_sum_usage(direct.get("usage", {}), *usages),
            ),
            {
                "eligible": True,
                "reason": "feedback_model_api_error",
                "error_type": type(exc).__name__,
                "plan_attempts": attempts,
                "receipt": receipt,
                "plan_calls": len(attempts),
                "final_feedback_calls": 1,
                "api_errors": api_errors,
                "fallback_used": True,
                "latency_seconds": time.perf_counter() - started,
            },
        )
    usages.append(final.usage)
    output = final.content.strip()
    score, prediction = score_output(output, case.expected, case.scorer)
    if prediction is None:
        return (
            _preserved_row(
                direct,
                route="direct_fallback_after_unparseable_feedback",
                usage=_sum_usage(direct.get("usage", {}), *usages),
            ),
            {
                "eligible": True,
                "reason": "feedback_parse_failure",
                "plan_attempts": attempts,
                "receipt": receipt,
                "plan_calls": len(attempts),
                "final_feedback_calls": 1,
                "api_errors": api_errors,
                "fallback_used": True,
                "latency_seconds": time.perf_counter() - started,
            },
        )
    if int(prediction) != receipt["result"]:
        return (
            _preserved_row(
                direct,
                route="direct_fallback_after_feedback_result_mismatch",
                usage=_sum_usage(direct.get("usage", {}), *usages),
            ),
            {
                "eligible": True,
                "reason": "feedback_result_mismatch",
                "plan_attempts": attempts,
                "receipt": receipt,
                "plan_calls": len(attempts),
                "final_feedback_calls": 1,
                "api_errors": api_errors,
                "fallback_used": True,
                "latency_seconds": time.perf_counter() - started,
            },
        )
    row = {
        **direct,
        "model": "qwen3.5-4b+grounded-calculator-recovery-v1",
        "selected_strategy": "grounded_calculator_recovery",
        "canary_route": "grounded_calculator_verified_feedback",
        "output": output,
        "prediction": prediction,
        "score": score,
        "status": "completed",
        "finish_reason": _finish_reason(final.raw),
        "latency_seconds": round(time.perf_counter() - started, 6),
        "usage": _sum_usage(direct.get("usage", {}), *usages),
        "stages": {
            **direct.get("stages", {}),
            "grounded_calculator": {
                "plan_max_tokens": config.plan_max_tokens,
                "final_max_tokens": config.final_max_tokens,
                "plan_attempts": len(attempts),
                "expression_sha256": receipt["expression_sha256"],
                "verified_result": receipt["result"],
                "feedback_sha256": hashlib.sha256(
                    feedback.encode()
                ).hexdigest(),
            },
        },
    }
    return (
        row,
        {
            "eligible": True,
            "reason": "verified_grounded_feedback",
            "plan_attempts": attempts,
            "receipt": receipt,
            "plan_calls": len(attempts),
            "final_feedback_calls": 1,
            "api_errors": api_errors,
            "fallback_used": False,
            "latency_seconds": time.perf_counter() - started,
        },
    )


def _finish_reason(raw: dict[str, Any]) -> str | None:
    choices = raw.get("choices", [])
    if not choices:
        return None
    return choices[0].get("finish_reason")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def evaluate_admission(
    config: GroundedCalculatorCanaryConfig,
    candidate_summary: dict[str, Any],
    comparison: dict[str, Any],
    receipts: dict[str, dict[str, Any]],
    direct_preservation_failures: list[str],
) -> dict[str, bool]:
    by_benchmark = candidate_summary["benchmarks"]
    overall = comparison["overall_micro"]
    gates = config.admission_gates
    return {
        "exact_211_case_identity": candidate_summary["total_cases"]
        == gates["cases"],
        "overall_correct_at_least_164": sum(
            row["correct"] for row in by_benchmark.values()
        )
        >= gates["minimum_overall_correct"],
        "gsm8k_at_least_90": by_benchmark["gsm8k"]["correct"]
        >= gates["benchmark_minimum_correct"]["gsm8k"],
        "mmlu_at_least_67": by_benchmark["mmlu"]["correct"]
        >= gates["benchmark_minimum_correct"]["mmlu"],
        "gpqa_diamond_at_least_6": by_benchmark["gpqa_diamond"]["correct"]
        >= gates["benchmark_minimum_correct"]["gpqa_diamond"],
        "parse_failures_at_most_2": sum(
            row["parse_failures"] for row in by_benchmark.values()
        )
        <= gates["maximum_parse_failures"],
        "candidate_api_errors_zero": candidate_summary["error_cases"]
        <= gates["maximum_api_errors"],
        "recovery_api_errors_zero": sum(
            row["api_errors"] for row in receipts.values()
        )
        == 0,
        "candidate_only_gt_base_only": (
            overall["paired_counts"]["candidate_only"]
            > overall["paired_counts"]["baseline_only"]
        ),
        "direct_preservation_exact": not direct_preservation_failures,
        "unsafe_executions_zero": all(
            row["receipt"].get("executor_uses_expected_answer") is False
            and row["receipt"].get("executor_uses_case_correctness") is False
            and row["receipt"].get("case_id_allowlist_used", False) is False
            for row in receipts.values()
            if "receipt" in row
        ),
    }


def run(config: GroundedCalculatorCanaryConfig) -> dict[str, Any]:
    frozen = verify_frozen_inputs(config, verify_service=True)
    cases = frozen["cases"]
    four_rows = frozen["four_b_rows"]
    four_by_id = _rows_by_id(four_rows)
    parent = frozen["parent_config"]
    plan_client = _client(
        parent,
        four_b=True,
        max_tokens=config.plan_max_tokens,
    )
    final_client = _client(
        parent,
        four_b=True,
        max_tokens=config.final_max_tokens,
    )
    candidate_rows = []
    receipts = {}
    for case in cases:
        candidate, receipt = recover_case(
            case,
            four_by_id[case.case_id],
            plan_client,
            final_client,
            config,
        )
        candidate_rows.append(candidate)
        receipts[case.case_id] = receipt

    candidate_path = Path(config.candidate_output_path)
    _write_jsonl(candidate_path, candidate_rows)
    comparison = compare_baselines(
        candidate_path,
        Path(config.four_b_raw_path),
        bootstrap_samples=10_000,
        bootstrap_seed=35,
    )
    versus_nine = compare_baselines(
        candidate_path,
        Path(config.nine_b_raw_path),
        bootstrap_samples=10_000,
        bootstrap_seed=35,
    )
    candidate_summary = summarize_baseline(candidate_path)
    direct_preservation_failures = []
    for candidate in candidate_rows:
        direct = four_by_id[candidate["case_id"]]
        if candidate["case_id"] not in frozen["eligible_case_ids"]:
            for field in (
                "output",
                "prediction",
                "score",
                "status",
                "finish_reason",
                "prompt_sha256",
                "system_prompt_sha256",
            ):
                if candidate.get(field) != direct.get(field):
                    direct_preservation_failures.append(
                        f"{candidate['case_id']}:{field}"
                    )
    admission = evaluate_admission(
        config,
        candidate_summary,
        comparison,
        receipts,
        direct_preservation_failures,
    )
    result = {
        "schema_version": RESULT_SCHEMA,
        "experiment_id": config.experiment_id,
        "config": config.__dict__,
        "identity": {
            "manifest_sha256": config.manifest_sha256,
            "case_manifest_sha256": config.case_manifest_sha256,
            "four_b_raw_sha256": config.four_b_raw_sha256,
            "nine_b_raw_sha256": config.nine_b_raw_sha256,
            "v2_config_sha256": config.v2_config_sha256,
            "v2_preregister_sha256": config.v2_preregister_sha256,
            "v2_report_sha256": config.v2_report_sha256,
            "service_receipt_sha256": config.service_receipt_sha256,
            "eligible_case_ids_sha256": frozen[
                "eligible_case_ids_sha256"
            ],
            "candidate_raw_sha256": sha256_file(candidate_path),
        },
        "candidate": candidate_summary,
        "comparisons": {
            "versus_frozen_four_b_direct": comparison,
            "versus_frozen_nine_b_direct": versus_nine,
        },
        "routing": {
            "eligible_rows": len(frozen["eligible_case_ids"]),
            "direct_preserve_rows": sum(
                not row["eligible"] for row in receipts.values()
            ),
            "verified_executions": sum(
                row.get("receipt", {}).get("executed", False)
                for row in receipts.values()
            ),
            "fallbacks": sum(
                row["fallback_used"] for row in receipts.values()
            ),
            "plan_calls": sum(
                row["plan_calls"] for row in receipts.values()
            ),
            "final_feedback_calls": sum(
                row["final_feedback_calls"] for row in receipts.values()
            ),
            "api_errors": sum(
                row["api_errors"] for row in receipts.values()
            ),
        },
        "receipts": receipts,
        "direct_preservation_failures": direct_preservation_failures,
        "admission_gates": admission,
        "decision": {
            "canary_passed": all(admission.values()),
            "complete_benchmark_preregistration_allowed": all(
                admission.values()
            ),
            "complete_benchmark_generation_allowed": False,
            "independent_holdout_allowed": False,
            "further_tuning_on_observed_canary_allowed": False,
        },
        "evaluation_boundary": {
            "training_eligible_cases": 0,
            "executor_uses_expected_answer": False,
            "executor_uses_case_correctness": False,
            "case_id_allowlist_used": False,
            "canary_rows_loaded": True,
            "canary_outputs_generated": True,
            "complete_benchmark_rows_loaded": False,
            "independent_holdout_rows_loaded": False,
        },
    }
    output = Path(config.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result
