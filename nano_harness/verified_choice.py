from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from nano_harness.analog_contract import sha256_file, summarize_rows


EXPRESSION_PATTERN = re.compile(
    r"(?<![\w.])([-+]?\d+)\s*([+\-*/])\s*([-+]?\d+)(?!\w|\.\d)"
)
OPTION_PATTERN = re.compile(
    r"(?m)^([A-D])\.\s*([-+]?(?:\d+(?:\.\d+)?|\.\d+))\s*$"
)
AVERAGE_PATTERN = re.compile(
    r"\baverage\b.*\b(?:two|2)\b|\b(?:two|2)\b.*\baverage\b",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class VerifiedChoiceConfig:
    schema_version: str
    experiment_id: str
    parser_version: str
    dataset_path: str
    dataset_sha256: str
    baseline_result_path: str
    baseline_result_sha256: str
    output_path: str
    supported_format_family: str
    supported_intent: str
    exact_option_match_required: bool
    ambiguous_fallback: str


def load_config(path: str | Path) -> VerifiedChoiceConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = set(VerifiedChoiceConfig.__dataclass_fields__)
    if set(raw) != expected:
        raise ValueError("verified choice config fields differ")
    config = VerifiedChoiceConfig(**raw)
    frozen = {
        "schema_version": "nano_harness_verified_choice_v1",
        "experiment_id": "anchored-v1-verified-choice-executor-v1",
        "parser_version": "explicit_two_expression_average_v1",
        "supported_format_family": "final_choice",
        "supported_intent": "average_of_two_expression_results",
        "exact_option_match_required": True,
        "ambiguous_fallback": "reuse_direct_output",
    }
    for field, expected_value in frozen.items():
        if getattr(config, field) != expected_value:
            raise ValueError(f"verified choice freezes {field}={expected_value}")
    return config


def _evaluate(left: int, operator: str, right: int) -> Fraction:
    if operator == "+":
        return Fraction(left + right)
    if operator == "-":
        return Fraction(left - right)
    if operator == "*":
        return Fraction(left * right)
    if operator == "/":
        if right == 0:
            raise ValueError("division by zero")
        return Fraction(left, right)
    raise ValueError(f"unsupported operator: {operator}")


def _fraction_text(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def verify_explicit_average_choice(prompt: str) -> dict[str, Any]:
    options = OPTION_PATTERN.findall(prompt)
    expressions = EXPRESSION_PATTERN.findall(prompt.split("\nA.", 1)[0])
    base = {
        "schema_version": "verified_choice_receipt_v1",
        "parser_version": "explicit_two_expression_average_v1",
        "eligible": False,
        "override": False,
        "reason": "",
    }
    if not AVERAGE_PATTERN.search(prompt):
        return {**base, "reason": "unsupported_intent"}
    if len(expressions) != 2:
        return {
            **base,
            "reason": "expression_count_not_two",
            "expression_count": len(expressions),
        }
    if len(options) != 4 or {letter for letter, _ in options} != set("ABCD"):
        return {
            **base,
            "reason": "options_not_four_unique_letters",
            "option_count": len(options),
        }
    option_values = {
        letter: Fraction(value)
        for letter, value in options
    }
    if len(set(option_values.values())) != 4:
        return {**base, "reason": "option_values_not_unique"}
    try:
        expression_values = [
            _evaluate(int(left), operator, int(right))
            for left, operator, right in expressions
        ]
    except (ValueError, ZeroDivisionError):
        return {**base, "reason": "unsafe_expression"}
    result = sum(expression_values, Fraction(0)) / 2
    matches = [
        letter for letter, value in option_values.items() if value == result
    ]
    receipt = {
        **base,
        "eligible": True,
        "expressions": [
            f"{left} {operator} {right}"
            for left, operator, right in expressions
        ],
        "expression_values": [
            _fraction_text(value) for value in expression_values
        ],
        "aggregate_expression": (
            f"({_fraction_text(expression_values[0])} + "
            f"{_fraction_text(expression_values[1])}) / 2"
        ),
        "result": _fraction_text(result),
        "option_values": {
            letter: _fraction_text(value)
            for letter, value in sorted(option_values.items())
        },
        "exact_matching_options": matches,
    }
    if len(matches) != 1:
        return {
            **receipt,
            "reason": "no_unique_exact_option_match",
        }
    return {
        **receipt,
        "override": True,
        "reason": "unique_exact_option_match",
        "selected_letter": matches[0],
    }


def run(config: VerifiedChoiceConfig) -> dict[str, Any]:
    dataset_path = Path(config.dataset_path)
    baseline_path = Path(config.baseline_result_path)
    if sha256_file(dataset_path) != config.dataset_sha256:
        raise ValueError("verified choice dataset identity mismatch")
    if sha256_file(baseline_path) != config.baseline_result_sha256:
        raise ValueError("verified choice baseline identity mismatch")
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    baseline_result = json.loads(baseline_path.read_text(encoding="utf-8"))
    if dataset.get("dataset_id") != "generic-choice-replay-v11":
        raise ValueError("verified choice requires generic choice replay v11")
    if (
        baseline_result.get("experiment_id")
        != "anchored-v1-choice-calculation-selector-v1"
        or baseline_result.get("evaluation_boundary", {}).get(
            "benchmark_rows_loaded"
        )
        is not False
    ):
        raise ValueError("verified choice baseline contract differs")

    samples = {
        str(sample["sample_id"]): sample
        for sample in dataset["samples"]
        if sample["split"] == "validation"
    }
    baseline_rows = baseline_result["baseline_rows"]
    if {row["sample_id"] for row in baseline_rows} != set(samples):
        raise ValueError("verified choice sample identities differ")

    candidate_rows = []
    receipts = {}
    for baseline in baseline_rows:
        sample = samples[baseline["sample_id"]]
        candidate = {
            **baseline,
            "route": "reuse_direct_output",
        }
        if sample["format_family"] == config.supported_format_family:
            prompt = str(sample["messages"][1]["content"])
            receipt = verify_explicit_average_choice(prompt)
            receipt["prompt_sha256"] = hashlib.sha256(
                prompt.encode("utf-8")
            ).hexdigest()
            receipts[baseline["sample_id"]] = receipt
            if receipt["override"]:
                output = f"FINAL: {receipt['selected_letter']}"
                target = str(sample["messages"][-1]["content"])
                candidate = {
                    **baseline,
                    "output": output,
                    "exact": output == target,
                    "semantic_valid": output == target,
                    "route": "verified_choice_override",
                }
        candidate_rows.append(candidate)

    result = {
        "schema_version": "nano_harness_verified_choice_result_v1",
        "experiment_id": config.experiment_id,
        "config": config.__dict__,
        "identity": {
            "dataset_sha256": sha256_file(dataset_path),
            "baseline_result_sha256": sha256_file(baseline_path),
        },
        "baseline": summarize_rows(baseline_rows),
        "candidate": summarize_rows(candidate_rows),
        "baseline_rows": baseline_rows,
        "candidate_rows": candidate_rows,
        "receipts": receipts,
        "routing": {
            "choice_rows": len(receipts),
            "verified_overrides": sum(
                receipt["override"] for receipt in receipts.values()
            ),
            "ambiguous_fallbacks": sum(
                receipt["eligible"] and not receipt["override"]
                for receipt in receipts.values()
            ),
            "ineligible_fallbacks": sum(
                not receipt["eligible"] for receipt in receipts.values()
            ),
            "non_choice_reused": sum(
                sample["format_family"] != config.supported_format_family
                for sample in samples.values()
            ),
        },
        "evaluation_boundary": {
            "target_used_by_parser": False,
            "benchmark_rows_loaded": False,
            "sealed_canary_run": False,
            "prior_full_suite_run": False,
            "independent_holdout_run": False,
        },
    }
    output = Path(config.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result
