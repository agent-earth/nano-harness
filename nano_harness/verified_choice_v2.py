from __future__ import annotations

import re
from fractions import Fraction
from typing import Any

from nano_harness.verified_choice import (
    OPTION_PATTERN,
    verify_explicit_average_choice,
)


HOST_PATTERN = re.compile(
    (
        r"coordinator attends a summit, registers (\d+) delegates, "
        r"and every delegate brings (\d+) guests\. Including the "
        r"coordinator, how many people attend\?"
    ),
    re.IGNORECASE,
)
VERBAL_AVERAGE_PATTERN = re.compile(
    (
        r"north depot processed (\d+) parcels and a south depot "
        r"processed (\d+) parcels\. What is the average number of "
        r"parcels processed by the two depots\?"
    ),
    re.IGNORECASE,
)


def _fraction_text(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _select_option(
    prompt: str,
    *,
    parser_version: str,
    proof_kind: str,
    proof_inputs: dict[str, int],
    expression: str,
    result: Fraction,
) -> dict[str, Any]:
    base = {
        "schema_version": "verified_choice_receipt_v2",
        "parser_version": parser_version,
        "proof_kind": proof_kind,
        "eligible": False,
        "override": False,
        "reason": "",
    }
    options = OPTION_PATTERN.findall(prompt)
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
    matches = [
        letter for letter, value in option_values.items() if value == result
    ]
    receipt = {
        **base,
        "eligible": True,
        "proof_inputs": proof_inputs,
        "expression": expression,
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


def verify_host_count_choice(prompt: str) -> dict[str, Any]:
    match = HOST_PATTERN.search(prompt)
    if match is None:
        return {
            "schema_version": "verified_choice_receipt_v2",
            "parser_version": "host_count_and_verbal_average_v2",
            "proof_kind": "host_count",
            "eligible": False,
            "override": False,
            "reason": "unsupported_host_count_intent",
        }
    invited, guests = (int(value) for value in match.groups())
    result = Fraction(1 + invited + invited * guests)
    return _select_option(
        prompt,
        parser_version="host_count_and_verbal_average_v2",
        proof_kind="host_count",
        proof_inputs={"invited": invited, "guests_per_invitee": guests},
        expression=f"1 + {invited} + {invited} * {guests}",
        result=result,
    )


def verify_verbal_average_choice(prompt: str) -> dict[str, Any]:
    match = VERBAL_AVERAGE_PATTERN.search(prompt)
    if match is None:
        return {
            "schema_version": "verified_choice_receipt_v2",
            "parser_version": "host_count_and_verbal_average_v2",
            "proof_kind": "verbal_average",
            "eligible": False,
            "override": False,
            "reason": "unsupported_verbal_average_intent",
        }
    north, south = (int(value) for value in match.groups())
    result = Fraction(north + south, 2)
    return _select_option(
        prompt,
        parser_version="host_count_and_verbal_average_v2",
        proof_kind="verbal_average",
        proof_inputs={"north": north, "south": south},
        expression=f"({north} + {south}) / 2",
        result=result,
    )


def verify_choice_v2(prompt: str) -> dict[str, Any]:
    host = verify_host_count_choice(prompt)
    if host["reason"] != "unsupported_host_count_intent":
        return host
    verbal = verify_verbal_average_choice(prompt)
    if verbal["reason"] != "unsupported_verbal_average_intent":
        return verbal
    v1 = verify_explicit_average_choice(prompt)
    return {
        **v1,
        "composite_parser_version": "host_count_and_verbal_average_v2",
        "proof_kind": "explicit_average_v1",
    }
