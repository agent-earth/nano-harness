from __future__ import annotations

import json
from typing import Any


def contract_errors(output: str, contract: dict[str, Any] | None) -> list[str]:
    if not contract:
        return []
    errors: list[str] = []
    value: Any = output
    if contract.get("format") == "json":
        try:
            value = json.loads(output)
        except json.JSONDecodeError as exc:
            return [f"output must be valid JSON: {exc.msg}"]
    expected_type = contract.get("type")
    if expected_type and not _matches_type(value, expected_type):
        return [f"output must be {expected_type}"]
    if isinstance(value, dict):
        required = contract.get("required", [])
        missing = [key for key in required if key not in value]
        if missing:
            errors.append(f"missing required keys: {missing}")
        if contract.get("exact_key_order") and list(value) != required:
            errors.append(f"keys must appear exactly in this order: {required}")
        for key, type_name in contract.get("property_types", {}).items():
            if key in value and not _matches_type(value[key], type_name):
                errors.append(f"{key} must be {type_name}")
    return errors


def should_audit(output: str, metadata: dict[str, Any]) -> tuple[bool, list[str]]:
    policy = metadata.get("audit_policy", "contract_failure")
    errors = contract_errors(output, metadata.get("output_contract"))
    if policy == "always":
        return True, errors
    if policy == "never":
        return False, errors
    if policy == "missing_patch":
        has_patch = "diff --git " in output or (
            "<<PATCH>>" in output and "<<END_PATCH>>" in output
        )
        return (
            not has_patch,
            ["candidate does not contain a patch"] if not has_patch else [],
        )
    return bool(errors), errors


def _matches_type(value: Any, type_name: str) -> bool:
    if type_name == "object":
        return isinstance(value, dict)
    if type_name == "array":
        return isinstance(value, list)
    if type_name == "string":
        return isinstance(value, str)
    if type_name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if type_name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "boolean":
        return isinstance(value, bool)
    if type_name == "null":
        return value is None
    raise ValueError(f"unsupported contract type: {type_name}")
