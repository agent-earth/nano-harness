from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from nano_harness.adapters.common import iter_jsonl, select_shard
from nano_harness.config import BenchmarkConfig
from nano_harness.types import Task, TaskResult


DEFAULT_TASKS = [
    {
        "task_id": "constraint-audit",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Using only these rules: ALPHA means 7; BETA means ALPHA + 5. "
                    "Return JSON with keys result then evidence, no markdown. What is BETA?"
                ),
            }
        ],
        "constraints": [
            "Return valid JSON.",
            "Keep key order result then evidence.",
            "The result field must be the JSON number 12, not a string.",
            "The evidence field must be a string containing the derivation.",
            "Use only supplied definitions.",
        ],
        "validator": "constraint_json",
        "audit_policy": "contract_failure",
        "output_contract": {
            "format": "json",
            "type": "object",
            "required": ["result", "evidence"],
            "exact_key_order": True,
            "property_types": {"result": "number", "evidence": "string"},
        },
    },
    {
        "task_id": "tool-recovery",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Look up account A-17, then report its verified status. Use the tool; "
                    "do not guess."
                ),
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "lookup_account",
                    "description": "Look up an account by id.",
                    "parameters": {
                        "type": "object",
                        "properties": {"account_id": {"type": "string"}},
                        "required": ["account_id"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
        "constraints": ["Use lookup_account.", "Report only observed status."],
        "validator": "observed_active_status",
        "audit_policy": "never",
    },
]


class SyntheticAdapter:
    name = "synthetic"

    def load(self, config: BenchmarkConfig) -> Iterable[Task]:
        records = iter_jsonl(config.source) if Path(config.source).exists() else DEFAULT_TASKS
        for index, record in select_shard(
            records,
            config.start,
            config.limit,
            config.shard_id,
            config.num_shards,
        ):
            yield Task(
                task_id=str(record.get("task_id", index)),
                benchmark=self.name,
                messages=list(record["messages"]),
                metadata={
                    "constraints": record.get("constraints", []),
                    "expected": record.get("expected"),
                    "validator": record.get("validator"),
                    "output_contract": record.get("output_contract"),
                    "audit_policy": record.get("audit_policy", "never"),
                },
                tools=list(record.get("tools", [])),
            )

    def serialize(self, result: TaskResult) -> dict:
        record = result.to_dict()
        validator = result.metadata.get("validator")
        if validator == "constraint_json":
            record["score"] = _validate_constraint_json(result.output)
        elif validator == "observed_active_status":
            used_tool = any(item.get("kind") == "tool" for item in result.trajectory)
            record["score"] = float(used_tool and "active" in result.output.lower())
        return record


def _validate_constraint_json(output: str) -> float:
    try:
        parsed = json.loads(output)
    except (json.JSONDecodeError, TypeError):
        return 0.0
    if list(parsed) != ["result", "evidence"]:
        return 0.0
    evidence = parsed.get("evidence")
    if parsed.get("result") != 12 or not isinstance(evidence, str):
        return 0.0
    compact = evidence.replace(" ", "")
    mentions_definition = "BETA" in evidence.upper() and "+5" in compact
    mentions_alpha_value = "7" in compact and "ALPHA" in evidence.upper()
    return float(mentions_definition and mentions_alpha_value)
