from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as parquet

from nano_harness.baseline import load_manifest, sha256_file
from nano_harness.semantic_skill_execution import (
    load_config as load_mechanism_config,
    route_prompt,
)


CONFIG_SCHEMA = "nano_harness_semantic_skill_applicability_v1"
RESULT_SCHEMA = "nano_harness_semantic_skill_applicability_result_v1"
LABELED_FACT_PATTERNS = {
    "implicit_scale_total": re.compile(
        r"\brows=(?P<rows>[0-9]+)\b.*"
        r"\bcolumns=(?P<columns>[0-9]+)\b.*"
        r"\bextra=(?P<extra>[0-9]+)\b"
    ),
    "first_strict_profit_period": re.compile(
        r"\bsetup_cost=(?P<setup_cost>[0-9]+)\b.*"
        r"\bunits_per_period=(?P<units_per_period>[0-9]+)\b.*"
        r"\bprice_per_unit=(?P<price_per_unit>[0-9]+)\b.*"
        r"\brecurring_cost=(?P<recurring_cost>[0-9]+)\b"
    ),
}


@dataclass(frozen=True)
class SemanticSkillApplicabilityConfig:
    schema_version: str
    experiment_id: str
    mechanism_config_path: str
    mechanism_config_sha256: str
    mechanism_report_path: str
    mechanism_report_sha256: str
    replication_report_path: str
    replication_report_sha256: str
    complete_manifest_path: str
    complete_manifest_sha256: str
    complete_case_manifest_path: str
    complete_case_manifest_sha256: str
    complete_report_path: str
    complete_report_sha256: str
    dataset_root: str
    output_path: str
    question_column_by_benchmark: dict[str, str]
    expected_cases_by_benchmark: dict[str, int]
    source_fact_extractor: str
    minimum_eligible_rows_for_transfer: int
    policy: dict[str, bool]
    execution_boundary: dict[str, bool]


def load_config(path: str | Path) -> SemanticSkillApplicabilityConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(raw) != set(SemanticSkillApplicabilityConfig.__dataclass_fields__):
        raise ValueError("semantic applicability config fields differ")
    config = SemanticSkillApplicabilityConfig(**raw)
    expected = {
        "schema_version": CONFIG_SCHEMA,
        "experiment_id": "qwen35-semantic-skill-applicability-v1",
        "mechanism_config_path": (
            "configs/harness/qwen35_semantic_skill_execution_v1.json"
        ),
        "mechanism_config_sha256": (
            "4e5785b9b4fc9dd5b3a76a95b438c2f2c3d505a2fae6231b81cfbf53ebbd6ad9"
        ),
        "mechanism_report_path": (
            "docs/results/qwen35_semantic_skill_execution_v1.public.json"
        ),
        "mechanism_report_sha256": (
            "fe53a512cbf0b6ada65ed3ae27c5f3dc90165e367cfecdb58307dd030d017d5f"
        ),
        "replication_report_path": (
            "docs/results/qwen35_semantic_skill_replication_v1.public.json"
        ),
        "replication_report_sha256": (
            "da143f04a30775f73a147374844deb3b7b0534a0d2bd40a55cc9af9fc14335f0"
        ),
        "complete_manifest_path": (
            "configs/baselines/qwen35_complete_direct_v1.yaml"
        ),
        "complete_manifest_sha256": (
            "6ec49d522892975e4532954d3bac7c7e5ed9b24e2c698700d5f8a61667753e90"
        ),
        "complete_case_manifest_path": (
            "configs/generated/qwen35_complete_direct_v1_cases.public.json"
        ),
        "complete_case_manifest_sha256": (
            "858656f58decf8bbc23c70101dabcffc6ef12e049771e043575927743c6cfd10"
        ),
        "complete_report_path": (
            "docs/results/qwen35_complete_direct_v1.public.json"
        ),
        "complete_report_sha256": (
            "c9e622872d6ecd87350475011dcd19534c4f574a2999e69541d9b66ae0985152"
        ),
        "dataset_root": "../../../datasets",
        "output_path": (
            "results/harness/qwen35-semantic-skill-applicability-v1/result.json"
        ),
        "question_column_by_benchmark": {
            "gsm8k": "question",
            "mmlu": "question",
            "gpqa_diamond": "question",
        },
        "expected_cases_by_benchmark": {
            "gsm8k": 1319,
            "mmlu": 14042,
            "gpqa_diamond": 198,
        },
        "source_fact_extractor": "exact_labeled_integer_fields_v1",
        "minimum_eligible_rows_for_transfer": 1,
        "policy": {
            "scan_only": True,
            "evaluation_only": True,
            "training_eligible": False,
            "loads_question_column_only": True,
            "loads_answer_columns": False,
            "loads_choices_column": False,
            "loads_model_outputs": False,
            "uses_expected_answer": False,
            "uses_case_correctness": False,
            "router_uses_case_metadata": False,
            "post_scan_route_or_extractor_change": False,
        },
        "execution_boundary": {
            "scan_started": False,
            "model_generation_started": False,
            "benchmark_generation_started": False,
            "canary_rerun_started": False,
            "holdout_accessed": False,
            "training_started": False,
            "this_commit_only_preregisters": True,
        },
    }
    for field, expected_value in expected.items():
        if getattr(config, field) != expected_value:
            raise ValueError(
                f"semantic applicability freezes {field}={expected_value}"
            )
    for path_value, digest in (
        (config.mechanism_config_path, config.mechanism_config_sha256),
        (config.mechanism_report_path, config.mechanism_report_sha256),
        (config.replication_report_path, config.replication_report_sha256),
        (config.complete_manifest_path, config.complete_manifest_sha256),
        (
            config.complete_case_manifest_path,
            config.complete_case_manifest_sha256,
        ),
        (config.complete_report_path, config.complete_report_sha256),
    ):
        if sha256_file(Path(path_value)) != digest:
            raise ValueError("semantic applicability evidence identity differs")
    mechanism = load_mechanism_config(config.mechanism_config_path)
    replication = json.loads(
        Path(config.replication_report_path).read_text(encoding="utf-8")
    )
    if (
        replication.get("decision", {}).get("replication_admitted") is not True
        or replication.get("decision", {}).get(
            "real_task_transfer_preregistration_allowed"
        )
        is not True
        or replication.get("decision", {}).get(
            "real_task_generation_allowed"
        )
        is not False
        or mechanism.policy["router_uses_case_metadata"] is not False
    ):
        raise ValueError("semantic applicability predecessor decision differs")
    return config


def extract_source_facts(
    prompt: str,
    route: dict[str, Any],
) -> dict[str, Any]:
    if not route.get("routed"):
        return {
            "extracted": False,
            "reason": route["reason"],
            "source_facts": None,
        }
    family = route["family"]
    match = LABELED_FACT_PATTERNS[family].search(prompt)
    if match is None:
        return {
            "extracted": False,
            "reason": "labeled_source_facts_missing",
            "source_facts": None,
        }
    facts: dict[str, Any] = {
        key: int(value) for key, value in match.groupdict().items()
    }
    if family == "implicit_scale_total":
        scale_word = route.get("semantic_operator")
        if scale_word not in {"double", "triple"}:
            return {
                "extracted": False,
                "reason": "semantic_operator_missing",
                "source_facts": None,
            }
        facts["scale_word"] = scale_word
    return {
        "extracted": True,
        "reason": "exact_labeled_source_facts",
        "source_facts": facts,
    }


def _sha256_lines(values: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode()).hexdigest()


def run(config: SemanticSkillApplicabilityConfig) -> dict[str, Any]:
    manifest = load_manifest(config.complete_manifest_path)
    public_cases = json.loads(
        Path(config.complete_case_manifest_path).read_text(encoding="utf-8")
    )
    by_benchmark: dict[str, list[dict[str, Any]]] = {}
    for row in public_cases:
        by_benchmark.setdefault(str(row["benchmark"]), []).append(row)
    if {
        name: len(rows) for name, rows in by_benchmark.items()
    } != config.expected_cases_by_benchmark:
        raise ValueError("semantic applicability public case counts differ")

    manifest_specs = {spec.name: spec for spec in manifest.datasets}
    route_counts: Counter[str] = Counter()
    extraction_failures: Counter[str] = Counter()
    eligible_case_ids: list[str] = []
    eligible_by_benchmark: Counter[str] = Counter()
    eligible_by_family: Counter[str] = Counter()
    question_hashes: list[str] = []
    for benchmark, rows in by_benchmark.items():
        spec = manifest_specs[benchmark]
        path = (Path(config.dataset_root) / spec.path).resolve()
        if sha256_file(path) != spec.sha256:
            raise ValueError(
                f"semantic applicability dataset differs: {benchmark}"
            )
        column = config.question_column_by_benchmark[benchmark]
        questions = parquet.read_table(path, columns=[column])[
            column
        ].to_pylist()
        for public in rows:
            question = str(questions[int(public["source_index"])])
            question_hashes.append(hashlib.sha256(question.encode()).hexdigest())
            route = route_prompt(question)
            route_counts[
                route["family"] if route["routed"] else route["reason"]
            ] += 1
            extraction = extract_source_facts(question, route)
            if not extraction["extracted"]:
                extraction_failures[extraction["reason"]] += 1
                continue
            case_id = str(public["case_id"])
            eligible_case_ids.append(case_id)
            eligible_by_benchmark[benchmark] += 1
            eligible_by_family[route["family"]] += 1

    total_cases = sum(config.expected_cases_by_benchmark.values())
    if len(question_hashes) != total_cases:
        raise ValueError("semantic applicability question coverage differs")
    transfer_allowed = (
        len(eligible_case_ids) >= config.minimum_eligible_rows_for_transfer
    )
    result = {
        "schema_version": RESULT_SCHEMA,
        "experiment_id": config.experiment_id,
        "identity": {
            "mechanism_config_sha256": config.mechanism_config_sha256,
            "mechanism_report_sha256": config.mechanism_report_sha256,
            "replication_report_sha256": config.replication_report_sha256,
            "complete_manifest_sha256": config.complete_manifest_sha256,
            "complete_case_manifest_sha256": (
                config.complete_case_manifest_sha256
            ),
            "complete_report_sha256": config.complete_report_sha256,
            "question_hashes_sha256": _sha256_lines(question_hashes),
            "eligible_case_ids_sha256": _sha256_lines(eligible_case_ids),
        },
        "scan": {
            "cases": total_cases,
            "cases_by_benchmark": config.expected_cases_by_benchmark,
            "route_counts": dict(sorted(route_counts.items())),
            "extraction_failures": dict(sorted(extraction_failures.items())),
            "eligible_rows": len(eligible_case_ids),
            "eligible_by_benchmark": dict(sorted(eligible_by_benchmark.items())),
            "eligible_by_family": dict(sorted(eligible_by_family.items())),
        },
        "decision": {
            "transfer_preregistration_allowed": transfer_allowed,
            "model_generation_allowed": False,
            "benchmark_generation_allowed": False,
            "canary_rerun_allowed": False,
            "holdout_allowed": False,
            "training_allowed": False,
            "post_scan_route_or_extractor_change_allowed": False,
            "next_action": (
                "Pre-register the exact eligible real-task routes, direct "
                "preservation outside those routes, fallback, budgets, and "
                "matched gates before any model generation."
                if transfer_allowed
                else "Close unchanged semantic-skill real-task transfer. "
                "Preserve zero-coverage evidence and change mechanism only "
                "on fresh non-benchmark data."
            ),
        },
        "scan_boundary": {
            "question_column_only": True,
            "answer_columns_loaded": False,
            "choices_column_loaded": False,
            "model_outputs_loaded": False,
            "expected_answers_used": False,
            "case_correctness_used": False,
            "raw_questions_published": False,
            "case_ids_published": False,
            "model_generation_started": False,
        },
    }
    output = Path(config.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result
