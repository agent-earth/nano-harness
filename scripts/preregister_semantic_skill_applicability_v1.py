#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pyarrow.parquet as parquet

from nano_harness.baseline import load_manifest, sha256_file
from nano_harness.semantic_skill_applicability import (
    LABELED_FACT_PATTERNS,
    load_config,
)
from nano_harness.semantic_skill_execution import (
    load_config as load_mechanism_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "configs/harness/qwen35_semantic_skill_applicability_v1.json"
)
JSON_OUTPUT = (
    ROOT
    / "docs/experiments/qwen35_semantic_skill_applicability_v1.preregister.json"
)
MARKDOWN_OUTPUT = (
    ROOT / "docs/experiments/qwen35_semantic_skill_applicability_v1.md"
)


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def build_receipt() -> dict:
    config = load_config(CONFIG)
    mechanism = load_mechanism_config(config.mechanism_config_path)
    manifest = load_manifest(config.complete_manifest_path)
    public_cases = json.loads(
        Path(config.complete_case_manifest_path).read_text(encoding="utf-8")
    )
    counts = {}
    for row in public_cases:
        benchmark = str(row["benchmark"])
        counts[benchmark] = counts.get(benchmark, 0) + 1
    if counts != config.expected_cases_by_benchmark:
        raise ValueError("semantic applicability public case counts differ")

    schema_audit = {}
    for spec in manifest.datasets:
        path = (Path(config.dataset_root) / spec.path).resolve()
        if sha256_file(path) != spec.sha256:
            raise ValueError(
                f"semantic applicability dataset differs: {spec.name}"
            )
        schema = parquet.read_schema(path)
        column = config.question_column_by_benchmark[spec.name]
        if column not in schema.names:
            raise ValueError(
                f"semantic applicability question column missing: {spec.name}"
            )
        schema_audit[spec.name] = {
            "dataset_sha256": spec.sha256,
            "question_column": column,
            "column_type": str(schema.field(column).type),
            "rows": parquet.ParquetFile(path).metadata.num_rows,
            "answer_column_not_requested": True,
            "choices_column_not_requested": True,
        }

    replication = json.loads(
        Path(config.replication_report_path).read_text(encoding="utf-8")
    )
    return {
        "schema_version": (
            "nano_harness_semantic_skill_applicability_preregister_v1"
        ),
        "experiment_id": config.experiment_id,
        "identity": {
            "code_revision": git_revision(),
            "config_sha256": sha256_file(CONFIG),
            "mechanism_config_sha256": config.mechanism_config_sha256,
            "mechanism_report_sha256": config.mechanism_report_sha256,
            "replication_report_sha256": config.replication_report_sha256,
            "complete_manifest_sha256": config.complete_manifest_sha256,
            "complete_case_manifest_sha256": (
                config.complete_case_manifest_sha256
            ),
            "complete_report_sha256": config.complete_report_sha256,
        },
        "surface": {
            "cases": sum(config.expected_cases_by_benchmark.values()),
            "cases_by_benchmark": config.expected_cases_by_benchmark,
            "case_manifest_has_expected_or_answer": any(
                "expected" in row or "answer" in row for row in public_cases
            ),
            "schema_audit": schema_audit,
        },
        "frozen_router": {
            "name": mechanism.skill_router,
            "route_markers": mechanism.route_markers,
            "router_uses_case_metadata": False,
            "minimum_eligible_rows_for_transfer": (
                config.minimum_eligible_rows_for_transfer
            ),
        },
        "frozen_extractor": {
            "name": config.source_fact_extractor,
            "patterns_sha256": {
                family: hashlib.sha256(pattern.pattern.encode()).hexdigest()
                for family, pattern in LABELED_FACT_PATTERNS.items()
            },
            "requires_all_labeled_integer_fields": True,
            "requires_route_before_extraction": True,
            "uses_expected_answer": False,
            "uses_case_correctness": False,
        },
        "predecessor": {
            "replication_admitted": replication["decision"][
                "replication_admitted"
            ],
            "harness_correct": replication["arms"][
                "four_b_semantic_skills"
            ]["correct"],
            "four_b_direct_correct": replication["arms"]["four_b_direct"][
                "correct"
            ],
            "nine_b_direct_correct": replication["arms"]["nine_b_direct"][
                "correct"
            ],
        },
        "scan_contract": {
            "read_columns_by_benchmark": {
                benchmark: [column]
                for benchmark, column in (
                    config.question_column_by_benchmark.items()
                )
            },
            "answer_columns_loaded": False,
            "choices_column_loaded": False,
            "model_outputs_loaded": False,
            "raw_questions_published": False,
            "case_ids_published": False,
            "published_outputs": [
                "aggregate route counts",
                "aggregate extractor failure counts",
                "eligible counts by benchmark and skill",
                "question-hash set SHA256",
                "eligible-case-id set SHA256",
            ],
        },
        "decision_policy": {
            "transfer_allowed": (
                "Only if exact unchanged router plus extractor yields at "
                "least one eligible row."
            ),
            "zero_coverage": (
                "Close this real-task transfer. Do not change markers or "
                "extractor after reading coverage."
            ),
            "positive_coverage": (
                "Separately pre-register exact eligible identities, direct "
                "preservation elsewhere, fallback, budgets, and matched gates "
                "before any model generation."
            ),
            "forbidden_after_scan": [
                "route_marker_change",
                "source_fact_extractor_change",
                "minimum_eligible_rows_change",
                "case_selection_change",
                "prompt_change",
                "tool_schema_change",
                "semantic_executor_change",
                "model_generation",
                "benchmark_generation",
                "canary_rerun",
                "holdout_access",
            ],
        },
        "execution_boundary": config.execution_boundary,
        "claim_boundary": (
            "This pre-registers a no-generation applicability scan. It may "
            "read only question columns and publish only aggregate counts and "
            "hashes. It is not a benchmark score or treatment result."
        ),
    }


def render_markdown(receipt: dict) -> str:
    return f"""# Qwen3.5 Semantic Skill Applicability Scan v1

## 目的

在任何 real-task model generation 前，先冻结 unchanged semantic router 和
source-fact extractor，再扫描完整 GSM8K/MMLU/GPQA question columns。

Scanner 不读 answer、MMLU choices 或任何模型 output，也不公开 raw question 或
case ID。

## Frozen Rule

- router：`{receipt['frozen_router']['name']}`；
- exact markers 与 parent semantic mechanism 完全一致；
- extractor：`{receipt['frozen_extractor']['name']}`；
- 必须先唯一 route，再完整提取所有 labeled integer fields；
- 任何缺字段、歧义或未命中都 direct-preserve，不暴露 tool。

## Surface

- GSM8K：1319；
- MMLU：14042；
- GPQA-Diamond：198；
- total：15559；
- complete case manifest 不含 expected/answer：
  `{str(not receipt['surface']['case_manifest_has_expected_or_answer']).lower()}`。

## Decision

- eligible rows >= 1：只允许另行预注册 exact real-task treatment；
- eligible rows = 0：关闭 unchanged semantic-skill real-task transfer；
- 观察 coverage 后禁止修改 markers、extractor、threshold、case selection、
  prompt、schema 或 executor；
- model generation、benchmark generation、canary rerun、holdout、training
  全部保持关闭。

## Boundary

- config SHA：`{receipt['identity']['config_sha256']}`；
- replication report SHA：
  `{receipt['identity']['replication_report_sha256']}`；
- scan started：false；
- model generation started：false；
- benchmark generation started：false。
"""


def main() -> None:
    receipt = build_receipt()
    JSON_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUTPUT.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    MARKDOWN_OUTPUT.write_text(
        render_markdown(receipt),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "experiment_id": receipt["experiment_id"],
                "identity": receipt["identity"],
                "surface": receipt["surface"],
                "frozen_router": receipt["frozen_router"],
                "frozen_extractor": receipt["frozen_extractor"],
                "scan_contract": receipt["scan_contract"],
                "execution_boundary": receipt["execution_boundary"],
                "json_output": str(JSON_OUTPUT),
                "markdown_output": str(MARKDOWN_OUTPUT),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
