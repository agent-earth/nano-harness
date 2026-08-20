#!/usr/bin/env python3

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from nano_harness.baseline import sha256_file
from nano_harness.semantic_skill_applicability import (
    load_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "configs/harness/qwen35_semantic_skill_applicability_v1.json"
)
PREREG = (
    ROOT
    / "docs/experiments/qwen35_semantic_skill_applicability_v1.preregister.json"
)
RAW = (
    ROOT / "results/harness/qwen35-semantic-skill-applicability-v1/result.json"
)
PUBLIC_JSON = (
    ROOT / "docs/results/qwen35_semantic_skill_applicability_v1.public.json"
)
MARKDOWN = ROOT / "docs/results/qwen35_semantic_skill_applicability_v1.md"
PREREG_REVISION = "0f3b1ed833ad64bf9c09c2f4468fa45262d3f53f"


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def build_report() -> dict:
    config = load_config(CONFIG)
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    raw = json.loads(RAW.read_text(encoding="utf-8"))
    if (
        prereg.get("schema_version")
        != "nano_harness_semantic_skill_applicability_preregister_v1"
        or prereg.get("identity", {}).get("config_sha256")
        != sha256_file(CONFIG)
        or prereg.get("identity", {}).get("code_revision")
        != "e25a0edd546db0634e3545afbef2aba784e91ebb"
        or raw.get("schema_version")
        != "nano_harness_semantic_skill_applicability_result_v1"
        or raw.get("identity", {}).get("mechanism_config_sha256")
        != config.mechanism_config_sha256
        or raw.get("identity", {}).get("mechanism_report_sha256")
        != config.mechanism_report_sha256
        or raw.get("identity", {}).get("replication_report_sha256")
        != config.replication_report_sha256
        or raw.get("identity", {}).get("complete_manifest_sha256")
        != config.complete_manifest_sha256
        or raw.get("identity", {}).get("complete_case_manifest_sha256")
        != config.complete_case_manifest_sha256
        or raw.get("identity", {}).get("complete_report_sha256")
        != config.complete_report_sha256
    ):
        raise ValueError("semantic applicability result identity differs")
    if (
        subprocess.run(
            [
                "git",
                "merge-base",
                "--is-ancestor",
                PREREG_REVISION,
                "HEAD",
            ],
            cwd=ROOT,
            check=False,
        ).returncode
        != 0
    ):
        raise ValueError("semantic applicability preregistration is not committed")
    expected_scan = {
        "cases": 15559,
        "cases_by_benchmark": config.expected_cases_by_benchmark,
        "route_counts": {"route_missing": 15559},
        "extraction_failures": {"route_missing": 15559},
        "eligible_rows": 0,
        "eligible_by_benchmark": {},
        "eligible_by_family": {},
    }
    expected_decision = {
        "transfer_preregistration_allowed": False,
        "model_generation_allowed": False,
        "benchmark_generation_allowed": False,
        "canary_rerun_allowed": False,
        "holdout_allowed": False,
        "training_allowed": False,
        "post_scan_route_or_extractor_change_allowed": False,
        "next_action": (
            "Close unchanged semantic-skill real-task transfer. Preserve "
            "zero-coverage evidence and change mechanism only on fresh "
            "non-benchmark data."
        ),
    }
    expected_boundary = {
        "question_column_only": True,
        "answer_columns_loaded": False,
        "choices_column_loaded": False,
        "model_outputs_loaded": False,
        "expected_answers_used": False,
        "case_correctness_used": False,
        "raw_questions_published": False,
        "case_ids_published": False,
        "model_generation_started": False,
    }
    if (
        raw.get("scan") != expected_scan
        or raw.get("decision") != expected_decision
        or raw.get("scan_boundary") != expected_boundary
        or raw.get("identity", {}).get("eligible_case_ids_sha256")
        != "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    ):
        raise ValueError("semantic applicability zero-coverage decision differs")
    return {
        "schema_version": (
            "nano_harness_semantic_skill_applicability_public_v1"
        ),
        "experiment_id": config.experiment_id,
        "identity": {
            "result_revision": git_revision(),
            "preregister_revision": PREREG_REVISION,
            "config_sha256": sha256_file(CONFIG),
            "preregister_sha256": sha256_file(PREREG),
            "raw_result_sha256": sha256_file(RAW),
            **raw["identity"],
        },
        "scan": raw["scan"],
        "decision": raw["decision"],
        "scan_boundary": raw["scan_boundary"],
        "mechanism": {
            "router": prereg["frozen_router"],
            "extractor": prereg["frozen_extractor"],
        },
        "claim_boundary": (
            "This is a question-only applicability scan, not a benchmark "
            "quality score. Zero coverage rejects only the unchanged exact-"
            "marker transfer. No answer, choice, model output, raw question, "
            "or case identity is published."
        ),
    }


def render_markdown(report: dict) -> str:
    return f"""# Qwen3.5 Semantic Skill Applicability v1 Result

## 结论

**Unchanged exact-marker semantic transfer 关闭。**

- scanned questions：15559；
- route missing：15559；
- eligible rows：0；
- model generation：0；
- answer columns / choices / model outputs：均未读取。

Parent mechanism 的 exact markers 和 exact labeled-field extractor 在完整
GSM8K、MMLU、GPQA question surface 上没有任何可执行覆盖。根据预注册规则，
现在不能基于已观察 prompts 扩 marker 或修改 extractor，也不能生成 benchmark
outputs。

## Coverage

```json
{json.dumps(report['scan'], indent=2, sort_keys=True)}
```

## Boundary

```json
{json.dumps(report['scan_boundary'], indent=2, sort_keys=True)}
```

只发布 aggregate counts 与 hash-set identities；不发布 raw questions 或 case
IDs。

## Evidence

- prereg commit：`{report['identity']['preregister_revision']}`；
- config SHA：`{report['identity']['config_sha256']}`；
- prereg SHA：`{report['identity']['preregister_sha256']}`；
- raw result SHA：`{report['identity']['raw_result_sha256']}`；
- question hash-set SHA：
  `{report['identity']['question_hashes_sha256']}`；
- empty eligible-set SHA：
  `{report['identity']['eligible_case_ids_sha256']}`。

## 下一步

只允许在 fresh non-benchmark surface 设计 model-selected semantic router，
用 enum-constrained route selection 后再单 skill exposure；必须先通过 fresh
local route/execute/fallback/zero-loss gate，再另行预注册 real-task scan 或
treatment。当前 benchmark generation、canary rerun、holdout 和 training
继续关闭。
"""


def main() -> None:
    report = build_report()
    PUBLIC_JSON.parent.mkdir(parents=True, exist_ok=True)
    PUBLIC_JSON.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    MARKDOWN.write_text(
        render_markdown(report),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "experiment_id": report["experiment_id"],
                "scan": report["scan"],
                "decision": report["decision"],
                "scan_boundary": report["scan_boundary"],
                "public_json": str(PUBLIC_JSON),
                "markdown": str(MARKDOWN),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
