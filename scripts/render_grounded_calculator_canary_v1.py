#!/usr/bin/env python3

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from nano_harness.baseline import sha256_file
from nano_harness.grounded_calculator_canary import (
    evaluate_admission,
    load_config,
    verify_frozen_inputs,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/harness/qwen35_grounded_calculator_canary_v1.json"
)
PREREG = (
    ROOT
    / "docs/experiments/qwen35_grounded_calculator_canary_v1.preregister.json"
)
RAW = (
    ROOT
    / "results/harness/qwen35-grounded-calculator-canary-v1/result.json"
)
CANDIDATE = (
    ROOT
    / "results/harness/qwen35-grounded-calculator-canary-v1/candidate.jsonl"
)
PUBLIC_JSON = (
    ROOT
    / "docs/results/qwen35_grounded_calculator_canary_v1.public.json"
)
MARKDOWN = (
    ROOT / "docs/results/qwen35_grounded_calculator_canary_v1.md"
)
PREREG_REVISION = "860f6fd01aeee47ea92bc7c8731059eef60e16fd"


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def _eligible_receipts(raw: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for case_id, receipt in raw["receipts"].items():
        if not receipt["eligible"]:
            continue
        plan_attempts = receipt.get("plan_attempts", [])
        execution = receipt.get("receipt", {})
        rows.append(
            {
                "case_id": case_id,
                "reason": receipt["reason"],
                "plan_calls": receipt["plan_calls"],
                "final_feedback_calls": receipt["final_feedback_calls"],
                "api_errors": receipt["api_errors"],
                "fallback_used": receipt["fallback_used"],
                "plan_attempts": [
                    {
                        "attempt": row["attempt"],
                        "output_sha256": row.get("output_sha256"),
                        "reason": row["reason"],
                        "executed": row["executed"],
                    }
                    for row in plan_attempts
                ],
                "execution": {
                    "executed": execution.get("executed", False),
                    "reason": execution.get("reason"),
                    "expression_sha256": execution.get(
                        "expression_sha256"
                    ),
                    "result": execution.get("result"),
                    "executor_uses_expected_answer": execution.get(
                        "executor_uses_expected_answer"
                    ),
                    "executor_uses_case_correctness": execution.get(
                        "executor_uses_case_correctness"
                    ),
                    "case_id_allowlist_used": execution.get(
                        "case_id_allowlist_used"
                    ),
                },
            }
        )
    return rows


def build_report() -> dict[str, Any]:
    config = load_config(CONFIG)
    frozen = verify_frozen_inputs(config, verify_service=False)
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    raw = json.loads(RAW.read_text(encoding="utf-8"))
    if (
        prereg.get("schema_version")
        != "nano_harness_grounded_calculator_canary_preregister_v1"
        or prereg.get("identity", {}).get("config_sha256")
        != sha256_file(CONFIG)
        or prereg.get("identity", {}).get("code_revision")
        != "612e5a44dcbda2c382e67cb0d049100e0943eea8"
    ):
        raise ValueError("grounded calculator preregistration differs")
    if (
        not subprocess.run(
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
        == 0
    ):
        raise ValueError("grounded calculator preregistration is not committed")
    if (
        raw.get("schema_version")
        != "nano_harness_grounded_calculator_canary_result_v1"
        or raw.get("experiment_id") != config.experiment_id
        or raw.get("identity", {}).get("candidate_raw_sha256")
        != sha256_file(CANDIDATE)
        or raw.get("identity", {}).get("eligible_case_ids_sha256")
        != frozen["eligible_case_ids_sha256"]
        or raw.get("evaluation_boundary")
        != {
            "training_eligible_cases": 0,
            "executor_uses_expected_answer": False,
            "executor_uses_case_correctness": False,
            "case_id_allowlist_used": False,
            "canary_rows_loaded": True,
            "canary_outputs_generated": True,
            "complete_benchmark_rows_loaded": False,
            "independent_holdout_rows_loaded": False,
        }
    ):
        raise ValueError("grounded calculator raw result boundary differs")
    recomputed_gates = evaluate_admission(
        config,
        raw["candidate"],
        raw["comparisons"]["versus_frozen_four_b_direct"],
        raw["receipts"],
        raw["direct_preservation_failures"],
    )
    if (
        recomputed_gates != raw["admission_gates"]
        or raw["decision"]["canary_passed"] is not False
        or all(recomputed_gates.values())
    ):
        raise ValueError("grounded calculator canary decision differs")

    versus_four = raw["comparisons"]["versus_frozen_four_b_direct"]
    versus_nine = raw["comparisons"]["versus_frozen_nine_b_direct"]
    candidate = raw["candidate"]
    eligible = _eligible_receipts(raw)
    if (
        raw["routing"]
        != {
            "eligible_rows": 2,
            "direct_preserve_rows": 209,
            "verified_executions": 1,
            "fallbacks": 1,
            "plan_calls": 3,
            "final_feedback_calls": 1,
            "api_errors": 0,
        }
        or len(eligible) != 2
        or raw["direct_preservation_failures"]
    ):
        raise ValueError("grounded calculator routing differs")
    return {
        "schema_version": (
            "nano_harness_grounded_calculator_canary_public_v1"
        ),
        "experiment_id": config.experiment_id,
        "identity": {
            "result_revision": git_revision(),
            "preregister_revision": PREREG_REVISION,
            "config_sha256": sha256_file(CONFIG),
            "preregister_sha256": sha256_file(PREREG),
            "raw_result_sha256": sha256_file(RAW),
            "candidate_raw_sha256": sha256_file(CANDIDATE),
            "manifest_sha256": config.manifest_sha256,
            "case_manifest_sha256": config.case_manifest_sha256,
            "four_b_raw_sha256": config.four_b_raw_sha256,
            "nine_b_raw_sha256": config.nine_b_raw_sha256,
            "v2_report_sha256": config.v2_report_sha256,
            "service_receipt_sha256": config.service_receipt_sha256,
            "eligible_case_ids_sha256": frozen[
                "eligible_case_ids_sha256"
            ],
        },
        "candidate": {
            "cases": candidate["total_cases"],
            "correct": sum(
                row["correct"]
                for row in candidate["benchmarks"].values()
            ),
            "macro_accuracy": candidate["macro_accuracy"],
            "parse_failures": sum(
                row["parse_failures"]
                for row in candidate["benchmarks"].values()
            ),
            "api_errors": candidate["error_cases"],
            "by_benchmark": {
                name: {
                    "cases": row["cases"],
                    "correct": row["correct"],
                    "accuracy": row["accuracy"],
                    "parse_failures": row["parse_failures"],
                    "api_errors": row["errors"],
                }
                for name, row in candidate["benchmarks"].items()
            },
        },
        "comparisons": {
            "versus_frozen_four_b_direct": {
                "candidate_correct": versus_four["overall_micro"][
                    "candidate_correct"
                ],
                "baseline_correct": versus_four["overall_micro"][
                    "baseline_correct"
                ],
                "delta": versus_four["overall_micro"]["delta"],
                "paired_counts": versus_four["overall_micro"][
                    "paired_counts"
                ],
                "paired_bootstrap_95_ci": versus_four["overall_micro"][
                    "paired_bootstrap_95_ci"
                ],
                "mcnemar_exact_p": versus_four["overall_micro"][
                    "mcnemar_exact_p"
                ],
            },
            "versus_frozen_nine_b_direct": {
                "candidate_correct": versus_nine["overall_micro"][
                    "candidate_correct"
                ],
                "baseline_correct": versus_nine["overall_micro"][
                    "baseline_correct"
                ],
                "delta": versus_nine["overall_micro"]["delta"],
                "paired_counts": versus_nine["overall_micro"][
                    "paired_counts"
                ],
                "paired_bootstrap_95_ci": versus_nine["overall_micro"][
                    "paired_bootstrap_95_ci"
                ],
                "mcnemar_exact_p": versus_nine["overall_micro"][
                    "mcnemar_exact_p"
                ],
            },
        },
        "routing": raw["routing"],
        "eligible_receipts": eligible,
        "failure_analysis": [
            {
                "failure_class": "implicit_semantic_constant_rejected",
                "count": 1,
                "explanation": (
                    "The model emitted an inferred multiplier that was not a "
                    "numeric literal in the prompt. The source-grounding "
                    "validator rejected both attempts and reused direct."
                ),
            },
            {
                "failure_class": "continuous_break_even_not_strict_year",
                "count": 1,
                "explanation": (
                    "The grounded expression computed the exact break-even "
                    "year, but the task required the first strictly profitable "
                    "integer year. Verified execution preserved the expression "
                    "result and therefore remained wrong."
                ),
            },
        ],
        "admission_gates": recomputed_gates,
        "decision": {
            "canary_passed": False,
            "complete_benchmark_preregistration_allowed": False,
            "complete_benchmark_generation_allowed": False,
            "independent_holdout_allowed": False,
            "further_tuning_or_rerun_on_observed_canary_allowed": False,
            "route_rejected": True,
            "next_action": (
                "Close this route. On a new disjoint local synthetic surface, "
                "test a typed semantic-skill executor that explicitly grounds "
                "linguistic operators and discrete boundary semantics. Do not "
                "reuse or rerun the observed 211-case canary."
            ),
        },
        "evaluation_boundary": raw["evaluation_boundary"],
        "claim_boundary": (
            "This is negative admission evidence for one pre-registered "
            "211-case canary run. It is not a complete benchmark, holdout, "
            "training, or final model-superiority result."
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    candidate = report["candidate"]
    versus_four = report["comparisons"]["versus_frozen_four_b_direct"]
    versus_nine = report["comparisons"]["versus_frozen_nine_b_direct"]
    decision = report["decision"]
    benchmark_rows = "\n".join(
        f"| {name} | {row['correct']}/{row['cases']} | "
        f"{row['parse_failures']} | {row['api_errors']} |"
        for name, row in candidate["by_benchmark"].items()
    )
    return f"""# Qwen3.5 Grounded Calculator Canary v1 Result

## 结论

**未通过，不允许跑完整 benchmark，也不允许在这 211 行上修改后重跑。**

- candidate：{candidate['correct']}/{candidate['cases']}；
- frozen 4B direct：{versus_four['baseline_correct']}/211；
- frozen 9B direct：{versus_nine['baseline_correct']}/211；
- candidate vs 4B：0 wins / 0 losses，delta
  {versus_four['delta']:+.4f}；
- 209 个非 eligible 行评分字段完全复用，0 regression；
- 2 个 GSM8K parse failures 中，1 个安全执行、1 个 fail-close；
- parse failures 2→1，但正确数仍是 163，没有达到 164 gate。

## 分任务

| Benchmark | Correct | Parse failures | API errors |
| --- | ---: | ---: | ---: |
{benchmark_rows}

## 两个 Recovery

1. 一个 plan 两次输出相同的隐含 multiplier。数字 `2` 不在题面中，strict
   source grounding 两次拒绝，原样回退 direct。这证明 fail-close 有效，但
   字面 numeric grounding 不能表示 “double” 这类语言算子。
2. 另一个 plan 通过 grounding 并精确执行出 break-even 结果 12，formatter
   也原样返回 12；题目要求的是第一个**严格盈利**的整数年份 13。问题不在
   算术执行，而在缺少离散边界语义。

不公开 prompt 或 raw output；plan/output 只记录 SHA 和失败类型。

## Gate

```json
{json.dumps(report['admission_gates'], indent=2, sort_keys=True)}
```

失败项：

- overall 163/211 < 164/211；
- relative to direct 4B，candidate-only 0 不大于 base-only 0。

## Evidence

- prereg commit：`{report['identity']['preregister_revision']}`；
- config SHA：`{report['identity']['config_sha256']}`；
- prereg SHA：`{report['identity']['preregister_sha256']}`；
- raw result SHA：`{report['identity']['raw_result_sha256']}`；
- candidate raw SHA：`{report['identity']['candidate_raw_sha256']}`；
- case manifest SHA：`{report['identity']['case_manifest_sha256']}`。

## 决策

- canary passed：`false`；
- complete benchmark preregistration：`false`；
- complete benchmark generation：`false`；
- independent holdout：`false`；
- observed-canary tuning/rerun：`false`。

下一步只能在**新的、互不重叠的 local synthetic surface**测试 typed semantic
skills：显式处理语言算子和离散边界语义。当前 211-case canary 已观察，永久
关闭对它的 route/prompt/grammar/budget 修改和重跑。
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
                "candidate": report["candidate"],
                "routing": report["routing"],
                "admission_gates": report["admission_gates"],
                "decision": report["decision"],
                "public_json": str(PUBLIC_JSON),
                "markdown": str(MARKDOWN),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
