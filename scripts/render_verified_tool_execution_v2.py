#!/usr/bin/env python3

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from nano_harness.baseline import sha256_file
from nano_harness.verified_tool_execution import (
    FAMILIES,
    build_cases,
    parse_and_execute_plan,
    public_case_contract,
    summarize_rows,
)
from nano_harness.verified_tool_execution_v2 import (
    load_config,
    parent_config,
)
from scripts.render_verified_tool_execution_v1 import paired_metrics


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/harness/qwen35_verified_tool_execution_v2.json"
PREREG = (
    ROOT
    / "docs/experiments/qwen35_verified_tool_execution_v2.preregister.json"
)
RAW = (
    ROOT / "results/harness/qwen35-verified-tool-execution-v2/result.json"
)
PUBLIC_JSON = (
    ROOT / "docs/results/qwen35_verified_tool_execution_v2.public.json"
)
MARKDOWN = ROOT / "docs/results/qwen35_verified_tool_execution_v2.md"


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def admission_gates(
    raw: dict,
    harness_vs_four: dict,
    harness_vs_nine: dict,
) -> dict[str, bool]:
    config = load_config(CONFIG)
    parent = parent_config(config)
    arms = raw["arms"]
    return {
        "all_rows_complete_and_parseable": all(
            arms[name]["cases"] == 256
            and arms[name]["parseable"] == 256
            for name in (
                "four_b_direct",
                "nine_b_direct",
                "four_b_skill_verified_tool",
            )
        ),
        "skill_routes_256": raw["routing"]["skill_routes"] == 256,
        "single_tool_exposures_256": (
            raw["routing"]["single_tool_exposures"] == 256
        ),
        "verified_executions_256": (
            raw["routing"]["verified_executions"] == 256
        ),
        "executor_contract_failures_zero": (
            raw["routing"]["fallbacks"] == 0
            and raw["routing"]["plan_retries"] == 0
            and raw["routing"]["final_feedback_calls"] == 256
        ),
        "harness_vs_four_b_accuracy_positive": (
            harness_vs_four["candidate_accuracy"]
            > harness_vs_four["baseline_accuracy"]
        ),
        "harness_vs_four_b_ci_lower_gt_zero": (
            harness_vs_four["paired_bootstrap_95_ci"][0] > 0
        ),
        "harness_vs_four_b_mcnemar_p_lt_005": (
            harness_vs_four["mcnemar_exact_p"]
            < parent.significance_alpha
        ),
        "harness_vs_four_b_minimum_wins": (
            harness_vs_four["paired_counts"]["candidate_only"]
            >= parent.minimum_harness_wins
        ),
        "harness_vs_four_b_maximum_losses": (
            harness_vs_four["paired_counts"]["baseline_only"]
            <= parent.maximum_harness_losses
        ),
        "harness_vs_nine_b_accuracy_positive": (
            harness_vs_nine["candidate_accuracy"]
            > harness_vs_nine["baseline_accuracy"]
        ),
        "harness_vs_nine_b_ci_lower_gt_zero": (
            harness_vs_nine["paired_bootstrap_95_ci"][0] > 0
        ),
        "harness_vs_nine_b_mcnemar_p_lt_005": (
            harness_vs_nine["mcnemar_exact_p"]
            < parent.significance_alpha
        ),
        "harness_vs_nine_b_minimum_wins": (
            harness_vs_nine["paired_counts"]["candidate_only"]
            >= parent.minimum_harness_wins
        ),
        "harness_vs_nine_b_maximum_losses": (
            harness_vs_nine["paired_counts"]["baseline_only"]
            <= parent.maximum_harness_losses
        ),
        "every_family_non_regression_vs_four_b_and_nine_b": all(
            arms["four_b_skill_verified_tool"]["by_family"][family][
                "correct"
            ]
            >= arms["four_b_direct"]["by_family"][family]["correct"]
            and arms["four_b_skill_verified_tool"]["by_family"][family][
                "correct"
            ]
            >= arms["nine_b_direct"]["by_family"][family]["correct"]
            for family in FAMILIES
        ),
    }


def build_report() -> dict:
    config = load_config(CONFIG)
    parent = parent_config(config)
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    raw = json.loads(RAW.read_text(encoding="utf-8"))
    cases = build_cases(parent)
    cases_by_id = {row["case_id"]: row for row in cases}
    if (
        prereg["schema_version"]
        != "nano_harness_verified_tool_execution_preregister_v2"
        or prereg["identity"]["config_sha256"] != sha256_file(CONFIG)
        or prereg["execution_boundary"]["evaluation_started"] is not False
        or raw["schema_version"]
        != "nano_harness_verified_tool_execution_result_v2"
        or raw["identity"]["case_contract"] != public_case_contract(cases)
        or raw["identity"]["parent_config_sha256"]
        != config.parent_config_sha256
        or raw["identity"]["prior_v1_report_sha256"]
        != config.prior_v1_report_sha256
        or raw["identity"]["service_receipt_sha256"]
        != config.service_receipt_sha256
    ):
        raise ValueError("verified tool v2 result identity differs")
    for key in ("four_b_rows", "nine_b_rows", "harness_rows"):
        rows = raw[key]
        if (
            len(rows) != len(cases)
            or {row["case_id"] for row in rows} != set(cases_by_id)
        ):
            raise ValueError(f"verified tool v2 {key} identities differ")
    if set(raw["harness_receipts"]) != set(cases_by_id):
        raise ValueError("verified tool v2 receipt identities differ")
    for case_id, receipt in raw["harness_receipts"].items():
        case = cases_by_id[case_id]
        if (
            receipt["skill_id"] != case["family"]
            or receipt["exposed_tools"] != [case["family"]]
            or not receipt["receipt"]["executed"]
            or receipt["fallback_used"]
            or not receipt["final_feedback_sent"]
        ):
            raise ValueError("verified tool v2 route receipt differs")
        replay = parse_and_execute_plan(
            receipt["plan_attempts"][-1]["output"],
            expected_tool=case["family"],
            source_facts=case["source_facts"],
        )
        if (
            replay != receipt["receipt"]
            or replay["result"] != case["expected"]
        ):
            raise ValueError("verified tool v2 execution receipt differs")
    expected_boundary = {
        "training_eligible_cases": 0,
        "executor_uses_expected_answer": False,
        "executor_uses_case_correctness": False,
        "benchmark_rows_loaded": False,
        "benchmark_outputs_loaded": False,
        "canary_rows_loaded": False,
        "independent_holdout_rows_loaded": False,
    }
    if (
        raw["evaluation_boundary"] != expected_boundary
        or summarize_rows(raw["four_b_rows"])
        != raw["arms"]["four_b_direct"]
        or summarize_rows(raw["nine_b_rows"])
        != raw["arms"]["nine_b_direct"]
        or summarize_rows(raw["harness_rows"])
        != raw["arms"]["four_b_skill_verified_tool"]
    ):
        raise ValueError("verified tool v2 aggregate boundary differs")
    harness_vs_four = paired_metrics(
        raw["harness_rows"],
        raw["four_b_rows"],
        seed=f"{parent.bootstrap_seed}:v2-harness-four",
        samples=parent.bootstrap_samples,
    )
    harness_vs_nine = paired_metrics(
        raw["harness_rows"],
        raw["nine_b_rows"],
        seed=f"{parent.bootstrap_seed}:v2-harness-nine",
        samples=parent.bootstrap_samples,
    )
    four_vs_nine = paired_metrics(
        raw["four_b_rows"],
        raw["nine_b_rows"],
        seed=f"{parent.bootstrap_seed}:v2-four-nine",
        samples=parent.bootstrap_samples,
    )
    gates = admission_gates(raw, harness_vs_four, harness_vs_nine)
    admitted = all(gates.values())
    return {
        "schema_version": (
            "nano_harness_verified_tool_execution_public_v2"
        ),
        "experiment_id": config.experiment_id,
        "identity": {
            "result_revision": git_revision(),
            "config_sha256": sha256_file(CONFIG),
            "preregister_sha256": sha256_file(PREREG),
            "raw_result_sha256": sha256_file(RAW),
            "case_contract_sha256": prereg["identity"][
                "case_contract_sha256"
            ],
            "parent_config_sha256": config.parent_config_sha256,
            "prior_v1_report_sha256": config.prior_v1_report_sha256,
            "service_receipt_sha256": config.service_receipt_sha256,
        },
        "data": {
            "cases": len(cases),
            "freshness": prereg["freshness"],
            "benchmark_canary_holdout_rows_or_outputs": 0,
        },
        "mechanism_delta": prereg["mechanism_delta"],
        "arms": raw["arms"],
        "routing": raw["routing"],
        "comparisons": {
            "harness_vs_four_b": harness_vs_four,
            "harness_vs_nine_b": harness_vs_nine,
            "four_b_vs_nine_b": four_vs_nine,
        },
        "decision": {
            "gates": gates,
            "local_harness_admitted": admitted,
            "canary_preregistration_allowed": admitted,
            "canary_generation_allowed": False,
            "benchmark_allowed": False,
            "independent_holdout_allowed": False,
            "further_tuning_on_observed_cases_allowed": False,
            "next_action": (
                "Pre-register the exact skill-routed harness transfer to the "
                "frozen 211-case canary. Do not generate canary outputs until "
                "the canary route, prompts, tool scope, fallback, parser, "
                "budgets, model identities, and gates are committed."
                if admitted
                else "Reject v2 local admission and preserve the evidence; "
                "do not tune or rerun on these cases."
            ),
        },
        "evaluation_boundary": raw["evaluation_boundary"],
        "claim_boundary": (
            "This result establishes only fresh synthetic local harness "
            "admission. It authorizes canary pre-registration, not canary "
            "generation, complete benchmark access, holdout access, or final "
            "4B/9B superiority claims."
        ),
    }


def render_markdown(report: dict) -> str:
    decision = report["decision"]
    return f"""# Qwen3.5 Skill-Routed Verified Tool Execution v2 Result

## 结论

- local harness admitted：
  `{str(decision['local_harness_admitted']).lower()}`；
- canary pre-registration allowed：
  `{str(decision['canary_preregistration_allowed']).lower()}`；
- canary generation allowed：`false`；
- complete benchmark allowed：`false`。

## Arms

| Arm | Correct | Accuracy | Parseable |
| --- | ---: | ---: | ---: |
{chr(10).join(
    f"| {name} | {metrics['correct']}/256 | {metrics['accuracy']:.4f} | "
    f"{metrics['parseable']}/256 |"
    for name, metrics in report['arms'].items()
)}

## Routing

```json
{json.dumps(report['routing'], indent=2, sort_keys=True)}
```

V2 完成 256/256 skill routes、single-tool exposures、verified executions 和
result-feedback calls，0 retry、0 fallback、0 contract failure。

## Paired comparisons

| Comparison | Delta | 95% CI | Wins | Losses | McNemar p |
| --- | ---: | --- | ---: | ---: | ---: |
{chr(10).join(
    f"| {name} | {row['delta']:+.4f} | "
    f"[{row['paired_bootstrap_95_ci'][0]:+.4f}, "
    f"{row['paired_bootstrap_95_ci'][1]:+.4f}] | "
    f"{row['paired_counts']['candidate_only']} | "
    f"{row['paired_counts']['baseline_only']} | "
    f"{row['mcnemar_exact_p']:.6g} |"
    for name, row in report['comparisons'].items()
)}

## Frozen gates

```json
{json.dumps(decision['gates'], indent=2, sort_keys=True)}
```

## Evidence

- config SHA：`{report['identity']['config_sha256']}`；
- prereg SHA：`{report['identity']['preregister_sha256']}`；
- raw result SHA：`{report['identity']['raw_result_sha256']}`；
- case contract SHA：`{report['identity']['case_contract_sha256']}`；
- parent V1 report SHA：
  `{report['identity']['prior_v1_report_sha256']}`。

通过只允许**另行预注册** 211-case canary；当前仍不能生成 canary outputs，
也不能访问完整 benchmark 或 independent holdout。
"""


def main() -> None:
    report = build_report()
    PUBLIC_JSON.parent.mkdir(parents=True, exist_ok=True)
    PUBLIC_JSON.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    MARKDOWN.write_text(render_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "experiment_id": report["experiment_id"],
                "decision": report["decision"],
                "arms": report["arms"],
                "routing": report["routing"],
                "comparisons": report["comparisons"],
                "public_json": str(PUBLIC_JSON),
                "markdown": str(MARKDOWN),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
