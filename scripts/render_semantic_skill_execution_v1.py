#!/usr/bin/env python3

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from nano_harness.baseline import sha256_file
from nano_harness.semantic_skill_execution import (
    FAMILIES,
    build_cases,
    load_config,
    parent_config,
    parse_and_execute_plan,
    public_case_contract,
    route_prompt,
    summarize_rows,
)
from scripts.render_verified_tool_execution_v1 import paired_metrics


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/harness/qwen35_semantic_skill_execution_v1.json"
PREREG = (
    ROOT
    / "docs/experiments/qwen35_semantic_skill_execution_v1.preregister.json"
)
RAW = (
    ROOT / "results/harness/qwen35-semantic-skill-execution-v1/result.json"
)
PUBLIC_JSON = (
    ROOT / "docs/results/qwen35_semantic_skill_execution_v1.public.json"
)
MARKDOWN = ROOT / "docs/results/qwen35_semantic_skill_execution_v1.md"
PREREG_REVISION = "53d07c610759db60b050ffd990de0d4c3a5c9a66"


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
    arms = raw["arms"]
    return {
        "all_rows_complete_and_parseable": all(
            arms[name]["cases"] == 256
            and arms[name]["parseable"] == 256
            for name in (
                "four_b_direct",
                "nine_b_direct",
                "four_b_semantic_skills",
            )
        ),
        "prompt_routes_256": raw["routing"]["prompt_routes"] == 256,
        "single_tool_exposures_256": (
            raw["routing"]["single_tool_exposures"] == 256
        ),
        "verified_executions_256": (
            raw["routing"]["verified_executions"] == 256
        ),
        "feedback_result_matches_256": (
            raw["routing"]["feedback_result_matches"] == 256
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
            < config.significance_alpha
        ),
        "harness_vs_four_b_minimum_wins": (
            harness_vs_four["paired_counts"]["candidate_only"]
            >= config.minimum_harness_wins
        ),
        "harness_vs_four_b_maximum_losses": (
            harness_vs_four["paired_counts"]["baseline_only"]
            <= config.maximum_harness_losses
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
            < config.significance_alpha
        ),
        "harness_vs_nine_b_minimum_wins": (
            harness_vs_nine["paired_counts"]["candidate_only"]
            >= config.minimum_harness_wins
        ),
        "harness_vs_nine_b_maximum_losses": (
            harness_vs_nine["paired_counts"]["baseline_only"]
            <= config.maximum_harness_losses
        ),
        "every_family_non_regression_vs_four_b_and_nine_b": all(
            arms["four_b_semantic_skills"]["by_family"][family]["correct"]
            >= arms["four_b_direct"]["by_family"][family]["correct"]
            and arms["four_b_semantic_skills"]["by_family"][family][
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
    cases = build_cases(config)
    cases_by_id = {row["case_id"]: row for row in cases}
    if (
        prereg.get("schema_version")
        != "nano_harness_semantic_skill_execution_preregister_v1"
        or prereg.get("identity", {}).get("config_sha256")
        != sha256_file(CONFIG)
        or prereg.get("identity", {}).get("code_revision")
        != "d3b67af19b1e70736ac630e87bc82290caa3c7ed"
        or raw.get("schema_version")
        != "nano_harness_semantic_skill_execution_result_v1"
        or raw.get("identity", {}).get("case_contract")
        != public_case_contract(cases)
        or raw.get("identity", {}).get("parent_config_sha256")
        != config.parent_config_sha256
        or raw.get("identity", {}).get("v2_report_sha256")
        != config.v2_report_sha256
        or raw.get("identity", {}).get("canary_rejection_sha256")
        != config.canary_rejection_sha256
        or raw.get("identity", {}).get("service_receipt_sha256")
        != config.service_receipt_sha256
    ):
        raise ValueError("semantic skill result identity differs")
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
        raise ValueError("semantic skill preregistration is not committed")
    for key in ("four_b_rows", "nine_b_rows", "harness_rows"):
        rows = raw[key]
        if (
            len(rows) != len(cases)
            or {row["case_id"] for row in rows} != set(cases_by_id)
        ):
            raise ValueError(f"semantic skill {key} identities differ")
    if set(raw["harness_receipts"]) != set(cases_by_id):
        raise ValueError("semantic skill receipt identities differ")
    for case_id, receipt in raw["harness_receipts"].items():
        case = cases_by_id[case_id]
        route = route_prompt(case["prompt"])
        if (
            route["family"] != case["family"]
            or route["router_uses_case_metadata"]
            or receipt["route"] != route
            or receipt["exposed_tools"] != [case["family"]]
            or not receipt["receipt"]["executed"]
            or receipt["fallback_used"]
            or not receipt["final_feedback_sent"]
            or not receipt["feedback_result_match"]
        ):
            raise ValueError("semantic skill route receipt differs")
        replay = parse_and_execute_plan(
            receipt["plan_attempts"][-1]["output"],
            route=route,
            source_facts=case["source_facts"],
        )
        if (
            replay != receipt["receipt"]
            or replay["result"] != case["expected"]
        ):
            raise ValueError("semantic skill execution receipt differs")
    expected_boundary = {
        "training_eligible_cases": 0,
        "router_uses_case_metadata": False,
        "executor_uses_expected_answer": False,
        "executor_uses_case_correctness": False,
        "benchmark_rows_loaded": False,
        "benchmark_outputs_loaded": False,
        "canary_rows_loaded": False,
        "canary_outputs_loaded": False,
        "independent_holdout_rows_loaded": False,
    }
    if (
        raw["evaluation_boundary"] != expected_boundary
        or summarize_rows(raw["four_b_rows"])
        != raw["arms"]["four_b_direct"]
        or summarize_rows(raw["nine_b_rows"])
        != raw["arms"]["nine_b_direct"]
        or summarize_rows(raw["harness_rows"])
        != raw["arms"]["four_b_semantic_skills"]
    ):
        raise ValueError("semantic skill aggregate boundary differs")
    harness_vs_four = paired_metrics(
        raw["harness_rows"],
        raw["four_b_rows"],
        seed=f"{config.bootstrap_seed}:harness-four",
        samples=config.bootstrap_samples,
    )
    harness_vs_nine = paired_metrics(
        raw["harness_rows"],
        raw["nine_b_rows"],
        seed=f"{config.bootstrap_seed}:harness-nine",
        samples=config.bootstrap_samples,
    )
    four_vs_nine = paired_metrics(
        raw["four_b_rows"],
        raw["nine_b_rows"],
        seed=f"{config.bootstrap_seed}:four-nine",
        samples=config.bootstrap_samples,
    )
    gates = admission_gates(raw, harness_vs_four, harness_vs_nine)
    admitted = all(gates.values())
    return {
        "schema_version": "nano_harness_semantic_skill_execution_public_v1",
        "experiment_id": config.experiment_id,
        "identity": {
            "result_revision": git_revision(),
            "preregister_revision": PREREG_REVISION,
            "config_sha256": sha256_file(CONFIG),
            "preregister_sha256": sha256_file(PREREG),
            "raw_result_sha256": sha256_file(RAW),
            "case_contract_sha256": prereg["identity"][
                "case_contract_sha256"
            ],
            "parent_config_sha256": config.parent_config_sha256,
            "v2_report_sha256": config.v2_report_sha256,
            "canary_rejection_sha256": config.canary_rejection_sha256,
            "service_receipt_sha256": config.service_receipt_sha256,
        },
        "data": {
            "cases": len(cases),
            "families": {family: 128 for family in FAMILIES},
            "freshness": prereg["freshness"],
            "benchmark_canary_holdout_rows_or_outputs": 0,
            "training_eligible_cases": 0,
        },
        "mechanism": prereg["mechanism"],
        "arms": raw["arms"],
        "routing": raw["routing"],
        "comparisons": {
            "harness_vs_four_b": harness_vs_four,
            "harness_vs_nine_b": harness_vs_nine,
            "four_b_vs_nine_b": four_vs_nine,
        },
        "interrupted_preflights": [
            {
                "stage": "remote_revision_check",
                "failure_kind": "github_http2_framing",
                "model_generation_started": False,
                "result_artifact_created": False,
            },
            {
                "stage": "local_service_health_check",
                "failure_kind": "localhost_sent_to_proxy",
                "model_generation_started": False,
                "result_artifact_created": False,
            },
        ],
        "decision": {
            "gates": gates,
            "local_semantic_skill_admitted": admitted,
            "fresh_local_replication_preregistration_allowed": admitted,
            "fresh_local_replication_generation_allowed": False,
            "canary_allowed": False,
            "benchmark_allowed": False,
            "independent_holdout_allowed": False,
            "training_allowed": False,
            "further_tuning_or_rerun_on_observed_cases_allowed": False,
            "next_action": (
                "Pre-register the same two typed semantic tools on a fresh "
                "history-disjoint paraphrase and numerical-regime replication. "
                "Keep the observed canary, complete benchmark, holdout, and "
                "training closed."
                if admitted
                else "Reject the semantic-skill mechanism and preserve the "
                "evidence; do not tune or rerun on these cases."
            ),
        },
        "evaluation_boundary": raw["evaluation_boundary"],
        "claim_boundary": (
            "This result establishes only fresh synthetic local mechanism "
            "admission for two typed semantic skills. The direct arms scoring "
            "zero and the harness scoring one are not benchmark superiority. "
            "Only a separately pre-registered fresh local replication is "
            "authorized; canary, benchmark, holdout, and training remain closed."
        ),
    }


def render_markdown(report: dict) -> str:
    decision = report["decision"]
    return f"""# Qwen3.5 Typed Semantic Skill Execution v1 Result

## 结论

- local semantic skill admitted：
  `{str(decision['local_semantic_skill_admitted']).lower()}`；
- fresh local replication preregistration allowed：
  `{str(decision['fresh_local_replication_preregistration_allowed']).lower()}`；
- fresh replication generation allowed：`false`；
- canary / benchmark / holdout / training：全部 `false`。

## 具体结果

| Arm | Correct | Accuracy | Parseable |
| --- | ---: | ---: | ---: |
{chr(10).join(
    f"| {name} | {metrics['correct']}/256 | {metrics['accuracy']:.4f} | "
    f"{metrics['parseable']}/256 |"
    for name, metrics in report['arms'].items()
)}

两个 family 均为 128 cases：

- implicit double/triple total：harness 128/128；
- first strictly profitable whole period：harness 128/128；
- 4B direct 和 9B direct 在两个 family 都是 0/128。

这说明 typed semantic skill 能补足“隐含语言算子”和“严格离散边界”，但这是
刻意构造的 local mechanism surface，**不是 benchmark 分数**。Direct 两臂为 0
也意味着下一步必须用 paraphrase 和不同数值分布做 fresh replication，不能直接
推广到真实任务。

## Routing

```json
{json.dumps(report['routing'], indent=2, sort_keys=True)}
```

256 prompt-only routes、single-tool exposures、verified executions 和 feedback
result matches；0 retry、0 fallback。Router 不读 case metadata，executor 不读
expected 或 correctness。

## Paired Comparisons

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

## Frozen Gates

```json
{json.dumps(decision['gates'], indent=2, sort_keys=True)}
```

## Interrupted Preflights

两次启动都在模型请求前 fail-close：

- GitHub HTTP/2 framing error；
- localhost health 被错误送到代理而返回 403。

两次均确认 result artifact 不存在，随后用 HTTP/1.1 和
`NO_PROXY=127.0.0.1,localhost` 完成唯一正式 run。它们不是额外实验臂。

## Evidence

- prereg commit：`{report['identity']['preregister_revision']}`；
- config SHA：`{report['identity']['config_sha256']}`；
- prereg SHA：`{report['identity']['preregister_sha256']}`；
- raw result SHA：`{report['identity']['raw_result_sha256']}`；
- case contract SHA：`{report['identity']['case_contract_sha256']}`；
- canary rejection SHA：
  `{report['identity']['canary_rejection_sha256']}`。

下一步只允许另行预注册 fresh history-disjoint paraphrase/numerical-regime
replication。已观察的 211-case canary 不重跑，complete benchmark、independent
holdout 和 training 继续关闭。
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
                "arms": report["arms"],
                "routing": report["routing"],
                "comparisons": report["comparisons"],
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
