#!/usr/bin/env python3

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from nano_harness.baseline import sha256_file
from nano_harness.semantic_skill_execution import (
    FAMILIES,
    load_config as load_mechanism_config,
    parse_and_execute_plan,
    route_prompt,
    summarize_rows,
)
from nano_harness.semantic_skill_replication import (
    build_cases,
    load_config,
    public_case_contract,
)
from scripts.render_semantic_skill_execution_v1 import admission_gates
from scripts.render_verified_tool_execution_v1 import paired_metrics


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "configs/harness/qwen35_semantic_skill_replication_v1.json"
)
PREREG = (
    ROOT
    / "docs/experiments/qwen35_semantic_skill_replication_v1.preregister.json"
)
RAW = (
    ROOT / "results/harness/qwen35-semantic-skill-replication-v1/result.json"
)
PUBLIC_JSON = (
    ROOT / "docs/results/qwen35_semantic_skill_replication_v1.public.json"
)
MARKDOWN = ROOT / "docs/results/qwen35_semantic_skill_replication_v1.md"
PREREG_REVISION = "ea13026b1e4f8692a564ed5e5153083971e7ad60"


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def build_report() -> dict:
    config = load_config(CONFIG)
    mechanism = load_mechanism_config(config.parent_config_path)
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    raw = json.loads(RAW.read_text(encoding="utf-8"))
    cases = build_cases(config)
    cases_by_id = {row["case_id"]: row for row in cases}
    if (
        prereg.get("schema_version")
        != "nano_harness_semantic_skill_replication_preregister_v1"
        or prereg.get("identity", {}).get("config_sha256")
        != sha256_file(CONFIG)
        or prereg.get("identity", {}).get("code_revision")
        != "3323ad8a8d972f823007ad4c7184779769507bb7"
        or raw.get("schema_version")
        != "nano_harness_semantic_skill_replication_result_v1"
        or raw.get("identity", {}).get("case_contract")
        != public_case_contract(cases)
        or raw.get("identity", {}).get("parent_config_sha256")
        != config.parent_config_sha256
        or raw.get("identity", {}).get("parent_preregister_sha256")
        != config.parent_preregister_sha256
        or raw.get("identity", {}).get("parent_report_sha256")
        != config.parent_report_sha256
    ):
        raise ValueError("semantic skill replication result identity differs")
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
        raise ValueError("semantic replication preregistration is not committed")
    expected_mechanism = prereg["mechanism_invariance"]
    actual_mechanism = {
        "bootstrap_samples": config.bootstrap_samples,
        "chat_template_kwargs": {"enable_thinking": False},
        "direct_max_tokens": config.direct_max_tokens,
        "final_max_tokens": config.final_max_tokens,
        "maximum_harness_losses": config.maximum_harness_losses,
        "minimum_harness_wins": config.minimum_harness_wins,
        "plan_max_tokens": config.plan_max_tokens,
        "plan_retry_limit": config.plan_retry_limit,
        "plan_structured_output_regex_by_family": (
            mechanism.plan_structured_output_regex_by_family
        ),
        "route_markers": mechanism.route_markers,
        "significance_alpha": config.significance_alpha,
        "skill_router": mechanism.skill_router,
        "temperature": 0.0,
    }
    if (
        raw.get("mechanism_identity")
        != {
            "skill_router": mechanism.skill_router,
            "route_markers": mechanism.route_markers,
            "plan_structured_output_regex_by_family": (
                mechanism.plan_structured_output_regex_by_family
            ),
            "direct_max_tokens": mechanism.direct_max_tokens,
            "plan_max_tokens": mechanism.plan_max_tokens,
            "final_max_tokens": mechanism.final_max_tokens,
            "plan_retry_limit": mechanism.plan_retry_limit,
        }
        or actual_mechanism != expected_mechanism
    ):
        raise ValueError("semantic replication mechanism identity differs")
    for key in ("four_b_rows", "nine_b_rows", "harness_rows"):
        rows = raw[key]
        if (
            len(rows) != len(cases)
            or {row["case_id"] for row in rows} != set(cases_by_id)
        ):
            raise ValueError(f"semantic replication {key} identities differ")
    if set(raw["harness_receipts"]) != set(cases_by_id):
        raise ValueError("semantic replication receipt identities differ")
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
            raise ValueError("semantic replication route receipt differs")
        replay = parse_and_execute_plan(
            receipt["plan_attempts"][-1]["output"],
            route=route,
            source_facts=case["source_facts"],
        )
        if (
            replay != receipt["receipt"]
            or replay["result"] != case["expected"]
        ):
            raise ValueError("semantic replication execution receipt differs")
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
        raise ValueError("semantic replication aggregate boundary differs")
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
    gate_input = {
        **raw,
        "config": {
            **raw["config"],
            "significance_alpha": config.significance_alpha,
            "minimum_harness_wins": config.minimum_harness_wins,
            "maximum_harness_losses": config.maximum_harness_losses,
        },
    }
    gates = admission_gates(gate_input, harness_vs_four, harness_vs_nine)
    admitted = all(gates.values())
    return {
        "schema_version": (
            "nano_harness_semantic_skill_replication_public_v1"
        ),
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
            "parent_preregister_sha256": config.parent_preregister_sha256,
            "parent_report_sha256": config.parent_report_sha256,
        },
        "data": {
            "cases": len(cases),
            "families": {family: 128 for family in FAMILIES},
            "freshness": prereg["freshness"],
            "prompt_regime": config.prompt_regime,
            "value_regime": config.value_regime,
            "benchmark_canary_holdout_rows_or_outputs": 0,
            "training_eligible_cases": 0,
        },
        "mechanism_invariance": expected_mechanism,
        "replication_delta": prereg["replication_delta"],
        "arms": raw["arms"],
        "routing": raw["routing"],
        "comparisons": {
            "harness_vs_four_b": harness_vs_four,
            "harness_vs_nine_b": harness_vs_nine,
            "four_b_vs_nine_b": four_vs_nine,
        },
        "decision": {
            "gates": gates,
            "replication_admitted": admitted,
            "real_task_transfer_preregistration_allowed": admitted,
            "real_task_generation_allowed": False,
            "canary_rerun_allowed": False,
            "benchmark_generation_allowed": False,
            "independent_holdout_allowed": False,
            "training_allowed": False,
            "further_tuning_or_rerun_on_observed_cases_allowed": False,
            "next_action": (
                "Pre-register a real-task transfer using the unchanged typed "
                "semantic skill mechanism and a prompt-only applicability "
                "router. Preserve direct output outside proven routes; keep "
                "generation closed until all real-task case identities, route "
                "eligibility, fallback, budgets, and gates are committed."
                if admitted
                else "Reject the semantic skill mechanism and preserve the "
                "evidence; do not tune or rerun this surface."
            ),
        },
        "evaluation_boundary": raw["evaluation_boundary"],
        "claim_boundary": (
            "This establishes fresh local replication only. It supports "
            "pre-registering a real-task transfer but is not benchmark, canary, "
            "holdout, training, or final model-superiority evidence."
        ),
    }


def render_markdown(report: dict) -> str:
    decision = report["decision"]
    return f"""# Qwen3.5 Typed Semantic Skill Replication v1 Result

## 结论

- replication admitted：`{str(decision['replication_admitted']).lower()}`；
- real-task transfer preregistration allowed：
  `{str(decision['real_task_transfer_preregistration_allowed']).lower()}`；
- real-task generation allowed：`false`；
- canary rerun / benchmark generation / holdout / training：全部 `false`。

## Fresh Replication

- unseen compact-display / kiosk contexts；
- small-integer numerical regime；
- parent case ID / prompt / source-fact overlap：0 / 0 / 0；
- prior benchmark prompt overlap：0；
- mechanism、models、services、budgets、retry、fallback 和 gates 不变。

| Arm | Correct | Accuracy | Parseable |
| --- | ---: | ---: | ---: |
{chr(10).join(
    f"| {name} | {metrics['correct']}/256 | {metrics['accuracy']:.4f} | "
    f"{metrics['parseable']}/256 |"
    for name, metrics in report['arms'].items()
)}

两个 semantic families 的 harness 都是 128/128。4B direct 为 5/256，
9B direct 为 4/256；因此这次不再是 parent 的双零 direct surface。

## Routing

```json
{json.dumps(report['routing'], indent=2, sort_keys=True)}
```

256 prompt routes、single-tool exposures、verified executions 和 feedback
matches，0 retry、0 fallback。

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

## Evidence

- prereg commit：`{report['identity']['preregister_revision']}`；
- config SHA：`{report['identity']['config_sha256']}`；
- prereg SHA：`{report['identity']['preregister_sha256']}`；
- raw result SHA：`{report['identity']['raw_result_sha256']}`；
- case contract SHA：`{report['identity']['case_contract_sha256']}`；
- parent report SHA：`{report['identity']['parent_report_sha256']}`。

通过只允许**另行预注册** real-task transfer。必须先冻结 real-task case
identities、prompt-only eligibility、direct-preserve 范围、fallback、budgets 和
gates；当前不能生成 benchmark outputs，也不能重跑已观察 211-case canary。
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
