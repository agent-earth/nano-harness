#!/usr/bin/env python3

from __future__ import annotations

import json
import subprocess
from collections import Counter
from pathlib import Path

from nano_harness.baseline import sha256_file
from nano_harness.semantic_model_router import (
    ALL_FAMILIES,
    NEGATIVE_FAMILIES,
    POSITIVE_FAMILIES,
    build_cases,
    load_config,
    parent_config,
    parse_route,
    public_case_contract,
    summarize_rows,
)
from scripts.render_verified_tool_execution_v1 import paired_metrics


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/harness/qwen35_semantic_model_router_v1.json"
PREREG = (
    ROOT / "docs/experiments/qwen35_semantic_model_router_v1.preregister.json"
)
RAW = ROOT / "results/harness/qwen35-semantic-model-router-v1/result.json"
PUBLIC_JSON = ROOT / "docs/results/qwen35_semantic_model_router_v1.public.json"
MARKDOWN = ROOT / "docs/results/qwen35_semantic_model_router_v1.md"
PREREG_REVISION = "6a30f64e32cb01957dcd03292180aece4e7a0cfb"


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def admission_gates(
    raw: dict,
    candidate_vs_four: dict,
    candidate_vs_nine: dict,
) -> dict[str, bool]:
    config = load_config(CONFIG)
    arms = raw["arms"]
    routing = raw["routing"]
    return {
        "all_rows_complete_and_parseable": all(
            arms[name]["cases"] == 256
            and arms[name]["parseable"] == 256
            for name in (
                "four_b_direct",
                "nine_b_direct",
                "four_b_model_router",
            )
        ),
        "router_outputs_parseable_256": all(
            receipt["router"]["selected_route"] is not None
            for receipt in raw["receipts"].values()
        ),
        "positive_route_recall_128": routing["positive_route_correct"] == 128,
        "negative_none_correct_128": routing["negative_none_correct"] == 128,
        "negative_false_positive_routes_zero": (
            routing["negative_false_positive_routes"] == 0
        ),
        "positive_verified_executions_128": (
            routing["verified_executions"] == 128
        ),
        "fallbacks_zero": routing["fallbacks"] == 0,
        "negative_candidate_exact_direct_parity": all(
            all(
                candidate.get(field) == direct.get(field)
                for field in (
                    "output",
                    "prediction",
                    "parseable",
                    "correct",
                )
            )
            for case, candidate, direct in zip(
                build_cases(config),
                raw["candidate_rows"],
                raw["four_b_rows"],
            )
            if not case["positive"]
        ),
        "candidate_vs_four_b_significant": (
            candidate_vs_four["candidate_accuracy"]
            > candidate_vs_four["baseline_accuracy"]
            and candidate_vs_four["paired_bootstrap_95_ci"][0] > 0
            and candidate_vs_four["mcnemar_exact_p"]
            < config.significance_alpha
        ),
        "candidate_vs_four_b_minimum_wins": (
            candidate_vs_four["paired_counts"]["candidate_only"]
            >= config.minimum_harness_wins
        ),
        "candidate_vs_four_b_maximum_losses": (
            candidate_vs_four["paired_counts"]["baseline_only"]
            <= config.maximum_harness_losses
        ),
        "candidate_vs_nine_b_significant": (
            candidate_vs_nine["candidate_accuracy"]
            > candidate_vs_nine["baseline_accuracy"]
            and candidate_vs_nine["paired_bootstrap_95_ci"][0] > 0
            and candidate_vs_nine["mcnemar_exact_p"]
            < config.significance_alpha
        ),
        "candidate_vs_nine_b_minimum_wins": (
            candidate_vs_nine["paired_counts"]["candidate_only"]
            >= config.minimum_harness_wins
        ),
        "candidate_vs_nine_b_maximum_losses": (
            candidate_vs_nine["paired_counts"]["baseline_only"]
            <= config.maximum_harness_losses
        ),
        "every_family_non_regression": all(
            arms["four_b_model_router"]["by_family"][family]["correct"]
            >= arms["four_b_direct"]["by_family"][family]["correct"]
            and arms["four_b_model_router"]["by_family"][family]["correct"]
            >= arms["nine_b_direct"]["by_family"][family]["correct"]
            for family in ALL_FAMILIES
        ),
    }


def build_report() -> dict:
    config = load_config(CONFIG)
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    raw = json.loads(RAW.read_text(encoding="utf-8"))
    cases = build_cases(config)
    by_id = {case["case_id"]: case for case in cases}
    if (
        prereg.get("schema_version")
        != "nano_harness_semantic_model_router_preregister_v1"
        or prereg.get("identity", {}).get("config_sha256")
        != sha256_file(CONFIG)
        or prereg.get("identity", {}).get("code_revision")
        != "f7cfa655397fb9bd675cf06d710be8b37084e5d2"
        or raw.get("schema_version")
        != "nano_harness_semantic_model_router_result_v1"
        or raw.get("identity", {}).get("case_contract")
        != public_case_contract(cases)
        or raw.get("identity", {}).get("mechanism_config_sha256")
        != config.mechanism_config_sha256
        or raw.get("identity", {}).get("applicability_report_sha256")
        != config.applicability_report_sha256
    ):
        raise ValueError("semantic model router result identity differs")
    if (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", PREREG_REVISION, "HEAD"],
            cwd=ROOT,
            check=False,
        ).returncode
        != 0
    ):
        raise ValueError("semantic model router preregistration is not committed")
    for key in ("four_b_rows", "nine_b_rows", "candidate_rows"):
        rows = raw[key]
        if (
            len(rows) != 256
            or {row["case_id"] for row in rows} != set(by_id)
        ):
            raise ValueError(f"semantic model router {key} identities differ")
    if set(raw["receipts"]) != set(by_id):
        raise ValueError("semantic model router receipt identities differ")
    confusion: Counter[tuple[str, str | None]] = Counter()
    for case_id, receipt in raw["receipts"].items():
        case = by_id[case_id]
        router = receipt["router"]
        if (
            parse_route(router["output"]) != router["selected_route"]
            or router["expected_route"] != case["expected_route"]
            or router["correct"]
            != (router["selected_route"] == case["expected_route"])
            or router["router_uses_case_metadata"]
            or router["router_uses_expected_answer"]
            or router["router_uses_case_correctness"]
        ):
            raise ValueError("semantic model router receipt differs")
        confusion[(router["expected_route"], router["selected_route"])] += 1
    expected_confusion = {
        ("NONE", "NONE"): 128,
        ("first_strict_profit_period", "first_strict_profit_period"): 64,
        ("implicit_scale_total", "NONE"): 64,
    }
    if dict(confusion) != expected_confusion:
        raise ValueError("semantic model router confusion differs")
    expected_boundary = {
        "training_eligible_cases": 0,
        "router_uses_case_metadata": False,
        "router_uses_expected_answer": False,
        "router_uses_case_correctness": False,
        "executor_uses_expected_answer": False,
        "executor_uses_case_correctness": False,
        "benchmark_rows_loaded": False,
        "canary_rows_loaded": False,
        "holdout_rows_loaded": False,
    }
    if (
        raw["evaluation_boundary"] != expected_boundary
        or summarize_rows(raw["four_b_rows"])
        != raw["arms"]["four_b_direct"]
        or summarize_rows(raw["nine_b_rows"])
        != raw["arms"]["nine_b_direct"]
        or summarize_rows(raw["candidate_rows"])
        != raw["arms"]["four_b_model_router"]
    ):
        raise ValueError("semantic model router aggregate boundary differs")
    parent = parent_config(config)
    candidate_vs_four = paired_metrics(
        raw["candidate_rows"],
        raw["four_b_rows"],
        seed=f"{config.bootstrap_seed}:candidate-four",
        samples=config.bootstrap_samples,
    )
    candidate_vs_nine = paired_metrics(
        raw["candidate_rows"],
        raw["nine_b_rows"],
        seed=f"{config.bootstrap_seed}:candidate-nine",
        samples=config.bootstrap_samples,
    )
    four_vs_nine = paired_metrics(
        raw["four_b_rows"],
        raw["nine_b_rows"],
        seed=f"{config.bootstrap_seed}:four-nine",
        samples=config.bootstrap_samples,
    )
    gates = admission_gates(raw, candidate_vs_four, candidate_vs_nine)
    admitted = all(gates.values())
    return {
        "schema_version": "nano_harness_semantic_model_router_public_v1",
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
            "mechanism_config_sha256": config.mechanism_config_sha256,
            "applicability_report_sha256": (
                config.applicability_report_sha256
            ),
        },
        "data": {
            "cases": 256,
            "positive_cases": 128,
            "negative_cases": 128,
            "family_counts": {family: 64 for family in ALL_FAMILIES},
            "freshness": prereg["freshness"],
            "benchmark_canary_holdout_rows_or_outputs": 0,
            "training_eligible_cases": 0,
        },
        "arms": raw["arms"],
        "routing": raw["routing"],
        "confusion": [
            {
                "expected_route": expected,
                "selected_route": selected,
                "cases": count,
            }
            for (expected, selected), count in sorted(
                confusion.items(), key=lambda item: str(item[0])
            )
        ],
        "comparisons": {
            "candidate_vs_four_b": candidate_vs_four,
            "candidate_vs_nine_b": candidate_vs_nine,
            "four_b_vs_nine_b": four_vs_nine,
        },
        "decision": {
            "gates": gates,
            "router_admitted": admitted,
            "real_question_model_scan_preregistration_allowed": False,
            "benchmark_generation_allowed": False,
            "canary_rerun_allowed": False,
            "independent_holdout_allowed": False,
            "training_allowed": False,
            "further_tuning_or_rerun_on_observed_cases_allowed": False,
            "router_precision_direction_supported": (
                raw["routing"]["negative_false_positive_routes"] == 0
            ),
            "router_recall_supported": (
                raw["routing"]["positive_route_correct"] == 128
            ),
            "next_action": (
                "Reject this router. Preserve zero-false-positive evidence, "
                "but do not tune prompt or rerun on these cases. On a fresh "
                "surface, test two independent binary skill detectors with a "
                "NONE-by-default composition and unchanged typed executors."
            ),
        },
        "evaluation_boundary": raw["evaluation_boundary"],
        "claim_boundary": (
            "This is negative fresh-local router evidence. It establishes "
            "perfect unsupported-task precision for this surface but only "
            "50% positive recall. It does not authorize a real question scan, "
            "benchmark generation, canary rerun, holdout, or training."
        ),
    }


def render_markdown(report: dict) -> str:
    decision = report["decision"]
    return f"""# Qwen3.5 Constrained Semantic Model Router v1 Result

## 结论

**拒绝，不允许 real question model scan。**

- unsupported negative：128/128 选 `NONE`，false positive 0；
- strict-profit positive：64/64 路由正确并执行成功；
- implicit-scale positive：0/64 路由正确，全部选 `NONE`；
- positive recall：64/128；
- candidate：64/256；4B direct：0/256；9B direct：0/256。

Router 的保守性方向成立，但召回 gate 明确失败。不能根据这些已观察样例改
prompt、枚举、预算或重跑。

## Confusion

```json
{json.dumps(report['confusion'], indent=2, sort_keys=True)}
```

## Routing

```json
{json.dumps(report['routing'], indent=2, sort_keys=True)}
```

所有 negative `NONE` rows 与 direct 保持一致；64 个 strict-profit rows 完成
verified execution。Implicit-scale rows安全地 direct-preserve，但没有得到预期
增益。

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

Aggregate 对 direct 显著，但不能覆盖 route recall 失败，故仍拒绝。

## Frozen Gates

```json
{json.dumps(decision['gates'], indent=2, sort_keys=True)}
```

## Evidence

- prereg commit：`{report['identity']['preregister_revision']}`；
- config SHA：`{report['identity']['config_sha256']}`；
- prereg SHA：`{report['identity']['preregister_sha256']}`；
- raw result SHA：`{report['identity']['raw_result_sha256']}`；
- case contract SHA：`{report['identity']['case_contract_sha256']}`。

下一机制只能在 fresh surface 测试两个独立 binary skill detectors，再用
`NONE` 默认组合；typed executors、source validation、feedback equality 和
fallback 保持不变。Benchmark generation、canary、holdout、training 继续关闭。
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
                "arms": report["arms"],
                "routing": report["routing"],
                "confusion": report["confusion"],
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
