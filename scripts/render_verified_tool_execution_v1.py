#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import math
import random
import subprocess
from pathlib import Path

from nano_harness.baseline import sha256_file
from nano_harness.verified_tool_execution import (
    FAMILIES,
    build_cases,
    load_config,
    parse_and_execute_plan,
    public_case_contract,
    summarize_rows,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/harness/qwen35_verified_tool_execution_v1.json"
PREREG = (
    ROOT
    / "docs/experiments/qwen35_verified_tool_execution_v1.preregister.json"
)
SERVICE_RECEIPT = (
    ROOT
    / "docs/experiments/"
    "qwen35_verified_tool_execution_services_v1.public.json"
)
RAW = (
    ROOT / "results/harness/qwen35-verified-tool-execution-v1/result.json"
)
PUBLIC_JSON = (
    ROOT / "docs/results/qwen35_verified_tool_execution_v1.public.json"
)
MARKDOWN = ROOT / "docs/results/qwen35_verified_tool_execution_v1.md"


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def paired_metrics(
    candidate: list[dict],
    baseline: list[dict],
    *,
    seed: str,
    samples: int,
) -> dict:
    candidate_by_id = {row["case_id"]: row for row in candidate}
    baseline_by_id = {row["case_id"]: row for row in baseline}
    if set(candidate_by_id) != set(baseline_by_id) or not candidate_by_id:
        raise ValueError("verified tool paired row identities differ")
    case_ids = sorted(candidate_by_id)
    deltas = [
        int(candidate_by_id[case_id]["correct"])
        - int(baseline_by_id[case_id]["correct"])
        for case_id in case_ids
    ]
    candidate_only = [
        case_id
        for case_id in case_ids
        if candidate_by_id[case_id]["correct"]
        and not baseline_by_id[case_id]["correct"]
    ]
    baseline_only = [
        case_id
        for case_id in case_ids
        if baseline_by_id[case_id]["correct"]
        and not candidate_by_id[case_id]["correct"]
    ]
    rng = random.Random(seed)
    estimates = sorted(
        sum(deltas[rng.randrange(len(deltas))] for _ in deltas)
        / len(deltas)
        for _ in range(samples)
    )
    discordant = len(candidate_only) + len(baseline_only)
    tail = min(len(candidate_only), len(baseline_only))
    p_value = (
        min(
            1.0,
            2.0
            * sum(
                math.comb(discordant, index)
                for index in range(tail + 1)
            )
            / (2**discordant),
        )
        if discordant
        else 1.0
    )
    return {
        "cases": len(case_ids),
        "candidate_accuracy": sum(
            candidate_by_id[case_id]["correct"] for case_id in case_ids
        )
        / len(case_ids),
        "baseline_accuracy": sum(
            baseline_by_id[case_id]["correct"] for case_id in case_ids
        )
        / len(case_ids),
        "delta": sum(deltas) / len(deltas),
        "paired_bootstrap_95_ci": [
            estimates[int(samples * 0.025)],
            estimates[min(samples - 1, int(samples * 0.975))],
        ],
        "mcnemar_exact_p": p_value,
        "paired_counts": {
            "candidate_only": len(candidate_only),
            "baseline_only": len(baseline_only),
            "both_correct": sum(
                candidate_by_id[case_id]["correct"]
                and baseline_by_id[case_id]["correct"]
                for case_id in case_ids
            ),
            "both_wrong": sum(
                not candidate_by_id[case_id]["correct"]
                and not baseline_by_id[case_id]["correct"]
                for case_id in case_ids
            ),
        },
        "candidate_only_case_ids": candidate_only,
        "baseline_only_case_ids": baseline_only,
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
    }


def admission_gates(
    raw: dict,
    harness_vs_four: dict,
    harness_vs_nine: dict,
    *,
    contract_failures: int,
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
                "four_b_verified_tool",
            )
        ),
        "harness_accuracy_gt_four_b_direct": (
            harness_vs_four["candidate_accuracy"]
            > harness_vs_four["baseline_accuracy"]
        ),
        "harness_vs_four_b_bootstrap_ci_lower_gt_zero": (
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
        "harness_accuracy_gt_nine_b_direct": (
            harness_vs_nine["candidate_accuracy"]
            > harness_vs_nine["baseline_accuracy"]
        ),
        "harness_vs_nine_b_bootstrap_ci_lower_gt_zero": (
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
            arms["four_b_verified_tool"]["by_family"][family]["correct"]
            >= arms["four_b_direct"]["by_family"][family]["correct"]
            and arms["four_b_verified_tool"]["by_family"][family]["correct"]
            >= arms["nine_b_direct"]["by_family"][family]["correct"]
            for family in FAMILIES
        ),
        "verified_execution_count_positive": (
            raw["routing"]["verified_executions"] > 0
        ),
        "executor_contract_failures_zero": contract_failures == 0,
    }


def build_report() -> dict:
    config = load_config(CONFIG)
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    service = json.loads(
        SERVICE_RECEIPT.read_text(encoding="utf-8")
    )
    raw = json.loads(RAW.read_text(encoding="utf-8"))
    cases = build_cases(config)
    cases_by_id = {row["case_id"]: row for row in cases}
    if (
        prereg["schema_version"]
        != "nano_harness_verified_tool_execution_preregister_v1"
        or prereg["identity"]["config_sha256"] != sha256_file(CONFIG)
        or prereg["execution_boundary"]["evaluation_started"] is not False
        or service["generation_started"] is not False
        or raw["schema_version"]
        != "nano_harness_verified_tool_execution_result_v1"
        or raw["identity"]["case_contract"] != public_case_contract(cases)
        or raw["identity"]["service_receipt_sha256"]
        != sha256_file(SERVICE_RECEIPT)
    ):
        raise ValueError("verified tool result identity differs")
    for key in ("four_b_rows", "nine_b_rows", "harness_rows"):
        rows = raw[key]
        if (
            len(rows) != len(cases)
            or {row["case_id"] for row in rows} != set(cases_by_id)
        ):
            raise ValueError(f"verified tool {key} identities differ")
    if set(raw["harness_receipts"]) != set(cases_by_id):
        raise ValueError("verified tool receipt identities differ")
    contract_failures = 0
    final_reasons = {}
    for case_id, receipt in raw["harness_receipts"].items():
        final = receipt["receipt"]
        final_reasons[final["reason"]] = (
            final_reasons.get(final["reason"], 0) + 1
        )
        if final["executed"]:
            replay = parse_and_execute_plan(
                receipt["plan_attempts"][-1]["output"],
                expected_tool=cases_by_id[case_id]["family"],
                source_facts=cases_by_id[case_id]["source_facts"],
            )
            if (
                replay != final
                or final["result"] != cases_by_id[case_id]["expected"]
                or receipt["fallback_used"]
                or not receipt["final_feedback_sent"]
            ):
                raise ValueError(
                    "verified tool executed receipt differs"
                )
        else:
            contract_failures += 1
            if (
                not receipt["fallback_used"]
                or receipt["final_feedback_sent"]
            ):
                raise ValueError(
                    "verified tool fallback receipt differs"
                )
    if (
        summarize_rows(raw["four_b_rows"]) != raw["arms"]["four_b_direct"]
        or summarize_rows(raw["nine_b_rows"])
        != raw["arms"]["nine_b_direct"]
        or summarize_rows(raw["harness_rows"])
        != raw["arms"]["four_b_verified_tool"]
        or raw["evaluation_boundary"]
        != {
            "training_eligible_cases": 0,
            "executor_uses_expected_answer": False,
            "executor_uses_case_correctness": False,
            "benchmark_rows_loaded": False,
            "benchmark_outputs_loaded": False,
            "canary_rows_loaded": False,
            "independent_holdout_rows_loaded": False,
        }
    ):
        raise ValueError("verified tool aggregate boundary differs")
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
    gates = admission_gates(
        raw,
        harness_vs_four,
        harness_vs_nine,
        contract_failures=contract_failures,
    )
    admitted = all(gates.values())
    return {
        "schema_version": (
            "nano_harness_verified_tool_execution_public_v1"
        ),
        "experiment_id": config.experiment_id,
        "identity": {
            "result_revision": git_revision(),
            "config_sha256": sha256_file(CONFIG),
            "preregister_sha256": sha256_file(PREREG),
            "service_receipt_sha256": sha256_file(SERVICE_RECEIPT),
            "raw_result_sha256": sha256_file(RAW),
            "case_contract_sha256": prereg["identity"][
                "case_contract_sha256"
            ],
            "four_b_model_config_sha256": (
                config.four_b_model_config_sha256
            ),
            "nine_b_model_config_sha256": (
                config.nine_b_model_config_sha256
            ),
        },
        "data": {
            "cases": len(cases),
            "family_counts": prereg["family_counts"],
            "prior_surface_prompt_overlap": prereg[
                "contamination_audit"
            ][
                "prior_surface_prompt_overlap"
            ],
            "benchmark_prompt_overlap": prereg["contamination_audit"][
                "benchmark_prompt_overlap"
            ],
            "benchmark_canary_holdout_rows_or_outputs": 0,
        },
        "arms": raw["arms"],
        "routing": {
            **raw["routing"],
            "executor_contract_failures": contract_failures,
            "final_reason_counts": dict(sorted(final_reasons.items())),
        },
        "comparisons": {
            "harness_vs_four_b": harness_vs_four,
            "harness_vs_nine_b": harness_vs_nine,
            "four_b_vs_nine_b": four_vs_nine,
        },
        "decision": {
            "gates": gates,
            "local_harness_admitted": admitted,
            "canary_allowed": admitted,
            "benchmark_allowed": False,
            "independent_holdout_allowed": False,
            "further_tuning_on_observed_cases_allowed": False,
            "next_action": (
                "Consume only this exact harness in a separately frozen "
                "211-case canary; do not change tools, prompts, retry, or "
                "budgets."
                if admitted
                else "Reject local admission and preserve the full evidence; "
                "do not tune or rerun on these cases."
            ),
        },
        "evaluation_boundary": raw["evaluation_boundary"],
        "claim_boundary": (
            "This result establishes only fresh synthetic local harness "
            "evidence. It does not authorize complete benchmark, holdout, or "
            "final 4B/9B superiority claims."
        ),
    }


def render_markdown(report: dict) -> str:
    comparisons = report["comparisons"]
    decision = report["decision"]
    return f"""# Qwen3.5 Verified Tool-Execution Harness v1 Result

## 结论

- local harness admitted：
  `{str(decision['local_harness_admitted']).lower()}`；
- 211-case canary allowed：
  `{str(decision['canary_allowed']).lower()}`；
- complete benchmark allowed：`false`；
- tuning/rerun on observed cases：`false`。

## Arms

| Arm | Correct | Accuracy | Parseable |
| --- | ---: | ---: | ---: |
{chr(10).join(
    f"| {name} | {metrics['correct']}/256 | {metrics['accuracy']:.4f} | "
    f"{metrics['parseable']}/256 |"
    for name, metrics in report['arms'].items()
)}

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
    for name, row in comparisons.items()
)}

## Routing

```json
{json.dumps(report['routing'], indent=2, sort_keys=True)}
```

192 cases completed plan → safe execute → verified-result feedback. 64
`labor_total` cases exhausted the frozen one-retry plan contract and fell back to
4B direct, so `executor_contract_failures_zero` fails even though the harness
has a large significant net gain.

## Frozen gates

```json
{json.dumps(decision['gates'], indent=2, sort_keys=True)}
```

## Evidence

- config SHA：`{report['identity']['config_sha256']}`；
- prereg SHA：`{report['identity']['preregister_sha256']}`；
- service receipt SHA：
  `{report['identity']['service_receipt_sha256']}`；
- raw result SHA：`{report['identity']['raw_result_sha256']}`；
- case contract SHA：`{report['identity']['case_contract_sha256']}`。

公开报告只包含聚合、case IDs、reason counts 和 SHA；不公开 prompt、facts、
expected、model outputs 或 full tool trajectories。Canary 和完整 benchmark 继续关闭。
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
                "comparisons": report["comparisons"],
                "routing": report["routing"],
                "public_json": str(PUBLIC_JSON),
                "markdown": str(MARKDOWN),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
