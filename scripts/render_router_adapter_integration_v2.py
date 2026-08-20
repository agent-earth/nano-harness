#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

from nano_harness.baseline import sha256_file
from nano_harness.router_adapter_integration import parse_route
from nano_harness.router_adapter_integration_v2 import (
    build_cases,
    load_config,
)
from nano_harness.semantic_model_router import (
    ALL_FAMILIES,
    summarize_rows,
)
from nano_harness.verified_tool_execution import public_case_contract
from scripts.render_verified_tool_execution_v1 import paired_metrics


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/harness/qwen35_router_adapter_integration_v2.json"
PREREG = (
    ROOT
    / "docs/experiments/qwen35_router_adapter_integration_v2.preregister.json"
)
SERVICE = (
    ROOT
    / "docs/experiments/"
    "qwen35_router_adapter_integration_v2_service.public.json"
)
RAW = ROOT / "results/harness/qwen35-router-adapter-integration-v2/result.json"
PUBLIC_JSON = (
    ROOT / "docs/results/qwen35_router_adapter_integration_v2.public.json"
)
MARKDOWN = ROOT / "docs/results/qwen35_router_adapter_integration_v2.md"


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def committed_preregister_sha256() -> str:
    content = subprocess.check_output(
        [
            "git",
            "show",
            "HEAD:docs/experiments/"
            "qwen35_router_adapter_integration_v2.preregister.json",
        ],
        cwd=ROOT,
    )
    return hashlib.sha256(content).hexdigest()


def admission_gates(
    raw: dict,
    candidate_vs_four: dict,
    candidate_vs_nine: dict,
) -> dict[str, bool]:
    config = load_config(CONFIG)
    arms = raw["arms"]
    routing = raw["routing"]
    return {
        "all_three_arms_complete_and_parseable_128": all(
            arms[name]["cases"] == 128
            and arms[name]["parseable"] == 128
            for name in (
                "four_b_direct",
                "nine_b_direct",
                "four_b_router_adapter_v2",
            )
        ),
        "router_outputs_parseable_128": all(
            receipt["router"]["label"] is not None
            for receipt in raw["receipts"].values()
        ),
        "router_a_recall_32": sum(
            receipt["router"]["expected_label"] == "A"
            and receipt["router"]["label"] == "A"
            for receipt in raw["receipts"].values()
        )
        == 32,
        "router_b_recall_32": sum(
            receipt["router"]["expected_label"] == "B"
            and receipt["router"]["label"] == "B"
            for receipt in raw["receipts"].values()
        )
        == 32,
        "router_c_precision_64": routing["negative_c_correct"] == 64,
        "negative_false_positive_routes_zero": (
            routing["negative_false_positive_routes"] == 0
        ),
        "positive_verified_executions_64": (
            routing["verified_executions"] == 64
        ),
        "positive_feedback_result_matches_64": sum(
            receipt.get("feedback_result_match", False)
            for receipt in raw["receipts"].values()
        )
        == 64,
        "fallbacks_zero": routing["fallbacks"] == 0,
        "negative_candidate_exact_direct_parity": all(
            all(
                candidate.get(field) == direct.get(field)
                for field in ("output", "prediction", "parseable", "correct")
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
            arms["four_b_router_adapter_v2"]["by_family"][family]["correct"]
            >= arms["four_b_direct"]["by_family"][family]["correct"]
            and arms["four_b_router_adapter_v2"]["by_family"][family]["correct"]
            >= arms["nine_b_direct"]["by_family"][family]["correct"]
            for family in ALL_FAMILIES
        ),
    }


def build_report() -> dict:
    config = load_config(CONFIG)
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    service = json.loads(SERVICE.read_text(encoding="utf-8"))
    raw = json.loads(RAW.read_text(encoding="utf-8"))
    cases = build_cases(config)
    by_id = {case["case_id"]: case for case in cases}
    if (
        prereg.get("schema_version")
        != "nano_harness_router_adapter_integration_preregister_v2"
        or prereg.get("identity", {}).get("config_sha256")
        != sha256_file(CONFIG)
        or committed_preregister_sha256() != sha256_file(PREREG)
        or service.get("schema_version")
        != "nano_harness_router_adapter_integration_v2_service"
        or service.get("preregister_sha256") != sha256_file(PREREG)
        or service.get("v2_generation_started") is not False
        or service.get("integration_v1_rerun") is not False
        or raw.get("schema_version")
        != "nano_harness_router_adapter_integration_result_v2"
        or raw.get("identity", {}).get("case_contract")
        != public_case_contract(cases)
        or raw.get("identity", {}).get("adapter_sha256")
        != config.adapter_tree_sha256
        or raw.get("identity", {}).get("parity_report_sha256")
        != config.parity_report_sha256
        or raw.get("v2_service_receipt") != service
    ):
        raise ValueError("router integration v2 result identity differs")
    for key in ("four_b_rows", "nine_b_rows", "candidate_rows"):
        rows = raw[key]
        if len(rows) != 128 or {row["case_id"] for row in rows} != set(by_id):
            raise ValueError(f"router integration v2 {key} differs")
    if set(raw["receipts"]) != set(by_id):
        raise ValueError("router integration v2 receipt identities differ")
    confusion: Counter[tuple[str, str | None]] = Counter()
    mapping = {
        "A": "implicit_scale_total",
        "B": "first_strict_profit_period",
        "C": "NONE",
    }
    for case_id, receipt in raw["receipts"].items():
        case = by_id[case_id]
        router = receipt["router"]
        if (
            parse_route(router["output"]) != router["label"]
            or router["selected_route"] != mapping.get(router["label"])
            or router["expected_label"] != case["expected_label"]
            or router["expected_route"] != case["expected_route"]
            or router["correct"]
            != (router["selected_route"] == case["expected_route"])
            or router["model"] != config.served_adapter_name
            or router["adapter_sha256"] != config.adapter_tree_sha256
            or router["uses_case_metadata"]
            or router["uses_expected_answer"]
            or router["uses_case_correctness"]
        ):
            raise ValueError("router integration v2 receipt differs")
        confusion[(case["expected_label"], router["label"])] += 1
    expected_boundary = {
        "training_eligible_cases": 0,
        "router_uses_case_metadata": False,
        "router_uses_expected_answer": False,
        "router_uses_case_correctness": False,
        "executor_uses_expected_answer": False,
        "executor_uses_case_correctness": False,
        "integration_v1_rows_or_outputs_loaded": False,
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
        != raw["arms"]["four_b_router_adapter_v2"]
    ):
        raise ValueError("router integration v2 aggregate boundary differs")
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
    decision_gates = admission_gates(
        raw, candidate_vs_four, candidate_vs_nine
    )
    admitted = all(decision_gates.values())
    return {
        "schema_version": (
            "nano_harness_router_adapter_integration_public_v2"
        ),
        "experiment_id": config.experiment_id,
        "identity": {
            "result_revision": git_revision(),
            "config_sha256": sha256_file(CONFIG),
            "preregister_sha256": sha256_file(PREREG),
            "service_receipt_sha256": sha256_file(SERVICE),
            "raw_result_sha256": sha256_file(RAW),
            "case_contract_sha256": prereg["identity"][
                "case_contract_sha256"
            ],
            "integration_v1_report_sha256": (
                config.integration_v1_report_sha256
            ),
            "parity_report_sha256": config.parity_report_sha256,
            "remapped_adapter_sha256": config.adapter_tree_sha256,
        },
        "data": {
            "cases": 128,
            "positive_cases": 64,
            "negative_cases": 64,
            "family_counts": {family: 32 for family in ALL_FAMILIES},
            "freshness": prereg["freshness"],
            "benchmark_canary_holdout_rows_or_outputs": 0,
            "integration_v1_rows_or_outputs_loaded": False,
            "training_eligible_cases": 0,
        },
        "arms": raw["arms"],
        "routing": raw["routing"],
        "confusion": [
            {
                "expected_label": expected,
                "selected_label": selected,
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
            "gates": decision_gates,
            "adapter_integration_v2_admitted": admitted,
            "question_only_scan_preregistration_allowed": admitted,
            "question_only_scan_generation_allowed": False,
            "integration_v1_rerun_allowed": False,
            "integration_v2_rerun_allowed": False,
            "benchmark_generation_allowed": False,
            "canary_allowed": False,
            "holdout_allowed": False,
            "training_or_rl_allowed": False,
            "further_tuning_allowed": False,
            "next_action": (
                "Pre-register exactly one question-only real-surface scan "
                "without answers, scoring, typed execution, or benchmark "
                "generation."
                if admitted
                else
                "Reject the remapped router transfer. Do not rerun or tune "
                "V1 or V2."
            ),
        },
        "evaluation_boundary": raw["evaluation_boundary"],
        "claim_boundary": (
            "This is fresh synthetic transfer evidence for the remapped "
            "adapter only. It is not benchmark, canary, holdout, training, "
            "RL, or final model-superiority evidence."
        ),
    }


def render_markdown(report: dict) -> str:
    decision = report["decision"]
    verdict = (
        "ADMIT"
        if decision["adapter_integration_v2_admitted"]
        else "REJECT"
    )
    return f"""# Qwen3.5 Router Adapter Integration v2 Result

## Verdict

**{verdict}.**

V2 uses the content-identical namespace-remapped adapter on 128 new prompts.
Neither V1 rows nor V1 outputs were loaded or regenerated.

## Arms

```json
{json.dumps(report['arms'], indent=2, sort_keys=True)}
```

## Routing

```json
{json.dumps(report['routing'], indent=2, sort_keys=True)}
```

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

## Boundaries

V1 and V2 cannot be rerun. Passing allows only a separately pre-registered
question-only scan. Benchmark, canary, holdout, training, and RL stay closed.

## Evidence

- prereg SHA: `{report['identity']['preregister_sha256']}`;
- service SHA: `{report['identity']['service_receipt_sha256']}`;
- raw SHA: `{report['identity']['raw_result_sha256']}`;
- remapped adapter SHA: `{report['identity']['remapped_adapter_sha256']}`.
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
                "comparisons": report["comparisons"],
                "decision": report["decision"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
