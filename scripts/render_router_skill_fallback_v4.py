#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

from nano_harness.baseline import sha256_file
from nano_harness.router_adapter_integration import parse_route
from nano_harness.router_skill_fallback_v4 import (
    C_FAMILIES,
    POSITIVE_FAMILIES,
    build_cases,
    load_config,
)
from nano_harness.verified_tool_execution import public_case_contract
from scripts.render_router_adapter_integration_v3 import (
    summarize_all_families,
)
from scripts.render_verified_tool_execution_v1 import paired_metrics


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/harness/qwen35_router_skill_fallback_v4.json"
PREREG = (
    ROOT
    / "docs/experiments/qwen35_router_skill_fallback_v4.preregister.json"
)
SERVICE = (
    ROOT
    / "docs/experiments/"
    "qwen35_router_skill_fallback_v4_service.public.json"
)
RAW = ROOT / "results/harness/qwen35-router-skill-fallback-v4/result.json"
PUBLIC_JSON = (
    ROOT / "docs/results/qwen35_router_skill_fallback_v4.public.json"
)
MARKDOWN = ROOT / "docs/results/qwen35_router_skill_fallback_v4.md"


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
            "qwen35_router_skill_fallback_v4.preregister.json",
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
    cases = build_cases(config)
    families = (*POSITIVE_FAMILIES, *C_FAMILIES)
    arms = {
        "four_b_direct": summarize_all_families(
            raw["four_b_rows"], families
        ),
        "nine_b_direct": summarize_all_families(
            raw["nine_b_rows"], families
        ),
        "four_b_router_skill_v4": summarize_all_families(
            raw["candidate_rows"], families
        ),
    }
    return {
        "all_three_arms_complete_and_parseable_160": all(
            row["cases"] == 160 and row["parseable"] == 160
            for row in arms.values()
        ),
        "router_outputs_parseable_and_correct_160": all(
            parse_route(receipt["router"]["output"])
            == receipt["router"]["expected_label"]
            for receipt in raw["receipts"].values()
        ),
        "ab_verified_executions_32": (
            raw["routing"]["ab_verified_executions"] == 32
        ),
        "c_skill_verified_executions_128": (
            raw["routing"]["c_skill_executions"] == 128
        ),
        "c_skill_result_exact_128": sum(
            case["family"] in C_FAMILIES and candidate["correct"]
            for case, candidate in zip(cases, raw["candidate_rows"])
        )
        == 128,
        "fallbacks_zero": raw["routing"]["fallbacks"] == 0,
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
            arms["four_b_router_skill_v4"]["by_family"][family]["correct"]
            >= arms["four_b_direct"]["by_family"][family]["correct"]
            and arms["four_b_router_skill_v4"]["by_family"][family]["correct"]
            >= arms["nine_b_direct"]["by_family"][family]["correct"]
            for family in families
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
        != "nano_harness_router_skill_fallback_preregister_v4"
        or prereg.get("identity", {}).get("config_sha256")
        != sha256_file(CONFIG)
        or committed_preregister_sha256() != sha256_file(PREREG)
        or service.get("schema_version")
        != "nano_harness_router_skill_fallback_v4_service"
        or service.get("preregister_sha256") != sha256_file(PREREG)
        or service.get("v4_generation_started") is not False
        or raw.get("schema_version")
        != "nano_harness_router_skill_fallback_result_v4"
        or raw.get("identity", {}).get("case_contract")
        != public_case_contract(cases)
        or raw.get("identity", {}).get("adapter_sha256")
        != config.adapter_tree_sha256
        or raw.get("v4_service_receipt") != service
    ):
        raise ValueError("router skill fallback v4 result identity differs")
    for key in ("four_b_rows", "nine_b_rows", "candidate_rows"):
        rows = raw[key]
        if len(rows) != 160 or {row["case_id"] for row in rows} != set(by_id):
            raise ValueError(f"router skill fallback v4 {key} differs")
    if set(raw["receipts"]) != set(by_id):
        raise ValueError("router skill fallback v4 receipt identities differ")
    for case_id, receipt in raw["receipts"].items():
        case = by_id[case_id]
        router = receipt["router"]
        if (
            parse_route(router["output"]) != router["label"]
            or router["expected_label"] != case["expected_label"]
            or router["correct"] is not True
            or router["model"] != config.served_adapter_name
            or router["adapter_sha256"] != config.adapter_tree_sha256
            or router["uses_case_metadata"]
            or router["uses_expected_answer"]
            or router["uses_case_correctness"]
        ):
            raise ValueError("router skill fallback v4 router receipt differs")
        if case["family"] in C_FAMILIES:
            skill = receipt.get("c_skill_receipt", {})
            if (
                skill.get("schema_version")
                != "nano_harness_router_c_skill_receipt_v4"
                or skill.get("executed") is not True
                or skill.get("executor_uses_expected_answer") is not False
                or skill.get("executor_uses_case_correctness") is not False
                or skill.get("selector_uses_case_metadata") is not False
                or receipt.get("fallback_used") is not False
            ):
                raise ValueError("router skill fallback v4 C receipt differs")
        else:
            tool = receipt.get("receipt", {})
            if (
                tool.get("executed") is not True
                or tool.get("executor_uses_expected_answer") is not False
                or tool.get("executor_uses_case_correctness") is not False
                or receipt.get("fallback_used") is not False
            ):
                raise ValueError("router skill fallback v4 AB receipt differs")
    expected_boundary = {
        "training_eligible_cases": 0,
        "router_uses_case_metadata": False,
        "router_uses_expected_answer": False,
        "router_uses_case_correctness": False,
        "skill_selector_uses_case_metadata": False,
        "executor_uses_expected_answer": False,
        "executor_uses_case_correctness": False,
        "integration_v1_v2_v3_rows_or_outputs_loaded": False,
        "benchmark_rows_loaded": False,
        "canary_rows_loaded": False,
        "holdout_rows_loaded": False,
    }
    if raw["evaluation_boundary"] != expected_boundary:
        raise ValueError("router skill fallback v4 boundary differs")
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
    families = (*POSITIVE_FAMILIES, *C_FAMILIES)
    arms = {
        "four_b_direct": summarize_all_families(
            raw["four_b_rows"], families
        ),
        "nine_b_direct": summarize_all_families(
            raw["nine_b_rows"], families
        ),
        "four_b_router_skill_v4": summarize_all_families(
            raw["candidate_rows"], families
        ),
    }
    routing_by_family = {
        family: {
            "cases": 16,
            "router_correct": sum(
                case["family"] == family
                and raw["receipts"][case["case_id"]]["router"]["correct"]
                for case in cases
            ),
            "verified_executions": sum(
                case["family"] == family
                and bool(
                    raw["receipts"][case["case_id"]]
                    .get("c_skill_receipt", {})
                    .get("executed")
                    or raw["receipts"][case["case_id"]]
                    .get("receipt", {})
                    .get("executed")
                )
                for case in cases
            ),
            "fallbacks": sum(
                case["family"] == family
                and bool(
                    raw["receipts"][case["case_id"]].get("fallback_used")
                )
                for case in cases
            ),
        }
        for family in families
    }
    return {
        "schema_version": "nano_harness_router_skill_fallback_public_v4",
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
            "integration_v3_report_sha256": (
                config.integration_v3_report_sha256
            ),
            "remapped_adapter_sha256": config.adapter_tree_sha256,
        },
        "data": {
            "cases": 160,
            "family_counts": {family: 16 for family in families},
            "freshness": prereg["freshness"],
            "benchmark_canary_holdout_rows_or_outputs": 0,
            "training_eligible_cases": 0,
        },
        "arms": arms,
        "routing": raw["routing"],
        "routing_by_family": routing_by_family,
        "comparisons": {
            "candidate_vs_four_b": candidate_vs_four,
            "candidate_vs_nine_b": candidate_vs_nine,
            "four_b_vs_nine_b": four_vs_nine,
        },
        "decision": {
            "gates": gates,
            "router_skill_fallback_v4_admitted": admitted,
            "benchmark_treatment_preregistration_allowed": admitted,
            "benchmark_generation_allowed": False,
            "v1_v2_v3_v4_rerun_allowed": False,
            "canary_allowed": False,
            "holdout_allowed": False,
            "training_or_rl_allowed": False,
            "further_tuning_allowed": False,
            "next_action": (
                "Pre-register one benchmark-agnostic treatment transfer on "
                "the frozen matched local suite."
                if admitted
                else
                "Reject V4. Do not rerun or tune V1-V4."
            ),
        },
        "evaluation_boundary": raw["evaluation_boundary"],
        "claim_boundary": (
            "This is fresh synthetic evidence for a router plus eight typed "
            "C-skill fallback. It is not benchmark or final superiority evidence."
        ),
    }


def render_markdown(report: dict) -> str:
    decision = report["decision"]
    verdict = (
        "ADMIT"
        if decision["router_skill_fallback_v4_admitted"]
        else "REJECT"
    )
    return f"""# Qwen3.5 Router Skill Fallback v4 Result

## Verdict

**{verdict}.**

V4 keeps the admitted router and A/B verifier, replacing only `C -> 4B direct`
with eight typed, deterministic skills on 160 history-disjoint prompts.

## Arms

```json
{json.dumps(report['arms'], indent=2, sort_keys=True)}
```

## Routing And Skills

```json
{json.dumps(report['routing_by_family'], indent=2, sort_keys=True)}
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

V1-V4 cannot be rerun. Passing permits only a separately pre-registered
benchmark treatment. Benchmark generation, canary, holdout, training, and RL
remain closed.

## Evidence

- prereg SHA: `{report['identity']['preregister_sha256']}`;
- raw SHA: `{report['identity']['raw_result_sha256']}`;
- V3 report SHA: `{report['identity']['integration_v3_report_sha256']}`.
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
