#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from nano_harness.baseline import sha256_file
from nano_harness.router_adapter_integration import parse_route
from nano_harness.router_skill_fallback_v4 import C_FAMILIES, POSITIVE_FAMILIES
from nano_harness.router_skill_registry_v5 import build_cases, load_config
from nano_harness.verified_tool_execution import public_case_contract
from scripts.render_router_adapter_integration_v3 import summarize_all_families
from scripts.render_verified_tool_execution_v1 import paired_metrics


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/harness/qwen35_router_skill_registry_v5.json"
PREREG = (
    ROOT / "docs/experiments/qwen35_router_skill_registry_v5.preregister.json"
)
SERVICE = (
    ROOT / "docs/experiments/qwen35_router_skill_registry_v5_service.public.json"
)
RAW = ROOT / "results/harness/qwen35-router-skill-registry-v5/result.json"
PUBLIC_JSON = ROOT / "docs/results/qwen35_router_skill_registry_v5.public.json"
MARKDOWN = ROOT / "docs/results/qwen35_router_skill_registry_v5.md"


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def committed_preregister_sha256() -> str:
    content = subprocess.check_output(
        [
            "git",
            "show",
            "HEAD:docs/experiments/"
            "qwen35_router_skill_registry_v5.preregister.json",
        ],
        cwd=ROOT,
    )
    return hashlib.sha256(content).hexdigest()


def build_report() -> dict:
    config = load_config(CONFIG)
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    service = json.loads(SERVICE.read_text(encoding="utf-8"))
    raw = json.loads(RAW.read_text(encoding="utf-8"))
    cases = build_cases(config)
    by_id = {case["case_id"]: case for case in cases}
    if (
        prereg.get("schema_version")
        != "nano_harness_router_skill_registry_preregister_v5"
        or prereg.get("identity", {}).get("config_sha256")
        != sha256_file(CONFIG)
        or committed_preregister_sha256() != sha256_file(PREREG)
        or service.get("schema_version")
        != "nano_harness_router_skill_registry_v5_service"
        or service.get("preregister_sha256") != sha256_file(PREREG)
        or service.get("v5_generation_started") is not False
        or raw.get("schema_version")
        != "nano_harness_router_skill_registry_result_v5"
        or raw.get("identity", {}).get("case_contract")
        != public_case_contract(cases)
        or raw.get("identity", {}).get("adapter_sha256")
        != config.adapter_tree_sha256
        or raw.get("v5_service_receipt") != service
    ):
        raise ValueError("V5 result identity differs")
    for key in ("four_b_rows", "nine_b_rows", "candidate_rows"):
        rows = raw[key]
        if len(rows) != 160 or {row["case_id"] for row in rows} != set(by_id):
            raise ValueError(f"V5 {key} differs")
    if set(raw["receipts"]) != set(by_id):
        raise ValueError("V5 receipt IDs differ")
    for case_id, receipt in raw["receipts"].items():
        case = by_id[case_id]
        router = receipt["router"]
        if (
            parse_route(router["output"]) != case["expected_label"]
            or router["correct"] is not True
            or router["uses_case_metadata"]
            or router["uses_expected_answer"]
            or router["uses_case_correctness"]
        ):
            raise ValueError("V5 router receipt differs")
        if case["family"] in C_FAMILIES:
            registry = receipt.get("registry", {})
            skill = receipt.get("skill_receipt", {})
            if (
                registry.get("unique_match") is not True
                or registry.get("applicable_skills") != [case["family"]]
                or registry.get("uses_case_metadata") is not False
                or registry.get("uses_expected_answer") is not False
                or registry.get("uses_case_correctness") is not False
                or skill.get("executed") is not True
                or skill.get("executor_uses_expected_answer") is not False
                or skill.get("executor_uses_case_correctness") is not False
                or receipt.get("fallback_used") is not False
            ):
                raise ValueError("V5 C receipt differs")
        elif (
            receipt.get("receipt", {}).get("executed") is not True
            or receipt.get("fallback_used") is not False
        ):
            raise ValueError("V5 AB receipt differs")
    expected_boundary = {
        "training_eligible_cases": 0,
        "router_uses_case_metadata": False,
        "router_uses_expected_answer": False,
        "router_uses_case_correctness": False,
        "skill_registry_uses_case_metadata": False,
        "skill_registry_uses_expected_answer": False,
        "skill_registry_uses_case_correctness": False,
        "executor_uses_expected_answer": False,
        "executor_uses_case_correctness": False,
        "v1_v2_v3_v4_rows_or_outputs_loaded": False,
        "benchmark_rows_loaded": False,
        "canary_rows_loaded": False,
        "holdout_rows_loaded": False,
    }
    if raw["evaluation_boundary"] != expected_boundary:
        raise ValueError("V5 boundary differs")

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
    families = (*POSITIVE_FAMILIES, *C_FAMILIES)
    arms = {
        "four_b_direct": summarize_all_families(raw["four_b_rows"], families),
        "nine_b_direct": summarize_all_families(raw["nine_b_rows"], families),
        "four_b_skill_registry_v5": summarize_all_families(
            raw["candidate_rows"], families
        ),
    }
    gates = {
        "all_three_arms_complete_and_parseable_160": all(
            row["cases"] == 160 and row["parseable"] == 160
            for row in arms.values()
        ),
        "router_correct_160": raw["routing"]["router_correct"] == 160,
        "registry_unique_128": raw["routing"]["registry_unique_matches"] == 128,
        "ab_verified_32": raw["routing"]["ab_verified_executions"] == 32,
        "c_single_skill_verified_128": (
            raw["routing"]["c_skill_executions"] == 128
        ),
        "fallbacks_zero": raw["routing"]["fallbacks"] == 0,
        "candidate_vs_four_significant_zero_loss": (
            candidate_vs_four["paired_bootstrap_95_ci"][0] > 0
            and candidate_vs_four["mcnemar_exact_p"] < config.significance_alpha
            and candidate_vs_four["paired_counts"]["baseline_only"] == 0
        ),
        "candidate_vs_nine_significant_zero_loss": (
            candidate_vs_nine["paired_bootstrap_95_ci"][0] > 0
            and candidate_vs_nine["mcnemar_exact_p"] < config.significance_alpha
            and candidate_vs_nine["paired_counts"]["baseline_only"] == 0
        ),
        "every_family_non_regression": all(
            arms["four_b_skill_registry_v5"]["by_family"][family]["correct"]
            >= arms["four_b_direct"]["by_family"][family]["correct"]
            and arms["four_b_skill_registry_v5"]["by_family"][family]["correct"]
            >= arms["nine_b_direct"]["by_family"][family]["correct"]
            for family in families
        ),
    }
    admitted = all(gates.values())
    return {
        "schema_version": "nano_harness_router_skill_registry_public_v5",
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
            "v4_report_sha256": config.v4_report_sha256,
            "adapter_sha256": config.adapter_tree_sha256,
        },
        "data": {
            "cases": 160,
            "family_counts": {family: 16 for family in families},
            "freshness": prereg["freshness"],
            "benchmark_canary_holdout_rows_or_outputs": 0,
        },
        "arms": arms,
        "routing": raw["routing"],
        "comparisons": {
            "candidate_vs_four_b": candidate_vs_four,
            "candidate_vs_nine_b": candidate_vs_nine,
            "four_b_vs_nine_b": four_vs_nine,
        },
        "decision": {
            "gates": gates,
            "router_skill_registry_v5_admitted": admitted,
            "benchmark_treatment_preregistration_allowed": admitted,
            "benchmark_generation_allowed": False,
            "v1_v2_v3_v4_v5_rerun_allowed": False,
            "training_or_rl_allowed": False,
            "next_action": (
                "Pre-register benchmark-agnostic treatment transfer."
                if admitted
                else "Reject V5; do not rerun or tune V1-V5."
            ),
        },
        "evaluation_boundary": raw["evaluation_boundary"],
        "claim_boundary": (
            "This is fresh synthetic evidence for a target-blind skill "
            "registry and single-schema extraction, not benchmark evidence."
        ),
    }


def render_markdown(report: dict) -> str:
    verdict = (
        "ADMIT"
        if report["decision"]["router_skill_registry_v5_admitted"]
        else "REJECT"
    )
    return f"""# Qwen3.5 Router Skill Registry v5 Result

## Verdict

**{verdict}.**

```json
{json.dumps(report['arms'], indent=2, sort_keys=True)}
```

```json
{json.dumps(report['comparisons'], indent=2, sort_keys=True)}
```

```json
{json.dumps(report['decision']['gates'], indent=2, sort_keys=True)}
```

V1-V5 cannot be rerun. Benchmark generation remains closed until a separate
treatment transfer is pre-registered.
"""


def main() -> None:
    report = build_report()
    PUBLIC_JSON.parent.mkdir(parents=True, exist_ok=True)
    PUBLIC_JSON.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
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
