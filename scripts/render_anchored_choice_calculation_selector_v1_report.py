#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/harness/anchored_v1_choice_calculation_selector_v1.json"
RESULT = (
    ROOT
    / "results/harness/anchored-v1-choice-calculation-selector-v1/local/result.json"
)
PRE_REGISTRATION_REVISION = "f747afe"
CONFIG_SHA256 = (
    "8727fb9f3d97c860f551fac2046822fe"
    "ec64898e9fe5005271f64df8e247674a"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def family_score(metrics: dict, family: str) -> int:
    return int(metrics["by_family"][family]["semantic_exact"])


def usage_total(rows: list[dict]) -> dict[str, float]:
    keys = {
        key
        for row in rows
        for key, value in row.get("usage", {}).items()
        if isinstance(value, (int, float))
    }
    return {
        key: sum(float(row.get("usage", {}).get(key, 0)) for row in rows)
        for key in sorted(keys)
    }


def main() -> None:
    if sha256_file(CONFIG) != CONFIG_SHA256:
        raise SystemExit("contract harness config differs from pre-registration")
    raw = json.loads(RESULT.read_text(encoding="utf-8"))
    if raw.get("schema_version") != "nano_harness_analog_contract_result_v1":
        raise SystemExit("unexpected contract harness result schema")
    if raw.get("experiment_id") != (
        "anchored-v1-choice-calculation-selector-v1"
    ):
        raise SystemExit("unexpected contract harness experiment")
    boundary = raw["evaluation_boundary"]
    if any(
        boundary.get(field) is not False
        for field in (
            "benchmark_rows_loaded",
            "sealed_canary_run",
            "prior_full_suite_run",
            "independent_holdout_run",
        )
    ):
        raise SystemExit("evaluation boundary differs")

    baseline = raw["baseline"]
    candidate = raw["candidate"]
    baseline_rows = {row["sample_id"]: row for row in raw["baseline_rows"]}
    candidate_rows = {row["sample_id"]: row for row in raw["candidate_rows"]}
    if set(baseline_rows) != set(candidate_rows) or len(baseline_rows) != 32:
        raise SystemExit("baseline and candidate rows are not aligned")

    choice_family = "capability_preservation_choice"
    numeric_family = "capability_preservation_numeric"
    process_family = "semantic_arithmetic_process"
    choice_ids = [
        sample_id
        for sample_id, row in candidate_rows.items()
        if row["route"] == "choice_calculation_selector"
    ]
    non_choice_ids = [
        sample_id
        for sample_id, row in candidate_rows.items()
        if row["route"] == "reuse_direct_baseline"
    ]
    if len(choice_ids) != 8 or len(non_choice_ids) != 24:
        raise SystemExit("route counts differ from pre-registration")

    selector_regex = re.compile(r"FINAL: [A-D]")
    selector_contract = all(
        selector_regex.fullmatch(str(candidate_rows[sample_id]["output"]))
        and candidate_rows[sample_id]["stages"]["selector"][
            "structured_outputs"
        ]["regex"]
        == r"FINAL: [A-D]"
        for sample_id in choice_ids
    )
    non_choice_equal = all(
        baseline_rows[sample_id]["output"]
        == candidate_rows[sample_id]["output"]
        for sample_id in non_choice_ids
    )
    changed_choice_ids = [
        sample_id
        for sample_id in choice_ids
        if baseline_rows[sample_id]["output"]
        != candidate_rows[sample_id]["output"]
    ]
    fixed_choice_ids = [
        sample_id
        for sample_id in choice_ids
        if not baseline_rows[sample_id]["exact"]
        and candidate_rows[sample_id]["exact"]
    ]
    regressed_choice_ids = [
        sample_id
        for sample_id in choice_ids
        if baseline_rows[sample_id]["exact"]
        and not candidate_rows[sample_id]["exact"]
    ]
    calculation_final_only_ids = [
        sample_id
        for sample_id in choice_ids
        if selector_regex.fullmatch(
            str(
                candidate_rows[sample_id]["stages"]["calculation"]["output"]
            )
        )
    ]

    checks = {
        "baseline_strict_22": baseline["exact"] == 22,
        "baseline_semantic_25": baseline["semantic_exact"] == 25,
        "baseline_numeric_11": family_score(baseline, numeric_family) == 11,
        "baseline_choice_6": family_score(baseline, choice_family) == 6,
        "baseline_process_8": family_score(baseline, process_family) == 8,
        "candidate_strict_at_least_22": candidate["exact"] >= 22,
        "candidate_semantic_at_least_25": candidate["semantic_exact"] >= 25,
        "candidate_numeric_at_least_11": (
            family_score(candidate, numeric_family) >= 11
        ),
        "candidate_choice_at_least_7": (
            family_score(candidate, choice_family) >= 7
        ),
        "candidate_process_equals_8": (
            family_score(candidate, process_family) == 8
        ),
        "non_choice_outputs_equal_24_of_24": non_choice_equal,
        "selector_contract_8_of_8": selector_contract,
    }
    if all(checks.values()):
        raise SystemExit("contract harness unexpectedly passes its frozen gate")
    if [key for key, value in checks.items() if not value] != [
        "candidate_choice_at_least_7"
    ]:
        raise SystemExit("contract harness failed outside the choice gate")
    if (
        len(changed_choice_ids) != 1
        or fixed_choice_ids
        or regressed_choice_ids
        or len(calculation_final_only_ids) != 8
    ):
        raise SystemExit("choice mechanism evidence differs")

    baseline_usage = usage_total(raw["baseline_rows"])
    candidate_usage = usage_total(raw["candidate_rows"])
    report = {
        "schema_version": (
            "nano_harness_public_anchored_choice_calculation_selector_v1"
        ),
        "experiment_id": raw["experiment_id"],
        "pre_registration_revision": PRE_REGISTRATION_REVISION,
        "passed": False,
        "identity": {
            "config_sha256": CONFIG_SHA256,
            "dataset_sha256": raw["identity"]["dataset_sha256"],
            "serving_receipt_sha256": raw["identity"][
                "serving_receipt_sha256"
            ],
            "serving_adapter_weights_sha256": raw["identity"][
                "serving_adapter_weights_sha256"
            ],
            "raw_result_sha256": sha256_file(RESULT),
        },
        "method": {
            "choice_treated_rows": raw["choice_treated_rows"],
            "non_choice_reused_rows": raw["non_choice_reused_rows"],
            "calculation_max_tokens": raw["config"][
                "calculation_max_tokens"
            ],
            "selector_max_tokens": raw["config"]["selector_max_tokens"],
            "selector_regex": raw["config"]["selector_regex"],
            "temperature": raw["config"]["temperature"],
            "thinking_enabled": False,
        },
        "baseline": baseline,
        "candidate": candidate,
        "local_gate": checks,
        "mechanism": {
            "choice_outputs_changed": len(changed_choice_ids),
            "changed_choice_sample_ids": sorted(changed_choice_ids),
            "choice_fixes": len(fixed_choice_ids),
            "choice_regressions": len(regressed_choice_ids),
            "calculation_final_only_rows": len(calculation_final_only_ids),
            "calculation_rows_with_explicit_work": (
                len(choice_ids) - len(calculation_final_only_ids)
            ),
            "finding": (
                "calculation_stage_obeyed_original_final_only_contract_"
                "instead_of_emitting_explicit_work"
            ),
        },
        "cost": {
            "baseline_usage": baseline_usage,
            "candidate_usage": candidate_usage,
            "candidate_to_baseline_total_token_ratio": (
                candidate_usage["total_tokens"] / baseline_usage["total_tokens"]
            ),
            "wall_seconds": raw["wall_seconds"],
        },
        "evaluation_boundary": {
            **boundary,
            "independent_holdout_prompts_loaded": False,
            "independent_holdout_references_loaded": False,
            "independent_quality_claim_allowed": False,
        },
        "decision": {
            "accepted_local_harness": False,
            "sealed_canary_allowed": False,
            "prior_full_suite_allowed": False,
            "independent_holdout_allowed": False,
            "merge_allowed": False,
            "scale_allowed": False,
            "rl_allowed": False,
            "post_hoc_prompt_or_budget_search_allowed": False,
            "next_action": (
                "Reject the exact calculation-selector prompt protocol and "
                "preserve anchored-v1. Replan from the observed instruction-"
                "hierarchy failure without changing this protocol post hoc."
            ),
        },
    }

    markdown = f"""# Anchored-v1 Choice Calculation Selector v1 Result

## Result

The harness is contract-safe but fails its frozen local improvement gate.

- baseline and candidate strict / semantic: {baseline['exact']}/32 /
  {baseline['semantic_exact']}/32 and {candidate['exact']}/32 /
  {candidate['semantic_exact']}/32;
- candidate numeric / choice / process semantic:
  {family_score(candidate, numeric_family)}/16,
  {family_score(candidate, choice_family)}/8,
  {family_score(candidate, process_family)}/8;
- selector regex compliance: 8/8;
- non-choice direct-output reuse: 24/24;
- choice fixes / regressions: 0 / 0;
- wall time: {raw['wall_seconds']:.1f} seconds.

The only failed gate is choice >=7/8: choice remains 6/8.

## Mechanism Evidence

All eight calculation stages return only a `FINAL: <letter>` line instead of
the requested explicit arithmetic. The original answer-only instruction
dominates the calculation-stage system prompt. The selector therefore receives
no independent calculation to verify. One wrong choice moves from one wrong
option to another; no case is fixed or regressed.

This is evidence against the exact prompt protocol. Do not alter its prompt,
budget, regex, or retry policy after observing this development result.

## Decision

Reject this local harness and preserve anchored-v1. The sealed canary, old
full-development suite, and independent holdout were not run. The holdout
remains unread.

## Reproduction Identity

- pre-registration revision: `{PRE_REGISTRATION_REVISION}`;
- config SHA256: `{CONFIG_SHA256}`;
- dataset SHA256: `{raw['identity']['dataset_sha256']}`;
- serving receipt SHA256: `{raw['identity']['serving_receipt_sha256']}`;
- serving adapter weights SHA256:
  `{raw['identity']['serving_adapter_weights_sha256']}`;
- raw local result SHA256: `{sha256_file(RESULT)}`.
"""
    output = ROOT / "docs/results"
    output.mkdir(parents=True, exist_ok=True)
    (
        output
        / "anchored_v1_choice_calculation_selector_v1.public.json"
    ).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (
        output / "anchored_v1_choice_calculation_selector_v1.md"
    ).write_text(markdown, encoding="utf-8")
    print(
        json.dumps(
            {
                "passed": False,
                "failed_gate": "candidate_choice_at_least_7",
                "choice": family_score(candidate, choice_family),
                "sealed_canary_allowed": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
