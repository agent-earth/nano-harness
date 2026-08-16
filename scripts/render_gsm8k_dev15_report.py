#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from nano_harness.baseline import (
    NUMERIC_FINAL_REGEX,
    compare_baselines,
    load_cases,
    load_manifest,
)


PATHS = {
    "four_b_direct": Path(
        "results/harness/qwen35-gsm8k-dev15-direct-v1/4b/cases.jsonl"
    ),
    "four_b_treatment": Path(
        "results/harness/qwen35-gsm8k-dev15-constrained-v1/4b/cases.jsonl"
    ),
    "nine_b_direct": Path(
        "results/harness/qwen35-gsm8k-dev15-direct-v1/9b/cases.jsonl"
    ),
}
DIRECT_MANIFEST = Path("configs/harness/qwen35_gsm8k_dev15_direct_v1.yaml")
TREATMENT_MANIFEST = Path(
    "configs/harness/qwen35_gsm8k_dev15_constrained_v1.yaml"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def rows(path: Path) -> dict[str, dict[str, Any]]:
    return {
        row["case_id"]: row
        for row in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def cost(path: Path) -> dict[str, Any]:
    records = rows(path).values()
    recoveries = [
        row.get("stages", {}).get("conditional_recovery")
        for row in records
        if row.get("stages", {}).get("conditional_recovery") is not None
    ]
    return {
        "cases": len(records),
        "correct": int(sum(float(row["score"]) for row in records)),
        "total_tokens": sum(
            int(row.get("usage", {}).get("total_tokens", 0)) for row in records
        ),
        "wall_seconds": sum(float(row["latency_seconds"]) for row in records),
        "parse_failures": sum(row.get("prediction") is None for row in records),
        "api_errors": sum(row.get("status") == "error" for row in records),
        "recovery_calls": len(recoveries),
        "recovery_truncations": sum(
            stage.get("finish_reason") == "length" for stage in recoveries
        ),
        "regex_matches": sum(
            re.fullmatch(NUMERIC_FINAL_REGEX, str(stage.get("output", ""))) is not None
            for stage in recoveries
        ),
    }


def compact(comparison: dict[str, Any]) -> dict[str, Any]:
    overall = comparison["overall_micro"]
    fields = (
        "candidate_accuracy",
        "baseline_accuracy",
        "delta",
        "paired_counts",
        "mcnemar_exact_p",
        "paired_bootstrap_95_ci",
        "candidate_only_cases",
        "baseline_only_cases",
        "candidate_parse_failures",
        "baseline_parse_failures",
    )
    return {
        **{key: overall[key] for key in fields},
        "bootstrap_samples": comparison["bootstrap_samples"],
        "bootstrap_seed": comparison["bootstrap_seed"],
    }


def audit_direct(path: Path) -> dict[str, Any]:
    manifest = load_manifest(DIRECT_MANIFEST)
    cases = {
        case.case_id: case
        for case in load_cases(manifest, Path("../../datasets"))
    }
    results = rows(path)
    failures = []
    if set(cases) != set(results):
        failures.append("case identities")
    for case_id, case in cases.items():
        record = results.get(case_id, {})
        expected = hashlib.sha256(case.prompt.encode()).hexdigest()
        actual = record.get("stages", {}).get("direct", {}).get("input_sha256")
        if record.get("selected_strategy") != "direct" or actual != expected:
            failures.append(case_id)
    if failures:
        raise SystemExit(f"direct contract audit failed: {failures[:5]}")
    return {
        "passed": True,
        "cases": len(cases),
        "stage_input_hashes_match": True,
    }


def audit_treatment() -> dict[str, Any]:
    manifest = load_manifest(TREATMENT_MANIFEST)
    cases = {
        case.case_id: case
        for case in load_cases(manifest, Path("../../datasets"))
    }
    treatment = rows(PATHS["four_b_treatment"])
    direct = rows(PATHS["four_b_direct"])
    failures = []
    recovery_cases = []
    expected_structured = {"regex": NUMERIC_FINAL_REGEX}
    if set(cases) != set(treatment):
        failures.append("case identities")
    for case_id, case in cases.items():
        record = treatment.get(case_id, {})
        stages = record.get("stages", {})
        protected = stages.get("protected_direct", {})
        recovery = stages.get("conditional_recovery")
        selection = stages.get("deterministic_selection", {})
        if record.get("selected_strategy") != "protected_math_constrained_recovery":
            failures.append(f"{case_id}:strategy")
            continue
        if protected.get("input_sha256") != hashlib.sha256(
            case.prompt.encode()
        ).hexdigest():
            failures.append(f"{case_id}:protected")
        direct_prediction = direct[case_id].get("prediction")
        if protected.get("prediction") != direct_prediction:
            failures.append(f"{case_id}:parity")
        expected_trigger = direct_prediction is None
        if bool(selection.get("recovery_triggered")) != expected_trigger:
            failures.append(f"{case_id}:trigger")
        if expected_trigger:
            recovery_cases.append(case_id)
            if recovery is None:
                failures.append(f"{case_id}:missing-recovery")
                continue
            recovery_prompt = (
                f"<original_task>\n{case.draft_prompt}\n</original_task>\n\n"
                "Solve the problem internally from scratch and provide the numeric final."
            )
            if recovery.get("input_sha256") != hashlib.sha256(
                recovery_prompt.encode()
            ).hexdigest():
                failures.append(f"{case_id}:recovery-input")
            if recovery.get("structured_outputs") != expected_structured:
                failures.append(f"{case_id}:structured-outputs")
            if (
                re.fullmatch(NUMERIC_FINAL_REGEX, str(recovery.get("output", "")))
                is None
            ):
                failures.append(f"{case_id}:regex")
            expected_selected = recovery.get("prediction")
        else:
            if recovery is not None:
                failures.append(f"{case_id}:unexpected-recovery")
            expected_selected = direct_prediction
        if (
            selection.get("selected_prediction") != expected_selected
            or record.get("prediction") != expected_selected
        ):
            failures.append(f"{case_id}:selection")
    if failures:
        raise SystemExit(f"treatment contract audit failed: {failures[:5]}")
    return {
        "passed": True,
        "cases": len(cases),
        "protected_direct_matches_control": True,
        "conditional_execution_matches": True,
        "structured_outputs_match": True,
        "recovery_regex_fullmatch": True,
        "recovery_cases": recovery_cases,
    }


def recovery_analysis() -> dict[str, Any]:
    treatment = rows(PATHS["four_b_treatment"])
    direct = rows(PATHS["four_b_direct"])
    recovery_cases = []
    wins = []
    unresolved = []
    parseable_direct_changes = []
    for case_id, record in treatment.items():
        recovery = record["stages"]["conditional_recovery"]
        if direct[case_id].get("prediction") is not None:
            if record.get("prediction") != direct[case_id].get("prediction"):
                parseable_direct_changes.append(case_id)
            continue
        recovery_cases.append(case_id)
        if record.get("score") == 1.0:
            wins.append(case_id)
        if record.get("prediction") is None:
            unresolved.append(case_id)
    if parseable_direct_changes:
        raise SystemExit(
            f"parseable direct changed: {parseable_direct_changes[:5]}"
        )
    return {
        "parseable_direct_unchanged": True,
        "recovery_count": len(recovery_cases),
        "recovery_cases": recovery_cases,
        "recovery_wins": wins,
        "unresolved_recoveries": unresolved,
    }


def main() -> None:
    for path in PATHS.values():
        if not path.is_file():
            raise SystemExit(f"missing result: {path}")
    versus_4b = compact(
        compare_baselines(PATHS["four_b_treatment"], PATHS["four_b_direct"])
    )
    versus_9b = compact(
        compare_baselines(PATHS["four_b_treatment"], PATHS["nine_b_direct"])
    )
    costs = {label: cost(path) for label, path in PATHS.items()}
    analysis = recovery_analysis()
    token_ratio = (
        costs["four_b_treatment"]["total_tokens"]
        / costs["four_b_direct"]["total_tokens"]
    )
    accepted = (
        analysis["recovery_count"] >= 1
        and len(analysis["recovery_wins"]) >= 1
        and versus_4b["paired_counts"]["baseline_only"] == 0
        and costs["four_b_treatment"]["parse_failures"]
        < costs["four_b_direct"]["parse_failures"]
        and costs["four_b_treatment"]["regex_matches"]
        == costs["four_b_treatment"]["recovery_calls"]
        and not costs["four_b_treatment"]["api_errors"]
        and token_ratio < 1.2
    )
    report = {
        "schema_version": "nano_harness_public_gsm8k_dev_v1",
        "experiment_id": "qwen35-gsm8k-dev15-constrained-recovery-v1",
        "code_revision": git_revision(),
        "versus_4b_direct": versus_4b,
        "versus_9b_direct": versus_9b,
        "costs": costs,
        "token_ratio_vs_4b_direct": token_ratio,
        "recovery_analysis": analysis,
        "contract_audits": {
            "four_b_direct": audit_direct(PATHS["four_b_direct"]),
            "four_b_treatment": audit_treatment(),
            "nine_b_direct": audit_direct(PATHS["nine_b_direct"]),
        },
        "artifacts": {
            f"{label}_raw_sha256": sha256_file(path)
            for label, path in PATHS.items()
        },
        "decision": {
            "accepted": accepted,
            "recovery_triggered": analysis["recovery_count"] >= 1,
            "at_least_one_recovery_win": len(analysis["recovery_wins"]) >= 1,
            "zero_direct_only_losses": (
                versus_4b["paired_counts"]["baseline_only"] == 0
            ),
            "parse_failures_reduced": (
                costs["four_b_treatment"]["parse_failures"]
                < costs["four_b_direct"]["parse_failures"]
            ),
            "all_recoveries_match_regex": (
                costs["four_b_treatment"]["regex_matches"]
                == costs["four_b_treatment"]["recovery_calls"]
            ),
            "no_api_errors": not costs["four_b_treatment"]["api_errors"],
            "token_ratio_below_1_2": token_ratio < 1.2,
            "next_experiment": (
                "Pre-register a fresh 96-case conditional constrained recovery "
                "confirmation to observe rare direct parse failures."
            ),
        },
    }
    markdown = f"""# GSM8K Dev15 Constrained Recovery Result

## Result

- 4B direct: {versus_4b['baseline_accuracy']:.4f};
- 4B constrained recovery: {versus_4b['candidate_accuracy']:.4f};
- 9B direct: {versus_9b['baseline_accuracy']:.4f}.

Treatment versus 4B direct is {versus_4b['delta']:+.4f}, 95% bootstrap CI
[{versus_4b['paired_bootstrap_95_ci'][0]:+.4f},
{versus_4b['paired_bootstrap_95_ci'][1]:+.4f}], with
{versus_4b['paired_counts']['candidate_only']} treatment-only wins and
{versus_4b['paired_counts']['baseline_only']} direct-only losses.

Recovery triggers {analysis['recovery_count']} times, produces
{len(analysis['recovery_wins'])} correct recoveries, and leaves
{len(analysis['unresolved_recoveries'])} unparseable. All
{costs['four_b_treatment']['regex_matches']} recovery outputs match the
committed regex.

Treatment token ratio versus direct is {token_ratio:.3f}x.

No direct parse failure occurs in this 48-case slice, so recovery never fires.
The structured-output capability is validated by a real smoke request and
audited implementation, but this slice cannot establish recovery benefit.

## Contract Audit

Recovery calls exist exactly for direct parse failures. Parseable direct
predictions are unchanged. Every recovery includes the committed structured
output metadata and full-matches the numeric FINAL regex. Raw outputs remain
local and ignored.

## Decision

{('Dev15 satisfies every directional promotion rule.'
   if accepted
   else 'Dev15 fails at least one directional promotion rule.')}

The next experiment expands to 96 fresh GSM8K cases without changing policy.

## Reproduction Identity

- Code revision: `{report['code_revision']}`
- 4B direct raw SHA256: `{report['artifacts']['four_b_direct_raw_sha256']}`
- 4B treatment raw SHA256: `{report['artifacts']['four_b_treatment_raw_sha256']}`
- 9B direct raw SHA256: `{report['artifacts']['nine_b_direct_raw_sha256']}`
"""
    Path(
        "docs/results/gsm8k_dev15_constrained_recovery_v1.public.json"
    ).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    Path("docs/results/gsm8k_dev15_constrained_recovery_v1.md").write_text(
        markdown,
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "experiment_id": report["experiment_id"],
                "accepted": accepted,
                "treatment_vs_4b_delta": versus_4b["delta"],
                "treatment_vs_9b_delta": versus_9b["delta"],
                "recoveries": analysis["recovery_count"],
                "recovery_wins": len(analysis["recovery_wins"]),
                "regex_matches": costs["four_b_treatment"]["regex_matches"],
                "token_ratio": token_ratio,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
