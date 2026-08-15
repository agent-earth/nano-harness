#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nano_harness.baseline import (
    compare_baselines,
    extract_prediction,
    load_cases,
    load_manifest,
)


PATHS = {
    "four_b_direct": Path(
        "results/harness/qwen35-gsm8k-dev9-direct-v1/4b/cases.jsonl"
    ),
    "four_b_treatment": Path(
        "results/harness/qwen35-gsm8k-dev9-protected-resolve-v1/4b/cases.jsonl"
    ),
    "nine_b_direct": Path(
        "results/harness/qwen35-gsm8k-dev9-direct-v1/9b/cases.jsonl"
    ),
}
DIRECT_MANIFEST = Path("configs/harness/qwen35_gsm8k_dev9_direct_v1.yaml")
TREATMENT_MANIFEST = Path(
    "configs/harness/qwen35_gsm8k_dev9_protected_resolve_v1.yaml"
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
    return {
        "cases": len(records),
        "correct": int(sum(float(row["score"]) for row in records)),
        "total_tokens": sum(
            int(row.get("usage", {}).get("total_tokens", 0)) for row in records
        ),
        "wall_seconds": sum(float(row["latency_seconds"]) for row in records),
        "parse_failures": sum(row.get("prediction") is None for row in records),
        "api_errors": sum(row.get("status") == "error" for row in records),
        "direct_truncations": sum(
            row.get("stages", {}).get("protected_direct", {}).get("finish_reason")
            == "length"
            for row in records
        ),
        "resolve_truncations": sum(
            row.get("stages", {})
            .get("independent_resolve", {})
            .get("finish_reason")
            == "length"
            for row in records
        ),
        "arbiter_truncations": sum(
            row.get("stages", {}).get("arbiter", {}).get("finish_reason")
            == "length"
            for row in records
        ),
    }


def compact(comparison: dict[str, Any]) -> dict[str, Any]:
    overall = comparison["overall_micro"]
    return {
        "candidate_accuracy": overall["candidate_accuracy"],
        "baseline_accuracy": overall["baseline_accuracy"],
        "delta": overall["delta"],
        "paired_counts": overall["paired_counts"],
        "mcnemar_exact_p": overall["mcnemar_exact_p"],
        "paired_bootstrap_95_ci": overall["paired_bootstrap_95_ci"],
        "candidate_only_cases": overall["candidate_only_cases"],
        "baseline_only_cases": overall["baseline_only_cases"],
        "candidate_parse_failures": overall["candidate_parse_failures"],
        "baseline_parse_failures": overall["baseline_parse_failures"],
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
        "selected_strategy": "direct",
        "stage_input_hashes_match": True,
    }


def audit_treatment() -> dict[str, Any]:
    manifest = load_manifest(TREATMENT_MANIFEST)
    cases = {
        case.case_id: case
        for case in load_cases(manifest, Path("../../datasets"))
    }
    results = rows(PATHS["four_b_treatment"])
    failures = []
    if set(cases) != set(results):
        failures.append("case identities")
    for case_id, case in cases.items():
        record = results.get(case_id, {})
        stages = record.get("stages", {})
        protected = stages.get("protected_direct", {})
        resolve = stages.get("independent_resolve", {})
        arbiter = stages.get("arbiter", {})
        if (
            record.get("selected_strategy") != "protected_math_arbiter"
            or not protected
            or not resolve
            or not arbiter.get("raw_output_sha256")
        ):
            failures.append(f"{case_id}:stages")
            continue
        if protected.get("input_sha256") != hashlib.sha256(
            case.prompt.encode()
        ).hexdigest():
            failures.append(f"{case_id}:protected")
        resolve_prompt = (
            f"<original_task>\n{case.draft_prompt}\n</original_task>\n\n"
            "Independently solve this math problem from scratch. Check units, "
            "rates, time periods, totals, and exactly what quantity is requested. "
            "Produce compact calculations and end with FINAL: <number>. Do not "
            "use tools."
        )
        if resolve.get("input_sha256") != hashlib.sha256(
            resolve_prompt.encode()
        ).hexdigest():
            failures.append(f"{case_id}:resolve")
        arbiter_prompt = (
            f"<original_task>\n{case.prompt}\n</original_task>\n\n"
            f"<protected_direct_answer>{protected.get('prediction')}"
            "</protected_direct_answer>\n"
            f"<independent_resolve>\n{resolve.get('output', '')}"
            "\n</independent_resolve>"
        )
        if arbiter.get("input_sha256") != hashlib.sha256(
            arbiter_prompt.encode()
        ).hexdigest():
            failures.append(f"{case_id}:arbiter")
    if failures:
        raise SystemExit(f"treatment contract audit failed: {failures[:5]}")
    return {
        "passed": True,
        "cases": len(cases),
        "selected_strategy": "protected_math_arbiter",
        "protected_direct_per_case": 1,
        "independent_resolve_per_case": 1,
        "stage_input_hashes_match": True,
        "raw_arbiter_hashes_present": True,
    }


def override_analysis() -> dict[str, Any]:
    treatment = rows(PATHS["four_b_treatment"])
    direct = rows(PATHS["four_b_direct"])
    protected_mismatches = []
    overrides = []
    wins = []
    losses = []
    neutral = []
    resolve_disagreements = []
    for case_id, record in treatment.items():
        stages = record["stages"]
        protected_prediction = stages["protected_direct"]["prediction"]
        resolve_prediction = stages["independent_resolve"]["prediction"]
        if protected_prediction != direct[case_id].get("prediction"):
            protected_mismatches.append(case_id)
        if resolve_prediction != protected_prediction:
            resolve_disagreements.append(case_id)
        if record.get("prediction") == protected_prediction:
            continue
        overrides.append(case_id)
        protected_score = float(protected_prediction == record["expected"])
        final_score = float(record["score"])
        if final_score > protected_score:
            wins.append(case_id)
        elif final_score < protected_score:
            losses.append(case_id)
        else:
            neutral.append(case_id)
    if protected_mismatches:
        raise SystemExit(
            f"protected direct differs from control: {protected_mismatches[:5]}"
        )
    return {
        "protected_direct_matches_control": True,
        "resolve_disagreement_count": len(resolve_disagreements),
        "resolve_disagreement_cases": resolve_disagreements,
        "override_count": len(overrides),
        "override_cases": overrides,
        "override_wins": wins,
        "override_losses": losses,
        "neutral_overrides": neutral,
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
    treatment_cost = cost(PATHS["four_b_treatment"])
    overrides = override_analysis()
    treatment_rows = rows(PATHS["four_b_treatment"])
    parse_failure_analysis = {
        "cases": [],
        "protected_correct": 0,
        "resolve_correct": 0,
        "both_wrong": 0,
    }
    for record in treatment_rows.values():
        if record.get("prediction") is not None:
            continue
        protected = record["stages"]["protected_direct"]["prediction"]
        resolve = record["stages"]["independent_resolve"]["prediction"]
        protected_correct = protected == record["expected"]
        resolve_correct = resolve == record["expected"]
        parse_failure_analysis["protected_correct"] += protected_correct
        parse_failure_analysis["resolve_correct"] += resolve_correct
        parse_failure_analysis["both_wrong"] += (
            not protected_correct and not resolve_correct
        )
        parse_failure_analysis["cases"].append(
            {
                "case_id": record["case_id"],
                "protected_correct": protected_correct,
                "resolve_correct": resolve_correct,
                "arbiter_finish_reason": record["stages"]["arbiter"][
                    "finish_reason"
                ],
            }
        )
    accepted = (
        versus_4b["candidate_accuracy"] > versus_4b["baseline_accuracy"]
        and versus_4b["paired_counts"]["candidate_only"] >= 1
        and versus_4b["paired_counts"]["baseline_only"] == 0
        and not treatment_cost["api_errors"]
        and not treatment_cost["parse_failures"]
    )
    report = {
        "schema_version": "nano_harness_public_gsm8k_dev_v1",
        "experiment_id": "qwen35-gsm8k-dev9-protected-resolve-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "code_revision": git_revision(),
        "versus_4b_direct": versus_4b,
        "versus_9b_direct": versus_9b,
        "costs": {label: cost(path) for label, path in PATHS.items()},
        "override_analysis": overrides,
        "parse_failure_analysis": parse_failure_analysis,
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
            "point_above_4b_direct": (
                versus_4b["candidate_accuracy"] > versus_4b["baseline_accuracy"]
            ),
            "at_least_one_treatment_only_win": (
                versus_4b["paired_counts"]["candidate_only"] >= 1
            ),
            "zero_direct_only_losses": (
                versus_4b["paired_counts"]["baseline_only"] == 0
            ),
            "no_api_errors": not treatment_cost["api_errors"],
            "no_parse_failures": not treatment_cost["parse_failures"],
            "next_experiment": (
                "Fresh dev10 with unchanged prompts and budgets; when the arbiter "
                "is unparseable, deterministically preserve the protected direct "
                "answer without rescoring dev9."
            ),
        },
    }
    markdown = f"""# GSM8K Dev9 Protected Re-solve Result

## Result

- 4B direct: {versus_4b['baseline_accuracy']:.4f};
- 4B protected re-solve: {versus_4b['candidate_accuracy']:.4f};
- 9B direct: {versus_9b['baseline_accuracy']:.4f}.

Treatment versus 4B direct is {versus_4b['delta']:+.4f}, 95% bootstrap CI
[{versus_4b['paired_bootstrap_95_ci'][0]:+.4f},
{versus_4b['paired_bootstrap_95_ci'][1]:+.4f}], with
{versus_4b['paired_counts']['candidate_only']} treatment-only wins and
{versus_4b['paired_counts']['baseline_only']} direct-only losses.

The independent re-solve disagrees with direct on
{overrides['resolve_disagreement_count']} cases. The arbiter overrides on
{overrides['override_count']} cases:
{len(overrides['override_wins'])} wins,
{len(overrides['override_losses'])} losses, and
{len(overrides['neutral_overrides'])} neutral.

Treatment uses {treatment_cost['total_tokens']} tokens and
{treatment_cost['wall_seconds']:.1f}s.

All {treatment_cost['parse_failures']} final parse failures are 64-token
arbiter length truncations. Their protected direct answers are correct on
{parse_failure_analysis['protected_correct']} cases; the independent re-solve
is correct on {parse_failure_analysis['resolve_correct']} cases; both are wrong
on {parse_failure_analysis['both_wrong']} case.

## Contract Audit

Protected-direct predictions match the independent 4B direct arm for all
cases. Re-solve independence, stage inputs, and raw arbiter hashes match the
committed protocol. Raw outputs remain local and ignored.

## Decision

{('Dev9 satisfies every directional promotion rule.'
   if accepted
   else 'Dev9 fails at least one directional promotion rule.')}

No dev9 output is rescored. The next fresh experiment keeps all prompts and
budgets unchanged and deterministically falls back to the protected direct
answer only when the arbiter final is unparseable.

## Reproduction Identity

- Code revision: `{report['code_revision']}`
- 4B direct raw SHA256: `{report['artifacts']['four_b_direct_raw_sha256']}`
- 4B treatment raw SHA256: `{report['artifacts']['four_b_treatment_raw_sha256']}`
- 9B direct raw SHA256: `{report['artifacts']['nine_b_direct_raw_sha256']}`
"""
    Path("docs/results/gsm8k_dev9_protected_resolve_v1.public.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    Path("docs/results/gsm8k_dev9_protected_resolve_v1.md").write_text(
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
                "resolve_disagreements": overrides["resolve_disagreement_count"],
                "overrides": overrides["override_count"],
                "override_wins": len(overrides["override_wins"]),
                "override_losses": len(overrides["override_losses"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
