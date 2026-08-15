#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nano_harness.baseline import compare_baselines, extract_prediction, load_cases, load_manifest


PATHS = {
    "four_b_direct": Path("results/harness/qwen35-gsm8k-dev10-direct-v1/4b/cases.jsonl"),
    "four_b_treatment": Path("results/harness/qwen35-gsm8k-dev10-fallback-v1/4b/cases.jsonl"),
    "nine_b_direct": Path("results/harness/qwen35-gsm8k-dev10-direct-v1/9b/cases.jsonl"),
}
DIRECT_MANIFEST = Path("configs/harness/qwen35_gsm8k_dev10_direct_v1.yaml")
TREATMENT_MANIFEST = Path("configs/harness/qwen35_gsm8k_dev10_fallback_v1.yaml")


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
    arbiters = [row.get("stages", {}).get("arbiter", {}) for row in records]
    return {
        "cases": len(records),
        "correct": int(sum(float(row["score"]) for row in records)),
        "total_tokens": sum(int(row.get("usage", {}).get("total_tokens", 0)) for row in records),
        "wall_seconds": sum(float(row["latency_seconds"]) for row in records),
        "parse_failures": sum(row.get("prediction") is None for row in records),
        "api_errors": sum(row.get("status") == "error" for row in records),
        "direct_truncations": sum(
            row.get("stages", {}).get("protected_direct", {}).get("finish_reason") == "length"
            for row in records
        ),
        "resolve_truncations": sum(
            row.get("stages", {}).get("independent_resolve", {}).get("finish_reason") == "length"
            for row in records
        ),
        "arbiter_truncations": sum(stage.get("finish_reason") == "length" for stage in arbiters),
        "fallback_count": sum(bool(stage.get("fallback_to_protected_applied")) for stage in arbiters),
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
    cases = {case.case_id: case for case in load_cases(manifest, Path("../../datasets"))}
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
    return {"passed": True, "cases": len(cases), "stage_input_hashes_match": True}


def audit_treatment() -> dict[str, Any]:
    manifest = load_manifest(TREATMENT_MANIFEST)
    if not manifest.fallback_to_protected_on_parse_failure:
        raise SystemExit("fallback is not enabled")
    cases = {case.case_id: case for case in load_cases(manifest, Path("../../datasets"))}
    results = rows(PATHS["four_b_treatment"])
    failures = []
    fallback_cases = []
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
            or not arbiter.get("raw_output_sha256")
            or "raw_output" not in arbiter
        ):
            failures.append(f"{case_id}:stages")
            continue
        if protected.get("input_sha256") != hashlib.sha256(case.prompt.encode()).hexdigest():
            failures.append(f"{case_id}:protected")
        resolve_prompt = (
            f"<original_task>\n{case.draft_prompt}\n</original_task>\n\n"
            "Independently solve this math problem from scratch. Check units, "
            "rates, time periods, totals, and exactly what quantity is requested. "
            "Produce compact calculations and end with FINAL: <number>. Do not use tools."
        )
        if resolve.get("input_sha256") != hashlib.sha256(resolve_prompt.encode()).hexdigest():
            failures.append(f"{case_id}:resolve")
        arbiter_prompt = (
            f"<original_task>\n{case.prompt}\n</original_task>\n\n"
            f"<protected_direct_answer>{protected.get('prediction')}</protected_direct_answer>\n"
            f"<independent_resolve>\n{resolve.get('output', '')}\n</independent_resolve>"
        )
        if arbiter.get("input_sha256") != hashlib.sha256(arbiter_prompt.encode()).hexdigest():
            failures.append(f"{case_id}:arbiter")
        raw_parseable = extract_prediction(arbiter["raw_output"], case.scorer) is not None
        protected_parseable = protected.get("prediction") is not None
        expected_fallback = not raw_parseable and protected_parseable
        if bool(arbiter.get("fallback_to_protected_applied")) != expected_fallback:
            failures.append(f"{case_id}:fallback")
        if expected_fallback:
            fallback_cases.append(case_id)
            if record.get("prediction") != protected.get("prediction"):
                failures.append(f"{case_id}:fallback-prediction")
    if failures:
        raise SystemExit(f"treatment contract audit failed: {failures[:5]}")
    return {
        "passed": True,
        "cases": len(cases),
        "stage_input_hashes_match": True,
        "raw_arbiter_hashes_present": True,
        "fallback_semantics_match": True,
        "fallback_cases": fallback_cases,
    }


def override_analysis() -> dict[str, Any]:
    treatment = rows(PATHS["four_b_treatment"])
    direct = rows(PATHS["four_b_direct"])
    mismatches = []
    resolve_disagreements = []
    overrides = []
    wins = []
    losses = []
    neutral = []
    fallback_cases = []
    for case_id, record in treatment.items():
        stages = record["stages"]
        protected = stages["protected_direct"]["prediction"]
        resolve = stages["independent_resolve"]["prediction"]
        if protected != direct[case_id].get("prediction"):
            mismatches.append(case_id)
        if resolve != protected:
            resolve_disagreements.append(case_id)
        if stages["arbiter"].get("fallback_to_protected_applied"):
            fallback_cases.append(case_id)
        if record.get("prediction") == protected:
            continue
        overrides.append(case_id)
        protected_score = float(protected == record["expected"])
        final_score = float(record["score"])
        if final_score > protected_score:
            wins.append(case_id)
        elif final_score < protected_score:
            losses.append(case_id)
        else:
            neutral.append(case_id)
    if mismatches:
        raise SystemExit(f"protected direct differs from control: {mismatches[:5]}")
    return {
        "protected_direct_matches_control": True,
        "resolve_disagreement_count": len(resolve_disagreements),
        "resolve_disagreement_cases": resolve_disagreements,
        "fallback_count": len(fallback_cases),
        "fallback_cases": fallback_cases,
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
    versus_4b = compact(compare_baselines(PATHS["four_b_treatment"], PATHS["four_b_direct"]))
    versus_9b = compact(compare_baselines(PATHS["four_b_treatment"], PATHS["nine_b_direct"]))
    treatment_cost = cost(PATHS["four_b_treatment"])
    overrides = override_analysis()
    treatment_rows = rows(PATHS["four_b_treatment"])
    fallback_analysis = {
        "protected_correct": 0,
        "resolve_correct": 0,
        "protected_wrong_resolve_correct": [],
        "protected_and_resolve_wrong": [],
    }
    for case_id in overrides["fallback_cases"]:
        record = treatment_rows[case_id]
        protected = record["stages"]["protected_direct"]["prediction"]
        resolve = record["stages"]["independent_resolve"]["prediction"]
        protected_correct = protected == record["expected"]
        resolve_correct = resolve == record["expected"]
        fallback_analysis["protected_correct"] += protected_correct
        fallback_analysis["resolve_correct"] += resolve_correct
        if not protected_correct and resolve_correct:
            fallback_analysis["protected_wrong_resolve_correct"].append(case_id)
        if not protected_correct and not resolve_correct:
            fallback_analysis["protected_and_resolve_wrong"].append(case_id)
    accepted = (
        versus_4b["candidate_accuracy"] > versus_4b["baseline_accuracy"]
        and versus_4b["paired_counts"]["candidate_only"] >= 1
        and versus_4b["paired_counts"]["baseline_only"] == 0
        and not treatment_cost["api_errors"]
        and not treatment_cost["parse_failures"]
    )
    report = {
        "schema_version": "nano_harness_public_gsm8k_dev_v1",
        "experiment_id": "qwen35-gsm8k-dev10-fallback-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "code_revision": git_revision(),
        "versus_4b_direct": versus_4b,
        "versus_9b_direct": versus_9b,
        "costs": {label: cost(path) for label, path in PATHS.items()},
        "override_analysis": overrides,
        "fallback_analysis": fallback_analysis,
        "contract_audits": {
            "four_b_direct": audit_direct(PATHS["four_b_direct"]),
            "four_b_treatment": audit_treatment(),
            "nine_b_direct": audit_direct(PATHS["nine_b_direct"]),
        },
        "artifacts": {f"{label}_raw_sha256": sha256_file(path) for label, path in PATHS.items()},
        "decision": {
            "accepted": accepted,
            "point_above_4b_direct": versus_4b["candidate_accuracy"] > versus_4b["baseline_accuracy"],
            "at_least_one_treatment_only_win": versus_4b["paired_counts"]["candidate_only"] >= 1,
            "zero_direct_only_losses": versus_4b["paired_counts"]["baseline_only"] == 0,
            "no_api_errors": not treatment_cost["api_errors"],
            "no_parse_failures": not treatment_cost["parse_failures"],
            "next_experiment": (
                "Fresh dev11 separates arbitration from formatting: an 8-token "
                "gate emits KEEP or USE_RESOLVE, and code deterministically emits "
                "the chosen FINAL number."
            ),
        },
    }
    markdown = f"""# GSM8K Dev10 Protected Fallback Result

## Result

- 4B direct: {versus_4b['baseline_accuracy']:.4f};
- 4B protected fallback: {versus_4b['candidate_accuracy']:.4f};
- 9B direct: {versus_9b['baseline_accuracy']:.4f}.

Treatment versus 4B direct is {versus_4b['delta']:+.4f}, 95% bootstrap CI
[{versus_4b['paired_bootstrap_95_ci'][0]:+.4f},
{versus_4b['paired_bootstrap_95_ci'][1]:+.4f}], with
{versus_4b['paired_counts']['candidate_only']} treatment-only wins and
{versus_4b['paired_counts']['baseline_only']} direct-only losses.

Fallback fires {overrides['fallback_count']} times. The independent re-solve
disagrees with direct on {overrides['resolve_disagreement_count']} cases; the
arbiter makes {overrides['override_count']} parseable overrides:
{len(overrides['override_wins'])} wins, {len(overrides['override_losses'])}
losses, and {len(overrides['neutral_overrides'])} neutral.

Within the fallback cases, protected direct is correct on
{fallback_analysis['protected_correct']} and the re-solve is correct on
{fallback_analysis['resolve_correct']}. There are
{len(fallback_analysis['protected_wrong_resolve_correct'])} cases where
protected is wrong but re-solve is correct, so unconditional protected
fallback discards usable repair evidence.

Treatment uses {treatment_cost['total_tokens']} tokens and
{treatment_cost['wall_seconds']:.1f}s.

## Contract Audit

Protected direct matches the independent 4B direct arm. Every fallback occurs
exactly when raw arbiter output is unparseable and protected direct is
parseable; final fallback predictions equal protected direct. Raw outputs
remain local and ignored.

## Decision

{('Dev10 satisfies every directional promotion rule.'
   if accepted
   else 'Dev10 fails at least one directional promotion rule.')}

The next fresh experiment separates decision from formatting: a short gate
must output only `KEEP` or `USE_RESOLVE`, then deterministic code emits the
selected numeric `FINAL:` line. No dev10 output is rescored.

## Reproduction Identity

- Code revision: `{report['code_revision']}`
- 4B direct raw SHA256: `{report['artifacts']['four_b_direct_raw_sha256']}`
- 4B treatment raw SHA256: `{report['artifacts']['four_b_treatment_raw_sha256']}`
- 9B direct raw SHA256: `{report['artifacts']['nine_b_direct_raw_sha256']}`
"""
    Path("docs/results/gsm8k_dev10_fallback_v1.public.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    Path("docs/results/gsm8k_dev10_fallback_v1.md").write_text(markdown, encoding="utf-8")
    print(
        json.dumps(
            {
                "experiment_id": report["experiment_id"],
                "accepted": accepted,
                "treatment_vs_4b_delta": versus_4b["delta"],
                "treatment_vs_9b_delta": versus_9b["delta"],
                "fallbacks": overrides["fallback_count"],
                "overrides": overrides["override_count"],
                "override_wins": len(overrides["override_wins"]),
                "override_losses": len(overrides["override_losses"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
