#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nano_harness.baseline import compare_baselines, load_cases, load_manifest


PATHS = {
    "four_b_direct": Path("results/harness/qwen35-gsm8k-dev11-direct-v1/4b/cases.jsonl"),
    "four_b_treatment": Path("results/harness/qwen35-gsm8k-dev11-gate-v1/4b/cases.jsonl"),
    "nine_b_direct": Path("results/harness/qwen35-gsm8k-dev11-direct-v1/9b/cases.jsonl"),
}
DIRECT_MANIFEST = Path("configs/harness/qwen35_gsm8k_dev11_direct_v1.yaml")
TREATMENT_MANIFEST = Path("configs/harness/qwen35_gsm8k_dev11_gate_v1.yaml")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
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
    gates = [row.get("stages", {}).get("decision_gate", {}) for row in records]
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
        "gate_truncations": sum(stage.get("finish_reason") == "length" for stage in gates),
        "use_resolve_count": sum(stage.get("decision") == "USE_RESOLVE" for stage in gates),
        "keep_count": sum(stage.get("decision") == "KEEP" for stage in gates),
        "invalid_raw_gate_outputs": sum(
            stage.get("raw_output") not in {"KEEP", "USE_RESOLVE"} for stage in gates
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
    cases = {case.case_id: case for case in load_cases(manifest, Path("../../datasets"))}
    results = rows(PATHS["four_b_treatment"])
    failures = []
    invalid_outputs = []
    if set(cases) != set(results):
        failures.append("case identities")
    for case_id, case in cases.items():
        record = results.get(case_id, {})
        stages = record.get("stages", {})
        protected = stages.get("protected_direct", {})
        resolve = stages.get("independent_resolve", {})
        gate = stages.get("decision_gate", {})
        if (
            record.get("selected_strategy") != "protected_math_gate"
            or not gate.get("raw_output_sha256")
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
        gate_prompt = (
            f"<original_task>\n{case.prompt}\n</original_task>\n\n"
            f"<protected_direct_answer>{protected.get('prediction')}</protected_direct_answer>\n"
            f"<independent_resolve>\n{resolve.get('output', '')}\n</independent_resolve>"
        )
        if gate.get("input_sha256") != hashlib.sha256(gate_prompt.encode()).hexdigest():
            failures.append(f"{case_id}:gate")
        raw = gate.get("raw_output")
        expected_decision = (
            "USE_RESOLVE"
            if raw == "USE_RESOLVE" and resolve.get("prediction") is not None
            else "KEEP"
        )
        if gate.get("decision") != expected_decision:
            failures.append(f"{case_id}:decision")
        selected = (
            resolve.get("prediction")
            if expected_decision == "USE_RESOLVE"
            else protected.get("prediction")
        )
        if gate.get("selected_prediction") != selected or record.get("prediction") != selected:
            failures.append(f"{case_id}:selection")
        if raw not in {"KEEP", "USE_RESOLVE"}:
            invalid_outputs.append(case_id)
    if failures:
        raise SystemExit(f"treatment contract audit failed: {failures[:5]}")
    return {
        "passed": True,
        "cases": len(cases),
        "stage_input_hashes_match": True,
        "deterministic_selection_matches": True,
        "invalid_raw_gate_output_cases": invalid_outputs,
    }


def decision_analysis() -> dict[str, Any]:
    treatment = rows(PATHS["four_b_treatment"])
    direct = rows(PATHS["four_b_direct"])
    mismatches = []
    use_cases = []
    wins = []
    losses = []
    neutral = []
    invalid = []
    for case_id, record in treatment.items():
        stages = record["stages"]
        protected = stages["protected_direct"]["prediction"]
        gate = stages["decision_gate"]
        if protected != direct[case_id].get("prediction"):
            mismatches.append(case_id)
        if gate["raw_output"] not in {"KEEP", "USE_RESOLVE"}:
            invalid.append(case_id)
        if gate["decision"] != "USE_RESOLVE":
            continue
        use_cases.append(case_id)
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
        "use_resolve_count": len(use_cases),
        "use_resolve_cases": use_cases,
        "use_resolve_wins": wins,
        "use_resolve_losses": losses,
        "neutral_use_resolve": neutral,
        "invalid_raw_gate_outputs": invalid,
    }


def main() -> None:
    for path in PATHS.values():
        if not path.is_file():
            raise SystemExit(f"missing result: {path}")
    versus_4b = compact(compare_baselines(PATHS["four_b_treatment"], PATHS["four_b_direct"]))
    versus_9b = compact(compare_baselines(PATHS["four_b_treatment"], PATHS["nine_b_direct"]))
    treatment_cost = cost(PATHS["four_b_treatment"])
    decisions = decision_analysis()
    accepted = (
        versus_4b["candidate_accuracy"] > versus_4b["baseline_accuracy"]
        and versus_4b["paired_counts"]["candidate_only"] >= 1
        and versus_4b["paired_counts"]["baseline_only"] == 0
        and not treatment_cost["api_errors"]
        and not treatment_cost["parse_failures"]
    )
    report = {
        "schema_version": "nano_harness_public_gsm8k_dev_v1",
        "experiment_id": "qwen35-gsm8k-dev11-gate-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "code_revision": git_revision(),
        "versus_4b_direct": versus_4b,
        "versus_9b_direct": versus_9b,
        "costs": {label: cost(path) for label, path in PATHS.items()},
        "decision_analysis": decisions,
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
        },
    }
    markdown = f"""# GSM8K Dev11 Decision Gate Result

## Result

- 4B direct: {versus_4b['baseline_accuracy']:.4f};
- 4B decision gate: {versus_4b['candidate_accuracy']:.4f};
- 9B direct: {versus_9b['baseline_accuracy']:.4f}.

Treatment versus 4B direct is {versus_4b['delta']:+.4f}, 95% bootstrap CI
[{versus_4b['paired_bootstrap_95_ci'][0]:+.4f},
{versus_4b['paired_bootstrap_95_ci'][1]:+.4f}], with
{versus_4b['paired_counts']['candidate_only']} treatment-only wins and
{versus_4b['paired_counts']['baseline_only']} direct-only losses.

The gate chooses `USE_RESOLVE` on {decisions['use_resolve_count']} cases:
{len(decisions['use_resolve_wins'])} wins,
{len(decisions['use_resolve_losses'])} losses, and
{len(decisions['neutral_use_resolve'])} neutral. It emits
{len(decisions['invalid_raw_gate_outputs'])} invalid raw decisions, all of
which fail closed to `KEEP`.

Treatment uses {treatment_cost['total_tokens']} tokens and
{treatment_cost['wall_seconds']:.1f}s.

## Contract Audit

Protected direct matches the independent 4B direct arm. Raw gate outputs,
decisions, selected predictions, stage inputs, and deterministic final
formatting match the committed protocol. Raw outputs remain local and ignored.

## Decision

{('Dev11 satisfies every directional promotion rule.'
   if accepted
   else 'Dev11 fails at least one directional promotion rule.')}

## Reproduction Identity

- Code revision: `{report['code_revision']}`
- 4B direct raw SHA256: `{report['artifacts']['four_b_direct_raw_sha256']}`
- 4B treatment raw SHA256: `{report['artifacts']['four_b_treatment_raw_sha256']}`
- 9B direct raw SHA256: `{report['artifacts']['nine_b_direct_raw_sha256']}`
"""
    Path("docs/results/gsm8k_dev11_gate_v1.public.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    Path("docs/results/gsm8k_dev11_gate_v1.md").write_text(markdown, encoding="utf-8")
    print(
        json.dumps(
            {
                "experiment_id": report["experiment_id"],
                "accepted": accepted,
                "treatment_vs_4b_delta": versus_4b["delta"],
                "treatment_vs_9b_delta": versus_9b["delta"],
                "use_resolve": decisions["use_resolve_count"],
                "use_resolve_wins": len(decisions["use_resolve_wins"]),
                "use_resolve_losses": len(decisions["use_resolve_losses"]),
                "invalid_gate_outputs": len(decisions["invalid_raw_gate_outputs"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
