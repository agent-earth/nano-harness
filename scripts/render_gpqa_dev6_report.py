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
    "four_b_direct": Path(
        "results/harness/qwen35-gpqa-dev6-direct-v1/4b/cases.jsonl"
    ),
    "four_b_treatment": Path(
        "results/harness/qwen35-gpqa-dev6-option-evidence-v1/4b/cases.jsonl"
    ),
    "nine_b_direct": Path(
        "results/harness/qwen35-gpqa-dev6-direct-v1/9b/cases.jsonl"
    ),
}
DIRECT_MANIFEST = Path("configs/harness/qwen35_gpqa_dev6_direct_v1.yaml")
TREATMENT_MANIFEST = Path(
    "configs/harness/qwen35_gpqa_dev6_option_evidence_v1.yaml"
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
    option_stages = [
        stage
        for row in records
        for stage in row.get("stages", {}).get("option_evidence", {}).values()
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
        "option_evaluator_calls": len(option_stages),
        "option_evaluator_truncations": sum(
            stage.get("finish_reason") == "length" for stage in option_stages
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


def audit_direct(result_path: Path) -> dict[str, Any]:
    manifest = load_manifest(DIRECT_MANIFEST)
    cases = {
        case.case_id: case for case in load_cases(manifest, Path("../../datasets"))
    }
    results = rows(result_path)
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
        case.case_id: case for case in load_cases(manifest, Path("../../datasets"))
    }
    results = rows(PATHS["four_b_treatment"])
    failures = []
    if set(cases) != set(results):
        failures.append("case identities")
    for case_id, case in cases.items():
        record = results.get(case_id, {})
        stages = record.get("stages", {})
        options = stages.get("option_evidence", {})
        if (
            record.get("selected_strategy") != "option_evidence_verify"
            or set(options) != {"A", "B", "C", "D"}
        ):
            failures.append(f"{case_id}:stages")
            continue
        evidence = []
        for letter in ("A", "B", "C", "D"):
            prompt = (
                f"<original_task>\n{case.draft_prompt}\n</original_task>\n\n"
                f"Evaluate option {letter} independently. State the strongest "
                "evidence for or against it, check the relevant facts or "
                f"calculation, and end with VERDICT {letter}: SUPPORT or "
                f"VERDICT {letter}: REJECT. Do not compare against another "
                "option's analysis and do not use tools."
            )
            stage = options[letter]
            if stage.get("input_sha256") != hashlib.sha256(prompt.encode()).hexdigest():
                failures.append(f"{case_id}:option-{letter}")
            evidence.append(
                f"<option_{letter}>\n{stage.get('output', '')}\n</option_{letter}>"
            )
        selector_prompt = (
            f"<original_task>\n{case.prompt}\n</original_task>\n\n"
            + "\n\n".join(evidence)
        )
        actual_selector = stages.get("selector", {}).get("input_sha256")
        if actual_selector != hashlib.sha256(selector_prompt.encode()).hexdigest():
            failures.append(f"{case_id}:selector")
    if failures:
        raise SystemExit(f"treatment contract audit failed: {failures[:5]}")
    return {
        "passed": True,
        "cases": len(cases),
        "selected_strategy": "option_evidence_verify",
        "option_evaluators_per_case": 4,
        "stage_input_hashes_match": True,
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
    treatment_rows = rows(PATHS["four_b_treatment"])
    parse_failure_details = []
    for record in treatment_rows.values():
        if record.get("prediction") is not None:
            continue
        output = str(record.get("output", "")).strip()
        parse_failure_details.append(
            {
                "case_id": record["case_id"],
                "output_kind": (
                    "bare_choice_letter"
                    if output.upper() in {"A", "B", "C", "D"}
                    else "other"
                ),
                "selector_finish_reason": (
                    record.get("stages", {}).get("selector", {}).get("finish_reason")
                ),
                "bare_letter_matches_reference": (
                    output.upper() == str(record["expected"]).upper()
                ),
            }
        )
    complete_option_calls = (
        treatment_cost["option_evaluator_calls"] == treatment_cost["cases"] * 4
    )
    accepted = (
        versus_4b["candidate_accuracy"] > versus_4b["baseline_accuracy"]
        and versus_4b["paired_counts"]["candidate_only"]
        > versus_4b["paired_counts"]["baseline_only"]
        and not treatment_cost["api_errors"]
        and not treatment_cost["parse_failures"]
        and complete_option_calls
    )
    report = {
        "schema_version": "nano_harness_public_gpqa_dev_v1",
        "experiment_id": "qwen35-gpqa-dev6-option-evidence-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "code_revision": git_revision(),
        "versus_4b_direct": versus_4b,
        "versus_9b_direct": versus_9b,
        "costs": {label: cost(path) for label, path in PATHS.items()},
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
            "net_paired_wins_over_4b": (
                versus_4b["paired_counts"]["candidate_only"]
                > versus_4b["paired_counts"]["baseline_only"]
            ),
            "no_api_errors": not treatment_cost["api_errors"],
            "no_parse_failures": not treatment_cost["parse_failures"],
            "all_option_evaluators_completed": complete_option_calls,
        },
        "failure_analysis": {
            "parse_failures": parse_failure_details,
            "interpretation": (
                "Option decomposition creates net corrective evidence, but the "
                "selector needs a pre-registered deterministic bare-letter "
                "normalizer before confirmation."
            ),
            "next_experiment": (
                "Fresh dev7 with unchanged option prompts and budgets plus strict "
                "bare-letter-to-FINAL normalization."
            ),
        },
    }
    markdown = f"""# GPQA Dev6 Option Evidence Result

## Result

- 4B direct: {versus_4b['baseline_accuracy']:.4f};
- 4B option evidence: {versus_4b['candidate_accuracy']:.4f};
- 9B direct: {versus_9b['baseline_accuracy']:.4f}.

Treatment versus 4B direct is {versus_4b['delta']:+.4f}, 95% bootstrap CI
[{versus_4b['paired_bootstrap_95_ci'][0]:+.4f},
{versus_4b['paired_bootstrap_95_ci'][1]:+.4f}], with
{versus_4b['paired_counts']['candidate_only']} treatment-only wins and
{versus_4b['paired_counts']['baseline_only']} direct-only losses.

Treatment uses {treatment_cost['total_tokens']} tokens and
{treatment_cost['wall_seconds']:.1f}s. It completed
{treatment_cost['option_evaluator_calls']} option evaluator calls with
{treatment_cost['option_evaluator_truncations']} truncations.

## Contract Audit

All case identities, strategies, four evaluator stages per case, actual option
input hashes, and selector input hashes match the committed protocol. Raw
outputs remain local and ignored.

## Decision

{('Dev6 satisfies every directional promotion rule.'
   if accepted
   else 'Dev6 fails at least one directional promotion rule.')}

The only treatment parse failure is a stopped selector that returned the bare
letter `D`; it matches that case's reference but is scored wrong under the
frozen `FINAL:` contract. No post-hoc rescoring is applied. The next fresh
experiment keeps all option prompts and budgets unchanged and adds only strict
deterministic normalization for a selector output that is exactly one choice
letter.

## Reproduction Identity

- Code revision: `{report['code_revision']}`
- 4B direct raw SHA256: `{report['artifacts']['four_b_direct_raw_sha256']}`
- 4B treatment raw SHA256: `{report['artifacts']['four_b_treatment_raw_sha256']}`
- 9B direct raw SHA256: `{report['artifacts']['nine_b_direct_raw_sha256']}`
"""
    Path("docs/results/gpqa_dev6_option_evidence_v1.public.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    Path("docs/results/gpqa_dev6_option_evidence_v1.md").write_text(
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
                "option_evaluator_truncations": treatment_cost[
                    "option_evaluator_truncations"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
