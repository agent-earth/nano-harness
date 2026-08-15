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
        "results/harness/qwen35-gpqa-dev8-direct-v1/4b/cases.jsonl"
    ),
    "four_b_treatment": Path(
        "results/harness/qwen35-gpqa-dev8-arbiter-v1/4b/cases.jsonl"
    ),
    "nine_b_direct": Path(
        "results/harness/qwen35-gpqa-dev8-direct-v1/9b/cases.jsonl"
    ),
}
DIRECT_MANIFEST = Path("configs/harness/qwen35_gpqa_dev8_direct_v1.yaml")
TREATMENT_MANIFEST = Path("configs/harness/qwen35_gpqa_dev8_arbiter_v1.yaml")


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
    arbiter_stages = [
        row.get("stages", {}).get("arbiter", {}) for row in records
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
        "bare_choice_normalizations": sum(
            bool(stage.get("normalized_bare_choice")) for stage in arbiter_stages
        ),
        "arbiter_raw_hashes": sum(
            bool(stage.get("raw_output_sha256")) for stage in arbiter_stages
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
        protected = stages.get("protected_direct", {})
        options = stages.get("option_evidence", {})
        arbiter = stages.get("arbiter", {})
        if (
            record.get("selected_strategy") != "option_evidence_arbiter"
            or set(options) != {"A", "B", "C", "D"}
            or not arbiter.get("raw_output_sha256")
        ):
            failures.append(f"{case_id}:stages")
            continue
        if protected.get("input_sha256") != hashlib.sha256(
            case.prompt.encode()
        ).hexdigest():
            failures.append(f"{case_id}:protected-direct")
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
        arbiter_prompt = (
            f"<original_task>\n{case.prompt}\n</original_task>\n\n"
            f"<protected_direct_candidate>\n{protected.get('output', '')}"
            "\n</protected_direct_candidate>\n\n"
            + "\n\n".join(evidence)
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
        "selected_strategy": "option_evidence_arbiter",
        "protected_direct_per_case": 1,
        "option_evaluators_per_case": 4,
        "stage_input_hashes_match": True,
        "raw_arbiter_hashes_present": True,
    }


def override_analysis() -> dict[str, Any]:
    treatment = rows(PATHS["four_b_treatment"])
    direct = rows(PATHS["four_b_direct"])
    overrides = []
    protected_mismatches = []
    override_wins = []
    override_losses = []
    for case_id, record in treatment.items():
        protected_output = (
            record.get("stages", {}).get("protected_direct", {}).get("output", "")
        )
        protected_prediction = extract_prediction(protected_output, "choice_exact")
        direct_prediction = direct[case_id].get("prediction")
        if protected_prediction != direct_prediction:
            protected_mismatches.append(case_id)
        if record.get("prediction") == protected_prediction:
            continue
        overrides.append(case_id)
        protected_score = float(protected_prediction == record["expected"])
        final_score = float(record["score"])
        if final_score > protected_score:
            override_wins.append(case_id)
        elif final_score < protected_score:
            override_losses.append(case_id)
    if protected_mismatches:
        raise SystemExit(
            f"protected direct differs from independent direct: "
            f"{protected_mismatches[:5]}"
        )
    return {
        "protected_direct_matches_independent_direct": True,
        "override_count": len(overrides),
        "override_cases": overrides,
        "override_wins": override_wins,
        "override_losses": override_losses,
        "neutral_overrides": len(overrides) - len(override_wins) - len(override_losses),
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
    complete = treatment_cost["option_evaluator_calls"] == treatment_cost["cases"] * 4
    accepted = (
        versus_4b["candidate_accuracy"] > versus_4b["baseline_accuracy"]
        and versus_4b["paired_counts"]["candidate_only"]
        > versus_4b["paired_counts"]["baseline_only"]
        and not treatment_cost["api_errors"]
        and not treatment_cost["parse_failures"]
        and complete
    )
    report = {
        "schema_version": "nano_harness_public_gpqa_dev_v1",
        "experiment_id": "qwen35-gpqa-dev8-conservative-arbiter-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "code_revision": git_revision(),
        "versus_4b_direct": versus_4b,
        "versus_9b_direct": versus_9b,
        "costs": {label: cost(path) for label, path in PATHS.items()},
        "override_analysis": overrides,
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
            "all_stages_completed": complete,
        },
    }
    markdown = f"""# GPQA Dev8 Conservative Arbiter Result

## Result

- 4B direct: {versus_4b['baseline_accuracy']:.4f};
- 4B conservative arbiter: {versus_4b['candidate_accuracy']:.4f};
- 9B direct: {versus_9b['baseline_accuracy']:.4f}.

Treatment versus 4B direct is {versus_4b['delta']:+.4f}, 95% bootstrap CI
[{versus_4b['paired_bootstrap_95_ci'][0]:+.4f},
{versus_4b['paired_bootstrap_95_ci'][1]:+.4f}], with
{versus_4b['paired_counts']['candidate_only']} treatment-only wins and
{versus_4b['paired_counts']['baseline_only']} direct-only losses.

The arbiter overrides protected direct on {overrides['override_count']} cases:
{len(overrides['override_wins'])} improve correctness,
{len(overrides['override_losses'])} reduce correctness, and
{overrides['neutral_overrides']} are neutral.

Treatment uses {treatment_cost['total_tokens']} tokens and
{treatment_cost['wall_seconds']:.1f}s, with
{treatment_cost['parse_failures']} final parse failures.

## Contract Audit

Protected-direct predictions match the independent 4B direct arm for all
cases. Case IDs, all six stage inputs, raw arbiter hashes, and strategy match
the committed protocol. Raw outputs remain local and ignored.

## Decision

{('Dev8 satisfies every directional promotion rule.'
   if accepted
   else 'Dev8 fails at least one directional promotion rule.')}

## Reproduction Identity

- Code revision: `{report['code_revision']}`
- 4B direct raw SHA256: `{report['artifacts']['four_b_direct_raw_sha256']}`
- 4B treatment raw SHA256: `{report['artifacts']['four_b_treatment_raw_sha256']}`
- 9B direct raw SHA256: `{report['artifacts']['nine_b_direct_raw_sha256']}`
"""
    Path("docs/results/gpqa_dev8_conservative_arbiter_v1.public.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    Path("docs/results/gpqa_dev8_conservative_arbiter_v1.md").write_text(
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
                "overrides": overrides["override_count"],
                "override_wins": len(overrides["override_wins"]),
                "override_losses": len(overrides["override_losses"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
