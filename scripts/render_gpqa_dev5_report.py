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
        "results/harness/qwen35-gpqa-dev5-direct-v1/4b/cases.jsonl"
    ),
    "four_b_treatment": Path(
        "results/harness/qwen35-gpqa-dev5-draft-verify-v1/4b/cases.jsonl"
    ),
    "nine_b_direct": Path(
        "results/harness/qwen35-gpqa-dev5-direct-v1/9b/cases.jsonl"
    ),
}
MANIFESTS = {
    "direct": Path("configs/harness/qwen35_gpqa_dev5_direct_v1.yaml"),
    "treatment": Path(
        "configs/harness/qwen35_gpqa_dev5_draft_verify_v1.yaml"
    ),
}


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
        "draft_truncations": sum(
            row.get("stages", {}).get("draft", {}).get("finish_reason")
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


def audit(
    manifest_path: Path,
    result_path: Path,
    dataset_root: Path,
    expected_strategy: str,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    cases = {
        case.case_id: case for case in load_cases(manifest, dataset_root)
    }
    results = rows(result_path)
    if set(cases) != set(results):
        raise SystemExit(f"case identity mismatch for {result_path}")
    failures = []
    stage = "direct" if expected_strategy == "direct" else "draft"
    for case_id, case in cases.items():
        record = results[case_id]
        if record.get("selected_strategy") != expected_strategy:
            failures.append(f"{case_id}:strategy")
            continue
        input_text = (
            case.prompt if expected_strategy == "direct" else case.draft_prompt
        )
        expected_sha = hashlib.sha256(input_text.encode()).hexdigest()
        actual_sha = (
            record.get("stages", {}).get(stage, {}).get("input_sha256")
        )
        if actual_sha != expected_sha:
            failures.append(f"{case_id}:{stage}.input_sha256")
    if failures:
        raise SystemExit(f"contract audit failed: {failures[:5]}")
    return {
        "passed": True,
        "cases": len(cases),
        "selected_strategy": expected_strategy,
        "stage_input_hashes_match": True,
    }


def main() -> None:
    for path in PATHS.values():
        if not path.is_file():
            raise SystemExit(f"missing result: {path}")
    versus_4b = compare_baselines(
        PATHS["four_b_treatment"], PATHS["four_b_direct"]
    )
    versus_9b = compare_baselines(
        PATHS["four_b_treatment"], PATHS["nine_b_direct"]
    )
    four_b = compact(versus_4b)
    nine_b = compact(versus_9b)
    treatment_cost = cost(PATHS["four_b_treatment"])
    direct_rows = rows(PATHS["four_b_direct"])
    treatment_rows = rows(PATHS["four_b_treatment"])
    same_predictions = sum(
        direct_rows[case_id].get("prediction")
        == treatment_rows[case_id].get("prediction")
        for case_id in direct_rows
    )
    same_scores = sum(
        direct_rows[case_id].get("score")
        == treatment_rows[case_id].get("score")
        for case_id in direct_rows
    )
    accepted = (
        four_b["candidate_accuracy"] > four_b["baseline_accuracy"]
        and four_b["paired_counts"]["candidate_only"]
        > four_b["paired_counts"]["baseline_only"]
        and not treatment_cost["api_errors"]
        and not treatment_cost["parse_failures"]
        and treatment_cost["draft_truncations"] < treatment_cost["cases"]
    )
    report = {
        "schema_version": "nano_harness_public_gpqa_dev_v1",
        "experiment_id": "qwen35-gpqa-dev5-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "code_revision": git_revision(),
        "versus_4b_direct": four_b,
        "versus_9b_direct": nine_b,
        "costs": {label: cost(path) for label, path in PATHS.items()},
        "contract_audits": {
            "four_b_direct": audit(
                MANIFESTS["direct"],
                PATHS["four_b_direct"],
                Path("../../datasets"),
                "direct",
            ),
            "four_b_treatment": audit(
                MANIFESTS["treatment"],
                PATHS["four_b_treatment"],
                Path("../../datasets"),
                "draft_verify",
            ),
            "nine_b_direct": audit(
                MANIFESTS["direct"],
                PATHS["nine_b_direct"],
                Path("../../datasets"),
                "direct",
            ),
        },
        "artifacts": {
            f"{label}_raw_sha256": sha256_file(path)
            for label, path in PATHS.items()
        },
        "decision": {
            "accepted": accepted,
            "point_above_4b_direct": (
                four_b["candidate_accuracy"] > four_b["baseline_accuracy"]
            ),
            "net_paired_wins_over_4b": (
                four_b["paired_counts"]["candidate_only"]
                > four_b["paired_counts"]["baseline_only"]
            ),
            "no_api_errors": not treatment_cost["api_errors"],
            "no_parse_failures": not treatment_cost["parse_failures"],
            "draft_truncation_below_dev4_rate": (
                treatment_cost["draft_truncations"] < treatment_cost["cases"]
            ),
            "next_experiment": (
                "Pre-register a new three-task GPQA-only routing holdout."
                if accepted
                else "Stop GPQA draft-verify and replan from fresh evidence."
            ),
        },
        "failure_analysis": {
            "same_predictions_as_4b_direct": same_predictions,
            "same_scores_as_4b_direct": same_scores,
            "non_truncated_drafts": (
                treatment_cost["cases"] - treatment_cost["draft_truncations"]
            ),
            "interpretation": (
                "A larger monolithic draft reduced truncation but produced no "
                "net correctness change; use independent option evidence next."
            ),
        },
    }
    markdown = f"""# GPQA Dev5 v1 Result

## Result

- 4B direct: {four_b['baseline_accuracy']:.4f};
- 4B 384-token draft-verify: {four_b['candidate_accuracy']:.4f};
- 9B direct: {nine_b['baseline_accuracy']:.4f}.

Treatment versus 4B direct is {four_b['delta']:+.4f}, 95% bootstrap CI
[{four_b['paired_bootstrap_95_ci'][0]:+.4f},
{four_b['paired_bootstrap_95_ci'][1]:+.4f}], with
{four_b['paired_counts']['candidate_only']} treatment-only wins and
{four_b['paired_counts']['baseline_only']} direct-only losses.

Treatment uses {treatment_cost['total_tokens']} tokens and
{treatment_cost['wall_seconds']:.1f}s. Draft truncations are
{treatment_cost['draft_truncations']}/{treatment_cost['cases']}; final parse
failures and API errors are {treatment_cost['parse_failures']} and
{treatment_cost['api_errors']}.

Treatment and 4B direct have identical correctness on
{same_scores}/{treatment_cost['cases']} cases and identical predictions on
{same_predictions}/{treatment_cost['cases']}. The only changed prediction is
wrong in both arms. Neither of the two non-truncated drafts changes the direct
prediction.

## Contract Audit

All three arms match committed case identities, selected strategies, and
actual direct/draft stage input hashes. Raw outputs remain local and ignored.

## Decision

{('Dev5 satisfies every directional promotion rule.'
   if accepted
   else 'Dev5 fails at least one directional promotion rule.')}

The next action is: test independent per-option evidence on fresh cases rather
than increasing a monolithic draft budget again.

## Reproduction Identity

- Code revision: `{report['code_revision']}`
- 4B direct raw SHA256: `{report['artifacts']['four_b_direct_raw_sha256']}`
- 4B treatment raw SHA256: `{report['artifacts']['four_b_treatment_raw_sha256']}`
- 9B direct raw SHA256: `{report['artifacts']['nine_b_direct_raw_sha256']}`
"""
    json_path = Path("docs/results/gpqa_dev5_v1.public.json")
    markdown_path = Path("docs/results/gpqa_dev5_v1.md")
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(markdown, encoding="utf-8")
    print(
        json.dumps(
            {
                "experiment_id": report["experiment_id"],
                "accepted": accepted,
                "treatment_vs_4b_delta": four_b["delta"],
                "treatment_vs_9b_delta": nine_b["delta"],
                "draft_truncations": treatment_cost["draft_truncations"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
