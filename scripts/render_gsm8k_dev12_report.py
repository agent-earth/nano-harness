#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from nano_harness.baseline import compare_baselines, load_cases, load_manifest


PATHS = {
    "four_b_direct": Path("results/harness/qwen35-gsm8k-dev12-direct-v1/4b/cases.jsonl"),
    "four_b_treatment": Path("results/harness/qwen35-gsm8k-dev12-majority-v1/4b/cases.jsonl"),
    "nine_b_direct": Path("results/harness/qwen35-gsm8k-dev12-direct-v1/9b/cases.jsonl"),
}
DIRECT_MANIFEST = Path("configs/harness/qwen35_gsm8k_dev12_direct_v1.yaml")
TREATMENT_MANIFEST = Path("configs/harness/qwen35_gsm8k_dev12_majority_v1.yaml")


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
    resolve_stages = [
        stage
        for row in records
        for stage in row.get("stages", {}).get("independent_resolves", {}).values()
    ]
    votes = [row.get("stages", {}).get("deterministic_vote", {}) for row in records]
    return {
        "cases": len(records),
        "correct": int(sum(float(row["score"]) for row in records)),
        "total_tokens": sum(int(row.get("usage", {}).get("total_tokens", 0)) for row in records),
        "wall_seconds": sum(float(row["latency_seconds"]) for row in records),
        "parse_failures": sum(row.get("prediction") is None for row in records),
        "api_errors": sum(row.get("status") == "error" for row in records),
        "resolve_calls": len(resolve_stages),
        "resolve_truncations": sum(stage.get("finish_reason") == "length" for stage in resolve_stages),
        "majority_count": sum(vote.get("selection_reason") == "numeric_majority" for vote in votes),
        "no_majority_count": sum(
            vote.get("selection_reason") == "no_majority_keep_direct" for vote in votes
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
    if set(cases) != set(results):
        failures.append("case identities")
    prompts = {}
    systems = {
        "resolve_a": (
            "Act as an independent forward math solver. Do not rely on another "
            "solution and make arithmetic dependencies explicit."
        ),
        "resolve_b": (
            "Act as an independent verification-oriented math solver. Use a "
            "different derivation and actively test the result for contradictions."
        ),
    }
    for case_id, case in cases.items():
        prompts["resolve_a"] = (
            f"<original_task>\n{case.draft_prompt}\n</original_task>\n\n"
            "Independently solve this math problem from scratch. Track each "
            "quantity and arithmetic dependency in forward order. Check units, "
            "rates, time periods, and the requested quantity. End with FINAL: "
            "<number>. Do not use tools."
        )
        prompts["resolve_b"] = (
            f"<original_task>\n{case.draft_prompt}\n</original_task>\n\n"
            "Independently solve this math problem using a verification-first "
            "approach. Derive the result, then check it by inverse calculation, "
            "estimation, unit analysis, and rereading exactly what is requested. "
            "End with FINAL: <number>. Do not use tools."
        )
        record = results.get(case_id, {})
        stages = record.get("stages", {})
        protected = stages.get("protected_direct", {})
        resolves = stages.get("independent_resolves", {})
        vote = stages.get("deterministic_vote", {})
        if (
            record.get("selected_strategy") != "protected_math_majority"
            or set(resolves) != {"resolve_a", "resolve_b"}
        ):
            failures.append(f"{case_id}:stages")
            continue
        if protected.get("input_sha256") != hashlib.sha256(case.prompt.encode()).hexdigest():
            failures.append(f"{case_id}:protected")
        for name in ("resolve_a", "resolve_b"):
            if resolves[name].get("input_sha256") != hashlib.sha256(prompts[name].encode()).hexdigest():
                failures.append(f"{case_id}:{name}")
        predictions = [
            protected.get("prediction"),
            resolves["resolve_a"].get("prediction"),
            resolves["resolve_b"].get("prediction"),
        ]
        counts = {}
        for prediction in predictions:
            if prediction is not None:
                counts[prediction] = counts.get(prediction, 0) + 1
        majority = next((value for value, count in counts.items() if count >= 2), None)
        selected = majority if majority is not None else protected.get("prediction")
        reason = "numeric_majority" if majority is not None else "no_majority_keep_direct"
        if (
            vote.get("predictions") != predictions
            or vote.get("counts") != counts
            or vote.get("majority_prediction") != majority
            or vote.get("selected_prediction") != selected
            or vote.get("selection_reason") != reason
            or record.get("prediction") != selected
        ):
            failures.append(f"{case_id}:vote")
    if systems["resolve_a"] == systems["resolve_b"]:
        failures.append("resolve systems are identical")
    if failures:
        raise SystemExit(f"treatment contract audit failed: {failures[:5]}")
    return {
        "passed": True,
        "cases": len(cases),
        "two_independent_resolves_per_case": True,
        "stage_input_hashes_match": True,
        "deterministic_vote_matches": True,
    }


def vote_analysis() -> dict[str, Any]:
    treatment = rows(PATHS["four_b_treatment"])
    direct = rows(PATHS["four_b_direct"])
    mismatches = []
    changed = []
    wins = []
    losses = []
    neutral = []
    majority_cases = []
    no_majority_cases = []
    for case_id, record in treatment.items():
        protected = record["stages"]["protected_direct"]["prediction"]
        vote = record["stages"]["deterministic_vote"]
        if protected != direct[case_id].get("prediction"):
            mismatches.append(case_id)
        if vote["selection_reason"] == "numeric_majority":
            majority_cases.append(case_id)
        else:
            no_majority_cases.append(case_id)
        if record.get("prediction") == protected:
            continue
        changed.append(case_id)
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
        "majority_count": len(majority_cases),
        "majority_cases": majority_cases,
        "no_majority_count": len(no_majority_cases),
        "no_majority_cases": no_majority_cases,
        "changed_from_direct_count": len(changed),
        "changed_from_direct_cases": changed,
        "vote_wins": wins,
        "vote_losses": losses,
        "neutral_changes": neutral,
    }


def main() -> None:
    for path in PATHS.values():
        if not path.is_file():
            raise SystemExit(f"missing result: {path}")
    versus_4b = compact(compare_baselines(PATHS["four_b_treatment"], PATHS["four_b_direct"]))
    versus_9b = compact(compare_baselines(PATHS["four_b_treatment"], PATHS["nine_b_direct"]))
    treatment_cost = cost(PATHS["four_b_treatment"])
    votes = vote_analysis()
    accepted = (
        versus_4b["candidate_accuracy"] > versus_4b["baseline_accuracy"]
        and versus_4b["paired_counts"]["candidate_only"] >= 1
        and versus_4b["paired_counts"]["baseline_only"] == 0
        and not treatment_cost["api_errors"]
        and not treatment_cost["parse_failures"]
    )
    report = {
        "schema_version": "nano_harness_public_gsm8k_dev_v1",
        "experiment_id": "qwen35-gsm8k-dev12-majority-v1",
        "code_revision": git_revision(),
        "versus_4b_direct": versus_4b,
        "versus_9b_direct": versus_9b,
        "costs": {label: cost(path) for label, path in PATHS.items()},
        "vote_analysis": votes,
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
    markdown = f"""# GSM8K Dev12 Deterministic Majority Result

## Result

- 4B direct: {versus_4b['baseline_accuracy']:.4f};
- 4B deterministic majority: {versus_4b['candidate_accuracy']:.4f};
- 9B direct: {versus_9b['baseline_accuracy']:.4f}.

Treatment versus 4B direct is {versus_4b['delta']:+.4f}, 95% bootstrap CI
[{versus_4b['paired_bootstrap_95_ci'][0]:+.4f},
{versus_4b['paired_bootstrap_95_ci'][1]:+.4f}], with
{versus_4b['paired_counts']['candidate_only']} treatment-only wins and
{versus_4b['paired_counts']['baseline_only']} direct-only losses.

A numeric majority exists on {votes['majority_count']} cases; no majority
keeps direct on {votes['no_majority_count']}. Voting changes the direct answer
on {votes['changed_from_direct_count']} cases:
{len(votes['vote_wins'])} wins, {len(votes['vote_losses'])} losses, and
{len(votes['neutral_changes'])} neutral.

Treatment uses {treatment_cost['total_tokens']} tokens and
{treatment_cost['wall_seconds']:.1f}s.

## Contract Audit

Protected direct matches the control, both isolated re-solve inputs match the
committed prompts, and every majority/count/selection is recomputed from
recorded normalized predictions. Raw outputs remain local and ignored.

## Decision

{('Dev12 satisfies every directional promotion rule.'
   if accepted
   else 'Dev12 fails at least one directional promotion rule.')}

## Reproduction Identity

- Code revision: `{report['code_revision']}`
- 4B direct raw SHA256: `{report['artifacts']['four_b_direct_raw_sha256']}`
- 4B treatment raw SHA256: `{report['artifacts']['four_b_treatment_raw_sha256']}`
- 9B direct raw SHA256: `{report['artifacts']['nine_b_direct_raw_sha256']}`
"""
    Path("docs/results/gsm8k_dev12_majority_v1.public.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    Path("docs/results/gsm8k_dev12_majority_v1.md").write_text(markdown, encoding="utf-8")
    print(
        json.dumps(
            {
                "experiment_id": report["experiment_id"],
                "accepted": accepted,
                "treatment_vs_4b_delta": versus_4b["delta"],
                "treatment_vs_9b_delta": versus_9b["delta"],
                "majority_count": votes["majority_count"],
                "changed_from_direct": votes["changed_from_direct_count"],
                "vote_wins": len(votes["vote_wins"]),
                "vote_losses": len(votes["vote_losses"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
