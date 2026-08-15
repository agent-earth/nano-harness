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
        "results/harness/qwen35-arbiter-holdout5-direct-v1/4b/cases.jsonl"
    ),
    "four_b_routed": Path(
        "results/harness/qwen35-arbiter-holdout5-routed-v1/4b/cases.jsonl"
    ),
    "nine_b_direct": Path(
        "results/harness/qwen35-arbiter-holdout5-direct-v1/9b/cases.jsonl"
    ),
}
DIRECT_MANIFEST = Path(
    "configs/harness/qwen35_arbiter_holdout5_direct_v1.yaml"
)
ROUTED_MANIFEST = Path(
    "configs/harness/qwen35_arbiter_holdout5_routed_v1.yaml"
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
    }


def compact(comparison: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "cases",
        "candidate_correct",
        "baseline_correct",
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
        "candidate_macro_accuracy": comparison["candidate_macro_accuracy"],
        "baseline_macro_accuracy": comparison["baseline_macro_accuracy"],
        "macro_delta": comparison["macro_delta"],
        "overall_micro": {
            key: comparison["overall_micro"][key] for key in fields
        },
        "benchmarks": {
            name: {key: metrics[key] for key in fields}
            for name, metrics in comparison["benchmarks"].items()
        },
        "bootstrap_samples": comparison["bootstrap_samples"],
        "bootstrap_seed": comparison["bootstrap_seed"],
    }


def audit_direct(result_path: Path) -> dict[str, Any]:
    manifest = load_manifest(DIRECT_MANIFEST)
    cases = {
        case.case_id: case
        for case in load_cases(manifest, Path("../../datasets"))
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


def audit_routed() -> dict[str, Any]:
    manifest = load_manifest(ROUTED_MANIFEST)
    cases = {
        case.case_id: case
        for case in load_cases(manifest, Path("../../datasets"))
    }
    results = rows(PATHS["four_b_routed"])
    failures = []
    direct_count = 0
    arbiter_count = 0
    if set(cases) != set(results):
        failures.append("case identities")
    for case_id, case in cases.items():
        record = results.get(case_id, {})
        stages = record.get("stages", {})
        if case.benchmark in {"gsm8k", "mmlu"}:
            direct_count += 1
            expected = hashlib.sha256(case.prompt.encode()).hexdigest()
            actual = stages.get("direct", {}).get("input_sha256")
            if record.get("selected_strategy") != "direct" or actual != expected:
                failures.append(f"{case_id}:direct")
            continue
        arbiter_count += 1
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
            if stage.get("input_sha256") != hashlib.sha256(
                prompt.encode()
            ).hexdigest():
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
        raise SystemExit(f"routed contract audit failed: {failures[:5]}")
    return {
        "passed": True,
        "cases": len(cases),
        "direct_cases": direct_count,
        "arbiter_cases": arbiter_count,
        "option_evaluators_per_arbiter_case": 4,
        "stage_input_hashes_match": True,
        "raw_arbiter_hashes_present": True,
    }


def override_analysis() -> dict[str, Any]:
    treatment = rows(PATHS["four_b_routed"])
    direct = rows(PATHS["four_b_direct"])
    overrides = []
    protected_mismatches = []
    wins = []
    losses = []
    for case_id, record in treatment.items():
        if record["benchmark"] != "gpqa_diamond":
            if record.get("prediction") != direct[case_id].get("prediction"):
                protected_mismatches.append(case_id)
            continue
        protected_output = (
            record.get("stages", {})
            .get("protected_direct", {})
            .get("output", "")
        )
        protected_prediction = extract_prediction(protected_output, "choice_exact")
        if protected_prediction != direct[case_id].get("prediction"):
            protected_mismatches.append(case_id)
        if record.get("prediction") == protected_prediction:
            continue
        overrides.append(case_id)
        protected_score = float(protected_prediction == record["expected"])
        final_score = float(record["score"])
        if final_score > protected_score:
            wins.append(case_id)
        elif final_score < protected_score:
            losses.append(case_id)
    if protected_mismatches:
        raise SystemExit(
            f"routed direct differs from direct control: {protected_mismatches[:5]}"
        )
    return {
        "routed_direct_matches_control": True,
        "gpqa_override_count": len(overrides),
        "gpqa_override_cases": overrides,
        "gpqa_override_wins": wins,
        "gpqa_override_losses": losses,
        "gpqa_neutral_overrides": len(overrides) - len(wins) - len(losses),
    }


def main() -> None:
    for path in PATHS.values():
        if not path.is_file():
            raise SystemExit(f"missing result: {path}")
    versus_4b = compact(
        compare_baselines(PATHS["four_b_routed"], PATHS["four_b_direct"])
    )
    versus_9b = compact(
        compare_baselines(PATHS["four_b_routed"], PATHS["nine_b_direct"])
    )
    treatment_cost = cost(PATHS["four_b_routed"])
    per_benchmark_non_regression = all(
        metrics["candidate_accuracy"] >= metrics["baseline_accuracy"]
        for metrics in versus_9b["benchmarks"].values()
    )
    overall = versus_9b["overall_micro"]
    accepted = (
        versus_9b["candidate_macro_accuracy"]
        > versus_9b["baseline_macro_accuracy"]
        and per_benchmark_non_regression
        and overall["paired_bootstrap_95_ci"][0] > 0
        and overall["mcnemar_exact_p"] < 0.05
        and not treatment_cost["api_errors"]
        and not treatment_cost["parse_failures"]
    )
    overrides = override_analysis()
    report = {
        "schema_version": "nano_harness_public_arbiter_holdout_v1",
        "holdout_id": "qwen35-arbiter-holdout5-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "code_revision": git_revision(),
        "versus_4b_direct": versus_4b,
        "versus_9b_direct": versus_9b,
        "costs": {label: cost(path) for label, path in PATHS.items()},
        "override_analysis": overrides,
        "contract_audits": {
            "four_b_direct": audit_direct(PATHS["four_b_direct"]),
            "four_b_routed": audit_routed(),
            "nine_b_direct": audit_direct(PATHS["nine_b_direct"]),
        },
        "artifacts": {
            f"{label}_raw_sha256": sha256_file(path)
            for label, path in PATHS.items()
        },
        "decision": {
            "accepted": accepted,
            "macro_above_9b": (
                versus_9b["candidate_macro_accuracy"]
                > versus_9b["baseline_macro_accuracy"]
            ),
            "per_benchmark_non_regression_vs_9b": per_benchmark_non_regression,
            "paired_micro_lower_bound_above_zero": (
                overall["paired_bootstrap_95_ci"][0] > 0
            ),
            "mcnemar_below_005": overall["mcnemar_exact_p"] < 0.05,
            "no_api_errors": not treatment_cost["api_errors"],
            "no_parse_failures": not treatment_cost["parse_failures"],
        },
    }
    names = {
        "gsm8k": "GSM8K",
        "mmlu": "MMLU",
        "gpqa_diamond": "GPQA-Diamond",
    }
    table = []
    for benchmark in ("gsm8k", "mmlu", "gpqa_diamond"):
        direct = versus_4b["benchmarks"][benchmark]
        nine = versus_9b["benchmarks"][benchmark]
        table.append(
            f"| {names[benchmark]} | {direct['baseline_accuracy']:.4f} | "
            f"{direct['candidate_accuracy']:.4f} | "
            f"{nine['baseline_accuracy']:.4f} |"
        )
    markdown = f"""# Conservative Arbiter Holdout5 Result

## Result

| Benchmark | 4B direct | 4B routed | 9B direct |
| --- | ---: | ---: | ---: |
{chr(10).join(table)}
| Macro | {versus_4b['baseline_macro_accuracy']:.4f} | {versus_4b['candidate_macro_accuracy']:.4f} | {versus_9b['baseline_macro_accuracy']:.4f} |

Routed 4B versus 9B direct has paired micro delta
{overall['delta']:+.4f}, 95% bootstrap CI
[{overall['paired_bootstrap_95_ci'][0]:+.4f},
{overall['paired_bootstrap_95_ci'][1]:+.4f}], exact McNemar
`p={overall['mcnemar_exact_p']:.8f}`.

The GPQA arbiter overrides protected direct on
{overrides['gpqa_override_count']} cases:
{len(overrides['gpqa_override_wins'])} improve correctness and
{len(overrides['gpqa_override_losses'])} reduce correctness.

## Contract Audit

All 72 identities and routed stages match the committed protocol. Routed
GSM8K/MMLU predictions and GPQA protected-direct predictions match the 4B
direct control before arbitration. Raw outputs remain local and ignored.

## Decision

{('Holdout5 satisfies every pre-registered harness acceptance rule.'
   if accepted
   else 'Holdout5 does not satisfy every pre-registered harness acceptance rule.')}

## Reproduction Identity

- Code revision: `{report['code_revision']}`
- 4B direct raw SHA256: `{report['artifacts']['four_b_direct_raw_sha256']}`
- 4B routed raw SHA256: `{report['artifacts']['four_b_routed_raw_sha256']}`
- 9B direct raw SHA256: `{report['artifacts']['nine_b_direct_raw_sha256']}`
"""
    Path("docs/results/arbiter_holdout5_v1.public.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    Path("docs/results/arbiter_holdout5_v1.md").write_text(
        markdown,
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "holdout_id": report["holdout_id"],
                "accepted": accepted,
                "routed_vs_9b_macro_delta": versus_9b["macro_delta"],
                "routed_vs_9b_micro_delta": overall["delta"],
                "gpqa_overrides": overrides["gpqa_override_count"],
                "gpqa_override_wins": len(overrides["gpqa_override_wins"]),
                "gpqa_override_losses": len(overrides["gpqa_override_losses"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
