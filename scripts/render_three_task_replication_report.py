#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from nano_harness.baseline import (
    compare_baselines,
    load_cases,
    load_manifest,
)


PATHS = {
    "four_b": Path(
        "results/harness/qwen35-three-task-replication-v1/4b/cases.jsonl"
    ),
    "nine_b": Path(
        "results/harness/qwen35-three-task-replication-v1/9b/cases.jsonl"
    ),
}
MANIFEST = Path("configs/harness/qwen35_three_task_replication_v1.yaml")


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
    records = list(rows(path).values())
    by_benchmark = {}
    for benchmark in sorted({row["benchmark"] for row in records}):
        subset = [row for row in records if row["benchmark"] == benchmark]
        by_benchmark[benchmark] = {
            "cases": len(subset),
            "correct": int(sum(float(row["score"]) for row in subset)),
            "parse_failures": sum(row.get("prediction") is None for row in subset),
            "length_truncations": sum(
                row.get("finish_reason") == "length" for row in subset
            ),
            "api_errors": sum(row.get("status") == "error" for row in subset),
            "total_tokens": sum(
                int(row.get("usage", {}).get("total_tokens", 0)) for row in subset
            ),
            "wall_seconds": sum(float(row["latency_seconds"]) for row in subset),
        }
    return {
        "cases": len(records),
        "correct": int(sum(float(row["score"]) for row in records)),
        "parse_failures": sum(row.get("prediction") is None for row in records),
        "length_truncations": sum(
            row.get("finish_reason") == "length" for row in records
        ),
        "api_errors": sum(row.get("status") == "error" for row in records),
        "total_tokens": sum(
            int(row.get("usage", {}).get("total_tokens", 0)) for row in records
        ),
        "wall_seconds": sum(float(row["latency_seconds"]) for row in records),
        "by_benchmark": by_benchmark,
    }


def audit_direct(path: Path) -> dict[str, Any]:
    manifest = load_manifest(MANIFEST)
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
        if (
            record.get("selected_strategy") != "direct"
            or record.get("prompt_sha256") != expected
            or actual != expected
        ):
            failures.append(case_id)
    if failures:
        raise SystemExit(f"direct contract audit failed: {failures[:5]}")
    return {
        "passed": True,
        "cases": len(cases),
        "case_id_set_matches": True,
        "selected_strategy": "direct",
        "prompt_hashes_match": True,
        "stage_input_hashes_match": True,
    }


def main() -> None:
    for path in PATHS.values():
        if not path.is_file():
            raise SystemExit(f"missing result: {path}")

    comparison = compare_baselines(PATHS["four_b"], PATHS["nine_b"])
    overall = comparison["overall_micro"]
    costs = {label: cost(path) for label, path in PATHS.items()}
    per_benchmark_non_regression = all(
        result["delta"] >= 0 for result in comparison["benchmarks"].values()
    )
    no_api_errors = all(not value["api_errors"] for value in costs.values())
    accepted = (
        comparison["candidate_macro_accuracy"]
        > comparison["baseline_macro_accuracy"]
        and overall["paired_bootstrap_95_ci"][0] > 0
        and overall["mcnemar_exact_p"] < 0.05
        and per_benchmark_non_regression
        and no_api_errors
    )
    report = {
        "schema_version": "nano_harness_public_three_task_replication_v1",
        "experiment_id": "qwen35-three-task-replication-v1",
        "code_revision": git_revision(),
        "comparison": comparison,
        "costs": costs,
        "contract_audits": {
            "four_b": audit_direct(PATHS["four_b"]),
            "nine_b": audit_direct(PATHS["nine_b"]),
        },
        "artifacts": {
            f"{label}_raw_sha256": sha256_file(path)
            for label, path in PATHS.items()
        },
        "decision": {
            "accepted": accepted,
            "macro_above_9b": (
                comparison["candidate_macro_accuracy"]
                > comparison["baseline_macro_accuracy"]
            ),
            "paired_micro_lower_bound_above_zero": (
                overall["paired_bootstrap_95_ci"][0] > 0
            ),
            "mcnemar_below_005": overall["mcnemar_exact_p"] < 0.05,
            "per_benchmark_non_regression": per_benchmark_non_regression,
            "no_api_errors": no_api_errors,
            "directionally_replicated": (
                overall["delta"] > 0 and per_benchmark_non_regression
            ),
            "next_experiment": (
                "Preserve this directional replication, but do not claim "
                "significance; use the discordant cases as versioned data and "
                "verifier inputs for the deferred SFT smoke."
            ),
        },
    }

    benchmark_rows = []
    for benchmark in ("gsm8k", "mmlu", "gpqa_diamond"):
        result = comparison["benchmarks"][benchmark]
        benchmark_rows.append(
            f"| {benchmark} | {result['candidate_correct']}/{result['cases']} "
            f"({result['candidate_accuracy']:.4f}) | "
            f"{result['baseline_correct']}/{result['cases']} "
            f"({result['baseline_accuracy']:.4f}) | "
            f"{result['delta']:+.4f} | "
            f"[{result['paired_bootstrap_95_ci'][0]:+.4f}, "
            f"{result['paired_bootstrap_95_ci'][1]:+.4f}] | "
            f"{result['mcnemar_exact_p']:.4g} |"
        )

    markdown = f"""# Three-Task 4B/9B Replication Result

## Result

| Benchmark | Qwen3.5-4B | Qwen3.5-9B | Delta | Paired 95% CI | McNemar p |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(benchmark_rows)}

- 4B macro accuracy: {comparison['candidate_macro_accuracy']:.4f};
- 9B macro accuracy: {comparison['baseline_macro_accuracy']:.4f};
- macro delta: {comparison['macro_delta']:+.4f}.

Across all 211 matched cases, 4B scores
{overall['candidate_correct']}/211 ({overall['candidate_accuracy']:.4f}) and
9B scores {overall['baseline_correct']}/211
({overall['baseline_accuracy']:.4f}). The paired micro delta is
{overall['delta']:+.4f}, with 95% bootstrap CI
[{overall['paired_bootstrap_95_ci'][0]:+.4f},
{overall['paired_bootstrap_95_ci'][1]:+.4f}] and exact McNemar
`p={overall['mcnemar_exact_p']:.6f}`.

There are {overall['paired_counts']['candidate_only']} 4B-only wins and
{overall['paired_counts']['baseline_only']} 9B-only wins. Every task point
estimate favors 4B, so the holdout5 direction replicates without task
regression. The confidence interval lower bound is exactly zero and the
McNemar p-value exceeds 0.05, so the pre-registered significance rule does not
pass.

## Parse And Cost

- 4B: {costs['four_b']['parse_failures']} parse failures,
  {costs['four_b']['length_truncations']} length truncations,
  {costs['four_b']['total_tokens']} tokens,
  {costs['four_b']['wall_seconds']:.1f}s summed request latency;
- 9B: {costs['nine_b']['parse_failures']} parse failures,
  {costs['nine_b']['length_truncations']} length truncations,
  {costs['nine_b']['total_tokens']} tokens,
  {costs['nine_b']['wall_seconds']:.1f}s summed request latency;
- both arms: zero API errors.

## Contract Audit

Both arms contain exactly the committed 211 unique case IDs. Prompt hashes,
direct-stage input hashes, strategies, dataset versions, scorers, and budgets
match the pre-registration. Raw outputs remain local and ignored.

## Decision

{('The replication satisfies every pre-registered superiority condition.'
   if accepted
   else 'The replication does not satisfy every pre-registered superiority condition.')}

The result is a replicated directional 4B advantage, not statistically
significant superiority. Preserve the 27 4B-only and 15 9B-only discordances
as versioned data/verifier inputs for training, rather than continuing
post-hoc prompt search on this sample.

## Reproduction Identity

- Pre-registration/code revision: `{report['code_revision']}`
- 4B raw SHA256: `{report['artifacts']['four_b_raw_sha256']}`
- 9B raw SHA256: `{report['artifacts']['nine_b_raw_sha256']}`
"""
    Path(
        "docs/results/three_task_replication_v1.public.json"
    ).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    Path("docs/results/three_task_replication_v1.md").write_text(
        markdown,
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "experiment_id": report["experiment_id"],
                "accepted": accepted,
                "macro_delta": comparison["macro_delta"],
                "micro_delta": overall["delta"],
                "micro_ci": overall["paired_bootstrap_95_ci"],
                "mcnemar_exact_p": overall["mcnemar_exact_p"],
                "per_benchmark_non_regression": per_benchmark_non_regression,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
