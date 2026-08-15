#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nano_harness.baseline import (
    _strategy_for_case,
    compare_baselines,
    load_cases,
    load_manifest,
    summarize_baseline,
)


SCOPES = {
    "dev4": {
        "direct_manifest": Path(
            "configs/harness/qwen35_routing_dev4_direct_v1.yaml"
        ),
        "routed_manifest": Path("configs/harness/qwen35_routing_dev4_v1.yaml"),
        "four_b_direct": Path(
            "results/harness/qwen35-routing-dev4-direct-v1/4b/cases.jsonl"
        ),
        "four_b_routed": Path(
            "results/harness/qwen35-routing-dev4-v1/4b/cases.jsonl"
        ),
        "nine_b_direct": Path(
            "results/harness/qwen35-routing-dev4-direct-v1/9b/cases.jsonl"
        ),
    },
    "holdout4": {
        "direct_manifest": Path(
            "configs/harness/qwen35_routing_holdout4_direct_v1.yaml"
        ),
        "routed_manifest": Path(
            "configs/harness/qwen35_routing_holdout4_v1.yaml"
        ),
        "four_b_direct": Path(
            "results/harness/qwen35-routing-holdout4-direct-v1/4b/cases.jsonl"
        ),
        "four_b_routed": Path(
            "results/harness/qwen35-routing-holdout4-v1/4b/cases.jsonl"
        ),
        "nine_b_direct": Path(
            "results/harness/qwen35-routing-holdout4-direct-v1/9b/cases.jsonl"
        ),
    },
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


def latest_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[str(row["case_id"])] = row
    return rows


def cost(path: Path) -> dict[str, Any]:
    rows = latest_rows(path).values()
    return {
        "cases": len(rows),
        "correct": int(sum(float(row["score"]) for row in rows)),
        "total_tokens": sum(
            int(row.get("usage", {}).get("total_tokens", 0)) for row in rows
        ),
        "wall_seconds": sum(float(row["latency_seconds"]) for row in rows),
        "parse_failures": sum(row.get("prediction") is None for row in rows),
        "api_errors": sum(row.get("status") == "error" for row in rows),
        "draft_truncations": sum(
            row.get("stages", {}).get("draft", {}).get("finish_reason")
            == "length"
            for row in rows
        ),
        "verifier_truncations": sum(
            row.get("stages", {}).get("verifier", {}).get("finish_reason")
            == "length"
            for row in rows
        ),
    }


def failure_analysis(
    routed_path: Path,
    direct_path: Path,
) -> dict[str, Any]:
    routed = latest_rows(routed_path)
    direct = latest_rows(direct_path)
    by_benchmark: dict[str, dict[str, Any]] = {}
    for case_id, routed_row in routed.items():
        benchmark = str(routed_row["benchmark"])
        metrics = by_benchmark.setdefault(
            benchmark,
            {
                "cases": 0,
                "draft_truncations": 0,
                "routed_only_wins": [],
                "direct_only_wins": [],
            },
        )
        metrics["cases"] += 1
        metrics["draft_truncations"] += (
            routed_row.get("stages", {})
            .get("draft", {})
            .get("finish_reason")
            == "length"
        )
        direct_score = float(direct[case_id]["score"])
        routed_score = float(routed_row["score"])
        if routed_score == 1.0 and direct_score == 0.0:
            metrics["routed_only_wins"].append(case_id)
        elif routed_score == 0.0 and direct_score == 1.0:
            metrics["direct_only_wins"].append(case_id)
    return by_benchmark


def compact_comparison(value: dict[str, Any]) -> dict[str, Any]:
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
        "candidate_macro_accuracy": value["candidate_macro_accuracy"],
        "baseline_macro_accuracy": value["baseline_macro_accuracy"],
        "macro_delta": value["macro_delta"],
        "overall_micro": {key: value["overall_micro"][key] for key in fields},
        "benchmarks": {
            name: {key: metrics[key] for key in fields}
            for name, metrics in value["benchmarks"].items()
        },
        "bootstrap_samples": value["bootstrap_samples"],
        "bootstrap_seed": value["bootstrap_seed"],
    }


def audit_contract(
    manifest_path: Path,
    result_path: Path,
    dataset_root: Path,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    cases = {case.case_id: case for case in load_cases(manifest, dataset_root)}
    rows = latest_rows(result_path)
    if set(rows) != set(cases):
        raise SystemExit(
            f"result identities differ for {result_path}: "
            f"expected={len(cases)}, actual={len(rows)}"
        )
    failures = []
    selected_counts: dict[str, int] = {}
    for case_id, case in cases.items():
        row = rows[case_id]
        expected_strategy = _strategy_for_case(manifest, case)
        selected_counts[expected_strategy] = (
            selected_counts.get(expected_strategy, 0) + 1
        )
        if row.get("selected_strategy") != expected_strategy:
            failures.append(f"{case_id}: selected_strategy")
            continue
        stage_name = "direct" if expected_strategy == "direct" else "draft"
        expected_input = (
            case.prompt if expected_strategy == "direct" else case.draft_prompt
        )
        actual_sha = row.get("stages", {}).get(stage_name, {}).get("input_sha256")
        expected_sha = hashlib.sha256(expected_input.encode()).hexdigest()
        if actual_sha != expected_sha:
            failures.append(f"{case_id}: {stage_name}.input_sha256")
    if failures:
        raise SystemExit(f"execution contract audit failed: {failures[:5]}")
    return {
        "passed": True,
        "cases": len(cases),
        "selected_strategy_counts": selected_counts,
        "stage_input_hashes_match": True,
    }


def render_markdown(report: dict[str, Any]) -> str:
    routed = report["versus_9b"]
    direct = report["versus_4b_direct"]
    lines = [
        f"# Benchmark Routing {report['scope'].title()} Result",
        "",
        "## Result",
        "",
        "| Benchmark | 4B direct | 4B routed | 9B direct |",
        "| --- | ---: | ---: | ---: |",
    ]
    display_names = {
        "gsm8k": "GSM8K",
        "mmlu": "MMLU",
        "gpqa_diamond": "GPQA-Diamond",
    }
    for benchmark in ("gsm8k", "mmlu", "gpqa_diamond"):
        lines.append(
            f"| {display_names[benchmark]} | "
            f"{direct['benchmarks'][benchmark]['baseline_accuracy']:.4f} | "
            f"{direct['benchmarks'][benchmark]['candidate_accuracy']:.4f} | "
            f"{routed['benchmarks'][benchmark]['baseline_accuracy']:.4f} |"
        )
    lines.extend(
        [
            f"| Macro | {direct['baseline_macro_accuracy']:.4f} | "
            f"{direct['candidate_macro_accuracy']:.4f} | "
            f"{routed['baseline_macro_accuracy']:.4f} |",
            "",
            "Against 9B direct, routed 4B has paired micro delta "
            f"{routed['overall_micro']['delta']:+.4f}, 95% bootstrap CI "
            f"[{routed['overall_micro']['paired_bootstrap_95_ci'][0]:+.4f}, "
            f"{routed['overall_micro']['paired_bootstrap_95_ci'][1]:+.4f}], "
            "and exact McNemar "
            f"`p={routed['overall_micro']['mcnemar_exact_p']:.8f}`.",
            "",
            "## Contract Audit",
            "",
            "All case identities, selected routes, and actual direct/draft stage "
            "input hashes match the committed manifests. Raw outputs remain local "
            "and ignored.",
            "",
            "## Decision",
            "",
            report["decision"]["summary"],
            "",
            "## Failure Analysis",
            "",
            (
                "Relative to 4B direct, routed execution has "
                f"{len(report['failure_analysis']['gpqa_diamond']['routed_only_wins'])} "
                "GPQA-only win and "
                f"{len(report['failure_analysis']['mmlu']['direct_only_wins'])} "
                "MMLU direct-only wins. GPQA draft truncations are "
                f"{report['failure_analysis']['gpqa_diamond']['draft_truncations']}/"
                f"{report['failure_analysis']['gpqa_diamond']['cases']}; MMLU "
                "draft truncations are "
                f"{report['failure_analysis']['mmlu']['draft_truncations']}/"
                f"{report['failure_analysis']['mmlu']['cases']}."
            ),
            "",
            "The next fresh-slice hypothesis keeps GSM8K and MMLU direct and "
            "tests a larger reasoning draft only for GPQA. Holdout4 remains "
            "unread because dev4 failed its pre-registered promotion rule.",
            "",
            "## Reproduction Identity",
            "",
            f"- Code revision: `{report['code_revision']}`",
            f"- 4B direct raw SHA256: "
            f"`{report['artifacts']['four_b_direct_raw_sha256']}`",
            f"- 4B routed raw SHA256: "
            f"`{report['artifacts']['four_b_routed_raw_sha256']}`",
            f"- 9B direct raw SHA256: "
            f"`{report['artifacts']['nine_b_direct_raw_sha256']}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=sorted(SCOPES), required=True)
    parser.add_argument("--dataset-root", default="../../datasets")
    args = parser.parse_args()
    paths = SCOPES[args.scope]
    for label in ("four_b_direct", "four_b_routed", "nine_b_direct"):
        if not paths[label].is_file():
            raise SystemExit(f"missing result: {paths[label]}")

    versus_4b = compare_baselines(
        paths["four_b_routed"], paths["four_b_direct"]
    )
    versus_9b = compare_baselines(
        paths["four_b_routed"], paths["nine_b_direct"]
    )
    routed_cost = cost(paths["four_b_routed"])
    overall = versus_9b["overall_micro"]
    per_benchmark_non_regression = all(
        metrics["candidate_accuracy"] >= metrics["baseline_accuracy"]
        for metrics in versus_9b["benchmarks"].values()
    )
    if args.scope == "dev4":
        accepted = (
            versus_4b["macro_delta"] >= 0
            and not routed_cost["api_errors"]
            and not routed_cost["parse_failures"]
        )
        summary = (
            "Dev4 supports the frozen holdout4 run."
            if accepted
            else "Dev4 rejects the routed policy; holdout4 must not run."
        )
    else:
        accepted = (
            versus_9b["macro_delta"] > 0
            and per_benchmark_non_regression
            and overall["paired_bootstrap_95_ci"][0] > 0
            and overall["mcnemar_exact_p"] < 0.05
            and not routed_cost["api_errors"]
            and not routed_cost["parse_failures"]
        )
        summary = (
            "Holdout4 satisfies every pre-registered harness acceptance rule."
            if accepted
            else "Holdout4 does not satisfy every pre-registered acceptance rule."
        )

    report = {
        "schema_version": "nano_harness_public_benchmark_routing_v1",
        "scope": args.scope,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "code_revision": git_revision(),
        "summaries": {
            label: summarize_baseline(paths[label])
            for label in ("four_b_direct", "four_b_routed", "nine_b_direct")
        },
        "versus_4b_direct": compact_comparison(versus_4b),
        "versus_9b": compact_comparison(versus_9b),
        "costs": {
            label: cost(paths[label])
            for label in ("four_b_direct", "four_b_routed", "nine_b_direct")
        },
        "failure_analysis": failure_analysis(
            paths["four_b_routed"],
            paths["four_b_direct"],
        ),
        "contract_audits": {
            "four_b_direct": audit_contract(
                paths["direct_manifest"],
                paths["four_b_direct"],
                Path(args.dataset_root),
            ),
            "four_b_routed": audit_contract(
                paths["routed_manifest"],
                paths["four_b_routed"],
                Path(args.dataset_root),
            ),
            "nine_b_direct": audit_contract(
                paths["direct_manifest"],
                paths["nine_b_direct"],
                Path(args.dataset_root),
            ),
        },
        "artifacts": {
            f"{label}_raw_sha256": sha256_file(paths[label])
            for label in ("four_b_direct", "four_b_routed", "nine_b_direct")
        },
        "decision": {
            "accepted": accepted,
            "summary": summary,
            "routed_vs_4b_macro_delta": versus_4b["macro_delta"],
            "routed_vs_9b_macro_delta": versus_9b["macro_delta"],
            "per_benchmark_non_regression_vs_9b": per_benchmark_non_regression,
            "paired_micro_lower_bound_above_zero": (
                overall["paired_bootstrap_95_ci"][0] > 0
            ),
            "mcnemar_below_005": overall["mcnemar_exact_p"] < 0.05,
            "no_api_errors": not routed_cost["api_errors"],
            "no_parse_failures": not routed_cost["parse_failures"],
        },
    }
    stem = f"benchmark_routing_{args.scope}_v1"
    json_path = Path(f"docs/results/{stem}.public.json")
    markdown_path = Path(f"docs/results/{stem}.md")
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "scope": args.scope,
                "accepted": accepted,
                "routed_vs_4b_macro_delta": versus_4b["macro_delta"],
                "routed_vs_9b_macro_delta": versus_9b["macro_delta"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
