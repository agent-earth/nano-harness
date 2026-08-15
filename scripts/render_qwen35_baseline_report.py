#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nano_harness.baseline import compare_baselines, summarize_baseline


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


def parse_failure_categories(path: Path) -> dict[str, int]:
    counts = {
        "parsed": 0,
        "length_truncation": 0,
        "missing_final_line": 0,
        "malformed_final_line": 0,
    }
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("prediction") is not None:
                counts["parsed"] += 1
            elif record.get("finish_reason") == "length":
                counts["length_truncation"] += 1
            elif "FINAL:" not in str(record.get("output", "")).upper():
                counts["missing_final_line"] += 1
            else:
                counts["malformed_final_line"] += 1
    return counts


def compact_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "total_cases": summary["total_cases"],
        "completed_cases": summary["completed_cases"],
        "error_cases": summary["error_cases"],
        "macro_accuracy": summary["macro_accuracy"],
        "benchmarks": summary["benchmarks"],
    }


def compact_comparison(comparison: dict[str, Any]) -> dict[str, Any]:
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
        "candidate_model": comparison["candidate_model"],
        "baseline_model": comparison["baseline_model"],
        "cases": comparison["cases"],
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


def render_markdown(report: dict[str, Any]) -> str:
    candidate = report["models"]["qwen3.5-4b"]
    baseline = report["models"]["qwen3.5-9b"]
    comparison = report["comparison"]
    rows = []
    for name in ("gsm8k", "mmlu", "gpqa_diamond"):
        candidate_metric = candidate["benchmarks"][name]
        baseline_metric = baseline["benchmarks"][name]
        delta = comparison["benchmarks"][name]["delta"]
        rows.append(
            f"| {name} | {candidate_metric['accuracy']:.4f} "
            f"({candidate_metric['correct']}/24) | "
            f"{baseline_metric['accuracy']:.4f} "
            f"({baseline_metric['correct']}/24) | {delta:+.4f} |"
        )

    micro = comparison["overall_micro"]
    ci = micro["paired_bootstrap_95_ci"]
    return (
        "# Qwen3.5 Local Baseline v5\n\n"
        "## Result\n\n"
        "| Benchmark | Qwen3.5-4B | Qwen3.5-9B | 4B - 9B |\n"
        "| --- | ---: | ---: | ---: |\n"
        + "\n".join(rows)
        + "\n"
        f"| Macro average | {candidate['macro_accuracy']:.4f} | "
        f"{baseline['macro_accuracy']:.4f} | "
        f"{comparison['macro_delta']:+.4f} |\n\n"
        f"Across all 72 paired cases, 4B scored {micro['candidate_accuracy']:.4f} "
        f"and 9B scored {micro['baseline_accuracy']:.4f}. The paired delta is "
        f"{micro['delta']:+.4f}, with a fixed-seed 95% bootstrap interval "
        f"[{ci[0]:+.4f}, {ci[1]:+.4f}] and exact McNemar "
        f"`p={micro['mcnemar_exact_p']:.4f}`. This baseline does not establish "
        "a significant model-quality difference.\n\n"
        "## Failure Evidence\n\n"
        f"- 4B parse categories: `{report['failure_categories']['qwen3.5-4b']}`.\n"
        f"- 9B parse categories: `{report['failure_categories']['qwen3.5-9b']}`.\n"
        "- v1 used a 256-token reasoning budget and truncated every GPQA output.\n"
        "- v2 used a 600-token reasoning budget but still truncated 39/48 GPQA outputs.\n"
        "- v3 answer-only removed truncation but reduced GSM8K accuracy sharply.\n"
        "- v4 used one global 600-token budget; it allowed answer-only contract drift.\n"
        "- v5 uses a 600-token reasoning budget for GSM8K and a 32-token "
        "answer-only budget for MMLU and GPQA, identically for both models.\n\n"
        "## Reproduction Identity\n\n"
        f"- Code revision: `{report['code_revision']}`\n"
        f"- Suite manifest SHA256: `{report['artifacts']['suite_manifest_sha256']}`\n"
        f"- Case manifest SHA256: `{report['artifacts']['case_manifest_sha256']}`\n"
        f"- 4B raw result SHA256: `{report['artifacts']['qwen3.5-4b_raw_sha256']}`\n"
        f"- 9B raw result SHA256: `{report['artifacts']['qwen3.5-9b_raw_sha256']}`\n\n"
        "Raw case outputs remain local and ignored. The public JSON report includes "
        "aggregate metrics and case IDs for paired failures, not task bodies or model "
        "response text.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results/baselines/qwen35-local-v5")
    parser.add_argument(
        "--manifest",
        default="configs/baselines/qwen35_local_v5.yaml",
    )
    parser.add_argument(
        "--case-manifest",
        default="configs/generated/qwen35_local_v5_cases.json",
    )
    parser.add_argument(
        "--json-output",
        default="docs/results/qwen35_local_v5.public.json",
    )
    parser.add_argument(
        "--markdown-output",
        default="docs/results/qwen35_local_v5.md",
    )
    args = parser.parse_args()

    root = Path(args.results_root)
    candidate_path = root / "4b" / "cases.jsonl"
    baseline_path = root / "9b" / "cases.jsonl"
    manifest_path = Path(args.manifest)
    case_manifest_path = Path(args.case_manifest)
    for path in (
        candidate_path,
        baseline_path,
        manifest_path,
        case_manifest_path,
    ):
        if not path.is_file():
            raise SystemExit(f"missing required artifact: {path}")

    candidate_summary = summarize_baseline(candidate_path)
    baseline_summary = summarize_baseline(baseline_path)
    if candidate_summary["total_cases"] != 72 or baseline_summary["total_cases"] != 72:
        raise SystemExit("both model runs must contain exactly 72 unique cases")
    if candidate_summary["error_cases"] or baseline_summary["error_cases"]:
        raise SystemExit("model API errors must be resolved before publication")

    comparison = compare_baselines(candidate_path, baseline_path)
    report = {
        "schema_version": "nano_harness_public_baseline_report_v1",
        "suite_id": "qwen35-local-v5",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "code_revision": git_revision(),
        "models": {
            "qwen3.5-4b": compact_summary(candidate_summary),
            "qwen3.5-9b": compact_summary(baseline_summary),
        },
        "comparison": compact_comparison(comparison),
        "failure_categories": {
            "qwen3.5-4b": parse_failure_categories(candidate_path),
            "qwen3.5-9b": parse_failure_categories(baseline_path),
        },
        "runtime": {
            "vllm": "0.19.1",
            "torch": "2.10.0+cu128",
            "transformers": "5.12.1",
            "gpu": "Tesla V100-SXM2-32GB",
            "dtype": "float16",
            "max_model_len": 1024,
            "gpu_memory_utilization": 0.85,
            "max_num_batched_tokens": 1024,
            "max_num_seqs": 1,
            "enforce_eager": True,
            "temperature": 0.0,
            "enable_thinking": False,
        },
        "artifacts": {
            "suite_manifest_sha256": sha256_file(manifest_path),
            "case_manifest_sha256": sha256_file(case_manifest_path),
            "qwen3.5-4b_raw_sha256": sha256_file(candidate_path),
            "qwen3.5-9b_raw_sha256": sha256_file(baseline_path),
        },
        "claim": {
            "significant_difference_established": False,
            "reason": (
                "The paired 95% bootstrap interval includes zero and exact "
                "McNemar p-value is not significant."
            ),
            "next_treatment": "harness-only iteration on the fixed v5 suite",
        },
    }

    json_output = Path(args.json_output)
    markdown_output = Path(args.markdown_output)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_output.write_text(render_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "suite_id": report["suite_id"],
                "candidate_macro_accuracy": comparison["candidate_macro_accuracy"],
                "baseline_macro_accuracy": comparison["baseline_macro_accuracy"],
                "macro_delta": comparison["macro_delta"],
                "json_output": str(json_output),
                "markdown_output": str(markdown_output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
