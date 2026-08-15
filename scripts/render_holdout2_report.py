#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nano_harness.baseline import compare_baselines, summarize_baseline


PATHS = {
    "four_b_direct": Path(
        "results/harness/qwen35-holdout2-direct-v1/4b/cases.jsonl"
    ),
    "four_b_treatment": Path(
        "results/harness/qwen35-holdout2-draft-verify-v1/4b/cases.jsonl"
    ),
    "nine_b_direct": Path(
        "results/harness/qwen35-holdout2-direct-v1/9b/cases.jsonl"
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


def compact_comparison(value: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "cases",
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


def cost(path: Path) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {
        "cases": len(rows),
        "correct": int(sum(row["score"] for row in rows)),
        "total_tokens": sum(row["usage"].get("total_tokens", 0) for row in rows),
        "wall_seconds": sum(row["latency_seconds"] for row in rows),
        "parse_failures": sum(row.get("prediction") is None for row in rows),
        "api_errors": sum(row.get("status") == "error" for row in rows),
        "draft_truncations": sum(
            row.get("stages", {}).get("draft", {}).get("finish_reason") == "length"
            for row in rows
        ),
        "verifier_truncations": sum(
            row.get("stages", {}).get("verifier", {}).get("finish_reason")
            == "length"
            for row in rows
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    versus_9b = report["versus_9b"]
    versus_4b = report["versus_4b_direct"]
    ci = versus_9b["overall_micro"]["paired_bootstrap_95_ci"]
    return f"""# Draft-Verify Holdout2 Result

## Primary Result

On the pre-registered 72-case holdout2, 4B draft-verify scores
{versus_9b['candidate_macro_accuracy']:.4f} versus 9B direct at
{versus_9b['baseline_macro_accuracy']:.4f}. The paired micro delta is
{versus_9b['overall_micro']['delta']:+.4f}, 95% bootstrap CI
[{ci[0]:+.4f}, {ci[1]:+.4f}], exact McNemar
`p={versus_9b['overall_micro']['mcnemar_exact_p']:.10f}`.

| Benchmark | 4B direct | 4B draft-verify | 9B direct |
| --- | ---: | ---: | ---: |
| GSM8K | {versus_4b['benchmarks']['gsm8k']['baseline_accuracy']:.4f} | {versus_4b['benchmarks']['gsm8k']['candidate_accuracy']:.4f} | {versus_9b['benchmarks']['gsm8k']['baseline_accuracy']:.4f} |
| MMLU | {versus_4b['benchmarks']['mmlu']['baseline_accuracy']:.4f} | {versus_4b['benchmarks']['mmlu']['candidate_accuracy']:.4f} | {versus_9b['benchmarks']['mmlu']['baseline_accuracy']:.4f} |
| GPQA-Diamond | {versus_4b['benchmarks']['gpqa_diamond']['baseline_accuracy']:.4f} | {versus_4b['benchmarks']['gpqa_diamond']['candidate_accuracy']:.4f} | {versus_9b['benchmarks']['gpqa_diamond']['baseline_accuracy']:.4f} |
| Macro | {versus_4b['baseline_macro_accuracy']:.4f} | {versus_4b['candidate_macro_accuracy']:.4f} | {versus_9b['baseline_macro_accuracy']:.4f} |

The overall lead is large and significant. MMLU and GPQA improve
significantly, and treatment final parse failures are zero.

## Acceptance Decision

Harness-stage acceptance is not yet satisfied because the pre-registered
task-group non-regression criterion fails on GSM8K: 4B draft-verify scores
22/24 while 9B direct scores 23/24. The observed -1 case delta has a bootstrap
interval including zero, but the criterion cannot be relaxed after seeing
holdout2.

The strategy remains frozen. The next experiment is a larger unseen
GSM8K-only confirmation, not another policy change.

## Cost

- 4B direct: {report['costs']['four_b_direct']['total_tokens']} tokens,
  {report['costs']['four_b_direct']['wall_seconds']:.1f}s.
- 4B draft-verify: {report['costs']['four_b_treatment']['total_tokens']} tokens,
  {report['costs']['four_b_treatment']['wall_seconds']:.1f}s.
- 9B direct: {report['costs']['nine_b_direct']['total_tokens']} tokens,
  {report['costs']['nine_b_direct']['wall_seconds']:.1f}s.

## Reproduction Identity

- Code revision: `{report['code_revision']}`
- 4B direct raw SHA256: `{report['artifacts']['four_b_direct_raw_sha256']}`
- 4B treatment raw SHA256: `{report['artifacts']['four_b_treatment_raw_sha256']}`
- 9B direct raw SHA256: `{report['artifacts']['nine_b_direct_raw_sha256']}`

Raw outputs remain local and ignored.
"""


def main() -> None:
    for path in PATHS.values():
        if not path.is_file():
            raise SystemExit(f"missing required result: {path}")

    versus_4b = compare_baselines(
        PATHS["four_b_treatment"], PATHS["four_b_direct"]
    )
    versus_9b = compare_baselines(
        PATHS["four_b_treatment"], PATHS["nine_b_direct"]
    )
    gsm8k = versus_9b["benchmarks"]["gsm8k"]
    acceptance = (
        versus_9b["overall_micro"]["paired_bootstrap_95_ci"][0] > 0
        and versus_9b["overall_micro"]["mcnemar_exact_p"] < 0.05
        and all(
            metrics["candidate_accuracy"] >= metrics["baseline_accuracy"]
            for metrics in versus_9b["benchmarks"].values()
        )
        and not cost(PATHS["four_b_treatment"])["api_errors"]
        and not cost(PATHS["four_b_treatment"])["parse_failures"]
    )
    report = {
        "schema_version": "nano_harness_public_harness_holdout_v1",
        "holdout_id": "qwen35-holdout2-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "code_revision": git_revision(),
        "summaries": {
            label: summarize_baseline(path) for label, path in PATHS.items()
        },
        "versus_4b_direct": compact_comparison(versus_4b),
        "versus_9b": compact_comparison(versus_9b),
        "costs": {label: cost(path) for label, path in PATHS.items()},
        "artifacts": {
            f"{label}_raw_sha256": sha256_file(path)
            for label, path in PATHS.items()
        },
        "decision": {
            "overall_significant_win": (
                versus_9b["overall_micro"]["paired_bootstrap_95_ci"][0] > 0
                and versus_9b["overall_micro"]["mcnemar_exact_p"] < 0.05
            ),
            "task_group_non_regression": all(
                metrics["candidate_accuracy"] >= metrics["baseline_accuracy"]
                for metrics in versus_9b["benchmarks"].values()
            ),
            "harness_acceptance_satisfied": acceptance,
            "gsm8k_delta": gsm8k["delta"],
            "next_experiment": "Larger unseen GSM8K-only confirmation.",
            "policy_frozen": True,
        },
    }
    json_path = Path("docs/results/holdout2_v1.public.json")
    markdown_path = Path("docs/results/holdout2_v1.md")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "holdout_id": report["holdout_id"],
                "overall_delta": versus_9b["overall_micro"]["delta"],
                "overall_p": versus_9b["overall_micro"]["mcnemar_exact_p"],
                "gsm8k_delta": gsm8k["delta"],
                "harness_acceptance_satisfied": acceptance,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
