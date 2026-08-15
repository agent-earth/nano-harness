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
    "dev_direct": Path("results/harness/qwen35-dev-direct-v1/4b/cases.jsonl"),
    "dev_treatment": Path(
        "results/harness/qwen35-dev-draft-verify-v1/4b/cases.jsonl"
    ),
    "eval_treatment": Path(
        "results/harness/qwen35-eval-draft-verify-v1/4b/cases.jsonl"
    ),
    "eval_4b_direct": Path("results/baselines/qwen35-local-v5/4b/cases.jsonl"),
    "eval_9b_direct": Path("results/baselines/qwen35-local-v5/9b/cases.jsonl"),
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


def cost(path: Path) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {
        "cases": len(rows),
        "correct": int(sum(row["score"] for row in rows)),
        "prompt_tokens": sum(row["usage"].get("prompt_tokens", 0) for row in rows),
        "completion_tokens": sum(
            row["usage"].get("completion_tokens", 0) for row in rows
        ),
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


def render_markdown(report: dict[str, Any]) -> str:
    dev = report["dev_comparison"]
    versus_9b = report["fixed_eval"]["versus_9b"]
    versus_4b = report["fixed_eval"]["versus_4b_direct"]
    ci = versus_9b["overall_micro"]["paired_bootstrap_95_ci"]
    return f"""# Draft-Verify v1 Result

## Decision

The treatment is retained as a promising harness component, but it does not
satisfy the harness-stage acceptance criterion. On the fixed 72-case suite,
4B draft-verify scores {versus_9b['candidate_macro_accuracy']:.4f} versus the
9B baseline at {versus_9b['baseline_macro_accuracy']:.4f}, a
{versus_9b['macro_delta']:+.4f} macro delta. The paired micro 95% bootstrap
interval is [{ci[0]:+.4f}, {ci[1]:+.4f}], so the lead is not significant.

## Disjoint Development Slice

- Direct 4B macro: {dev['baseline_macro_accuracy']:.4f}
- Draft-verify 4B macro: {dev['candidate_macro_accuracy']:.4f}
- Paired delta: {dev['macro_delta']:+.4f}
- Paired counts:
  `{dev['overall_micro']['paired_counts']}`
- Parse failures: {report['costs']['dev_direct']['parse_failures']} direct to
  {report['costs']['dev_treatment']['parse_failures']} treatment
- Tokens: {report['costs']['dev_direct']['total_tokens']} direct to
  {report['costs']['dev_treatment']['total_tokens']} treatment

The 18 development cases are disjoint from the fixed 72 evaluation cases.

## Fixed Evaluation

| Benchmark | 4B direct | 4B draft-verify | 9B direct |
| --- | ---: | ---: | ---: |
| GSM8K | {versus_4b['benchmarks']['gsm8k']['baseline_accuracy']:.4f} | {versus_4b['benchmarks']['gsm8k']['candidate_accuracy']:.4f} | {versus_9b['benchmarks']['gsm8k']['baseline_accuracy']:.4f} |
| MMLU | {versus_4b['benchmarks']['mmlu']['baseline_accuracy']:.4f} | {versus_4b['benchmarks']['mmlu']['candidate_accuracy']:.4f} | {versus_9b['benchmarks']['mmlu']['baseline_accuracy']:.4f} |
| GPQA-Diamond | {versus_4b['benchmarks']['gpqa_diamond']['baseline_accuracy']:.4f} | {versus_4b['benchmarks']['gpqa_diamond']['candidate_accuracy']:.4f} | {versus_9b['benchmarks']['gpqa_diamond']['baseline_accuracy']:.4f} |
| Macro | {versus_4b['baseline_macro_accuracy']:.4f} | {versus_4b['candidate_macro_accuracy']:.4f} | {versus_9b['baseline_macro_accuracy']:.4f} |

Draft-verify improves GSM8K and MMLU by two cases each versus 4B direct, but
loses two GPQA cases. It uses about twice the tokens while reducing wall-clock
time in this single-sequence local setup. The next iteration should preserve
the verifier for math/knowledge tasks and test a GPQA-specific repair on a new
development slice, not tune again on the observed fixed evaluation cases.

## Reproduction Identity

- Code revision: `{report['code_revision']}`
- Dev control manifest SHA256: `{report['artifacts']['dev_direct_manifest_sha256']}`
- Dev treatment manifest SHA256: `{report['artifacts']['dev_treatment_manifest_sha256']}`
- Eval treatment manifest SHA256: `{report['artifacts']['eval_treatment_manifest_sha256']}`
- Dev direct raw SHA256: `{report['artifacts']['dev_direct_raw_sha256']}`
- Dev treatment raw SHA256: `{report['artifacts']['dev_treatment_raw_sha256']}`
- Eval treatment raw SHA256: `{report['artifacts']['eval_treatment_raw_sha256']}`

Raw outputs remain local and ignored. The public JSON contains metrics,
artifact digests, and case IDs only.
"""


def main() -> None:
    for path in PATHS.values():
        if not path.is_file():
            raise SystemExit(f"missing required result: {path}")

    dev_comparison = compare_baselines(
        PATHS["dev_treatment"],
        PATHS["dev_direct"],
    )
    versus_4b = compare_baselines(
        PATHS["eval_treatment"],
        PATHS["eval_4b_direct"],
    )
    versus_9b = compare_baselines(
        PATHS["eval_treatment"],
        PATHS["eval_9b_direct"],
    )
    report = {
        "schema_version": "nano_harness_public_harness_iteration_v1",
        "iteration_id": "draft-verify-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "code_revision": git_revision(),
        "hypothesis": (
            "Separate bounded candidate reasoning from a strict final verifier "
            "to improve correctness and format reliability."
        ),
        "dev_comparison": compact_comparison(dev_comparison),
        "fixed_eval": {
            "treatment_summary": summarize_baseline(PATHS["eval_treatment"]),
            "versus_4b_direct": compact_comparison(versus_4b),
            "versus_9b": compact_comparison(versus_9b),
        },
        "costs": {
            label: cost(path)
            for label, path in PATHS.items()
        },
        "artifacts": {
            "dev_direct_manifest_sha256": sha256_file(
                Path("configs/harness/qwen35_dev_direct_v1.yaml")
            ),
            "dev_treatment_manifest_sha256": sha256_file(
                Path("configs/harness/qwen35_dev_draft_verify_v1.yaml")
            ),
            "eval_treatment_manifest_sha256": sha256_file(
                Path("configs/harness/qwen35_eval_draft_verify_v1.yaml")
            ),
            "dev_direct_raw_sha256": sha256_file(PATHS["dev_direct"]),
            "dev_treatment_raw_sha256": sha256_file(PATHS["dev_treatment"]),
            "eval_treatment_raw_sha256": sha256_file(PATHS["eval_treatment"]),
        },
        "decision": {
            "promising_component": True,
            "harness_acceptance_satisfied": False,
            "reasons": [
                "Fixed-suite 4B macro exceeds 9B by one case.",
                "The paired confidence interval includes zero.",
                "GPQA regresses by two cases versus 4B direct.",
            ],
            "next_experiment": "GPQA repair on a fresh disjoint dev2 slice",
        },
    }
    output_json = Path("docs/results/draft_verify_v1.public.json")
    output_markdown = Path("docs/results/draft_verify_v1.md")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_markdown.write_text(render_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "iteration_id": report["iteration_id"],
                "dev_macro_delta": dev_comparison["macro_delta"],
                "eval_vs_9b_macro_delta": versus_9b["macro_delta"],
                "harness_acceptance_satisfied": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
