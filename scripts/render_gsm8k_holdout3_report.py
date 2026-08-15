#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from nano_harness.baseline import compare_baselines, summarize_baseline


PATHS = {
    "draft": Path(
        "results/harness/qwen35-gsm8k-holdout3-draft-verify-v1/4b/cases.jsonl"
    ),
    "dual": Path(
        "results/harness/qwen35-gsm8k-holdout3-dual-solve-v1/4b/cases.jsonl"
    ),
    "nine_b": Path(
        "results/harness/qwen35-gsm8k-holdout3-direct-v1/9b/cases.jsonl"
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


def cost(path: Path) -> dict:
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
        "second_solve_truncations": sum(
            row.get("stages", {}).get("second_solve", {}).get("finish_reason")
            == "length"
            for row in rows
        ),
        "verifier_truncations": sum(
            row.get("stages", {}).get("verifier", {}).get("finish_reason")
            == "length"
            for row in rows
        ),
    }


def compact(value: dict) -> dict:
    overall = value["overall_micro"]
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
        "bootstrap_samples": value["bootstrap_samples"],
        "bootstrap_seed": value["bootstrap_seed"],
    }


def main() -> None:
    for path in PATHS.values():
        if not path.is_file():
            raise SystemExit(f"missing result: {path}")
    versus_draft = compare_baselines(PATHS["dual"], PATHS["draft"])
    versus_9b = compare_baselines(PATHS["dual"], PATHS["nine_b"])
    overall_9b = versus_9b["overall_micro"]
    dual_cost = cost(PATHS["dual"])
    acceptance = (
        overall_9b["candidate_accuracy"] >= overall_9b["baseline_accuracy"]
        and overall_9b["paired_bootstrap_95_ci"][0] > -0.05
        and not dual_cost["api_errors"]
        and not dual_cost["parse_failures"]
    )
    report = {
        "schema_version": "nano_harness_public_gsm8k_holdout_v1",
        "holdout_id": "qwen35-gsm8k-holdout3-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "code_revision": git_revision(),
        "summaries": {
            label: summarize_baseline(path) for label, path in PATHS.items()
        },
        "versus_draft_verify": compact(versus_draft),
        "versus_9b": compact(versus_9b),
        "costs": {label: cost(path) for label, path in PATHS.items()},
        "artifacts": {
            f"{label}_raw_sha256": sha256_file(path)
            for label, path in PATHS.items()
        },
        "decision": {
            "improves_over_draft_verify": (
                versus_draft["overall_micro"]["paired_bootstrap_95_ci"][0] > 0
            ),
            "point_at_least_9b": (
                overall_9b["candidate_accuracy"] >= overall_9b["baseline_accuracy"]
            ),
            "non_inferiority_lower_bound_above_minus_005": (
                overall_9b["paired_bootstrap_95_ci"][0] > -0.05
            ),
            "holdout_acceptance_satisfied": acceptance,
            "next_experiment": (
                "Benchmark-aware routing on a fresh slice: direct math path and "
                "draft-verify knowledge/science path."
            ),
        },
    }
    draft = report["versus_draft_verify"]
    nine_b = report["versus_9b"]
    markdown = f"""# GSM8K Dual-Solve Holdout3 Result

## Result

On 96 unseen GSM8K cases:

- 4B draft-verify: {draft['baseline_accuracy']:.4f};
- 4B dual-solve: {draft['candidate_accuracy']:.4f};
- 9B direct: {nine_b['baseline_accuracy']:.4f}.

Dual-solve improves over draft-verify by {draft['delta']:+.4f}, with 95%
bootstrap CI [{draft['paired_bootstrap_95_ci'][0]:+.4f},
{draft['paired_bootstrap_95_ci'][1]:+.4f}]. It has
{draft['paired_counts']['candidate_only']} dual-only wins and
{draft['paired_counts']['baseline_only']} draft-only wins.

Against 9B, dual-solve is {nine_b['delta']:+.4f}, 95% CI
[{nine_b['paired_bootstrap_95_ci'][0]:+.4f},
{nine_b['paired_bootstrap_95_ci'][1]:+.4f}]. It scores 92/96 versus 94/96.

## Decision

Dual-solve is retained as evidence that independent re-solving repairs some
draft errors, but holdout acceptance fails:

- its point estimate remains below 9B;
- the CI lower bound (-0.0521) is slightly below the -0.05 non-inferiority
  margin;
- it uses {report['costs']['dual']['total_tokens']} tokens versus
  {report['costs']['nine_b']['total_tokens']} for 9B direct.

The next experiment must use a fresh slice. Benchmark-aware routing is the next
falsifiable direction: preserve direct/reasoning behavior for math while using
draft-verify only where it has repeatedly improved MMLU and GPQA. No holdout3
tuning is allowed.

## Reproduction Identity

- Code revision: `{report['code_revision']}`
- Draft raw SHA256: `{report['artifacts']['draft_raw_sha256']}`
- Dual raw SHA256: `{report['artifacts']['dual_raw_sha256']}`
- 9B raw SHA256: `{report['artifacts']['nine_b_raw_sha256']}`

Raw stage texts remain local and ignored.
"""
    json_path = Path("docs/results/gsm8k_holdout3_v1.public.json")
    markdown_path = Path("docs/results/gsm8k_holdout3_v1.md")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(markdown, encoding="utf-8")
    print(
        json.dumps(
            {
                "holdout_id": report["holdout_id"],
                "dual_vs_draft_delta": draft["delta"],
                "dual_vs_9b_delta": nine_b["delta"],
                "acceptance": acceptance,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
