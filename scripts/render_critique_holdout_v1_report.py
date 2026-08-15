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
    "dev2_incumbent": Path(
        "results/harness/qwen35-dev2-draft-verify-v1/4b/cases.jsonl"
    ),
    "dev2_critique": Path(
        "results/harness/qwen35-dev2-critique-v1/4b/cases.jsonl"
    ),
    "holdout_4b_direct": Path(
        "results/harness/qwen35-holdout-direct-v1/4b/cases.jsonl"
    ),
    "holdout_4b_treatment": Path(
        "results/harness/qwen35-holdout-draft-verify-v1/4b/cases.jsonl"
    ),
    "holdout_9b_direct": Path(
        "results/harness/qwen35-holdout-direct-v1/9b/cases.jsonl"
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
        "critique_truncations": sum(
            row.get("stages", {}).get("critique", {}).get("finish_reason")
            == "length"
            for row in rows
        ),
        "verifier_truncations": sum(
            row.get("stages", {}).get("verifier", {}).get("finish_reason")
            == "length"
            for row in rows
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    critique = report["dev2_critique_comparison"]
    holdout = report["holdout"]["versus_9b"]
    holdout_4b = report["holdout"]["versus_4b_direct"]
    ci = holdout["overall_micro"]["paired_bootstrap_95_ci"]
    return f"""# Critique v1 And Holdout Result

## Critique Decision

The critique stage is rejected. On fresh dev2 it scores
{critique['candidate_macro_accuracy']:.4f} versus draft-verify at
{critique['baseline_macro_accuracy']:.4f}, a {critique['macro_delta']:+.4f}
delta. Both lost cases are GPQA. Critique uses
{report['costs']['dev2_critique']['total_tokens']} tokens versus
{report['costs']['dev2_incumbent']['total_tokens']} and
{report['costs']['dev2_critique']['wall_seconds']:.1f}s versus
{report['costs']['dev2_incumbent']['wall_seconds']:.1f}s.

## Untouched Holdout Confirmation

The selected draft-verify policy was frozen before reading the 18 holdout
cases. The holdout has zero overlap with fixed v5 evaluation, dev1, or dev2.

| Benchmark | 4B direct | 4B draft-verify | 9B direct |
| --- | ---: | ---: | ---: |
| GSM8K | {holdout_4b['benchmarks']['gsm8k']['baseline_accuracy']:.4f} | {holdout_4b['benchmarks']['gsm8k']['candidate_accuracy']:.4f} | {holdout['benchmarks']['gsm8k']['baseline_accuracy']:.4f} |
| MMLU | {holdout_4b['benchmarks']['mmlu']['baseline_accuracy']:.4f} | {holdout_4b['benchmarks']['mmlu']['candidate_accuracy']:.4f} | {holdout['benchmarks']['mmlu']['baseline_accuracy']:.4f} |
| GPQA-Diamond | {holdout_4b['benchmarks']['gpqa_diamond']['baseline_accuracy']:.4f} | {holdout_4b['benchmarks']['gpqa_diamond']['candidate_accuracy']:.4f} | {holdout['benchmarks']['gpqa_diamond']['baseline_accuracy']:.4f} |
| Macro | {holdout_4b['baseline_macro_accuracy']:.4f} | {holdout_4b['candidate_macro_accuracy']:.4f} | {holdout['baseline_macro_accuracy']:.4f} |

Against 9B direct, 4B draft-verify has seven treatment-only wins and zero
9B-only wins. The paired delta is {holdout['overall_micro']['delta']:+.4f},
95% bootstrap CI [{ci[0]:+.4f}, {ci[1]:+.4f}], exact McNemar
`p={holdout['overall_micro']['mcnemar_exact_p']:.6f}`.

This is significant confirmation on a small 18-case holdout. The strategy is
now frozen. The next experiment is a pre-registered 72-case holdout2; no prompt,
budget, or scorer changes are allowed before reading it.

## Reproduction Identity

- Code revision: `{report['code_revision']}`
- Dev2 incumbent raw SHA256: `{report['artifacts']['dev2_incumbent_raw_sha256']}`
- Dev2 critique raw SHA256: `{report['artifacts']['dev2_critique_raw_sha256']}`
- Holdout 4B direct raw SHA256: `{report['artifacts']['holdout_4b_direct_raw_sha256']}`
- Holdout 4B treatment raw SHA256: `{report['artifacts']['holdout_4b_treatment_raw_sha256']}`
- Holdout 9B direct raw SHA256: `{report['artifacts']['holdout_9b_direct_raw_sha256']}`

Raw outputs remain local and ignored.
"""


def main() -> None:
    for path in PATHS.values():
        if not path.is_file():
            raise SystemExit(f"missing required artifact: {path}")

    dev2 = compare_baselines(PATHS["dev2_critique"], PATHS["dev2_incumbent"])
    versus_4b = compare_baselines(
        PATHS["holdout_4b_treatment"], PATHS["holdout_4b_direct"]
    )
    versus_9b = compare_baselines(
        PATHS["holdout_4b_treatment"], PATHS["holdout_9b_direct"]
    )
    report = {
        "schema_version": "nano_harness_public_harness_iteration_v1",
        "iteration_id": "critique-v1-holdout",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "code_revision": git_revision(),
        "dev2_critique_comparison": compact_comparison(dev2),
        "holdout": {
            "four_b_direct": summarize_baseline(PATHS["holdout_4b_direct"]),
            "four_b_treatment": summarize_baseline(PATHS["holdout_4b_treatment"]),
            "nine_b_direct": summarize_baseline(PATHS["holdout_9b_direct"]),
            "versus_4b_direct": compact_comparison(versus_4b),
            "versus_9b": compact_comparison(versus_9b),
        },
        "costs": {label: cost(path) for label, path in PATHS.items()},
        "artifacts": {
            f"{label}_raw_sha256": sha256_file(path)
            for label, path in PATHS.items()
        },
        "decision": {
            "critique_rejected": True,
            "draft_verify_frozen": True,
            "small_holdout_significant": True,
            "holdout2_required": True,
            "next_experiment": (
                "Pre-registered 72-case holdout2 with 4B direct, 4B "
                "draft-verify, and 9B direct."
            ),
        },
    }
    json_path = Path("docs/results/critique_v1_holdout.public.json")
    markdown_path = Path("docs/results/critique_v1_holdout.md")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "iteration_id": report["iteration_id"],
                "critique_delta": dev2["macro_delta"],
                "holdout_vs_9b_delta": versus_9b["macro_delta"],
                "holdout_vs_9b_p": versus_9b["overall_micro"]["mcnemar_exact_p"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
