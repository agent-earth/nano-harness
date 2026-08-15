#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from nano_harness.baseline import compare_baselines, summarize_baseline


CANDIDATE = Path(
    "results/harness/qwen35-gsm8k-confirm-draft-verify-v1/4b/cases.jsonl"
)
BASELINE = Path("results/harness/qwen35-gsm8k-confirm-direct-v1/9b/cases.jsonl")


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
        "verifier_truncations": sum(
            row.get("stages", {}).get("verifier", {}).get("finish_reason")
            == "length"
            for row in rows
        ),
    }


def main() -> None:
    if not CANDIDATE.is_file() or not BASELINE.is_file():
        raise SystemExit("confirmation results are incomplete")
    comparison = compare_baselines(CANDIDATE, BASELINE)
    overall = comparison["overall_micro"]
    candidate_cost = cost(CANDIDATE)
    baseline_cost = cost(BASELINE)
    candidate_rows = {
        row["case_id"]: row
        for row in map(json.loads, CANDIDATE.read_text(encoding="utf-8").splitlines())
    }
    baseline_rows = {
        row["case_id"]: row
        for row in map(json.loads, BASELINE.read_text(encoding="utf-8").splitlines())
    }
    baseline_only = overall["baseline_only_cases"]
    failure_mechanisms = {
        "draft_length_truncation": sum(
            candidate_rows[case_id]["stages"]["draft"]["finish_reason"] == "length"
            for case_id in baseline_only
        ),
        "completed_wrong_draft": sum(
            candidate_rows[case_id]["stages"]["draft"]["finish_reason"] != "length"
            for case_id in baseline_only
        ),
        "verifier_corrected_to_reference": sum(
            candidate_rows[case_id]["prediction"] == candidate_rows[case_id]["expected"]
            for case_id in baseline_only
        ),
        "baseline_reference_matches": sum(
            baseline_rows[case_id]["prediction"] == baseline_rows[case_id]["expected"]
            for case_id in baseline_only
        ),
    }
    lower_bound = overall["paired_bootstrap_95_ci"][0]
    acceptance = (
        overall["candidate_accuracy"] >= overall["baseline_accuracy"]
        and lower_bound > -0.05
        and not candidate_cost["api_errors"]
        and not candidate_cost["parse_failures"]
    )
    report = {
        "schema_version": "nano_harness_public_gsm8k_confirmation_v1",
        "confirmation_id": "qwen35-gsm8k-confirm-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "code_revision": git_revision(),
        "candidate_summary": summarize_baseline(CANDIDATE),
        "baseline_summary": summarize_baseline(BASELINE),
        "comparison": {
            "candidate_accuracy": overall["candidate_accuracy"],
            "baseline_accuracy": overall["baseline_accuracy"],
            "delta": overall["delta"],
            "paired_counts": overall["paired_counts"],
            "mcnemar_exact_p": overall["mcnemar_exact_p"],
            "paired_bootstrap_95_ci": overall["paired_bootstrap_95_ci"],
            "candidate_only_cases": overall["candidate_only_cases"],
            "baseline_only_cases": overall["baseline_only_cases"],
        },
        "failure_mechanisms": failure_mechanisms,
        "costs": {
            "four_b_draft_verify": candidate_cost,
            "nine_b_direct": baseline_cost,
        },
        "artifacts": {
            "four_b_raw_sha256": sha256_file(CANDIDATE),
            "nine_b_raw_sha256": sha256_file(BASELINE),
        },
        "decision": {
            "point_non_regression": (
                overall["candidate_accuracy"] >= overall["baseline_accuracy"]
            ),
            "non_inferiority_lower_bound_above_minus_005": lower_bound > -0.05,
            "confirmation_satisfied": acceptance,
            "policy_frozen_for_confirmation": True,
            "next_experiment": (
                "Independent math re-solve verifier on a fresh GSM8K dev3 slice."
            ),
        },
    }
    ci = overall["paired_bootstrap_95_ci"]
    markdown = f"""# GSM8K Confirmation Result

## Result

On 96 unseen GSM8K cases, 4B draft-verify scores
{overall['candidate_accuracy']:.4f} and 9B direct scores
{overall['baseline_accuracy']:.4f}. The paired delta is
{overall['delta']:+.4f}, bootstrap 95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}],
exact McNemar `p={overall['mcnemar_exact_p']:.6f}`.

The pre-registered confirmation fails:

- the 4B point estimate is lower than 9B;
- the CI lower bound is below the -0.05 non-inferiority margin;
- six cases are 9B-only wins and none are 4B-only.

## Failure Mechanism

Among the six 9B-only cases:

- {failure_mechanisms['draft_length_truncation']} 4B drafts hit the 256-token limit;
- {failure_mechanisms['completed_wrong_draft']} 4B drafts stopped with incorrect reasoning;
- the strict verifier corrected {failure_mechanisms['verifier_corrected_to_reference']} to the reference.

The verifier reliably formats but does not independently repair math reasoning.
The next experiment must use fresh dev3 cases and an independent math re-solve
verifier. No tuning is allowed on these 96 confirmation cases.

## Cost And Identity

- 4B draft-verify: {candidate_cost['total_tokens']} tokens,
  {candidate_cost['wall_seconds']:.1f}s.
- 9B direct: {baseline_cost['total_tokens']} tokens,
  {baseline_cost['wall_seconds']:.1f}s.
- Code revision: `{report['code_revision']}`
- 4B raw SHA256: `{report['artifacts']['four_b_raw_sha256']}`
- 9B raw SHA256: `{report['artifacts']['nine_b_raw_sha256']}`

Raw prompts and outputs remain local and ignored.
"""
    json_path = Path("docs/results/gsm8k_confirmation_v1.public.json")
    markdown_path = Path("docs/results/gsm8k_confirmation_v1.md")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(markdown, encoding="utf-8")
    print(
        json.dumps(
            {
                "confirmation_id": report["confirmation_id"],
                "delta": overall["delta"],
                "ci": ci,
                "confirmation_satisfied": acceptance,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
