#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from nano_harness.baseline import load_cases, load_manifest
from nano_harness.verified_choice import verify_explicit_average_choice


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/harness/anchored_v1_verified_choice_canary_v1.json"
RESULT = (
    ROOT
    / "results/harness/anchored-v1-verified-choice-canary-v1/"
    "candidate/result.json"
)
PRE_REGISTRATION_REVISION = "8ffba19"
CONFIG_SHA256 = (
    "3cc141f3b1505f7e9fd222c7fad21e4f"
    "220ece8ed16e73952d3c67e082a36c2b"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if sha256_file(CONFIG) != CONFIG_SHA256:
        raise SystemExit("verified choice canary config differs")
    raw = json.loads(RESULT.read_text(encoding="utf-8"))
    if (
        raw.get("schema_version")
        != "nano_harness_verified_choice_canary_result_v1"
    ):
        raise SystemExit("unexpected verified choice canary schema")
    boundary = raw["evaluation_boundary"]
    if boundary != {
        "target_used_by_parser": False,
        "sealed_canary_run": True,
        "quality_claim_allowed": False,
        "training_eligible": False,
        "prior_full_suite_run": False,
        "independent_holdout_run": False,
    }:
        raise SystemExit("verified choice canary boundary differs")

    config = raw["config"]
    manifest = load_manifest(config["manifest_path"])
    cases = load_cases(manifest, Path(config["dataset_root"]))
    by_case = {case.case_id: case for case in cases}
    baseline = {row["case_id"]: row for row in raw["baseline_rows"]}
    candidate = {row["case_id"]: row for row in raw["candidate_rows"]}
    if (
        len(cases) != 40
        or set(by_case) != set(baseline)
        or set(by_case) != set(candidate)
    ):
        raise SystemExit("verified choice canary row identities differ")

    choice_ids = {
        case.case_id for case in cases if case.scorer == "choice_exact"
    }
    if set(raw["receipts"]) != choice_ids or len(choice_ids) != 24:
        raise SystemExit("verified choice canary receipt coverage differs")
    for case_id in sorted(choice_ids):
        if (
            verify_explicit_average_choice(by_case[case_id].prompt)
            != raw["receipts"][case_id]
        ):
            raise SystemExit("verified choice canary receipt is not reproducible")

    changed = [
        case_id
        for case_id in by_case
        if baseline[case_id]["output"] != candidate[case_id]["output"]
    ]
    regressions = [
        case_id
        for case_id in by_case
        if baseline[case_id]["score"] and not candidate[case_id]["score"]
    ]
    baseline_metrics = raw["baseline"]
    candidate_metrics = raw["candidate"]
    expected = {
        "gsm8k": 15,
        "mmlu": 13,
        "gpqa_diamond": 4,
    }
    checks = {
        "baseline_total_32": baseline_metrics["correct"] == 32,
        "candidate_total_at_least_32": candidate_metrics["correct"] >= 32,
        "baseline_task_identity": all(
            baseline_metrics["by_benchmark"][name]["correct"] == score
            for name, score in expected.items()
        ),
        "task_non_regression": all(
            candidate_metrics["by_benchmark"][name]["correct"] >= score
            for name, score in expected.items()
        ),
        "zero_regressions": not regressions,
        "all_fallback_outputs_equal_direct": len(changed) == 0,
        "zero_api_errors": all(
            row["api_errors"] == 0
            for row in candidate_metrics["by_benchmark"].values()
        ),
        "zero_parse_failures": all(
            row["parse_failures"] == 0
            for row in candidate_metrics["by_benchmark"].values()
        ),
        "zero_length_truncations": all(
            row["length_truncations"] == 0
            for row in candidate_metrics["by_benchmark"].values()
        ),
        "target_blind_parser": boundary["target_used_by_parser"] is False,
    }
    if not all(checks.values()):
        raise SystemExit(
            "verified choice canary gate failed: "
            + ",".join(key for key, value in checks.items() if not value)
        )
    reasons = {
        reason: sum(
            receipt["reason"] == reason
            for receipt in raw["receipts"].values()
        )
        for reason in sorted(
            {receipt["reason"] for receipt in raw["receipts"].values()}
        )
    }
    if reasons != {"unsupported_intent": 24}:
        raise SystemExit("verified choice canary routing reasons differ")

    report = {
        "schema_version": "nano_harness_public_anchored_verified_choice_canary_v1",
        "experiment_id": raw["experiment_id"],
        "pre_registration_revision": PRE_REGISTRATION_REVISION,
        "passed": True,
        "policy": {
            "source_split": "sealed_eval_canary",
            "training_eligible": False,
            "quality_claim_allowed": False,
            "case_level_publication_allowed": False,
        },
        "identity": {
            "config_sha256": CONFIG_SHA256,
            "raw_result_sha256": sha256_file(RESULT),
            **raw["identity"],
        },
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "routing": {
            **raw["routing"],
            "reason_counts": reasons,
            "changed_outputs": len(changed),
            "regressions": len(regressions),
        },
        "canary_gate": checks,
        "evaluation_boundary": {
            **boundary,
            "independent_holdout_prompts_loaded": False,
            "independent_holdout_references_loaded": False,
        },
        "decision": {
            "canary_passed": True,
            "full_development_suite_allowed": True,
            "independent_holdout_allowed": False,
            "merge_allowed": False,
            "scale_allowed": False,
            "rl_allowed": False,
            "next_action": (
                "Apply the exact target-blind verified-choice executor to the "
                "old 211-case development suite. Require per-task base-4B "
                "non-regression before opening the independent holdout."
            ),
        },
    }
    markdown = f"""# Anchored-v1 Verified Choice Canary v1 Result

## Result

The exact verified-choice executor passes the old sealed canary:

- GSM8K: {candidate_metrics['by_benchmark']['gsm8k']['correct']}/16;
- MMLU: {candidate_metrics['by_benchmark']['mmlu']['correct']}/16;
- GPQA-Diamond:
  {candidate_metrics['by_benchmark']['gpqa_diamond']['correct']}/8;
- total: {candidate_metrics['correct']}/40;
- API errors, parse failures, and truncations: 0 / 0 / 0.

All 24 choice prompts are outside parser v1's explicit arithmetic-average
intent. The executor therefore performs zero overrides and reuses 40/40
anchored-v1 outputs byte-for-byte. There are zero regressions.

## Boundary

This post-v6-calibrated canary remains a regression gate only. It is not
independent quality evidence and no case-level payload is published or
training eligible.

Passing permits only the old 211-case development suite. The independent
holdout, merge, scale, and RL remain blocked.

## Identity

- pre-registration revision: `{PRE_REGISTRATION_REVISION}`;
- config SHA256: `{CONFIG_SHA256}`;
- canary manifest SHA256: `{raw['identity']['manifest']}`;
- anchored-v1 raw SHA256: `{raw['identity']['baseline_raw']}`;
- local pass receipt SHA256: `{raw['identity']['local_pass_report']}`;
- raw applicator result SHA256: `{sha256_file(RESULT)}`.
"""
    output = ROOT / "docs/results"
    output.mkdir(parents=True, exist_ok=True)
    (
        output / "anchored_v1_verified_choice_canary_v1.public.json"
    ).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "anchored_v1_verified_choice_canary_v1.md").write_text(
        markdown,
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "passed": True,
                "correct": candidate_metrics["correct"],
                "overrides": raw["routing"]["verified_overrides"],
                "full_development_suite_allowed": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
