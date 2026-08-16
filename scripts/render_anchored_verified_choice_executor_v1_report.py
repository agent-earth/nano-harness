#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from nano_harness.verified_choice import verify_explicit_average_choice


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/harness/anchored_v1_verified_choice_executor_v1.json"
DATASET = ROOT / "../nano-data-pipeline/datasets/generic_choice_replay_v11.json"
RESULT = (
    ROOT
    / "results/harness/anchored-v1-verified-choice-executor-v1/local/result.json"
)
PRE_REGISTRATION_REVISION = "41441c3"
CONFIG_SHA256 = (
    "b6aa27605fdbf25db5a470cfe95f0fee"
    "704f1c249c9069ef6b12401afa5d5178"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def family_score(metrics: dict, family: str) -> int:
    return int(metrics["by_family"][family]["semantic_exact"])


def main() -> None:
    if sha256_file(CONFIG) != CONFIG_SHA256:
        raise SystemExit("verified choice config differs from pre-registration")
    raw = json.loads(RESULT.read_text(encoding="utf-8"))
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    if raw.get("schema_version") != "nano_harness_verified_choice_result_v1":
        raise SystemExit("unexpected verified choice result schema")
    boundary = raw["evaluation_boundary"]
    if boundary != {
        "target_used_by_parser": False,
        "benchmark_rows_loaded": False,
        "sealed_canary_run": False,
        "prior_full_suite_run": False,
        "independent_holdout_run": False,
    }:
        raise SystemExit("verified choice evaluation boundary differs")

    samples = {
        str(sample["sample_id"]): sample
        for sample in dataset["samples"]
        if sample["split"] == "validation"
    }
    baseline_rows = {row["sample_id"]: row for row in raw["baseline_rows"]}
    candidate_rows = {row["sample_id"]: row for row in raw["candidate_rows"]}
    if (
        set(samples) != set(baseline_rows)
        or set(samples) != set(candidate_rows)
        or len(samples) != 32
    ):
        raise SystemExit("verified choice row identities differ")

    receipts = raw["receipts"]
    choice_ids = {
        sample_id
        for sample_id, sample in samples.items()
        if sample["format_family"] == "final_choice"
    }
    if set(receipts) != choice_ids or len(receipts) != 8:
        raise SystemExit("verified choice receipts do not cover choice rows")
    for sample_id in sorted(choice_ids):
        prompt = str(samples[sample_id]["messages"][1]["content"])
        reproduced = verify_explicit_average_choice(prompt)
        receipt = dict(receipts[sample_id])
        prompt_sha256 = receipt.pop("prompt_sha256")
        if (
            prompt_sha256 != hashlib.sha256(prompt.encode()).hexdigest()
            or reproduced != receipt
        ):
            raise SystemExit(f"receipt is not reproducible: {sample_id}")

    changed = [
        sample_id
        for sample_id in samples
        if baseline_rows[sample_id]["output"]
        != candidate_rows[sample_id]["output"]
    ]
    fixed = [
        sample_id
        for sample_id in changed
        if not baseline_rows[sample_id]["exact"]
        and candidate_rows[sample_id]["exact"]
    ]
    regressed = [
        sample_id
        for sample_id in changed
        if baseline_rows[sample_id]["exact"]
        and not candidate_rows[sample_id]["exact"]
    ]
    fallback_ids = [
        sample_id
        for sample_id, row in candidate_rows.items()
        if row["route"] == "reuse_direct_output"
    ]
    if any(
        baseline_rows[sample_id]["output"]
        != candidate_rows[sample_id]["output"]
        for sample_id in fallback_ids
    ):
        raise SystemExit("a fallback output differs from direct")

    choice = "capability_preservation_choice"
    numeric = "capability_preservation_numeric"
    process = "semantic_arithmetic_process"
    baseline = raw["baseline"]
    candidate = raw["candidate"]
    checks = {
        "baseline_strict_22": baseline["exact"] == 22,
        "baseline_semantic_25": baseline["semantic_exact"] == 25,
        "baseline_numeric_11": family_score(baseline, numeric) == 11,
        "baseline_choice_6": family_score(baseline, choice) == 6,
        "baseline_process_8": family_score(baseline, process) == 8,
        "candidate_strict_at_least_22": candidate["exact"] >= 22,
        "candidate_semantic_at_least_25": candidate["semantic_exact"] >= 25,
        "candidate_numeric_at_least_11": family_score(candidate, numeric) >= 11,
        "candidate_choice_at_least_7": family_score(candidate, choice) >= 7,
        "candidate_process_equals_8": family_score(candidate, process) == 8,
        "fallback_outputs_equal_direct": len(fallback_ids) == 27,
        "one_fix_zero_regressions": len(fixed) == 1 and not regressed,
        "target_blind_parser": boundary["target_used_by_parser"] is False,
    }
    if not all(checks.values()):
        raise SystemExit(
            "verified choice local gate failed: "
            + ",".join(key for key, value in checks.items() if not value)
        )
    if len(changed) != 1 or changed != fixed:
        raise SystemExit("verified choice output deltas differ")

    report = {
        "schema_version": "nano_harness_public_anchored_verified_choice_v1",
        "experiment_id": raw["experiment_id"],
        "pre_registration_revision": PRE_REGISTRATION_REVISION,
        "passed": True,
        "identity": {
            "config_sha256": CONFIG_SHA256,
            "dataset_sha256": raw["identity"]["dataset_sha256"],
            "baseline_result_sha256": raw["identity"][
                "baseline_result_sha256"
            ],
            "raw_result_sha256": sha256_file(RESULT),
        },
        "method": {
            "parser_version": raw["config"]["parser_version"],
            "supported_intent": raw["config"]["supported_intent"],
            "exact_option_match_required": raw["config"][
                "exact_option_match_required"
            ],
            "ambiguous_fallback": raw["config"]["ambiguous_fallback"],
            "model_calls": 0,
            "target_used_by_parser": False,
        },
        "baseline": baseline,
        "candidate": candidate,
        "routing": raw["routing"],
        "local_gate": checks,
        "case_delta": {
            "changed_sample_ids": changed,
            "fixed_sample_ids": fixed,
            "regressed_sample_ids": regressed,
            "fallback_rows": len(fallback_ids),
        },
        "fixed_case_receipt": receipts[fixed[0]],
        "evaluation_boundary": {
            **boundary,
            "independent_holdout_prompts_loaded": False,
            "independent_holdout_references_loaded": False,
            "independent_quality_claim_allowed": False,
        },
        "decision": {
            "accepted_local_harness": True,
            "sealed_canary_allowed": True,
            "prior_full_suite_allowed": False,
            "independent_holdout_allowed": False,
            "merge_allowed": False,
            "scale_allowed": False,
            "rl_allowed": False,
            "next_action": (
                "Apply the exact target-blind verified-choice executor to the "
                "old sealed 40-case regression canary. A canary pass permits "
                "the old 211-case development suite, not the independent holdout."
            ),
        },
    }
    receipt = receipts[fixed[0]]
    markdown = f"""# Anchored-v1 Verified Choice Executor v1 Result

## Result

The target-blind verified executor passes every frozen local gate:

- baseline strict / semantic: {baseline['exact']}/32 /
  {baseline['semantic_exact']}/32;
- candidate strict / semantic: {candidate['exact']}/32 /
  {candidate['semantic_exact']}/32;
- candidate numeric / choice / process semantic:
  {family_score(candidate, numeric)}/16,
  {family_score(candidate, choice)}/8,
  {family_score(candidate, process)}/8;
- one choice fix, zero regressions;
- 27 fallback outputs remain byte-identical to direct;
- zero model calls and no target use during parsing.

## Mechanism Evidence

The fixed row contains expressions `{receipt['expressions'][0]}` and
`{receipt['expressions'][1]}`. Exact `Fraction` evaluation yields
{receipt['expression_values'][0]} and {receipt['expression_values'][1]}, whose
average is {receipt['result']}. Exactly one option has that value, so the
executor selects `{receipt['selected_letter']}`. Fractional results with no
exact option remain on direct output; no rounding or nearest-option heuristic
is allowed.

## Decision

Passing authorizes only the old sealed 40-case regression canary. The old
211-case development suite, independent holdout, merge, scale, and RL remain
blocked.

## Reproduction Identity

- pre-registration revision: `{PRE_REGISTRATION_REVISION}`;
- config SHA256: `{CONFIG_SHA256}`;
- dataset SHA256: `{raw['identity']['dataset_sha256']}`;
- baseline result SHA256: `{raw['identity']['baseline_result_sha256']}`;
- raw local result SHA256: `{sha256_file(RESULT)}`.
"""
    output = ROOT / "docs/results"
    output.mkdir(parents=True, exist_ok=True)
    (output / "anchored_v1_verified_choice_executor_v1.public.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "anchored_v1_verified_choice_executor_v1.md").write_text(
        markdown,
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "passed": True,
                "strict": candidate["exact"],
                "semantic": candidate["semantic_exact"],
                "choice": family_score(candidate, choice),
                "sealed_canary_allowed": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
