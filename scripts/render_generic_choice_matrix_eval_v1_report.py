#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import random
from pathlib import Path

from nano_harness.verified_choice import verify_explicit_average_choice


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/harness/generic_choice_capability_matrix_eval_v1.json"
MATRIX = (
    ROOT.parent
    / "nano-data-pipeline/datasets/generic_choice_capability_matrix_v1.json"
)
RESULT = ROOT / "results/harness/generic-choice-capability-matrix-eval-v1/result.json"
PRE_REGISTRATION_REVISION = "815700d"
CONFIG_SHA256 = (
    "f40be9cc5f1a98fb101724bae43a3ede"
    "1f477f4cf2d7a4956fa7efc42a2d7257"
)
RESULT_SHA256 = "fcb145d64dce23e4558d4004dea3b11f4ef1aa442e9026118abbf6966985d31b"


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def paired_metrics(
    candidate: list[dict],
    baseline: list[dict],
) -> dict:
    by_id = {row["case_id"]: row for row in baseline if row["scored"]}
    candidate_by_id = {
        row["case_id"]: row for row in candidate if row["scored"]
    }
    if set(by_id) != set(candidate_by_id) or len(by_id) != 32:
        raise SystemExit("matrix scored row identities differ")
    ids = sorted(by_id)
    values = [
        int(bool(candidate_by_id[case_id]["correct"]))
        - int(bool(by_id[case_id]["correct"]))
        for case_id in ids
    ]
    candidate_only = [
        case_id
        for case_id in ids
        if candidate_by_id[case_id]["correct"] and not by_id[case_id]["correct"]
    ]
    baseline_only = [
        case_id
        for case_id in ids
        if by_id[case_id]["correct"] and not candidate_by_id[case_id]["correct"]
    ]
    rng = random.Random("generic-choice-matrix-v1")
    samples = 10_000
    estimates = sorted(
        sum(values[rng.randrange(len(values))] for _ in values) / len(values)
        for _ in range(samples)
    )
    discordant = len(candidate_only) + len(baseline_only)
    tail = min(len(candidate_only), len(baseline_only))
    p_value = (
        min(
            1.0,
            2.0
            * sum(math.comb(discordant, index) for index in range(tail + 1))
            / (2**discordant),
        )
        if discordant
        else 1.0
    )
    return {
        "cases": len(ids),
        "delta": sum(values) / len(values),
        "paired_bootstrap_95_ci": [
            estimates[int(samples * 0.025)],
            estimates[int(samples * 0.975)],
        ],
        "mcnemar_exact_p": p_value,
        "candidate_only": len(candidate_only),
        "baseline_only": len(baseline_only),
        "candidate_only_case_ids": candidate_only,
        "baseline_only_case_ids": baseline_only,
    }


def main() -> None:
    if sha256_file(CONFIG) != CONFIG_SHA256:
        raise SystemExit("matrix eval config differs from pre-registration")
    if sha256_file(RESULT) != RESULT_SHA256:
        raise SystemExit("matrix raw result identity differs")
    raw = json.loads(RESULT.read_text(encoding="utf-8"))
    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    if raw.get("schema_version") != "nano_harness_choice_matrix_eval_result_v1":
        raise SystemExit("unexpected matrix result schema")
    if raw["evaluation_boundary"] != {
        "training_eligible_cases": 0,
        "target_used_by_executor_parser": False,
        "benchmark_rows_loaded": False,
        "canary_rows_loaded": False,
        "independent_holdout_rows_loaded": False,
    }:
        raise SystemExit("matrix evaluation boundary differs")

    cases = {row["case_id"]: row for row in matrix["cases"]}
    four = {row["case_id"]: row for row in raw["four_b_rows"]}
    nine = {row["case_id"]: row for row in raw["nine_b_rows"]}
    executor = {row["case_id"]: row for row in raw["executor_rows"]}
    if (
        len(cases) != 48
        or set(cases) != set(four)
        or set(cases) != set(nine)
        or set(cases) != set(executor)
    ):
        raise SystemExit("matrix arm row identities differ")
    for case_id, case in cases.items():
        if verify_explicit_average_choice(case["prompt"]) != raw[
            "executor_receipts"
        ][case_id]:
            raise SystemExit("matrix executor receipt is not reproducible")

    expected_route_matches = sum(
        (
            case["expected_route"] == "verified_override"
            and raw["executor_receipts"][case_id]["override"]
        )
        or (
            case["expected_route"] != "verified_override"
            and not raw["executor_receipts"][case_id]["override"]
        )
        for case_id, case in cases.items()
    )
    ambiguity_ids = [
        case_id for case_id, case in cases.items() if case["reference"] is None
    ]
    ambiguity_overrides = sum(
        executor[case_id]["verified_choice_route"] != "reuse_direct_output"
        for case_id in ambiguity_ids
    )
    ambiguity_parity = sum(
        four[case_id]["output"] == executor[case_id]["output"]
        for case_id in ambiguity_ids
    )
    comparison = paired_metrics(
        list(executor.values()),
        list(four.values()),
    )
    fixed_by_family = {}
    for case_id in comparison["candidate_only_case_ids"]:
        family = cases[case_id]["family"]
        fixed_by_family[family] = fixed_by_family.get(family, 0) + 1

    nine_outputs = [row["output"] for row in nine.values()]
    nine_arm = {
        "valid_for_quality_comparison": False,
        "parseable": sum(row["parseable"] for row in nine.values()),
        "outputs_at_token_cap": sum(
            row["usage"].get("completion_tokens") == 32 for row in nine.values()
        ),
        "outputs_containing_final": sum(
            "FINAL" in output.upper() for output in nine_outputs
        ),
        "failure": "reasoning_truncated_before_final_under_frozen_32_token_budget",
    }
    checks = {
        "matrix_cases_48": len(cases) == 48,
        "scored_cases_32": sum(case["reference"] is not None for case in cases.values())
        == 32,
        "expected_route_matches_48": expected_route_matches == 48,
        "executor_correct_25": raw["arms"]["four_b_verified_executor"]["correct"]
        == 25,
        "four_b_direct_correct_19": raw["arms"]["four_b_direct"]["correct"] == 19,
        "six_fixes_zero_regressions": (
            comparison["candidate_only"] == 6
            and comparison["baseline_only"] == 0
        ),
        "ambiguity_zero_overrides": ambiguity_overrides == 0,
        "ambiguity_parity_16": ambiguity_parity == 16,
        "nine_b_arm_invalid": (
            nine_arm["parseable"] == 0
            and nine_arm["outputs_at_token_cap"] == 48
            and nine_arm["outputs_containing_final"] == 0
        ),
        "training_eligible_zero": matrix["summary"]["training_eligible_cases"] == 0,
    }
    if not all(checks.values()):
        raise SystemExit(
            "matrix result contract failed: "
            + ",".join(key for key, value in checks.items() if not value)
        )

    report = {
        "schema_version": "nano_harness_public_choice_matrix_eval_v1",
        "experiment_id": raw["experiment_id"],
        "pre_registration_revision": PRE_REGISTRATION_REVISION,
        "passed": True,
        "identity": {
            "config_sha256": CONFIG_SHA256,
            "matrix_sha256": raw["identity"]["matrix_sha256"],
            "four_b_serving_receipt_sha256": raw["identity"][
                "four_b_serving_receipt_sha256"
            ],
            "raw_result_sha256": RESULT_SHA256,
        },
        "arms": {
            "four_b_direct": raw["arms"]["four_b_direct"],
            "four_b_verified_executor": raw["arms"]["four_b_verified_executor"],
            "nine_b_direct": {
                **raw["arms"]["nine_b_direct"],
                **nine_arm,
            },
        },
        "executor_vs_four_b_direct": {
            **comparison,
            "fixed_by_family": fixed_by_family,
        },
        "routing": {
            **raw["routing"],
            "ambiguity_overrides": ambiguity_overrides,
            "ambiguity_direct_parity": ambiguity_parity,
        },
        "validation": checks,
        "evaluation_boundary": {
            **raw["evaluation_boundary"],
            "matrix_training_allowed": False,
            "nine_b_quality_comparison_allowed": False,
            "independent_quality_claim_allowed": False,
        },
        "decision": {
            "verified_executor_capability_supported": True,
            "four_b_significantly_improves_over_direct_on_matrix": (
                comparison["paired_bootstrap_95_ci"][0] > 0
                and comparison["mcnemar_exact_p"] < 0.05
            ),
            "four_b_exceeds_nine_b_claim_allowed": False,
            "independent_holdout_allowed": False,
            "merge_allowed": False,
            "scale_allowed": False,
            "rl_allowed": False,
            "next_action": (
                "Pre-register a matched matrix v2 that constrains both 4B and "
                "9B direct decoding to FINAL: [A-D]. Preserve the v1 executor "
                "arm and matrix; do not rerun or reinterpret the invalid 9B arm."
            ),
        },
    }
    direct = report["arms"]["four_b_direct"]
    routed = report["arms"]["four_b_verified_executor"]
    markdown = f"""# Generic Choice Capability Matrix Evaluation v1

## Result

On 32 scored fresh cases:

- anchored-v1 direct: {direct['correct']}/32;
- anchored-v1 plus verified executor: {routed['correct']}/32;
- delta: {comparison['delta']:+.4f};
- paired bootstrap 95% CI:
  [{comparison['paired_bootstrap_95_ci'][0]:+.4f},
  {comparison['paired_bootstrap_95_ci'][1]:+.4f}];
- exact McNemar p={comparison['mcnemar_exact_p']:.5f};
- fixes / regressions: {comparison['candidate_only']} /
  {comparison['baseline_only']}.

All six fixes are in explicit-average cases: direct improves from 2/8 to 8/8.
The executor makes zero overrides on 16 ambiguity cases and preserves all 16
direct outputs. Parser expected-route agreement is 48/48.

## Invalid 9B Arm

The frozen 9B direct arm is not a valid quality baseline: 48/48 outputs hit the
32-token cap while emitting reasoning, zero contain `FINAL`, and zero are
parseable. Report this as a contract/budget failure, not 9B quality accuracy.
No claim that 4B exceeds 9B is allowed from v1.

## Decision

The fresh matrix supports the verified-execution mechanism relative to 4B
direct. Pre-register a matched v2 with the same `FINAL: [A-D]` constrained
decoding for both 4B and 9B direct arms. Do not rerun or reinterpret v1's 9B
arm. The independent holdout, merge, scale, and RL remain blocked.

## Identity

- pre-registration revision: `{PRE_REGISTRATION_REVISION}`;
- config SHA256: `{CONFIG_SHA256}`;
- matrix SHA256: `{report['identity']['matrix_sha256']}`;
- raw result SHA256: `{RESULT_SHA256}`.
"""
    output = ROOT / "docs/results"
    output.mkdir(parents=True, exist_ok=True)
    (output / "generic_choice_capability_matrix_eval_v1.public.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "generic_choice_capability_matrix_eval_v1.md").write_text(
        markdown,
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "passed": True,
                "four_b_direct": direct["correct"],
                "four_b_executor": routed["correct"],
                "nine_b_arm_valid": False,
                "next_action": "matched_regex_v2",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
