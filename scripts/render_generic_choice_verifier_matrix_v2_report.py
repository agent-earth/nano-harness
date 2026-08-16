#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path

from nano_harness.verified_choice_v2 import verify_choice_v2


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/harness/generic_choice_verifier_matrix_eval_v2.json"
MATRIX = (
    ROOT.parent
    / "nano-data-pipeline/datasets/generic_choice_verifier_matrix_v2.json"
)
RESULT = (
    ROOT
    / "results/harness/generic-choice-verifier-matrix-eval-v2/result.json"
)
PRE_REGISTRATION_REVISION = "5a38882"
CONFIG_SHA256 = (
    "00e5b2125edee0c8e54c771746952375"
    "d6227bb69aa518bb195b642e61dd1ef1"
)
RESULT_SHA256 = "e6b208e1b3d816fc540d6a7c7e9315338751c09804b248691c633186491c4c87"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def paired_metrics(
    left: list[dict],
    right: list[dict],
    *,
    seed: str,
) -> dict:
    left_by_id = {row["case_id"]: row for row in left if row["scored"]}
    right_by_id = {row["case_id"]: row for row in right if row["scored"]}
    if set(left_by_id) != set(right_by_id) or len(left_by_id) != 16:
        raise SystemExit("verifier matrix scored row identities differ")
    ids = sorted(left_by_id)
    values = [
        int(bool(left_by_id[case_id]["correct"]))
        - int(bool(right_by_id[case_id]["correct"]))
        for case_id in ids
    ]
    left_only = [
        case_id
        for case_id in ids
        if left_by_id[case_id]["correct"]
        and not right_by_id[case_id]["correct"]
    ]
    right_only = [
        case_id
        for case_id in ids
        if right_by_id[case_id]["correct"]
        and not left_by_id[case_id]["correct"]
    ]
    rng = random.Random(seed)
    samples = 10_000
    estimates = sorted(
        sum(values[rng.randrange(len(values))] for _ in values) / len(values)
        for _ in range(samples)
    )
    discordant = len(left_only) + len(right_only)
    tail = min(len(left_only), len(right_only))
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
        "left_only": len(left_only),
        "right_only": len(right_only),
        "left_only_case_ids": left_only,
        "right_only_case_ids": right_only,
    }


def family_counts(case_ids: list[str], cases: dict[str, dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case_id in case_ids:
        family = str(cases[case_id]["family"])
        counts[family] = counts.get(family, 0) + 1
    return dict(sorted(counts.items()))


def main() -> None:
    if sha256_file(CONFIG) != CONFIG_SHA256:
        raise SystemExit("verifier matrix config differs from pre-registration")
    if sha256_file(RESULT) != RESULT_SHA256:
        raise SystemExit("verifier matrix raw result identity differs")
    raw = json.loads(RESULT.read_text(encoding="utf-8"))
    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    if (
        raw.get("schema_version")
        != "nano_harness_choice_verifier_matrix_result_v2"
    ):
        raise SystemExit("unexpected verifier matrix result schema")
    expected_boundary = {
        "training_eligible_cases": 0,
        "target_used_by_executor_parser": False,
        "benchmark_rows_loaded": False,
        "canary_rows_loaded": False,
        "independent_holdout_rows_loaded": False,
    }
    if raw["evaluation_boundary"] != expected_boundary:
        raise SystemExit("verifier matrix boundary differs")

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
        raise SystemExit("verifier matrix arm identities differ")
    for case_id, case in cases.items():
        if (
            verify_choice_v2(case["prompt"])
            != raw["executor_receipts"][case_id]
        ):
            raise SystemExit("verifier matrix receipt is not reproducible")
        for row in (four[case_id], nine[case_id]):
            if (
                row["structured_outputs"]
                != {"regex": raw["config"]["structured_output_regex"]}
                or not row["parseable"]
            ):
                raise SystemExit("verifier matrix constrained contract differs")

    executor_four = paired_metrics(
        list(executor.values()),
        list(four.values()),
        seed="verifier-v2-executor-four",
    )
    executor_nine = paired_metrics(
        list(executor.values()),
        list(nine.values()),
        seed="verifier-v2-executor-nine",
    )
    four_nine = paired_metrics(
        list(four.values()),
        list(nine.values()),
        seed="verifier-v2-four-nine",
    )
    for comparison in (executor_four, executor_nine, four_nine):
        comparison["left_only_by_family"] = family_counts(
            comparison["left_only_case_ids"],
            cases,
        )
        comparison["right_only_by_family"] = family_counts(
            comparison["right_only_case_ids"],
            cases,
        )

    ambiguity_ids = [
        case_id for case_id, case in cases.items() if case["reference"] is None
    ]
    ambiguity_overrides = sum(
        executor[case_id]["verified_choice_route"] != "reuse_direct_output"
        for case_id in ambiguity_ids
    )
    ambiguity_parity = sum(
        executor[case_id]["output"] == four[case_id]["output"]
        for case_id in ambiguity_ids
    )
    checks = {
        "all_arms_parseable_48": all(
            raw["arms"][name]["parseable"] == 48
            for name in (
                "four_b_constrained",
                "nine_b_constrained",
                "four_b_verified_executor_v2",
            )
        ),
        "four_b_correct_8": raw["arms"]["four_b_constrained"]["correct"] == 8,
        "nine_b_correct_11": raw["arms"]["nine_b_constrained"]["correct"] == 11,
        "executor_correct_16": (
            raw["arms"]["four_b_verified_executor_v2"]["correct"] == 16
        ),
        "executor_eight_wins_zero_losses_vs_four": (
            executor_four["left_only"] == 8
            and executor_four["right_only"] == 0
        ),
        "executor_five_wins_zero_losses_vs_nine": (
            executor_nine["left_only"] == 5
            and executor_nine["right_only"] == 0
        ),
        "executor_vs_four_significant": (
            executor_four["paired_bootstrap_95_ci"][0] > 0
            and executor_four["mcnemar_exact_p"] < 0.05
        ),
        "executor_vs_nine_ci_positive": (
            executor_nine["paired_bootstrap_95_ci"][0] > 0
        ),
        "executor_vs_nine_mcnemar_not_below_005": (
            executor_nine["mcnemar_exact_p"] >= 0.05
        ),
        "four_vs_nine_not_significant": (
            four_nine["paired_bootstrap_95_ci"][0] <= 0
            and four_nine["paired_bootstrap_95_ci"][1] >= 0
            and four_nine["mcnemar_exact_p"] >= 0.05
        ),
        "ambiguity_zero_overrides": ambiguity_overrides == 0,
        "ambiguity_parity_32": ambiguity_parity == 32,
        "expected_route_matches_48": raw["routing"]["expected_route_matches"]
        == 48,
        "training_eligible_zero": matrix["summary"]["training_eligible_cases"]
        == 0,
    }
    if not all(checks.values()):
        raise SystemExit(
            "verifier matrix result contract failed: "
            + ",".join(key for key, value in checks.items() if not value)
        )

    report = {
        "schema_version": "nano_harness_public_choice_verifier_matrix_v2",
        "experiment_id": raw["experiment_id"],
        "pre_registration_revision": PRE_REGISTRATION_REVISION,
        "passed": True,
        "identity": {
            "config_sha256": CONFIG_SHA256,
            "raw_result_sha256": RESULT_SHA256,
            **raw["identity"],
        },
        "arms": raw["arms"],
        "comparisons": {
            "executor_vs_four_b": executor_four,
            "executor_vs_nine_b": executor_nine,
            "four_b_vs_nine_b": four_nine,
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
            "independent_quality_claim_allowed": False,
        },
        "decision": {
            "executor_significantly_improves_over_four_b_direct": True,
            "executor_ci_positive_over_nine_b": True,
            "executor_significantly_exceeds_nine_b": False,
            "replication_required_for_nine_b_significance": True,
            "benchmark_superiority_claim_allowed": False,
            "independent_holdout_allowed": False,
            "merge_allowed": False,
            "scale_allowed": False,
            "rl_allowed": False,
            "next_action": (
                "Pre-register a larger history-disjoint replication of exact "
                "host-count and verbal-average cases to test the 5-0 executor "
                "advantage over 9B with adequate McNemar power. Preserve all "
                "ambiguity families as safety-only and training-forbidden."
            ),
        },
    }
    arms = report["arms"]
    markdown = f"""# Generic Choice Verifier Matrix v2 Result

## Matched Result

All arms are 48/48 parseable. On 16 scored fresh exact cases:

- anchored-v1 constrained direct:
  {arms['four_b_constrained']['correct']}/16;
- 9B constrained direct: {arms['nine_b_constrained']['correct']}/16;
- anchored-v1 plus verifier v2:
  {arms['four_b_verified_executor_v2']['correct']}/16.

Executor versus 4B direct:

- delta {executor_four['delta']:+.4f};
- 95% CI [{executor_four['paired_bootstrap_95_ci'][0]:+.4f},
  {executor_four['paired_bootstrap_95_ci'][1]:+.4f}];
- McNemar p={executor_four['mcnemar_exact_p']:.5f};
- eight wins and zero losses.

Executor versus 9B:

- delta {executor_nine['delta']:+.4f};
- 95% CI [{executor_nine['paired_bootstrap_95_ci'][0]:+.4f},
  {executor_nine['paired_bootstrap_95_ci'][1]:+.4f}];
- McNemar p={executor_nine['mcnemar_exact_p']:.5f};
- five wins and zero losses.

The CI is positive, but p=0.0625 misses the pre-registered 0.05 dual
significance criterion. Do not claim significant superiority over 9B.

## Safety And Scope

Verifier v2 scores 8/8 on host-count and 8/8 on verbal-average exact cases.
It makes zero overrides on all 32 ambiguity cases and preserves all 32 direct
outputs. Expected-route agreement is 48/48.

This is fresh generic mechanism evidence, not benchmark or holdout evidence.
The matrix is forbidden for every training use. Holdout, merge, scale, and RL
remain blocked.

## Decision

Pre-register a larger history-disjoint exact-case replication to test the 5-0
trend with enough McNemar power. Preserve ambiguity families as safety-only;
do not train on any matrix row.

## Identity

- pre-registration revision: `{PRE_REGISTRATION_REVISION}`;
- config SHA256: `{CONFIG_SHA256}`;
- matrix SHA256: `{raw['identity']['matrix_sha256']}`;
- raw result SHA256: `{RESULT_SHA256}`.
"""
    output = ROOT / "docs/results"
    output.mkdir(parents=True, exist_ok=True)
    (
        output / "generic_choice_verifier_matrix_eval_v2.public.json"
    ).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "generic_choice_verifier_matrix_eval_v2.md").write_text(
        markdown,
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "passed": True,
                "four_b": arms["four_b_constrained"]["correct"],
                "nine_b": arms["nine_b_constrained"]["correct"],
                "executor": arms["four_b_verified_executor_v2"]["correct"],
                "executor_vs_nine_significant": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
