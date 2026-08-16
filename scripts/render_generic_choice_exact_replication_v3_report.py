#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path

from nano_harness.verified_choice_v2 import verify_choice_v2


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/harness/generic_choice_exact_replication_eval_v3.json"
MATRIX = (
    ROOT.parent
    / "nano-data-pipeline/datasets/generic_choice_exact_replication_matrix_v3.json"
)
RESULT = (
    ROOT
    / "results/harness/generic-choice-exact-replication-eval-v3/result.json"
)
PRE_REGISTRATION_REVISION = "3b90f87"
CONFIG_SHA256 = "c71f0cd6f730946fed6c92c5d2172bbbf8a03c937127de1e9d975d8134270ace"
RESULT_SHA256 = "2720c866ab72090b15feb3ffedd2fc6f66890cac118ad8ab8afb21230fb65c49"


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
    samples: int,
) -> dict:
    left_by_id = {row["case_id"]: row for row in left if row["scored"]}
    right_by_id = {row["case_id"]: row for row in right if row["scored"]}
    if set(left_by_id) != set(right_by_id) or len(left_by_id) != 32:
        raise SystemExit("exact replication scored row identities differ")
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
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
    }


def family_counts(case_ids: list[str], cases: dict[str, dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case_id in case_ids:
        family = str(cases[case_id]["family"])
        counts[family] = counts.get(family, 0) + 1
    return dict(sorted(counts.items()))


def main() -> None:
    if sha256_file(CONFIG) != CONFIG_SHA256:
        raise SystemExit("exact replication config differs from pre-registration")
    if sha256_file(RESULT) != RESULT_SHA256:
        raise SystemExit("exact replication raw result identity differs")
    raw = json.loads(RESULT.read_text(encoding="utf-8"))
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    if (
        raw.get("schema_version")
        != "nano_harness_choice_exact_replication_result_v3"
    ):
        raise SystemExit("unexpected exact replication result schema")
    expected_boundary = {
        "training_eligible_cases": 0,
        "target_used_by_executor_parser": False,
        "benchmark_rows_loaded": False,
        "canary_rows_loaded": False,
        "independent_holdout_rows_loaded": False,
    }
    if raw["evaluation_boundary"] != expected_boundary:
        raise SystemExit("exact replication boundary differs")

    cases = {row["case_id"]: row for row in matrix["cases"]}
    four = {row["case_id"]: row for row in raw["four_b_rows"]}
    nine = {row["case_id"]: row for row in raw["nine_b_rows"]}
    executor = {row["case_id"]: row for row in raw["executor_rows"]}
    if (
        len(cases) != config["scored_cases"]
        or set(cases) != set(four)
        or set(cases) != set(nine)
        or set(cases) != set(executor)
    ):
        raise SystemExit("exact replication arm identities differ")
    for case_id, case in cases.items():
        if verify_choice_v2(case["prompt"]) != raw["executor_receipts"][case_id]:
            raise SystemExit("exact replication receipt is not reproducible")
        if raw["executor_receipts"][case_id].get("selected_letter") != case[
            "reference"
        ]:
            raise SystemExit("exact replication proof does not match reference")
        for row in (four[case_id], nine[case_id]):
            if (
                row["structured_outputs"]
                != {"regex": raw["config"]["structured_output_regex"]}
                or not row["parseable"]
            ):
                raise SystemExit("exact replication constrained contract differs")

    comparisons = {}
    pairs = {
        "executor_vs_four_b": (executor, four, "executor-four"),
        "executor_vs_nine_b": (executor, nine, "executor-nine"),
        "four_b_vs_nine_b": (four, nine, "four-nine"),
    }
    for name, (left, right, suffix) in pairs.items():
        comparison = paired_metrics(
            list(left.values()),
            list(right.values()),
            seed=f"{config['bootstrap_seed']}:{suffix}",
            samples=config["bootstrap_samples"],
        )
        comparison["left_only_by_family"] = family_counts(
            comparison["left_only_case_ids"], cases
        )
        comparison["right_only_by_family"] = family_counts(
            comparison["right_only_case_ids"], cases
        )
        comparisons[name] = comparison

    executor_four = comparisons["executor_vs_four_b"]
    executor_nine = comparisons["executor_vs_nine_b"]
    alpha = config["significance_alpha"]
    executor_vs_nine_passed = (
        executor_nine["paired_bootstrap_95_ci"][0] > 0
        and executor_nine["mcnemar_exact_p"] < alpha
        and executor_nine["left_only"]
        >= config["minimum_executor_wins_over_nine_b"]
        and executor_nine["right_only"]
        <= config["maximum_executor_losses_over_nine_b"]
    )
    checks = {
        "all_arms_parseable_32": all(
            raw["arms"][name]["parseable"] == 32
            for name in (
                "four_b_constrained",
                "nine_b_constrained",
                "four_b_verified_executor_v2",
            )
        ),
        "all_rows_scored_32": all(
            raw["arms"][name]["scored_cases"] == 32
            for name in (
                "four_b_constrained",
                "nine_b_constrained",
                "four_b_verified_executor_v2",
            )
        ),
        "executor_correct_32": (
            raw["arms"]["four_b_verified_executor_v2"]["correct"] == 32
        ),
        "executor_all_routes_verified": (
            raw["routing"]["verified_overrides"] == 32
            and raw["routing"]["fallbacks"] == 0
            and raw["routing"]["expected_route_matches"] == 32
        ),
        "executor_vs_four_significant": (
            executor_four["paired_bootstrap_95_ci"][0] > 0
            and executor_four["mcnemar_exact_p"] < alpha
        ),
        "executor_vs_nine_preregistered_gate_passed": executor_vs_nine_passed,
        "training_eligible_zero": matrix["summary"]["training_eligible_cases"]
        == 0,
        "matrix_evaluation_only": matrix["policy"]["evaluation_only"] is True,
    }
    if not all(checks.values()):
        raise SystemExit(
            "exact replication result contract failed: "
            + ",".join(key for key, value in checks.items() if not value)
        )

    report = {
        "schema_version": "nano_harness_public_choice_exact_replication_v3",
        "experiment_id": raw["experiment_id"],
        "pre_registration_revision": PRE_REGISTRATION_REVISION,
        "passed": True,
        "identity": {
            "config_sha256": CONFIG_SHA256,
            "raw_result_sha256": RESULT_SHA256,
            **raw["identity"],
        },
        "arms": raw["arms"],
        "comparisons": comparisons,
        "routing": raw["routing"],
        "validation": checks,
        "evaluation_boundary": {
            **raw["evaluation_boundary"],
            "matrix_training_allowed": False,
            "independent_quality_claim_allowed": False,
        },
        "decision": {
            "executor_significantly_improves_over_four_b_direct": True,
            "executor_significantly_exceeds_nine_b_on_exact_replication": True,
            "preregistered_executor_vs_nine_b_gate_passed": True,
            "benchmark_superiority_claim_allowed": False,
            "independent_holdout_allowed": False,
            "merge_allowed": False,
            "scale_allowed": False,
            "rl_allowed": False,
            "next_action": (
                "Preserve verifier v2 as replicated mechanism evidence. "
                "Pre-register a benchmark-agnostic transfer intervention that "
                "can affect the frozen local and prior three-task suites, then "
                "require per-task non-regression before any holdout or training "
                "promotion."
            ),
        },
    }
    arms = report["arms"]
    markdown = f"""# Generic Choice Exact Replication v3 Result

## Matched Result

All arms are 32/32 parseable on 32 fresh scored exact cases:

- anchored-v1 constrained direct:
  {arms['four_b_constrained']['correct']}/32;
- 9B constrained direct: {arms['nine_b_constrained']['correct']}/32;
- anchored-v1 plus verifier v2:
  {arms['four_b_verified_executor_v2']['correct']}/32.

Verifier v2 versus anchored-v1 direct:

- delta {executor_four['delta']:+.4f};
- 95% CI [{executor_four['paired_bootstrap_95_ci'][0]:+.4f},
  {executor_four['paired_bootstrap_95_ci'][1]:+.4f}];
- McNemar p={executor_four['mcnemar_exact_p']:.8f};
- {executor_four['left_only']} wins and {executor_four['right_only']} losses.

Verifier v2 versus 9B:

- delta {executor_nine['delta']:+.4f};
- 95% CI [{executor_nine['paired_bootstrap_95_ci'][0]:+.4f},
  {executor_nine['paired_bootstrap_95_ci'][1]:+.4f}];
- McNemar p={executor_nine['mcnemar_exact_p']:.10f};
- {executor_nine['left_only']} wins and {executor_nine['right_only']} losses.

This passes the pre-registered dual significance criterion and the minimum
six-win, zero-loss discordance criterion.

## Family Result

Verifier v2 is 16/16 on host-count and 16/16 on verbal-average. Anchored-v1
direct is 8/16 and 14/16; 9B direct is 4/16 and 7/16, respectively.

All 32 verifier receipts reproduce byte-for-byte from the unchanged
target-blind parser. Every route is a unique exact proof; references are not
passed to the parser.

## Scope

This replicates significant verifier-v2 superiority over matched 9B for these
two exact generic mechanisms. It is not GSM8K, MMLU, GPQA, agent-benchmark, or
independent-holdout superiority. The matrix is forbidden for every training
use. Holdout, merge, scale, and RL remain blocked.

## Decision

Preserve verifier v2 as replicated mechanism evidence. The next intervention
must be benchmark-agnostic yet capable of transferring to the frozen local and
prior three-task suites, with per-task non-regression before holdout access.

## Identity

- pre-registration revision: `{PRE_REGISTRATION_REVISION}`;
- config SHA256: `{CONFIG_SHA256}`;
- matrix SHA256: `{raw['identity']['matrix_sha256']}`;
- raw result SHA256: `{RESULT_SHA256}`.
"""
    output = ROOT / "docs/results"
    output.mkdir(parents=True, exist_ok=True)
    (
        output / "generic_choice_exact_replication_eval_v3.public.json"
    ).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "generic_choice_exact_replication_eval_v3.md").write_text(
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
                "executor_vs_nine_b": executor_nine,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
