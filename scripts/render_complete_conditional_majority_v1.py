#!/usr/bin/env python3

from __future__ import annotations

import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from nano_harness.baseline import sha256_file
from nano_harness.complete_conditional_majority import load_config
from nano_harness.orca_recovered_self_consistency import (
    parse_recovered_final,
)
from nano_harness.orca_self_consistency import score_prediction
from nano_harness.v5_complete_treatment import jsonl_rows
from scripts.render_orca_self_consistency_replication_v2 import (
    four_b_preservation_gates,
)
from scripts.render_orca_self_consistency_v1 import paired_metrics
from scripts.run_complete_conditional_majority_shard_v1 import EXECUTION


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/campaign/qwen35_complete_conditional_majority_v1.json"
)
PREREGISTER = (
    ROOT
    / "docs/experiments/"
    "qwen35_complete_conditional_majority_v1.preregister.json"
)
RAW_RESULT = (
    ROOT / "results/full/qwen35-complete-conditional-majority-v1/result.json"
)
PUBLIC = (
    ROOT
    / "docs/results/qwen35_complete_conditional_majority_v1.public.json"
)
MARKDOWN = (
    ROOT / "docs/results/qwen35_complete_conditional_majority_v1.md"
)


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {}
    for row in rows:
        case_id = str(row["case_id"])
        if case_id in result:
            raise ValueError(f"duplicate complete candidate case: {case_id}")
        result[case_id] = row
    return result


def _correct_rows(
    rows: list[dict[str, Any]],
    *,
    recovered_numeric: bool,
) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        correct = bool(row["score"])
        prediction = row.get("prediction")
        if recovered_numeric:
            prediction = parse_recovered_final(str(row["output"]))
            correct = score_prediction(prediction, str(row["expected"]))
        result.append(
            {
                "case_id": row["case_id"],
                "prediction": prediction,
                "correct": correct,
            }
        )
    return result


def comparison(
    candidate: list[dict[str, Any]],
    baseline: list[dict[str, Any]],
    *,
    benchmark: str,
    bootstrap_samples: int,
    bootstrap_seed: str,
) -> dict[str, Any]:
    candidate_rows = [
        row for row in candidate if row["benchmark"] == benchmark
    ]
    baseline_rows = [
        row for row in baseline if row["benchmark"] == benchmark
    ]
    recovered = benchmark == "gsm8k"
    return paired_metrics(
        _correct_rows(candidate_rows, recovered_numeric=recovered),
        _correct_rows(baseline_rows, recovered_numeric=recovered),
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )


def gsm8k_gates(
    versus_four: dict[str, Any],
    versus_nine: dict[str, Any],
    *,
    receipts: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    four_b: list[dict[str, Any]],
    alpha: float,
    minimum_candidate_only_wins: int,
) -> dict[str, Any]:
    candidate_by_id = _by_id(candidate)
    four_by_id = _by_id(four_b)
    route_deltas = {}
    for strict_parseable in (False, True):
        case_ids = [
            row["case_id"]
            for row in receipts
            if bool(row["receipt"]["direct_strict_parseable"])
            is strict_parseable
        ]
        candidate_correct = sum(
            score_prediction(
                parse_recovered_final(candidate_by_id[case_id]["output"]),
                str(candidate_by_id[case_id]["expected"]),
            )
            for case_id in case_ids
        )
        four_correct = sum(
            score_prediction(
                parse_recovered_final(four_by_id[case_id]["output"]),
                str(four_by_id[case_id]["expected"]),
            )
            for case_id in case_ids
        )
        route_deltas[
            "strict_parseable" if strict_parseable else "strict_parse_failure"
        ] = {
            "cases": len(case_ids),
            "candidate_correct": candidate_correct,
            "four_b_correct": four_correct,
            "delta_correct": candidate_correct - four_correct,
        }

    preservation = four_b_preservation_gates(
        {
            **versus_four,
            "by_stratum": {
                name: {"delta": row["delta_correct"]}
                for name, row in route_deltas.items()
            },
        }
    )
    superiority = {
        "point_delta_positive": versus_nine["delta"] > 0,
        "bootstrap_ci_lower_positive": (
            versus_nine["paired_bootstrap_95_ci"][0] > 0
        ),
        "mcnemar_p_below_bonferroni_alpha": (
            versus_nine["mcnemar_exact_p"] < alpha
        ),
        "minimum_candidate_only_wins": (
            versus_nine["paired_counts"]["candidate_only"]
            >= minimum_candidate_only_wins
        ),
        "candidate_only_exceeds_baseline_only": (
            versus_nine["paired_counts"]["candidate_only"]
            > versus_nine["paired_counts"]["baseline_only"]
        ),
    }
    return {
        "route_metrics": route_deltas,
        "four_b_preservation": preservation,
        "nine_b_superiority": superiority,
        "admitted": all(preservation.values()) and all(superiority.values()),
    }


def holm_bonferroni(
    p_values: dict[str, float],
    *,
    alpha: float,
) -> dict[str, Any]:
    ordered = sorted(p_values.items(), key=lambda row: (row[1], row[0]))
    rows = []
    preceding_pass = True
    for index, (benchmark, value) in enumerate(ordered):
        threshold = alpha / (len(ordered) - index)
        local_pass = value < threshold
        rejected = preceding_pass and local_pass
        rows.append(
            {
                "benchmark": benchmark,
                "rank": index + 1,
                "p_value": value,
                "threshold": threshold,
                "rejected": rejected,
            }
        )
        preceding_pass = rejected
    return {
        "familywise_alpha": alpha,
        "family_size": len(ordered),
        "all_rejected": all(row["rejected"] for row in rows),
        "ordered_tests": rows,
    }


def build_report() -> dict[str, Any]:
    config = load_config(CONFIG)
    preregister = json.loads(PREREGISTER.read_text(encoding="utf-8"))
    raw = json.loads(RAW_RESULT.read_text(encoding="utf-8"))
    output = config["output"]
    gsm8k_path = ROOT / output["gsm8k_candidate_path"]
    receipts_path = ROOT / output["gsm8k_receipts_path"]
    complete_path = ROOT / output["complete_candidate_path"]
    if (
        preregister.get("schema_version")
        != "nano_harness_complete_conditional_majority_preregister_v1"
        or preregister.get("identity", {}).get("config_sha256")
        != sha256_file(CONFIG)
        or raw.get("schema_version")
        != "nano_harness_complete_conditional_majority_raw_v1"
        or raw.get("identity", {}).get("config_sha256")
        != sha256_file(CONFIG)
        or raw.get("identity", {}).get("execution_sha256")
        != sha256_file(EXECUTION)
        or raw.get("identity", {}).get("gsm8k_candidate_sha256")
        != sha256_file(gsm8k_path)
        or raw.get("identity", {}).get("gsm8k_receipts_sha256")
        != sha256_file(receipts_path)
        or raw.get("identity", {}).get("complete_candidate_sha256")
        != sha256_file(complete_path)
    ):
        raise ValueError("complete conditional majority identity differs")

    candidate = jsonl_rows(complete_path)
    four_b = jsonl_rows(ROOT / config["baseline"]["four_b_raw_path"])
    nine_b = jsonl_rows(ROOT / config["baseline"]["nine_b_raw_path"])
    prior = jsonl_rows(
        ROOT / config["predecessors"]["prior_complete_candidate_path"]
    )
    receipts = jsonl_rows(receipts_path)
    candidate_by_id = _by_id(candidate)
    four_by_id = _by_id(four_b)
    nine_by_id = _by_id(nine_b)
    prior_by_id = _by_id(prior)
    receipt_by_id = _by_id(receipts)
    expected_ids = set(four_by_id)
    gsm8k_ids = {
        case_id
        for case_id, row in candidate_by_id.items()
        if row["benchmark"] == "gsm8k"
    }
    if (
        len(candidate) != 15_559
        or set(candidate_by_id) != expected_ids
        or set(nine_by_id) != expected_ids
        or set(prior_by_id) != expected_ids
        or len(receipts) != 1_319
        or set(receipt_by_id) != gsm8k_ids
    ):
        raise ValueError("complete conditional majority case sets differ")
    for case_id, row in candidate_by_id.items():
        if row["benchmark"] == "mmlu":
            source = four_by_id[case_id]
        elif row["benchmark"] == "gpqa_diamond":
            source = prior_by_id[case_id]
        else:
            continue
        if any(
            row.get(key) != source.get(key)
            for key in ("output", "prediction", "score", "expected")
        ):
            raise ValueError(
                "complete conditional majority preserved endpoint differs"
            )

    stats = config["statistics"]
    comparisons = {"versus_four_b": {}, "versus_nine_b": {}}
    for benchmark in ("gsm8k", "mmlu", "gpqa_diamond"):
        comparisons["versus_four_b"][benchmark] = comparison(
            candidate,
            four_b,
            benchmark=benchmark,
            bootstrap_samples=stats["bootstrap_samples"],
            bootstrap_seed=(
                f"{stats['bootstrap_seed']}:four:{benchmark}"
            ),
        )
        comparisons["versus_nine_b"][benchmark] = comparison(
            candidate,
            nine_b,
            benchmark=benchmark,
            bootstrap_samples=stats["bootstrap_samples"],
            bootstrap_seed=(
                f"{stats['bootstrap_seed']}:nine:{benchmark}"
            ),
        )

    gsm8k_candidate = [
        row for row in candidate if row["benchmark"] == "gsm8k"
    ]
    gsm8k_four = [
        row for row in four_b if row["benchmark"] == "gsm8k"
    ]
    gsm8k = gsm8k_gates(
        comparisons["versus_four_b"]["gsm8k"],
        comparisons["versus_nine_b"]["gsm8k"],
        receipts=receipts,
        candidate=gsm8k_candidate,
        four_b=gsm8k_four,
        alpha=stats["bonferroni_alpha"],
        minimum_candidate_only_wins=stats["minimum_candidate_only_wins"],
    )
    p_values = {
        benchmark: comparisons["versus_nine_b"][benchmark][
            "mcnemar_exact_p"
        ]
        for benchmark in ("gsm8k", "mmlu", "gpqa_diamond")
    }
    holm = holm_bonferroni(
        p_values,
        alpha=stats["final_three_benchmark_family"]["familywise_alpha"],
    )
    other_non_regression = all(
        comparisons["versus_four_b"][benchmark]["delta"] >= 0
        for benchmark in ("mmlu", "gpqa_diamond")
    )
    admitted = gsm8k["admitted"] and other_non_regression and holm[
        "all_rejected"
    ]
    receipt_values = [row["receipt"] for row in receipts]
    diagnostics = {
        "gsm8k": {
            "cases": len(receipts),
            "overrides": sum(
                bool(row["override"]) for row in receipt_values
            ),
            "fallbacks": sum(
                bool(row["fallback"]) for row in receipt_values
            ),
            "direct_strict_parseable": sum(
                bool(row["direct_strict_parseable"])
                for row in receipt_values
            ),
            "direct_strict_parse_failure": sum(
                not bool(row["direct_strict_parseable"])
                for row in receipt_values
            ),
            "minimum_vote_route_counts": dict(
                sorted(
                    Counter(
                        str(row["minimum_votes"])
                        for row in receipt_values
                    ).items()
                )
            ),
            "consensus_vote_counts": dict(
                sorted(
                    Counter(
                        str(row["consensus_votes"])
                        for row in receipt_values
                    ).items()
                )
            ),
        },
        "new_model_requests": {
            "gsm8k": 1_319 * 5,
            "mmlu": 0,
            "gpqa_diamond": 0,
        },
    }
    return {
        "schema_version": (
            "nano_harness_complete_conditional_majority_public_v1"
        ),
        "experiment_id": config["experiment_id"],
        "identity": {
            "result_revision": git_revision(),
            "config_sha256": sha256_file(CONFIG),
            "execution_sha256": sha256_file(EXECUTION),
            "preregister_sha256": sha256_file(PREREGISTER),
            "raw_result_sha256": sha256_file(RAW_RESULT),
            "gsm8k_candidate_sha256": sha256_file(gsm8k_path),
            "gsm8k_receipts_sha256": sha256_file(receipts_path),
            "complete_candidate_sha256": sha256_file(complete_path),
            "four_b_raw_sha256": config["baseline"]["four_b_raw_sha256"],
            "nine_b_raw_sha256": config["baseline"]["nine_b_raw_sha256"],
            "prior_complete_candidate_sha256": config["predecessors"][
                "prior_complete_candidate_sha256"
            ],
        },
        "comparisons": comparisons,
        "gsm8k_gate": gsm8k,
        "holm_bonferroni": holm,
        "diagnostics": diagnostics,
        "decision": {
            "complete_candidate_admitted": admitted,
            "complete_benchmarks_significantly_won": sum(
                row["rejected"] for row in holm["ordered_tests"]
            ),
            "all_benchmarks_non_regressing_vs_four_b": (
                gsm8k["admitted"] and other_non_regression
            ),
            "twenty_seven_b_parity_preregistration_allowed": admitted,
            "rerun_or_tuning_allowed": False,
            "next_action": (
                "Pre-register matched 27B parity on a declared subset."
                if admitted
                else (
                    "Publish negative evidence. Do not rerun or tune this "
                    "complete candidate."
                )
            ),
        },
        "evaluation_boundary": raw["evaluation_boundary"],
        "claim_boundary": (
            "This is a sequential complete GSM8K/MMLU/GPQA result. MMLU and "
            "GPQA reuse previously frozen endpoints; GSM8K is the only new "
            "generation. It is not independent three-benchmark replication "
            "or 27B parity."
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    rows = []
    for benchmark in ("gsm8k", "mmlu", "gpqa_diamond"):
        four = report["comparisons"]["versus_four_b"][benchmark]
        nine = report["comparisons"]["versus_nine_b"][benchmark]
        rows.append(
            f"| {benchmark} | {four['candidate_correct']}/{four['cases']} | "
            f"{four['baseline_correct']}/{four['cases']} | "
            f"{nine['baseline_correct']}/{nine['cases']} | "
            f"{nine['delta']:+.4f} | "
            f"[{nine['paired_bootstrap_95_ci'][0]:+.4f}, "
            f"{nine['paired_bootstrap_95_ci'][1]:+.4f}] | "
            f"{nine['mcnemar_exact_p']:.6g} |"
        )
    verdict = (
        "ADMIT"
        if report["decision"]["complete_candidate_admitted"]
        else "REJECT"
    )
    gsm8k = report["diagnostics"]["gsm8k"]
    return f"""# Qwen3.5 Complete Conditional-Majority v1 Result

## Verdict

**{verdict}.**

| Benchmark | Candidate | Direct 4B | Direct 9B | Delta vs 9B | 95% CI | McNemar p |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
{chr(10).join(rows)}

## What Ran

Only GSM8K received new model calls: five 4B samples for each of 1,319 cases.
MMLU preserved frozen 4B direct output, and GPQA reused the frozen V5
conservative-consensus endpoint. GSM8K had {gsm8k['overrides']} answer
replacements and {gsm8k['fallbacks']} fallbacks.

GSM8K uses the target-blind recovered parser for candidate, 4B, and 9B.
Its second complete treatment attempt is judged at Bonferroni
`alpha=0.025`. The final three-benchmark family uses Holm-Bonferroni at
familywise `alpha=0.05`.

## Decision Boundary

This is sequential evidence. MMLU and GPQA were not rerun, and this is not an
independent three-benchmark replication or a 27B comparison. Raw outputs stay
local and may not enter training, reward, or verifier data. No rerun or
post-observation tuning is allowed.
"""


def main() -> None:
    report = build_report()
    PUBLIC.parent.mkdir(parents=True, exist_ok=True)
    PUBLIC.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    MARKDOWN.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
