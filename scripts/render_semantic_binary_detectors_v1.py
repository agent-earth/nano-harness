#!/usr/bin/env python3

from __future__ import annotations

import json
import subprocess
from collections import Counter
from pathlib import Path

from nano_harness.baseline import sha256_file
from nano_harness.semantic_binary_detectors import (
    ALL_FAMILIES,
    build_cases,
    load_config,
    parent_config,
    parse_detection,
)
from nano_harness.semantic_model_router import summarize_rows
from nano_harness.verified_tool_execution import public_case_contract
from scripts.render_verified_tool_execution_v1 import paired_metrics


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/harness/qwen35_semantic_binary_detectors_v1.json"
PREREG = (
    ROOT
    / "docs/experiments/qwen35_semantic_binary_detectors_v1.preregister.json"
)
RAW = ROOT / "results/harness/qwen35-semantic-binary-detectors-v1/result.json"
PUBLIC_JSON = (
    ROOT / "docs/results/qwen35_semantic_binary_detectors_v1.public.json"
)
MARKDOWN = ROOT / "docs/results/qwen35_semantic_binary_detectors_v1.md"
PREREG_REVISION = "83835e3f9a13ad5145fd95205b49e2ff66ee6a31"


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def admission_gates(
    raw: dict,
    candidate_vs_four: dict,
    candidate_vs_nine: dict,
) -> dict[str, bool]:
    config = load_config(CONFIG)
    routing = raw["routing"]
    arms = raw["arms"]
    return {
        "all_rows_complete_and_parseable": all(
            arms[name]["cases"] == 128 and arms[name]["parseable"] == 128
            for name in (
                "four_b_direct",
                "nine_b_direct",
                "four_b_binary_detectors",
            )
        ),
        "all_detector_outputs_parseable": all(
            detection["yes"] is not None
            for receipt in raw["receipts"].values()
            for detection in receipt["detections"].values()
        ),
        "detector_composition_correct_128": (
            routing["detector_correct"] == 128
        ),
        "positive_route_recall_64": routing["positive_route_correct"] == 64,
        "negative_none_correct_64": routing["negative_none_correct"] == 64,
        "negative_false_positive_routes_zero": (
            routing["negative_false_positive_routes"] == 0
        ),
        "conflicts_zero": routing["conflicts"] == 0,
        "positive_verified_executions_64": (
            routing["verified_executions"] == 64
        ),
        "fallbacks_zero": routing["fallbacks"] == 0,
        "negative_candidate_exact_direct_parity": all(
            all(
                candidate.get(field) == direct.get(field)
                for field in ("output", "prediction", "parseable", "correct")
            )
            for case, candidate, direct in zip(
                build_cases(config),
                raw["candidate_rows"],
                raw["four_b_rows"],
            )
            if not case["positive"]
        ),
        "candidate_vs_four_b_significant": (
            candidate_vs_four["candidate_accuracy"]
            > candidate_vs_four["baseline_accuracy"]
            and candidate_vs_four["paired_bootstrap_95_ci"][0] > 0
            and candidate_vs_four["mcnemar_exact_p"]
            < config.significance_alpha
        ),
        "candidate_vs_four_b_minimum_wins": (
            candidate_vs_four["paired_counts"]["candidate_only"]
            >= config.minimum_harness_wins
        ),
        "candidate_vs_four_b_maximum_losses": (
            candidate_vs_four["paired_counts"]["baseline_only"]
            <= config.maximum_harness_losses
        ),
        "candidate_vs_nine_b_significant": (
            candidate_vs_nine["candidate_accuracy"]
            > candidate_vs_nine["baseline_accuracy"]
            and candidate_vs_nine["paired_bootstrap_95_ci"][0] > 0
            and candidate_vs_nine["mcnemar_exact_p"]
            < config.significance_alpha
        ),
        "candidate_vs_nine_b_minimum_wins": (
            candidate_vs_nine["paired_counts"]["candidate_only"]
            >= config.minimum_harness_wins
        ),
        "candidate_vs_nine_b_maximum_losses": (
            candidate_vs_nine["paired_counts"]["baseline_only"]
            <= config.maximum_harness_losses
        ),
        "every_family_non_regression": all(
            arms["four_b_binary_detectors"]["by_family"][family]["correct"]
            >= arms["four_b_direct"]["by_family"][family]["correct"]
            and arms["four_b_binary_detectors"]["by_family"][family]["correct"]
            >= arms["nine_b_direct"]["by_family"][family]["correct"]
            for family in ALL_FAMILIES
        ),
    }


def build_report() -> dict:
    config = load_config(CONFIG)
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    raw = json.loads(RAW.read_text(encoding="utf-8"))
    cases = build_cases(config)
    by_id = {case["case_id"]: case for case in cases}
    if (
        prereg.get("schema_version")
        != "nano_harness_semantic_binary_detectors_preregister_v1"
        or prereg.get("identity", {}).get("config_sha256")
        != sha256_file(CONFIG)
        or prereg.get("identity", {}).get("code_revision")
        != "10141bb6fcc81cf82e0f62fbd3d3e1233fb802bb"
        or raw.get("schema_version")
        != "nano_harness_semantic_binary_detectors_result_v1"
        or raw.get("identity", {}).get("case_contract")
        != public_case_contract(cases)
        or raw.get("identity", {}).get("mechanism_config_sha256")
        != config.mechanism_config_sha256
        or raw.get("identity", {}).get("multiclass_report_sha256")
        != config.multiclass_report_sha256
    ):
        raise ValueError("semantic binary detector result identity differs")
    if (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", PREREG_REVISION, "HEAD"],
            cwd=ROOT,
            check=False,
        ).returncode
        != 0
    ):
        raise ValueError("semantic binary detector preregistration is not committed")
    for key in ("four_b_rows", "nine_b_rows", "candidate_rows"):
        rows = raw[key]
        if len(rows) != 128 or {row["case_id"] for row in rows} != set(by_id):
            raise ValueError(f"semantic binary detector {key} identities differ")
    if set(raw["receipts"]) != set(by_id):
        raise ValueError("semantic binary detector receipt identities differ")
    confusion: Counter[tuple[str, str]] = Counter()
    detector_counts: Counter[tuple[str, bool | None]] = Counter()
    for case_id, receipt in raw["receipts"].items():
        case = by_id[case_id]
        if (
            receipt["expected_route"] != case["expected_detector"]
            or receipt["detector_correct"]
            != (receipt["selected_route"] == case["expected_detector"])
        ):
            raise ValueError("semantic binary detector composition differs")
        confusion[(receipt["expected_route"], receipt["selected_route"])] += 1
        for family, detection in receipt["detections"].items():
            if (
                parse_detection(detection["output"]) != detection["yes"]
                or detection["uses_case_metadata"]
                or detection["uses_expected_answer"]
                or detection["uses_case_correctness"]
            ):
                raise ValueError("semantic binary detector receipt differs")
            detector_counts[(family, detection["yes"])] += 1
    expected_confusion = {
        ("NONE", "NONE"): 64,
        ("first_strict_profit_period", "NONE"): 32,
        ("implicit_scale_total", "NONE"): 32,
    }
    if dict(confusion) != expected_confusion:
        raise ValueError("semantic binary detector confusion differs")
    expected_boundary = {
        "training_eligible_cases": 0,
        "detectors_use_case_metadata": False,
        "detectors_use_expected_answer": False,
        "detectors_use_case_correctness": False,
        "executor_uses_expected_answer": False,
        "executor_uses_case_correctness": False,
        "benchmark_rows_loaded": False,
        "canary_rows_loaded": False,
        "holdout_rows_loaded": False,
    }
    if (
        raw["evaluation_boundary"] != expected_boundary
        or summarize_rows(raw["four_b_rows"]) != raw["arms"]["four_b_direct"]
        or summarize_rows(raw["nine_b_rows"]) != raw["arms"]["nine_b_direct"]
        or summarize_rows(raw["candidate_rows"])
        != raw["arms"]["four_b_binary_detectors"]
    ):
        raise ValueError("semantic binary detector aggregate boundary differs")
    candidate_vs_four = paired_metrics(
        raw["candidate_rows"],
        raw["four_b_rows"],
        seed=f"{config.bootstrap_seed}:candidate-four",
        samples=config.bootstrap_samples,
    )
    candidate_vs_nine = paired_metrics(
        raw["candidate_rows"],
        raw["nine_b_rows"],
        seed=f"{config.bootstrap_seed}:candidate-nine",
        samples=config.bootstrap_samples,
    )
    four_vs_nine = paired_metrics(
        raw["four_b_rows"],
        raw["nine_b_rows"],
        seed=f"{config.bootstrap_seed}:four-nine",
        samples=config.bootstrap_samples,
    )
    gates = admission_gates(raw, candidate_vs_four, candidate_vs_nine)
    return {
        "schema_version": "nano_harness_semantic_binary_detectors_public_v1",
        "experiment_id": config.experiment_id,
        "identity": {
            "result_revision": git_revision(),
            "preregister_revision": PREREG_REVISION,
            "config_sha256": sha256_file(CONFIG),
            "preregister_sha256": sha256_file(PREREG),
            "raw_result_sha256": sha256_file(RAW),
            "case_contract_sha256": prereg["identity"][
                "case_contract_sha256"
            ],
            "mechanism_config_sha256": config.mechanism_config_sha256,
            "multiclass_report_sha256": config.multiclass_report_sha256,
        },
        "data": {
            "cases": 128,
            "positive_cases": 64,
            "negative_cases": 64,
            "family_counts": {family: 32 for family in ALL_FAMILIES},
            "freshness": prereg["freshness"],
            "benchmark_canary_holdout_rows_or_outputs": 0,
            "training_eligible_cases": 0,
        },
        "arms": raw["arms"],
        "routing": raw["routing"],
        "confusion": [
            {
                "expected_route": expected,
                "selected_route": selected,
                "cases": count,
            }
            for (expected, selected), count in sorted(
                confusion.items(), key=lambda item: str(item[0])
            )
        ],
        "detector_counts": [
            {"detector": family, "yes": yes, "cases": count}
            for (family, yes), count in sorted(
                detector_counts.items(), key=lambda item: str(item[0])
            )
        ],
        "comparisons": {
            "candidate_vs_four_b": candidate_vs_four,
            "candidate_vs_nine_b": candidate_vs_nine,
            "four_b_vs_nine_b": four_vs_nine,
        },
        "decision": {
            "gates": gates,
            "detectors_admitted": False,
            "real_question_detector_scan_preregistration_allowed": False,
            "benchmark_generation_allowed": False,
            "canary_rerun_allowed": False,
            "independent_holdout_allowed": False,
            "training_allowed": False,
            "further_tuning_or_rerun_on_observed_cases_allowed": False,
            "negative_precision_supported": (
                raw["routing"]["negative_false_positive_routes"] == 0
            ),
            "positive_recall_supported": (
                raw["routing"]["positive_route_correct"] == 64
            ),
            "next_action": (
                "Reject binary detectors. Both emitted NO for every case. "
                "Do not tune or rerun. Move router learning into a separately "
                "pre-registered synthetic SFT classification objective with "
                "broad paraphrase positives and unsupported negatives; keep "
                "typed executors frozen."
            ),
        },
        "evaluation_boundary": raw["evaluation_boundary"],
        "claim_boundary": (
            "This is negative fresh-local detector evidence. It establishes "
            "zero unsupported false positives only because both detectors "
            "always returned NO; positive recall is zero. It authorizes no "
            "real scan, benchmark generation, canary, holdout, or training."
        ),
    }


def render_markdown(report: dict) -> str:
    return f"""# Qwen3.5 Semantic Binary Detectors v1 Result

## 结论

**拒绝。两个 detector 对128条全部输出 NO。**

- negative NONE：64/64；
- negative false positive：0；
- positive recall：0/64；
- verified executions：0；
- candidate / 4B direct / 9B direct：均 0/128。

零误报来自恒 NO，不是可用 precision-recall tradeoff。按冻结 gate 不允许
real question scan，也不能调 prompt 或重跑。

## Confusion

```json
{json.dumps(report['confusion'], indent=2, sort_keys=True)}
```

## Detector Outputs

```json
{json.dumps(report['detector_counts'], indent=2, sort_keys=True)}
```

## Frozen Gates

```json
{json.dumps(report['decision']['gates'], indent=2, sort_keys=True)}
```

## Evidence

- prereg commit：`{report['identity']['preregister_revision']}`；
- config SHA：`{report['identity']['config_sha256']}`；
- prereg SHA：`{report['identity']['preregister_sha256']}`；
- raw result SHA：`{report['identity']['raw_result_sha256']}`；
- case contract SHA：`{report['identity']['case_contract_sha256']}`。

## 下一步

不再搜索 inference prompt。改为另行预注册 synthetic router SFT
classification objective：大量 paraphrase positives + unsupported negatives，typed
executors、validators、feedback equality 和 fallback 全部冻结。真实 benchmark、
canary、holdout、training generation 在该 SFT/data contract 提交前继续关闭。
"""


def main() -> None:
    report = build_report()
    PUBLIC_JSON.parent.mkdir(parents=True, exist_ok=True)
    PUBLIC_JSON.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    MARKDOWN.write_text(render_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "experiment_id": report["experiment_id"],
                "arms": report["arms"],
                "routing": report["routing"],
                "confusion": report["confusion"],
                "detector_counts": report["detector_counts"],
                "decision": report["decision"],
                "public_json": str(PUBLIC_JSON),
                "markdown": str(MARKDOWN),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
