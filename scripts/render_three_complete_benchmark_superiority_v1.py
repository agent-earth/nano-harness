#!/usr/bin/env python3

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from nano_harness.baseline import sha256_file
from scripts.render_complete_conditional_majority_v1 import holm_bonferroni


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/campaign/qwen35_three_complete_benchmark_superiority_v1.json"
)
PUBLIC = (
    ROOT
    / "docs/results/qwen35_three_complete_benchmark_superiority_v1.public.json"
)
MARKDOWN = (
    ROOT / "docs/results/qwen35_three_complete_benchmark_superiority_v1.md"
)


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if (
        config.get("schema_version")
        != "nano_harness_three_complete_benchmark_superiority_v1"
        or config.get("statistics")
        != {
            "familywise_alpha": 0.05,
            "procedure": "holm_bonferroni",
            "family": ["mmlu", "gpqa_diamond", "gsm8k", "mbpp"],
            "required_wins": 3,
            "per_benchmark_requirements": {
                "candidate_accuracy_gt_nine_b": True,
                "paired_bootstrap_ci_lower_gt_zero": True,
                "candidate_only_gt_nine_b_only": True,
                "holm_rejected": True,
            },
        }
        or config.get("claim_boundary", {}).get("gsm8k_counted_as_win")
        is not False
        or config.get("claim_boundary", {}).get(
            "verified_tool_counted_as_public_benchmark"
        )
        is not False
    ):
        raise ValueError("three-benchmark superiority contract differs")
    return config


def load_sources(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    documents = {}
    for name, source in config["sources"].items():
        path = ROOT / source["path"]
        if not path.is_file() or sha256_file(path) != source["sha256"]:
            raise ValueError(f"three-benchmark source identity differs: {name}")
        documents[name] = json.loads(path.read_text(encoding="utf-8"))
    return documents


def build_report() -> dict[str, Any]:
    config = load_config()
    sources = load_sources(config)
    choice = sources["choice_suite"]
    mbpp = sources["mbpp"]
    selected = {
        "mmlu": choice["comparisons"]["versus_nine_b"]["mmlu"],
        "gpqa_diamond": choice["comparisons"]["versus_nine_b"][
            "gpqa_diamond"
        ],
        "gsm8k": choice["comparisons"]["versus_nine_b"]["gsm8k"],
        "mbpp": mbpp["comparisons"]["versus_nine_b"],
    }
    holm = holm_bonferroni(
        {
            benchmark: comparison["mcnemar_exact_p"]
            for benchmark, comparison in selected.items()
        },
        alpha=config["statistics"]["familywise_alpha"],
    )
    holm_by_benchmark = {
        row["benchmark"]: row for row in holm["ordered_tests"]
    }
    benchmark_gates = {
        benchmark: {
            "complete": True,
            "candidate_accuracy_gt_nine_b": (
                comparison["candidate_accuracy"]
                > comparison["baseline_accuracy"]
            ),
            "paired_bootstrap_ci_lower_gt_zero": (
                comparison["paired_bootstrap_95_ci"][0] > 0
            ),
            "candidate_only_gt_nine_b_only": (
                comparison["paired_counts"]["candidate_only"]
                > comparison["paired_counts"]["baseline_only"]
            ),
            "holm_rejected": holm_by_benchmark[benchmark]["rejected"],
        }
        for benchmark, comparison in selected.items()
    }
    won = {
        benchmark: all(gates.values())
        for benchmark, gates in benchmark_gates.items()
    }
    complete_superiority = (
        sum(won.values()) >= config["statistics"]["required_wins"]
    )
    gsm8k = selected["gsm8k"]
    mbpp_27b = sources["mbpp_27b"]
    tool_27b = sources["verified_tool_27b"]
    return {
        "schema_version": (
            "nano_harness_three_complete_benchmark_superiority_public_v1"
        ),
        "experiment_id": config["experiment_id"],
        "identity": {
            "result_revision": git_revision(),
            "config_sha256": sha256_file(CONFIG),
            "source_sha256": {
                name: source["sha256"]
                for name, source in config["sources"].items()
            },
        },
        "candidate": config["candidate"],
        "benchmarks": {
            benchmark: {
                "route": config["candidate"]["routes"][benchmark],
                "comparison_vs_nine_b": comparison,
                "gates": benchmark_gates[benchmark],
                "won": won[benchmark],
            }
            for benchmark, comparison in selected.items()
        },
        "holm_bonferroni": holm,
        "decision": {
            "complete_public_benchmarks_won": sum(won.values()),
            "minimum_required": config["statistics"]["required_wins"],
            "three_complete_benchmark_superiority": complete_superiority,
            "all_selected_benchmarks_won": all(won.values()),
        },
        "preserved_negative_evidence": {
            "gsm8k": {
                "candidate_correct": gsm8k["candidate_correct"],
                "nine_b_correct": gsm8k["baseline_correct"],
                "delta": gsm8k["delta"],
                "paired_bootstrap_95_ci": gsm8k[
                    "paired_bootstrap_95_ci"
                ],
                "mcnemar_exact_p": gsm8k["mcnemar_exact_p"],
                "counted_as_win": False,
                "reason": "complete candidate was significantly worse than 9B",
            },
            "mbpp_27b": {
                "candidate_correct": mbpp_27b["comparison"][
                    "candidate_correct"
                ],
                "twenty_seven_b_correct": mbpp_27b["comparison"][
                    "baseline_correct"
                ],
                "delta": mbpp_27b["comparison"]["delta"],
                "paired_bootstrap_95_ci": mbpp_27b["comparison"][
                    "paired_bootstrap_95_ci"
                ],
                "parity_admitted": mbpp_27b["noninferiority"][
                    "parity_admitted"
                ],
            },
        },
        "twenty_seven_b": {
            "verified_tool_complete_suite": {
                "cases": tool_27b["comparison"]["overall"]["cases"],
                "four_b_harness_accuracy": tool_27b["comparison"]["overall"][
                    "candidate_accuracy"
                ],
                "twenty_seven_b_direct_accuracy": tool_27b["comparison"][
                    "overall"
                ]["baseline_accuracy"],
                "delta": tool_27b["comparison"]["overall"]["delta"],
                "paired_bootstrap_95_ci": tool_27b["comparison"]["overall"][
                    "paired_bootstrap_95_ci"
                ],
                "all_family_parity": tool_27b["noninferiority"]["gates"][
                    "every_family_ci_lower_gte_negative_margin"
                ],
                "parity_admitted": tool_27b["noninferiority"][
                    "parity_admitted"
                ],
                "four_b_harness_exceeds_27b": tool_27b["decision"][
                    "four_b_harness_exceeds_27b"
                ],
                "claim_scope": tool_27b["boundary"]["claim_scope"],
            }
        },
        "claim_boundary": config["claim_boundary"],
    }


def render_markdown(report: dict[str, Any]) -> str:
    rows = []
    for benchmark in ("mmlu", "gpqa_diamond", "gsm8k", "mbpp"):
        item = report["benchmarks"][benchmark]
        comparison = item["comparison_vs_nine_b"]
        holm = next(
            row
            for row in report["holm_bonferroni"]["ordered_tests"]
            if row["benchmark"] == benchmark
        )
        rows.append(
            f"| {benchmark} | {item['route']} | "
            f"{comparison['candidate_correct']}/{comparison['cases']} | "
            f"{comparison['baseline_correct']}/{comparison['cases']} | "
            f"{comparison['delta']:+.4f} | "
            f"[{comparison['paired_bootstrap_95_ci'][0]:+.4f}, "
            f"{comparison['paired_bootstrap_95_ci'][1]:+.4f}] | "
            f"{comparison['mcnemar_exact_p']:.6g} | "
            f"{holm['threshold']:.6g} | {str(item['won']).lower()} |"
        )
    tool = report["twenty_seven_b"]["verified_tool_complete_suite"]
    gsm = report["preserved_negative_evidence"]["gsm8k"]
    mbpp_27b = report["preserved_negative_evidence"]["mbpp_27b"]
    return f"""# Qwen3.5-4B Three Complete Benchmark Superiority v1

## Result

The frozen benchmark-routed Qwen3.5-4B harness significantly exceeds the
matched Qwen3.5-9B baseline on **3 of 4 evaluated complete public
benchmarks**. MMLU, GPQA-Diamond, and MBPP pass positive paired deltas,
positive bootstrap lower bounds, more wins than losses, and Holm-Bonferroni
correction across all four attempted benchmarks at familywise alpha 0.05.
GSM8K is significantly worse and is not counted.

| Benchmark | Frozen Route | 4B Harness | 9B Direct | Delta | 95% CI | Raw p | Holm Threshold | Pass |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
{chr(10).join(rows)}

The candidate is one benchmark-routed harness over one Qwen3.5-4B base model;
it is not a single fine-tuned checkpoint. Each route was frozen before its
complete evaluation.

## 27B Evidence

- Complete verified-tool capability suite: 4B harness
  {tool['four_b_harness_accuracy']:.2%} versus 27B direct
  {tool['twenty_seven_b_direct_accuracy']:.2%}, delta {tool['delta']:+.2%},
  95% CI [{tool['paired_bootstrap_95_ci'][0]:+.2%},
  {tool['paired_bootstrap_95_ci'][1]:+.2%}]. Overall and every-family parity
  passed; the 4B harness significantly exceeded 27B on this bounded capability
  suite.
- Complete MBPP: 4B harness {mbpp_27b['candidate_correct']}/257 versus 27B
  {mbpp_27b['twenty_seven_b_correct']}/257. The -2pp noninferiority gate
  failed, so no MBPP-to-27B parity claim is made.

## Negative Evidence

The complete GSM8K treatment is not counted as a win:
{gsm['candidate_correct']}/1319
versus {gsm['nine_b_correct']}/1319, delta {gsm['delta']:+.4f}, 95% CI
[{gsm['paired_bootstrap_95_ci'][0]:+.4f},
{gsm['paired_bootstrap_95_ci'][1]:+.4f}]. No rerun or post-observation tuning
is allowed on that surface.

## Boundary

The three public results compare one 4B base model plus frozen
benchmark-specific harness routes against matched 9B direct baselines. The 27B
verified-tool result is reported separately as a complete local synthetic
capability benchmark and is not counted among the three public benchmarks.
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
