#!/usr/bin/env python3

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from nano_harness.baseline import compare_baselines, sha256_file, summarize_baseline
from nano_harness.v5_complete_treatment import load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/campaign/qwen35_v5_complete_treatment_v1.json"
PREREG = (
    ROOT / "docs/experiments/qwen35_v5_complete_treatment_v1.preregister.json"
)
RAW_RESULT = ROOT / "results/full/qwen35-v5-complete-treatment-v1/result.json"
PUBLIC_JSON = ROOT / "docs/results/qwen35_v5_complete_treatment_v1.public.json"
MARKDOWN = ROOT / "docs/results/qwen35_v5_complete_treatment_v1.md"


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def build_report() -> dict:
    config = load_config(CONFIG)
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    raw = json.loads(RAW_RESULT.read_text(encoding="utf-8"))
    candidate = ROOT / config["output"]["candidate_path"]
    receipts = candidate.with_suffix(".receipts.jsonl")
    if (
        prereg.get("schema_version")
        != "nano_harness_v5_complete_treatment_preregister_v1"
        or prereg.get("identity", {}).get("config_sha256")
        != sha256_file(CONFIG)
        or raw.get("schema_version")
        != "nano_harness_v5_complete_treatment_result_v1"
        or raw.get("identity", {}).get("config_sha256")
        != sha256_file(CONFIG)
        or raw.get("identity", {}).get("candidate_sha256")
        != sha256_file(candidate)
        or raw.get("identity", {}).get("receipts_sha256")
        != sha256_file(receipts)
    ):
        raise ValueError("V5 complete treatment result identity differs")
    summary = summarize_baseline(candidate)
    versus_four = compare_baselines(
        candidate,
        ROOT / config["baseline"]["four_b_raw_path"],
        bootstrap_samples=config["statistics"]["bootstrap_samples"],
        bootstrap_seed=config["statistics"]["bootstrap_seed"] + ":four",
    )
    versus_nine = compare_baselines(
        candidate,
        ROOT / config["baseline"]["nine_b_raw_path"],
        bootstrap_samples=config["statistics"]["bootstrap_samples"],
        bootstrap_seed=config["statistics"]["bootstrap_seed"] + ":nine",
    )
    gates = {}
    for benchmark in ("gsm8k", "mmlu", "gpqa_diamond"):
        four = versus_four["benchmarks"][benchmark]
        nine = versus_nine["benchmarks"][benchmark]
        gates[f"{benchmark}_non_regression_vs_four_b"] = four["delta"] >= 0
        gates[f"{benchmark}_superior_to_nine_b"] = (
            nine["delta"] > 0
            and nine["paired_bootstrap_95_ci"][0] > 0
            and nine["mcnemar_exact_p"] < config["statistics"]["alpha"]
            and nine["paired_counts"]["candidate_only"]
            > nine["paired_counts"]["baseline_only"]
        )
    gates["all_rows_complete"] = (
        summary["completed_cases"] == 15_559 and summary["error_cases"] == 0
    )
    gates["all_three_complete_benchmarks_won"] = sum(
        gates[f"{benchmark}_superior_to_nine_b"]
        for benchmark in ("gsm8k", "mmlu", "gpqa_diamond")
    ) == 3
    admitted = all(gates.values())
    return {
        "schema_version": "nano_harness_v5_complete_treatment_public_v1",
        "experiment_id": config["experiment_id"],
        "identity": {
            "result_revision": git_revision(),
            "config_sha256": sha256_file(CONFIG),
            "preregister_sha256": sha256_file(PREREG),
            "raw_result_sha256": sha256_file(RAW_RESULT),
            "candidate_sha256": sha256_file(candidate),
            "receipts_sha256": sha256_file(receipts),
            "case_manifest_sha256": config["baseline"]["case_manifest_sha256"],
            "four_b_raw_sha256": config["baseline"]["four_b_raw_sha256"],
            "nine_b_raw_sha256": config["baseline"]["nine_b_raw_sha256"],
            "v5_report_sha256": config["treatment"]["v5_report_sha256"],
        },
        "summary": summary,
        "comparisons": {
            "versus_four_b": versus_four,
            "versus_nine_b": versus_nine,
        },
        "decision": {
            "gates": gates,
            "complete_treatment_admitted": admitted,
            "complete_benchmarks_significantly_won": sum(
                gates[f"{benchmark}_superior_to_nine_b"]
                for benchmark in ("gsm8k", "mmlu", "gpqa_diamond")
            ),
            "twenty_seven_b_parity_preregistration_allowed": admitted,
            "rl_or_opd_allowed": False,
            "rerun_or_tuning_allowed": False,
            "next_action": (
                "Pre-register matched 27B parity on GSM8K and MMLU."
                if admitted
                else "Publish negative evidence; do not rerun or tune this treatment."
            ),
        },
        "evaluation_boundary": raw["evaluation_boundary"],
        "claim_boundary": (
            "This is the complete matched GSM8K/MMLU/GPQA treatment result. "
            "It is not 27B parity or an agent-benchmark result."
        ),
    }


def render_markdown(report: dict) -> str:
    verdict = (
        "ADMIT"
        if report["decision"]["complete_treatment_admitted"]
        else "REJECT"
    )
    rows = []
    for benchmark in ("gsm8k", "mmlu", "gpqa_diamond"):
        four = report["comparisons"]["versus_four_b"]["benchmarks"][benchmark]
        nine = report["comparisons"]["versus_nine_b"]["benchmarks"][benchmark]
        rows.append(
            f"| {benchmark} | {four['candidate_correct']}/{four['cases']} | "
            f"{four['baseline_correct']}/{four['cases']} | "
            f"{nine['baseline_correct']}/{nine['cases']} | "
            f"{nine['delta']:+.4f} | "
            f"[{nine['paired_bootstrap_95_ci'][0]:+.4f}, "
            f"{nine['paired_bootstrap_95_ci'][1]:+.4f}] | "
            f"{nine['mcnemar_exact_p']:.6g} |"
        )
    return f"""# Qwen3.5 V5 Complete Treatment v1 Result

## Verdict

**{verdict}.**

| Benchmark | Candidate | Direct 4B | Direct 9B | Delta vs 9B | 95% CI | McNemar p |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
{chr(10).join(rows)}

## Gates

```json
{json.dumps(report['decision']['gates'], indent=2, sort_keys=True)}
```

No rerun, prompt, route, parser, budget, consensus, or scorer change is allowed
after this complete result.
"""


def main() -> None:
    report = build_report()
    PUBLIC_JSON.parent.mkdir(parents=True, exist_ok=True)
    PUBLIC_JSON.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    MARKDOWN.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
