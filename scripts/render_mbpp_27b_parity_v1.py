#!/usr/bin/env python3

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from nano_harness.baseline import sha256_file
from nano_harness.mbpp_27b_parity import load_config
from scripts.render_orca_self_consistency_v1 import paired_metrics


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/campaign/mbpp_27b_parity_v1.json"
PREREGISTER = ROOT / "docs/experiments/mbpp_27b_parity_v1.preregister.json"
RAW_RESULT = ROOT / "results/harness/mbpp-27b-parity-v1/result.json"
RAW_CASES = ROOT / "results/harness/mbpp-27b-parity-v1/cases.jsonl"
PUBLIC = ROOT / "docs/results/mbpp_27b_parity_v1.public.json"
MARKDOWN = ROOT / "docs/results/mbpp_27b_parity_v1.md"


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def arm_rows(
    rows: list[dict[str, Any]], arm: str
) -> list[dict[str, Any]]:
    if arm == "candidate":
        return [
            {
                "case_id": row["case_id"],
                "prediction": "pass" if row[arm]["correct"] else "fail",
                "correct": bool(row[arm]["correct"]),
            }
            for row in rows
        ]
    return [
        {
            "case_id": row["case_id"],
            "prediction": (
                "pass" if row[arm]["test_result"]["full_pass"] else "fail"
            ),
            "correct": bool(row[arm]["test_result"]["full_pass"]),
        }
        for row in rows
    ]


def build_report() -> dict[str, Any]:
    config = load_config(CONFIG)
    preregister = json.loads(PREREGISTER.read_text(encoding="utf-8"))
    raw = json.loads(RAW_RESULT.read_text(encoding="utf-8"))
    if (
        preregister.get("schema_version")
        != "nano_harness_mbpp_27b_parity_preregister_v1"
        or preregister.get("identity", {}).get("config_sha256")
        != sha256_file(CONFIG)
        or raw.get("schema_version")
        != "nano_harness_mbpp_27b_parity_merged_v1"
        or raw.get("identity", {}).get("config_sha256")
        != sha256_file(CONFIG)
        or raw.get("identity", {}).get("raw_sha256")
        != sha256_file(RAW_CASES)
        or raw.get("identity", {}).get("candidate_raw_sha256")
        != config["candidate"]["raw_sha256"]
        or len(raw.get("identity", {}).get("shards", []))
        != config["execution"]["num_shards"]
        or any(
            row.get("sha256")
            != sha256_file(
                ROOT
                / config["output_dir"]
                / f"shard-{row.get('shard_id')}.jsonl"
            )
            for row in raw.get("identity", {}).get("shards", [])
        )
        or raw.get("surface")
        != {
            "benchmark": "mbpp",
            "split": "sanitized_test",
            "complete_benchmark": True,
            "cases": 257,
            "four_b_generation_repeated": False,
            "nine_b_generation_repeated": False,
        }
    ):
        raise ValueError("MBPP 27B parity result identity differs")
    rows = read_jsonl(RAW_CASES)
    if len(rows) != 257 or len({row["case_id"] for row in rows}) != 257:
        raise ValueError("MBPP 27B parity rows differ")
    comparison = paired_metrics(
        arm_rows(rows, "candidate"),
        arm_rows(rows, "twenty_seven_b"),
        bootstrap_samples=config["statistics"]["bootstrap_samples"],
        bootstrap_seed=config["statistics"]["bootstrap_seed"],
    )
    margin = config["statistics"]["noninferiority_margin"]
    gates = {
        "all_cases_complete": len(rows) == config["dataset"]["test_rows"],
        "twenty_seven_b_parse_failures_zero": (
            comparison["baseline_parse_failures"] == 0
        ),
        "paired_bootstrap_ci_lower_gte_negative_margin": (
            comparison["paired_bootstrap_95_ci"][0] >= -margin
        ),
    }
    admitted = all(gates.values())
    return {
        "schema_version": "nano_harness_mbpp_27b_parity_public_v1",
        "experiment_id": config["experiment_id"],
        "identity": {
            "result_revision": git_revision(),
            "config_sha256": sha256_file(CONFIG),
            "preregister_sha256": sha256_file(PREREGISTER),
            "raw_result_sha256": sha256_file(RAW_RESULT),
            "raw_cases_sha256": sha256_file(RAW_CASES),
            "candidate_raw_sha256": config["candidate"]["raw_sha256"],
            "serving_report_sha256": config["twenty_seven_b"][
                "serving_report_sha256"
            ],
        },
        "comparison": comparison,
        "noninferiority": {
            "margin": margin,
            "gates": gates,
            "parity_admitted": admitted,
        },
        "diagnostics": raw["diagnostics"],
        "decision": {
            "mbpp_complete_parity_with_27b": admitted,
            "rerun_or_tuning_allowed": False,
            "next_action": (
                "Count MBPP as one complete 27B-parity benchmark and continue the remaining acceptance gaps."
                if admitted
                else (
                    "Freeze MBPP 27B parity as negative evidence and continue "
                    "on a separately preregistered benchmark."
                )
            ),
        },
        "boundary": {
            "benchmark": "mbpp",
            "split": "sanitized_test",
            "complete_benchmark": True,
            "four_b_candidate_reused": True,
            "four_b_generation_repeated": False,
            "nine_b_generation_repeated": False,
            "reference_solution_used_for_generation": False,
            "claim_scope": "complete MBPP parity against Qwen3.5-27B",
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    comparison = report["comparison"]
    verdict = (
        "PARITY ADMITTED"
        if report["noninferiority"]["parity_admitted"]
        else "PARITY REJECTED"
    )
    return f"""# MBPP 27B Parity v1

## Verdict

**{verdict}.**

| Arm | Correct | Accuracy |
| --- | ---: | ---: |
| Frozen 4B harness | {comparison['candidate_correct']}/257 | {comparison['candidate_accuracy']:.4f} |
| Qwen3.5-27B BF16 direct | {comparison['baseline_correct']}/257 | {comparison['baseline_accuracy']:.4f} |

- paired delta, 4B harness minus 27B: {comparison['delta']:+.4f};
- paired-bootstrap 95% CI:
  [{comparison['paired_bootstrap_95_ci'][0]:+.4f},
  {comparison['paired_bootstrap_95_ci'][1]:+.4f}];
- noninferiority margin: -0.0200;
- candidate-only / 27B-only: {comparison['paired_counts']['candidate_only']} /
  {comparison['paired_counts']['baseline_only']};
- exact McNemar p: {comparison['mcnemar_exact_p']:.6g}.

This is a complete 257-case MBPP parity comparison. The 4B arm is the frozen
one-shot test output and was not regenerated. The 27B arm is direct generation
from the validated BF16 TP=2 service. No benchmark row or output may enter
training, reward, verifier fitting, or post-observation tuning.
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
