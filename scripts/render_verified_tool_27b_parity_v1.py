#!/usr/bin/env python3

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from nano_harness.baseline import sha256_file
from nano_harness.verified_tool_27b_parity import load_config
from nano_harness.verified_tool_execution import FAMILIES
from scripts.render_verified_tool_execution_v1 import paired_metrics


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/campaign/verified_tool_27b_parity_v1.json"
PREREGISTER = (
    ROOT / "docs/experiments/verified_tool_27b_parity_v1.preregister.json"
)
RAW_RESULT = (
    ROOT / "results/harness/verified-tool-27b-parity-v1/result.json"
)
RAW_CASES = (
    ROOT / "results/harness/verified-tool-27b-parity-v1/cases.jsonl"
)
PUBLIC = ROOT / "docs/results/verified_tool_27b_parity_v1.public.json"
MARKDOWN = ROOT / "docs/results/verified_tool_27b_parity_v1.md"


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
    return [
        {
            "case_id": row["case_id"],
            "correct": bool(row[arm]["correct"]),
            "prediction": row[arm]["prediction"],
        }
        for row in rows
    ]


def build_report() -> dict[str, Any]:
    config = load_config(CONFIG)
    preregister = json.loads(PREREGISTER.read_text(encoding="utf-8"))
    raw = json.loads(RAW_RESULT.read_text(encoding="utf-8"))
    if (
        preregister.get("schema_version")
        != "nano_harness_verified_tool_27b_parity_preregister_v1"
        or preregister.get("identity", {}).get("config_sha256")
        != sha256_file(CONFIG)
        or raw.get("schema_version")
        != "nano_harness_verified_tool_27b_merged_v1"
        or raw.get("identity", {}).get("config_sha256")
        != sha256_file(CONFIG)
        or raw.get("identity", {}).get("raw_sha256")
        != sha256_file(RAW_CASES)
        or raw.get("identity", {}).get("source_raw_sha256")
        != config["source"]["raw_sha256"]
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
            "cases": 256,
            "four_b_harness_reused": True,
            "four_b_generation_repeated": False,
            "nine_b_generation_repeated": False,
            "suite_changed": False,
        }
    ):
        raise ValueError("verified-tool 27B result identity differs")
    rows = read_jsonl(RAW_CASES)
    if len(rows) != 256 or len({row["case_id"] for row in rows}) != 256:
        raise ValueError("verified-tool 27B rows differ")
    stats = config["statistics"]
    candidate = arm_rows(rows, "four_b_harness")
    baseline = arm_rows(rows, "twenty_seven_b_direct")
    overall = paired_metrics(
        candidate,
        baseline,
        seed=f"{stats['bootstrap_seed']}:overall",
        samples=stats["bootstrap_samples"],
    )
    by_family = {}
    for family in FAMILIES:
        selected = [row for row in rows if row["family"] == family]
        by_family[family] = paired_metrics(
            arm_rows(selected, "four_b_harness"),
            arm_rows(selected, "twenty_seven_b_direct"),
            seed=f"{stats['bootstrap_seed']}:{family}",
            samples=stats["bootstrap_samples"],
        )
    margin = stats["noninferiority_margin"]
    gates = {
        "all_cases_complete": len(rows) == config["source"]["cases"],
        "four_b_harness_unchanged": (
            sum(row["four_b_harness"]["correct"] for row in rows)
            == config["source"]["harness_correct"]
        ),
        "twenty_seven_b_parseable_256": (
            raw["diagnostics"]["twenty_seven_b_parseable"] == 256
        ),
        "overall_ci_lower_gte_negative_margin": (
            overall["paired_bootstrap_95_ci"][0] >= -margin
        ),
        "every_family_ci_lower_gte_negative_margin": all(
            comparison["paired_bootstrap_95_ci"][0] >= -margin
            for comparison in by_family.values()
        ),
    }
    admitted = all(gates.values())
    return {
        "schema_version": "nano_harness_verified_tool_27b_parity_public_v1",
        "experiment_id": config["experiment_id"],
        "identity": {
            "result_revision": git_revision(),
            "config_sha256": sha256_file(CONFIG),
            "preregister_sha256": sha256_file(PREREGISTER),
            "raw_result_sha256": sha256_file(RAW_RESULT),
            "raw_cases_sha256": sha256_file(RAW_CASES),
            "source_raw_sha256": config["source"]["raw_sha256"],
            "case_contract_sha256": config["source"][
                "case_contract_sha256"
            ],
            "serving_report_sha256": config["twenty_seven_b"][
                "serving_report_sha256"
            ],
        },
        "comparison": {
            "overall": overall,
            "by_family": by_family,
        },
        "noninferiority": {
            "margin": margin,
            "gates": gates,
            "parity_admitted": admitted,
        },
        "diagnostics": raw["diagnostics"],
        "decision": {
            "complete_verified_tool_parity_with_27b": admitted,
            "four_b_harness_exceeds_27b": (
                overall["delta"] > 0
                and overall["paired_bootstrap_95_ci"][0] > 0
            ),
            "rerun_or_tuning_allowed": False,
            "next_action": (
                "Count verified-tool execution as a complete 27B-parity capability benchmark and continue the remaining complete-benchmark acceptance gaps."
                if admitted
                else (
                    "Freeze verified-tool 27B parity as negative evidence; "
                    "do not tune or rerun this suite."
                )
            ),
        },
        "boundary": {
            "complete_suite": True,
            "synthetic_evaluation_only": True,
            "benchmark_rows_or_outputs_used": False,
            "four_b_harness_reused": True,
            "four_b_generation_repeated": False,
            "nine_b_generation_repeated": False,
            "claim_scope": (
                "complete local verified-tool capability parity, not an "
                "external benchmark score"
            ),
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    overall = report["comparison"]["overall"]
    verdict = (
        "PARITY ADMITTED"
        if report["noninferiority"]["parity_admitted"]
        else "PARITY REJECTED"
    )
    family_rows = "\n".join(
        f"| {family} | {row['candidate_accuracy']:.4f} | "
        f"{row['baseline_accuracy']:.4f} | {row['delta']:+.4f} | "
        f"[{row['paired_bootstrap_95_ci'][0]:+.4f}, "
        f"{row['paired_bootstrap_95_ci'][1]:+.4f}] |"
        for family, row in report["comparison"]["by_family"].items()
    )
    return f"""# Verified-Tool 27B Parity v1

## Verdict

**{verdict}.**

| Scope | 4B Harness | 27B Direct | Delta | 95% CI |
| --- | ---: | ---: | ---: | --- |
| overall | {overall['candidate_accuracy']:.4f} | {overall['baseline_accuracy']:.4f} | {overall['delta']:+.4f} | [{overall['paired_bootstrap_95_ci'][0]:+.4f}, {overall['paired_bootstrap_95_ci'][1]:+.4f}] |
{family_rows}

The frozen 4B harness result was reused without model generation. Only the 27B
direct arm was generated. Parity requires the overall and every-family 95%
paired-bootstrap lower bounds to be at least -0.02.

This is a complete local synthetic capability benchmark, not an external
benchmark score. It contains no MMLU, GSM8K, GPQA, MBPP, canary, or holdout
rows or outputs.
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
