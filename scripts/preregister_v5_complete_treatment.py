#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

from nano_harness.baseline import sha256_file
from nano_harness.v5_complete_treatment import jsonl_ids, load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/campaign/qwen35_v5_complete_treatment_v1.json"
JSON_OUTPUT = (
    ROOT / "docs/experiments/qwen35_v5_complete_treatment_v1.preregister.json"
)
MARKDOWN_OUTPUT = (
    ROOT / "docs/experiments/qwen35_v5_complete_treatment_v1.md"
)


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def build_receipt() -> dict:
    config = load_config(CONFIG)
    paths = {}
    for section, values in (
        ("baseline", config["baseline"]),
        ("treatment", config["treatment"]),
    ):
        for key, value in values.items():
            if key.endswith("_path"):
                digest_key = key.removesuffix("_path") + "_sha256"
                path = ROOT / value
                if (
                    not path.is_file()
                    or sha256_file(path) != values[digest_key]
                ):
                    raise ValueError(f"V5 treatment {section}.{key} differs")
                paths[f"{section}.{key}"] = path

    public_cases = json.loads(
        paths["baseline.case_manifest_path"].read_text(encoding="utf-8")
    )
    if len(public_cases) != 15_559:
        raise ValueError("V5 complete case count differs")
    ids = [str(row["case_id"]) for row in public_cases]
    counts = Counter(str(row["benchmark"]) for row in public_cases)
    if (
        len(ids) != len(set(ids))
        or counts
        != {
            "gsm8k": 1_319,
            "mmlu": 14_042,
            "gpqa_diamond": 198,
        }
    ):
        raise ValueError("V5 complete case identity differs")
    four_ids = jsonl_ids(paths["baseline.four_b_raw_path"])
    nine_ids = jsonl_ids(paths["baseline.nine_b_raw_path"])
    if set(four_ids) != set(ids) or set(nine_ids) != set(ids):
        raise ValueError("V5 complete raw case set differs")

    baseline = json.loads(
        paths["baseline.report_path"].read_text(encoding="utf-8")
    )
    v5 = json.loads(
        paths["treatment.v5_report_path"].read_text(encoding="utf-8")
    )
    if (
        baseline.get("decision", {}).get("direct_baseline_accepted")
        is not True
        or baseline.get("decision", {}).get(
            "complete_benchmarks_significantly_won"
        )
        != 1
        or baseline.get("identity", {}).get("case_contract_sha256")
        != "858656f58decf8bbc23c70101dabcffc6ef12e049771e043575927743c6cfd10"
        or v5.get("decision", {}).get(
            "router_skill_registry_v5_admitted"
        )
        is not True
        or v5.get("decision", {}).get(
            "benchmark_treatment_preregistration_allowed"
        )
        is not True
        or v5.get("decision", {}).get("benchmark_generation_allowed")
        is not False
        or v5.get("decision", {}).get(
            "v1_v2_v3_v4_v5_rerun_allowed"
        )
        is not False
    ):
        raise ValueError("V5 complete treatment predecessor gate differs")

    return {
        "schema_version": "nano_harness_v5_complete_treatment_preregister_v1",
        "experiment_id": config["experiment_id"],
        "identity": {
            "code_revision": git_revision(),
            "config_sha256": sha256_file(CONFIG),
            "case_manifest_sha256": config["baseline"][
                "case_manifest_sha256"
            ],
            "case_ids_sha256": hashlib.sha256(
                "\n".join(sorted(ids)).encode()
            ).hexdigest(),
            "four_b_raw_sha256": config["baseline"]["four_b_raw_sha256"],
            "nine_b_raw_sha256": config["baseline"]["nine_b_raw_sha256"],
            "baseline_report_sha256": config["baseline"]["report_sha256"],
            "v5_report_sha256": config["treatment"]["v5_report_sha256"],
        },
        "surface": {
            "cases": len(ids),
            "by_benchmark": dict(sorted(counts.items())),
            "case_set_matches_both_direct_arms": True,
            "prompts_or_outputs_published": False,
        },
        "treatment": {
            "routes": config["routes"],
            "mmlu_direct_preserved": True,
            "gsm8k_fail_closed_to_direct": True,
            "gpqa_fail_closed_to_direct": True,
        },
        "statistics": config["statistics"],
        "acceptance": {
            "all_15559_rows_complete_and_parseable": True,
            "per_benchmark_non_regression_vs_direct_4b": True,
            "gsm8k_superior_to_9b_ci_positive_mcnemar_lt_005": True,
            "mmlu_superior_to_9b_ci_positive_mcnemar_lt_005": True,
            "gpqa_superior_to_9b_ci_positive_mcnemar_lt_005": True,
            "minimum_complete_benchmarks_significantly_won": 3,
            "candidate_only_gt_nine_b_only_each_benchmark": True,
            "api_errors_zero": True,
        },
        "decision_policy": {
            "passed": (
                "Publish complete treatment evidence and separately open "
                "27B parity preregistration."
            ),
            "failed": (
                "Publish negative evidence. Do not rerun or tune this "
                "complete treatment."
            ),
            "forbidden_after_observation": [
                "route_change",
                "prompt_change",
                "parser_or_scorer_change",
                "budget_or_replica_change",
                "consensus_or_override_rule_change",
                "case_selection_change",
                "model_or_adapter_change",
                "rerun",
                "training_on_benchmark_rows_or_outputs",
            ],
        },
        "execution_boundary": config["execution_boundary"],
        "claim_boundary": (
            "This pre-registers one complete matched benchmark treatment. "
            "It validates identities and counts only, starts no generation, "
            "and does not use benchmark outputs for training or routing."
        ),
    }


def render_markdown(receipt: dict) -> str:
    return f"""# Qwen3.5 V5 Complete Treatment v1

This freezes one matched treatment over the existing 15,559 complete direct
case set. It starts no generation.

- GSM8K 1,319: three grounded calculator plans; override direct only when at
  least two safe executions agree.
- MMLU 14,042: preserve frozen 4B direct exactly.
- GPQA-Diamond 198: override direct only when two option reviews and one
  confirmation agree on the same non-direct option.
- every failure or disagreement preserves frozen 4B direct.
- config SHA: `{receipt['identity']['config_sha256']}`;
- case IDs SHA: `{receipt['identity']['case_ids_sha256']}`;
- V5 report SHA: `{receipt['identity']['v5_report_sha256']}`.

Passing requires strict superiority over 9B on each complete benchmark,
positive paired bootstrap CI, McNemar p<0.05, and no regression versus direct
4B. Observation freezes the treatment permanently.
"""


def main() -> None:
    receipt = build_receipt()
    JSON_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUTPUT.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    MARKDOWN_OUTPUT.write_text(render_markdown(receipt), encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
