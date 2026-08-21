#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from nano_harness.baseline import sha256_file
from nano_harness.complete_conditional_majority import load_config
from nano_harness.v5_complete_treatment import jsonl_ids


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/campaign/qwen35_complete_conditional_majority_v1.json"
)
JSON_OUTPUT = (
    ROOT
    / "docs/experiments/"
    "qwen35_complete_conditional_majority_v1.preregister.json"
)
MARKDOWN_OUTPUT = (
    ROOT
    / "docs/experiments/qwen35_complete_conditional_majority_v1.md"
)


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _load_public(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_receipt() -> dict[str, Any]:
    config = load_config(CONFIG)
    paths: dict[str, Path] = {}
    for section in ("baseline", "predecessors"):
        for key, value in config[section].items():
            if not key.endswith("_path"):
                continue
            digest_key = key.removesuffix("_path") + "_sha256"
            path = ROOT / value
            if (
                not path.is_file()
                or sha256_file(path) != config[section][digest_key]
            ):
                raise ValueError(
                    f"complete conditional majority {section}.{key} differs"
                )
            paths[f"{section}.{key}"] = path

    public_cases = _load_public(paths["baseline.case_manifest_path"])
    counts = Counter(str(row["benchmark"]) for row in public_cases)
    case_ids = [str(row["case_id"]) for row in public_cases]
    gsm8k_ids = sorted(
        str(row["case_id"])
        for row in public_cases
        if row["benchmark"] == "gsm8k"
    )
    four_ids = jsonl_ids(paths["baseline.four_b_raw_path"])
    nine_ids = jsonl_ids(paths["baseline.nine_b_raw_path"])
    prior_candidate_ids = jsonl_ids(
        paths["predecessors.prior_complete_candidate_path"]
    )
    if (
        len(case_ids) != 15_559
        or len(case_ids) != len(set(case_ids))
        or counts
        != {
            "gsm8k": 1_319,
            "mmlu": 14_042,
            "gpqa_diamond": 198,
        }
        or set(four_ids) != set(case_ids)
        or set(nine_ids) != set(case_ids)
        or set(prior_candidate_ids) != set(case_ids)
        or len(gsm8k_ids) != 1_319
    ):
        raise ValueError("complete conditional majority case identity differs")

    baseline = _load_public(paths["baseline.report_path"])
    local_v4 = _load_public(paths["predecessors.local_v4_report_path"])
    prior_complete = _load_public(
        paths["predecessors.prior_complete_report_path"]
    )
    if (
        baseline.get("decision", {}).get("direct_baseline_accepted")
        is not True
        or local_v4.get("decision", {}).get("candidate_admitted")
        is not True
        or local_v4.get("decision", {}).get(
            "complete_benchmark_allowed"
        )
        is not True
        or local_v4.get("decision", {}).get("rerun_or_tuning_allowed")
        is not False
        or prior_complete.get("decision", {}).get(
            "complete_treatment_admitted"
        )
        is not False
        or prior_complete.get("decision", {}).get("gates", {}).get(
            "mmlu_superior_to_nine_b"
        )
        is not True
        or prior_complete.get("decision", {}).get("gates", {}).get(
            "gpqa_diamond_superior_to_nine_b"
        )
        is not True
        or prior_complete.get("decision", {}).get(
            "rerun_or_tuning_allowed"
        )
        is not False
    ):
        raise ValueError(
            "complete conditional majority predecessor gate differs"
        )

    return {
        "schema_version": (
            "nano_harness_complete_conditional_majority_preregister_v1"
        ),
        "experiment_id": config["experiment_id"],
        "identity": {
            "code_revision": git_revision(),
            "config_sha256": sha256_file(CONFIG),
            "suite_manifest_sha256": config["baseline"][
                "suite_manifest_sha256"
            ],
            "case_manifest_sha256": config["baseline"][
                "case_manifest_sha256"
            ],
            "complete_case_ids_sha256": hashlib.sha256(
                "\n".join(sorted(case_ids)).encode()
            ).hexdigest(),
            "gsm8k_case_ids_sha256": hashlib.sha256(
                "\n".join(gsm8k_ids).encode()
            ).hexdigest(),
            "four_b_raw_sha256": config["baseline"][
                "four_b_raw_sha256"
            ],
            "nine_b_raw_sha256": config["baseline"][
                "nine_b_raw_sha256"
            ],
            "local_v4_report_sha256": config["predecessors"][
                "local_v4_report_sha256"
            ],
            "prior_complete_report_sha256": config["predecessors"][
                "prior_complete_report_sha256"
            ],
            "prior_complete_candidate_sha256": config["predecessors"][
                "prior_complete_candidate_sha256"
            ],
        },
        "surface": {
            "complete_cases": len(case_ids),
            "by_benchmark": dict(sorted(counts.items())),
            "new_model_generation": {
                "gsm8k": 1_319,
                "mmlu": 0,
                "gpqa_diamond": 0,
            },
            "case_sets_match_frozen_direct_and_prior_candidate": True,
            "prompts_answers_or_outputs_published": False,
        },
        "candidate": {
            "gsm8k": config["routes"]["gsm8k"],
            "mmlu": config["routes"]["mmlu"],
            "gpqa_diamond": config["routes"]["gpqa_diamond"],
            "single_frozen_composition": True,
        },
        "statistics": config["statistics"],
        "decision_rule": {
            "gsm8k_admitted": (
                "all four_b_preservation gates and all nine_b_superiority "
                "gates pass at Bonferroni alpha 0.025"
            ),
            "complete_candidate_admitted": (
                "GSM8K is admitted and the frozen MMLU/GPQA endpoints remain "
                "superior to 9B after Holm-Bonferroni correction across the "
                "three complete benchmark p-values"
            ),
            "twenty_seven_b_preregistration_allowed": (
                "complete_candidate_admitted"
            ),
            "rerun_or_tuning_allowed": False,
        },
        "forbidden_after_observation": [
            "prompt_or_parser_change",
            "budget_temperature_seed_or_vote_threshold_change",
            "case_selection_change",
            "model_or_adapter_change",
            "prior_endpoint_replacement",
            "rerun",
            "training_reward_or_verifier_use_of_benchmark_rows_or_outputs",
        ],
        "execution_boundary": config["execution_boundary"],
        "claim_boundary": (
            "This freezes one sequential complete-benchmark candidate and "
            "starts no model generation. The MMLU and GPQA endpoints are "
            "prior frozen evidence; only GSM8K will receive new requests. "
            "No quality or 27B claim is established by this receipt."
        ),
    }


def render_markdown(receipt: dict[str, Any]) -> str:
    return f"""# Qwen3.5 Complete Conditional-Majority v1

This pre-registers one complete three-benchmark candidate. It starts no model
generation.

## Frozen Candidate

- GSM8K, 1,319 cases: run the admitted target-blind recovered parser and
  conditional-majority v4 policy. Five 4B solves are sampled per case. A
  strict-parse failure may be replaced by 3-of-5 agreement; an already strict
  direct answer requires unanimous 5-of-5 agreement. Otherwise preserve the
  frozen recovered 4B direct answer.
- MMLU, 14,042 cases: preserve the frozen 4B direct result.
- GPQA-Diamond, 198 cases: reuse the frozen V5 conservative-consensus result;
  make no new GPQA request.

The complete candidate is frozen before new GSM8K generation. Existing V5
GSM8K outputs are not reused.

## Sequential Inference

This is the second and final complete GSM8K treatment attempt. GSM8K admission
uses Bonferroni `alpha=0.025` over the two complete attempts. The final
three-benchmark claim additionally applies Holm-Bonferroni at familywise
`alpha=0.05` to the three frozen benchmark p-values.

## Identity

- config SHA: `{receipt['identity']['config_sha256']}`;
- complete case IDs SHA:
  `{receipt['identity']['complete_case_ids_sha256']}`;
- GSM8K case IDs SHA: `{receipt['identity']['gsm8k_case_ids_sha256']}`;
- local v4 report SHA:
  `{receipt['identity']['local_v4_report_sha256']}`.

## Boundary

The preregistration reads only public reports, public case metadata, and raw
case IDs. It does not read benchmark prompts, answers, or model outputs. Raw
artifacts remain local and cannot enter training, reward, or verifier data.
After observation, no rerun or policy change is allowed.
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
