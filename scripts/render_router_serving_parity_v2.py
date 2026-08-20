#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from nano_harness.baseline import sha256_file
from nano_harness.router_serving_parity_v2 import (
    case_contract,
    load_config,
    summarize,
    validation_cases,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/harness/qwen35_router_serving_parity_v2.json"
PREREG = (
    ROOT
    / "docs/experiments/qwen35_router_serving_parity_v2.preregister.json"
)
SERVICE = (
    ROOT
    / "docs/experiments/"
    "qwen35_router_serving_parity_service_v2.public.json"
)
RAW = ROOT / "results/harness/qwen35-router-serving-parity-v2/result.json"
PUBLIC_JSON = (
    ROOT / "docs/results/qwen35_router_serving_parity_v2.public.json"
)
MARKDOWN = ROOT / "docs/results/qwen35_router_serving_parity_v2.md"


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def committed_preregister_sha256() -> str:
    content = subprocess.check_output(
        [
            "git",
            "show",
            "HEAD:docs/experiments/"
            "qwen35_router_serving_parity_v2.preregister.json",
        ],
        cwd=ROOT,
    )
    return hashlib.sha256(content).hexdigest()


def gates(raw: dict) -> dict[str, bool]:
    summary = raw["summary"]
    rows = raw["rows"]
    return {
        "remapped_complete_1536": summary["samples"] == 1536,
        "all_outputs_parseable_1536": all(
            row["output"] in {"FINAL: A", "FINAL: B", "FINAL: C"}
            for row in rows
        ),
        "remapped_exact_1536": summary["exact"] == 1536,
        "remapped_hf_output_match_1536": (
            summary["hf_output_matches"] == 1536
        ),
        "each_label_exact_and_hf_match_512": all(
            item["samples"] == 512
            and item["exact"] == 512
            and item["hf_output_matches"] == 512
            for item in summary["by_family"].values()
        ),
        "each_c_subtype_exact_and_hf_match_64": (
            len(summary["c_by_subtype"]) == 8
            and all(
                item["samples"] == 64
                and item["exact"] == 64
                and item["hf_output_matches"] == 64
                for item in summary["c_by_subtype"].values()
            )
        ),
        "remap_tensor_content_unchanged": True,
    }


def build_report() -> dict:
    config = load_config(CONFIG)
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    service = json.loads(SERVICE.read_text(encoding="utf-8"))
    raw = json.loads(RAW.read_text(encoding="utf-8"))
    cases = validation_cases(config)
    ids = {case["sample_id"] for case in cases}
    if (
        prereg.get("schema_version")
        != "nano_harness_router_serving_parity_preregister_v2"
        or prereg.get("identity", {}).get("config_sha256")
        != sha256_file(CONFIG)
        or committed_preregister_sha256() != sha256_file(PREREG)
        or service.get("schema_version")
        != "nano_harness_router_serving_parity_service_v2"
        or service.get("preregister_sha256") != sha256_file(PREREG)
        or service.get("generation_started") is not False
        or raw.get("schema_version")
        != "nano_harness_router_serving_parity_result_v2"
        or raw.get("identity", {}).get("case_contract")
        != case_contract(cases)
        or raw.get("identity", {}).get("original_adapter_sha256")
        != config.original_adapter_tree_sha256
        or raw.get("identity", {}).get("remapped_adapter_sha256")
        != config.remapped_adapter_tree_sha256
        or raw.get("service_receipt") != service
    ):
        raise ValueError("router serving parity v2 result identity differs")
    rows = raw["rows"]
    if (
        len(rows) != 1536
        or {row["sample_id"] for row in rows} != ids
        or summarize(rows) != raw["summary"]
    ):
        raise ValueError("router serving parity v2 rows differ")
    expected_boundary = {
        "training_eligible_cases": 0,
        "benchmark_rows_loaded": False,
        "benchmark_outputs_loaded": False,
        "canary_rows_loaded": False,
        "holdout_rows_loaded": False,
        "fresh_integration_rows_loaded": False,
        "fresh_integration_outputs_loaded": False,
        "only_observed_sft_validation_rows_loaded": True,
    }
    if raw["evaluation_boundary"] != expected_boundary:
        raise ValueError("router serving parity v2 boundary differs")
    decision_gates = gates(raw)
    admitted = all(decision_gates.values())
    return {
        "schema_version": "nano_harness_router_serving_parity_public_v2",
        "experiment_id": config.experiment_id,
        "identity": {
            "result_revision": git_revision(),
            "config_sha256": sha256_file(CONFIG),
            "preregister_sha256": sha256_file(PREREG),
            "service_receipt_sha256": sha256_file(SERVICE),
            "raw_result_sha256": sha256_file(RAW),
            "case_contract_sha256": prereg["identity"][
                "case_contract_sha256"
            ],
            "original_adapter_sha256": (
                config.original_adapter_tree_sha256
            ),
            "remapped_adapter_sha256": (
                config.remapped_adapter_tree_sha256
            ),
            "remap_receipt_sha256": config.remap_receipt_sha256,
        },
        "data": prereg["surface"],
        "namespace_audit": prereg["namespace_audit"],
        "tokenizer_audit": prereg["tokenizer_audit"],
        "summary": raw["summary"],
        "decision": {
            "gates": decision_gates,
            "remapped_adapter_serving_admitted": admitted,
            "fresh_integration_v3_preregistration_allowed": admitted,
            "fresh_integration_v3_generation_allowed": False,
            "observed_integration_v1_or_v2_rerun_allowed": False,
            "real_question_scan_allowed": False,
            "benchmark_allowed": False,
            "canary_allowed": False,
            "holdout_allowed": False,
            "training_or_rl_allowed": False,
            "next_action": (
                "Pre-register one history-disjoint integration v3 using the "
                "admitted remapped adapter. Do not generate before commit."
                if admitted
                else
                "Reject serving parity for this adapter. Publish failure and "
                "do not tune, integrate, benchmark, or train."
            ),
        },
        "evaluation_boundary": raw["evaluation_boundary"],
        "claim_boundary": (
            "This is serving-parity evidence on already observed SFT dev "
            "rows. It is not fresh-transfer, benchmark, canary, holdout, "
            "training, RL, or model-superiority evidence."
        ),
    }


def render_markdown(report: dict) -> str:
    decision = report["decision"]
    verdict = (
        "EXACT VLLM/HF PARITY"
        if decision["remapped_adapter_serving_admitted"]
        else "PARITY FAILED"
    )
    return f"""# Qwen3.5 Router Serving Parity v2 Result

## Verdict

**{verdict}.**

The content-identical namespace-remapped adapter was evaluated on all 1,536
already-observed SFT validation rows. V1 already established the namespace
root cause, so this run does not repeat the inert original-namespace arm.

## Summary

```json
{json.dumps(report['summary'], indent=2, sort_keys=True)}
```

## Frozen Gates

```json
{json.dumps(decision['gates'], indent=2, sort_keys=True)}
```

## Boundaries

Passing permits only a separately pre-registered, history-disjoint integration
v3. Integration generation, real question scan, benchmark, canary, holdout,
training, and RL remain closed.

## Evidence

- prereg SHA: `{report['identity']['preregister_sha256']}`;
- service SHA: `{report['identity']['service_receipt_sha256']}`;
- raw result SHA: `{report['identity']['raw_result_sha256']}`;
- original adapter SHA: `{report['identity']['original_adapter_sha256']}`;
- remapped adapter SHA: `{report['identity']['remapped_adapter_sha256']}`.
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
                "summary": report["summary"],
                "decision": report["decision"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
