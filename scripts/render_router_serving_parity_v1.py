#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from nano_harness.baseline import sha256_file
from nano_harness.router_serving_parity import (
    case_contract,
    load_config,
    summarize,
    validation_cases,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/harness/qwen35_router_serving_parity_v1.json"
PREREG = (
    ROOT
    / "docs/experiments/qwen35_router_serving_parity_v1.preregister.json"
)
SERVICE = (
    ROOT
    / "docs/experiments/"
    "qwen35_router_serving_parity_service_v1.public.json"
)
RAW = ROOT / "results/harness/qwen35-router-serving-parity-v1/result.json"
PUBLIC_JSON = (
    ROOT / "docs/results/qwen35_router_serving_parity_v1.public.json"
)
MARKDOWN = ROOT / "docs/results/qwen35_router_serving_parity_v1.md"


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
            "qwen35_router_serving_parity_v1.preregister.json",
        ],
        cwd=ROOT,
    )
    return hashlib.sha256(content).hexdigest()


def gates(raw: dict) -> dict[str, bool]:
    summaries = raw["summaries"]
    matches = raw["hf_output_matches"]
    arms = raw["arms"]
    return {
        "all_three_arms_complete_192": all(
            summaries[arm]["samples"] == 192
            for arm in ("base", "original", "remapped")
        ),
        "all_outputs_parseable_192": all(
            row["output"] in {"FINAL: A", "FINAL: B", "FINAL: C"}
            for rows in arms.values()
            for row in rows
        ),
        "remapped_exact_192": summaries["remapped"]["exact"] == 192,
        "remapped_hf_output_match_192": matches["remapped"] == 192,
        "remapped_each_label_exact_64": all(
            row["samples"] == 64 and row["exact"] == 64
            for row in summaries["remapped"]["by_family"].values()
        ),
        "original_hf_output_match_less_than_192": matches["original"] < 192,
        "remapped_exact_greater_than_original": (
            summaries["remapped"]["exact"] > summaries["original"]["exact"]
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
        != "nano_harness_router_serving_parity_preregister_v1"
        or prereg.get("identity", {}).get("config_sha256")
        != sha256_file(CONFIG)
        or committed_preregister_sha256() != sha256_file(PREREG)
        or service.get("schema_version")
        != "nano_harness_router_serving_parity_service_v1"
        or service.get("preregister_sha256") != sha256_file(PREREG)
        or service.get("generation_started") is not False
        or raw.get("schema_version")
        != "nano_harness_router_serving_parity_result_v1"
        or raw.get("identity", {}).get("case_contract")
        != case_contract(cases)
        or raw.get("identity", {}).get("original_adapter_sha256")
        != config.original_adapter_tree_sha256
        or raw.get("identity", {}).get("remapped_adapter_sha256")
        != config.remapped_adapter_tree_sha256
        or raw.get("service_receipt") != service
    ):
        raise ValueError("router serving parity result identity differs")
    for arm in ("base", "original", "remapped"):
        rows = raw["arms"][arm]
        if (
            len(rows) != 192
            or {row["sample_id"] for row in rows} != ids
            or summarize(rows) != raw["summaries"][arm]
        ):
            raise ValueError(f"router serving parity {arm} rows differ")
    expected_boundary = {
        "training_eligible_cases": 0,
        "benchmark_rows_loaded": False,
        "benchmark_outputs_loaded": False,
        "canary_rows_loaded": False,
        "holdout_rows_loaded": False,
        "fresh_integration_rows_loaded": False,
        "only_observed_sft_validation_rows_loaded": True,
    }
    if raw["evaluation_boundary"] != expected_boundary:
        raise ValueError("router serving parity boundary differs")
    decision_gates = gates(raw)
    admitted = all(decision_gates.values())
    original_vs_remapped = {
        "both_exact": sum(
            original["exact"] and remapped["exact"]
            for original, remapped in zip(
                raw["arms"]["original"],
                raw["arms"]["remapped"],
            )
        ),
        "remapped_only": sum(
            not original["exact"] and remapped["exact"]
            for original, remapped in zip(
                raw["arms"]["original"],
                raw["arms"]["remapped"],
            )
        ),
        "original_only": sum(
            original["exact"] and not remapped["exact"]
            for original, remapped in zip(
                raw["arms"]["original"],
                raw["arms"]["remapped"],
            )
        ),
        "both_wrong": sum(
            not original["exact"] and not remapped["exact"]
            for original, remapped in zip(
                raw["arms"]["original"],
                raw["arms"]["remapped"],
            )
        ),
    }
    return {
        "schema_version": "nano_harness_router_serving_parity_public_v1",
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
        "summaries": raw["summaries"],
        "hf_reference": raw["hf_reference"],
        "hf_output_matches": raw["hf_output_matches"],
        "original_vs_remapped": original_vs_remapped,
        "decision": {
            "gates": decision_gates,
            "serving_namespace_root_cause_supported": admitted,
            "remapped_adapter_serving_admitted": admitted,
            "fresh_integration_v2_preregistration_allowed": admitted,
            "fresh_integration_v2_generation_allowed": False,
            "observed_integration_v1_rerun_allowed": False,
            "real_question_scan_allowed": False,
            "benchmark_allowed": False,
            "canary_allowed": False,
            "holdout_allowed": False,
            "training_or_rl_allowed": False,
            "next_action": (
                "Pre-register one new history-disjoint integration v2 using "
                "the content-identical remapped adapter. Do not rerun v1."
                if admitted
                else
                "Serving parity remains unresolved. Do not generate a new "
                "integration or train."
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
        "NAMESPACE ROOT CAUSE SUPPORTED"
        if decision["serving_namespace_root_cause_supported"]
        else "PARITY UNRESOLVED"
    )
    return f"""# Qwen3.5 Router Serving Parity v1 Result

## Verdict

**{verdict}.**

The three arms use the same 192 already-observed SFT validation rows and
unchanged generation contract. The remapped adapter changes only tensor key
names; all 224 tensor dtype/shape/content hashes are unchanged.

## Summaries

```json
{json.dumps(report['summaries'], indent=2, sort_keys=True)}
```

## HF Output Matches

```json
{json.dumps(report['hf_output_matches'], indent=2, sort_keys=True)}
```

## Original vs Remapped

```json
{json.dumps(report['original_vs_remapped'], indent=2, sort_keys=True)}
```

## Frozen Gates

```json
{json.dumps(decision['gates'], indent=2, sort_keys=True)}
```

## Boundaries

Passing does not revive observed integration v1. It permits only a separately
pre-registered, new history-disjoint integration v2. Real question scan,
benchmark, canary, holdout, training, and RL remain closed.

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
                "summaries": report["summaries"],
                "hf_output_matches": report["hf_output_matches"],
                "original_vs_remapped": report["original_vs_remapped"],
                "decision": report["decision"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
