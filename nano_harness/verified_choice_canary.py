from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nano_harness.baseline import (
    case_manifest,
    load_cases,
    load_manifest,
    score_output,
)
from nano_harness.verified_choice import sha256_file, verify_explicit_average_choice


@dataclass(frozen=True)
class VerifiedChoiceCanaryConfig:
    schema_version: str
    experiment_id: str
    parser_version: str
    manifest_path: str
    manifest_sha256: str
    case_manifest_path: str
    case_manifest_sha256: str
    dataset_root: str
    baseline_raw_path: str
    baseline_raw_sha256: str
    baseline_public_report_path: str
    baseline_public_report_sha256: str
    local_pass_report_path: str
    local_pass_report_sha256: str
    output_path: str
    exact_option_match_required: bool
    ambiguous_fallback: str


def load_config(path: str | Path) -> VerifiedChoiceCanaryConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = set(VerifiedChoiceCanaryConfig.__dataclass_fields__)
    if set(raw) != expected:
        raise ValueError("verified choice canary config fields differ")
    config = VerifiedChoiceCanaryConfig(**raw)
    frozen = {
        "schema_version": "nano_harness_verified_choice_canary_v1",
        "experiment_id": "anchored-v1-verified-choice-canary-v1",
        "parser_version": "explicit_two_expression_average_v1",
        "exact_option_match_required": True,
        "ambiguous_fallback": "reuse_direct_output",
    }
    for field, expected_value in frozen.items():
        if getattr(config, field) != expected_value:
            raise ValueError(
                f"verified choice canary freezes {field}={expected_value}"
            )
    return config


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_benchmark = {}
    for benchmark in sorted({str(row["benchmark"]) for row in rows}):
        subset = [row for row in rows if row["benchmark"] == benchmark]
        by_benchmark[benchmark] = {
            "cases": len(subset),
            "correct": sum(bool(row["score"]) for row in subset),
            "parse_failures": sum(row["prediction"] is None for row in subset),
            "api_errors": sum(row["status"] == "error" for row in subset),
            "length_truncations": sum(
                row.get("finish_reason") == "length" for row in subset
            ),
        }
    return {
        "cases": len(rows),
        "correct": sum(bool(row["score"]) for row in rows),
        "by_benchmark": by_benchmark,
    }


def run(config: VerifiedChoiceCanaryConfig) -> dict[str, Any]:
    paths = {
        "manifest": Path(config.manifest_path),
        "case_manifest": Path(config.case_manifest_path),
        "baseline_raw": Path(config.baseline_raw_path),
        "baseline_public_report": Path(config.baseline_public_report_path),
        "local_pass_report": Path(config.local_pass_report_path),
    }
    expected_hashes = {
        "manifest": config.manifest_sha256,
        "case_manifest": config.case_manifest_sha256,
        "baseline_raw": config.baseline_raw_sha256,
        "baseline_public_report": config.baseline_public_report_sha256,
        "local_pass_report": config.local_pass_report_sha256,
    }
    for name, path in paths.items():
        if sha256_file(path) != expected_hashes[name]:
            raise ValueError(f"verified choice canary {name} identity mismatch")

    manifest = load_manifest(paths["manifest"])
    cases = load_cases(manifest, Path(config.dataset_root))
    committed_cases = json.loads(
        paths["case_manifest"].read_text(encoding="utf-8")
    )
    if case_manifest(cases) != committed_cases or len(cases) != 40:
        raise ValueError("verified choice canary cases differ")
    committed_by_id = {
        str(item["case_id"]): item for item in committed_cases
    }
    baseline_public = json.loads(
        paths["baseline_public_report"].read_text(encoding="utf-8")
    )
    local_pass = json.loads(
        paths["local_pass_report"].read_text(encoding="utf-8")
    )
    if (
        baseline_public.get("passed") is not True
        or baseline_public.get("candidate", {}).get("correct") != 32
        or local_pass.get("passed") is not True
        or local_pass.get("decision", {}).get("sealed_canary_allowed") is not True
    ):
        raise ValueError("verified choice canary staged receipts do not authorize")

    baseline_rows = _jsonl(paths["baseline_raw"])
    by_id = {str(row["case_id"]): row for row in baseline_rows}
    if len(baseline_rows) != 40 or set(by_id) != {case.case_id for case in cases}:
        raise ValueError("verified choice canary raw case set differs")
    for case in cases:
        row = by_id[case.case_id]
        committed = committed_by_id[case.case_id]
        if (
            row.get("suite_id") != manifest.suite_id
            or row.get("benchmark") != case.benchmark
            or row.get("model") != "qwen3.5-4b-anchor-v1"
            or row.get("strategy") != "direct"
            or row.get("source_index") != case.source_index
            or row.get("max_tokens") != case.max_tokens
            or row.get("expected") != case.expected
            or row.get("prompt_sha256") != committed["prompt_sha256"]
            or row.get("system_prompt_sha256")
            != committed["system_prompt_sha256"]
            or row.get("selected_strategy") != "direct"
            or row.get("status") != "completed"
        ):
            raise ValueError(f"verified choice canary row parity failed: {case.case_id}")

    receipts = {}
    routed_outputs = {}
    for case in cases:
        baseline = by_id[case.case_id]
        output = str(baseline["output"])
        route = "reuse_direct_output"
        if case.scorer == "choice_exact":
            receipt = verify_explicit_average_choice(case.prompt)
            receipts[case.case_id] = receipt
            if receipt["override"]:
                output = f"FINAL: {receipt['selected_letter']}"
                route = "verified_choice_override"
        routed_outputs[case.case_id] = {
            "output": output,
            "route": route,
        }

    candidate_rows = []
    for case in cases:
        baseline = by_id[case.case_id]
        routed = routed_outputs[case.case_id]
        score, prediction = score_output(
            routed["output"],
            case.expected,
            case.scorer,
        )
        candidate_rows.append(
            {
                **baseline,
                "model": "qwen3.5-4b-anchor-v1+verified-choice-v1",
                "output": routed["output"],
                "prediction": prediction,
                "score": score,
                "verified_choice_route": routed["route"],
            }
        )

    result = {
        "schema_version": "nano_harness_verified_choice_canary_result_v1",
        "experiment_id": config.experiment_id,
        "config": config.__dict__,
        "identity": {
            name: sha256_file(path) for name, path in paths.items()
        },
        "baseline": _aggregate(baseline_rows),
        "candidate": _aggregate(candidate_rows),
        "baseline_rows": baseline_rows,
        "candidate_rows": candidate_rows,
        "receipts": receipts,
        "routing": {
            "choice_rows": len(receipts),
            "verified_overrides": sum(
                receipt["override"] for receipt in receipts.values()
            ),
            "fallback_rows": sum(
                row["verified_choice_route"] == "reuse_direct_output"
                for row in candidate_rows
            ),
        },
        "evaluation_boundary": {
            "target_used_by_parser": False,
            "sealed_canary_run": True,
            "quality_claim_allowed": False,
            "training_eligible": False,
            "prior_full_suite_run": False,
            "independent_holdout_run": False,
        },
    }
    output = Path(config.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result
