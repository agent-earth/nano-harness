from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nano_harness.baseline import sha256_file
from nano_harness.orca_self_consistency import (
    Config,
    _read_jsonl,
    _sha256_lines,
    run_selection,
)


CONFIG_SCHEMA = "nano_harness_orca_self_consistency_replication_v2"


def load_config(path: str | Path) -> Config:
    config_path = Path(path).resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("unsupported self-consistency replication schema")
    if (
        raw["experiment_id"]
        != "orca-math-self-consistency-replication-v2"
        or raw["cases_by_stratum"]
        != {"short": 40, "medium": 80, "long": 40}
        or raw["direct"]
        != {"temperature": 0.0, "top_p": 1.0, "max_tokens": 384}
        or raw["candidate"]
        != {
            "replicas": 5,
            "minimum_agreement": 4,
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 384,
            "seed_base": 2026082100,
            "fallback": "frozen_four_b_direct",
        }
        or raw["statistics"]
        != {
            "bootstrap_samples": 10_000,
            "bootstrap_seed": 20260824,
            "alpha": 0.05,
            "minimum_candidate_only_wins": 6,
            "four_b_noninferiority_margin": 0.0,
        }
    ):
        raise ValueError("self-consistency replication contract differs")
    return Config(path=config_path, raw=raw)


def select_cases(config: Config) -> dict[str, Any]:
    raw = config.raw
    dataset_path = config.resolve(raw["dataset_path"])
    if sha256_file(dataset_path) != raw["dataset_sha256"]:
        raise ValueError("replication dataset identity differs")
    preregister_paths = [
        config.resolve(raw["prior_dpo_v1_preregister_path"]),
        config.resolve(raw["prior_dpo_v2_preregister_path"]),
        config.resolve(raw["prior_self_consistency_preregister_path"]),
    ]
    expected_sha256 = [
        raw["prior_dpo_v1_preregister_sha256"],
        raw["prior_dpo_v2_preregister_sha256"],
        raw["prior_self_consistency_preregister_sha256"],
    ]
    for path, expected in zip(preregister_paths, expected_sha256):
        if sha256_file(path) != expected:
            raise ValueError("replication prior preregister identity differs")
    result_path = config.resolve(raw["prior_self_consistency_result_path"])
    if sha256_file(result_path) != raw["prior_self_consistency_result_sha256"]:
        raise ValueError("replication prior result identity differs")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if (
        result.get("schema_version")
        != "nano_harness_orca_self_consistency_public_v1"
        or result.get("decision", {}).get("candidate_admitted") is not False
        or result.get("decision", {}).get("versus_nine_b_gates", {}).get(
            "bootstrap_ci_lower_positive"
        )
        is not True
        or result.get("decision", {}).get("versus_nine_b_gates", {}).get(
            "mcnemar_below_alpha"
        )
        is not True
        or result.get("comparisons", {})
        .get("versus_four_b", {})
        .get("paired_counts", {})
        .get("baseline_only")
        != 0
    ):
        raise ValueError("replication prior result boundary differs")

    prior_ids = set()
    for path in preregister_paths:
        receipt = json.loads(path.read_text(encoding="utf-8"))
        selection = receipt["selection"]
        if "train_ids" in selection:
            prior_ids.update(selection["train_ids"])
            prior_ids.update(selection["dev_ids"])
        else:
            prior_ids.update(selection["case_ids"])
    rows = [
        row
        for row in _read_jsonl(dataset_path)
        if row["sample_id"] not in prior_ids and row["split"] == "train"
    ]
    rows.sort(key=lambda row: row["sample_id"])
    by_stratum = {
        stratum: [row for row in rows if row["stratum"] == stratum]
        for stratum in ("short", "medium", "long")
    }
    if {
        key: len(value) for key, value in by_stratum.items()
    } != raw["cases_by_stratum"]:
        raise ValueError("replication does not cover every remaining row")
    selected = [
        row
        for stratum in ("short", "medium", "long")
        for row in by_stratum[stratum]
    ]
    selected.sort(key=lambda row: row["sample_id"])
    case_ids = [row["sample_id"] for row in selected]
    if len(selected) != 160 or set(case_ids) & prior_ids:
        raise ValueError("replication selection differs")
    return {
        "cases": selected,
        "case_ids": case_ids,
        "case_ids_sha256": _sha256_lines(case_ids),
        "prior_ids_sha256": _sha256_lines(sorted(prior_ids)),
    }


def run(config: Config) -> dict[str, Any]:
    return run_selection(config, select_cases(config))
