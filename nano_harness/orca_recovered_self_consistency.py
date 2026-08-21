from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from nano_harness.baseline import sha256_file
from nano_harness.orca_self_consistency import (
    Config,
    _number,
    _rank,
    _read_jsonl,
    _sha256_lines,
    parse_final,
    run_selection,
)


CONFIG_SCHEMA = "nano_harness_orca_recovered_self_consistency_v3"
PLAIN_NUMBER = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"[-+]?(?:[0-9]+(?:\.[0-9]+)?|[0-9]+/[0-9]+)"
)
LATEX_FRACTION = re.compile(
    r"\\frac\s*\{\s*([-+]?[0-9]+(?:\.[0-9]+)?)\s*\}"
    r"\s*\{\s*([0-9]+(?:\.[0-9]+)?)\s*\}"
)


def load_config(path: str | Path) -> Config:
    config_path = Path(path).resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("unsupported recovered self-consistency schema")
    if (
        raw["experiment_id"] != "orca-math-recovered-self-consistency-v3"
        or raw["cases_by_stratum"]
        != {"short": 24, "medium": 48, "long": 24}
        or raw["parser"]
        != {
            "strict_final_first": True,
            "fallback": "last_numeric_token_in_last_1500_chars",
            "target_blind": True,
        }
        or raw["direct"]
        != {"temperature": 0.0, "top_p": 1.0, "max_tokens": 384}
        or raw["candidate"]
        != {
            "replicas": 5,
            "minimum_agreement": 4,
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 384,
            "seed_base": 2026082500,
            "fallback": "frozen_four_b_recovered_direct",
        }
        or raw["statistics"]
        != {
            "bootstrap_samples": 10_000,
            "bootstrap_seed": 20260825,
            "alpha": 0.05,
            "minimum_candidate_only_wins": 6,
        }
    ):
        raise ValueError("recovered self-consistency contract differs")
    return Config(path=config_path, raw=raw)


def parse_recovered_final(output: str) -> str | None:
    strict = parse_final(output)
    if strict is not None:
        return strict
    tail = output[-1_500:]
    candidates: list[tuple[int, str]] = []
    latex_spans = []
    for match in LATEX_FRACTION.finditer(tail):
        latex_spans.append(match.span())
        candidates.append(
            (match.start(), f"{match.group(1)}/{match.group(2)}")
        )
    for match in PLAIN_NUMBER.finditer(tail):
        if any(start <= match.start() < end for start, end in latex_spans):
            continue
        candidates.append((match.start(), match.group(0)))
    if not candidates:
        return None
    value = max(candidates, key=lambda row: row[0])[1]
    return value if _number(value) is not None else None


def select_cases(config: Config) -> dict[str, Any]:
    raw = config.raw
    source_path = config.resolve(raw["source_dataset_path"])
    preference_path = config.resolve(raw["preference_dataset_path"])
    if (
        sha256_file(source_path) != raw["source_dataset_sha256"]
        or sha256_file(preference_path)
        != raw["preference_dataset_sha256"]
    ):
        raise ValueError("recovered self-consistency dataset identity differs")
    sft_path = config.resolve(raw["prior_sft_preregister_path"])
    v1_path = config.resolve(raw["prior_self_consistency_result_path"])
    v2_path = config.resolve(raw["prior_replication_result_path"])
    if (
        sha256_file(sft_path) != raw["prior_sft_preregister_sha256"]
        or sha256_file(v1_path)
        != raw["prior_self_consistency_result_sha256"]
        or sha256_file(v2_path)
        != raw["prior_replication_result_sha256"]
    ):
        raise ValueError("recovered self-consistency prior identity differs")
    v1 = json.loads(v1_path.read_text(encoding="utf-8"))
    v2 = json.loads(v2_path.read_text(encoding="utf-8"))
    if (
        v1.get("decision", {}).get("candidate_admitted") is not False
        or v2.get("decision", {}).get("replication_admitted") is not False
        or v2.get("boundary", {}).get(
            "pooled_result_overrides_replication_gate"
        )
        is not False
    ):
        raise ValueError("recovered self-consistency prior boundary differs")
    sft = json.loads(sft_path.read_text(encoding="utf-8"))
    excluded_source_ids = set(sft["selection"]["train_sample_ids"]) | set(
        sft["selection"]["dev_sample_ids"]
    )
    excluded_source_ids.update(
        row["source_sample_id"] for row in _read_jsonl(preference_path)
    )
    rows = [
        row
        for row in _read_jsonl(source_path)
        if row["split"] == "dev"
        and row["sample_id"] not in excluded_source_ids
    ]
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault(row["stratum"], []).append(row)
    selected = []
    for stratum in ("short", "medium", "long"):
        ranked = sorted(
            buckets[stratum],
            key=lambda row: (
                _rank(raw["selection_seed"], row["sample_id"]),
                row["sample_id"],
            ),
        )
        selected.extend(ranked[: raw["cases_by_stratum"][stratum]])
    selected.sort(key=lambda row: row["sample_id"])
    case_ids = [row["sample_id"] for row in selected]
    if (
        len(selected) != 96
        or len(set(case_ids)) != 96
        or set(case_ids) & excluded_source_ids
    ):
        raise ValueError("recovered self-consistency selection differs")
    return {
        "cases": selected,
        "case_ids": case_ids,
        "case_ids_sha256": _sha256_lines(case_ids),
        "excluded_source_ids_sha256": _sha256_lines(
            sorted(excluded_source_ids)
        ),
    }


def run(config: Config) -> dict[str, Any]:
    return run_selection(
        config,
        select_cases(config),
        messages_for_row=lambda row: row["messages"][:-1],
        expected_for_row=lambda row: str(row["numeric_answer"]),
        parser=parse_recovered_final,
    )
