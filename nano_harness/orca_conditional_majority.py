from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nano_harness.baseline import sha256_file
from nano_harness.orca_recovered_self_consistency import (
    load_config as load_v3_config,
    parse_recovered_final,
)
from nano_harness.orca_self_consistency import (
    Config,
    _number,
    _rank,
    _read_jsonl,
    _request,
    _sha256_lines,
    parse_final,
    score_prediction,
)
from openai import OpenAI


CONFIG_SCHEMA = "nano_harness_orca_conditional_majority_v4"


def load_config(path: str | Path) -> Config:
    config_path = Path(path).resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("unsupported conditional majority schema")
    if (
        raw["experiment_id"] != "orca-math-conditional-majority-v4"
        or raw["cases_by_stratum"]
        != {"short": 24, "medium": 48, "long": 24}
        or raw["parser"]
        != {
            "strict_final_first": True,
            "fallback": "last_numeric_token_in_last_1500_chars",
            "target_blind": True,
        }
        or raw["override_rule"]
        != {
            "direct_strict_parse_failure_minimum_votes": 3,
            "direct_strict_parseable_minimum_votes": 5,
        }
        or raw["direct"]
        != {"temperature": 0.0, "top_p": 1.0, "max_tokens": 384}
        or raw["candidate"]
        != {
            "replicas": 5,
            "minimum_agreement": 3,
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 384,
            "seed_base": 2026082600,
            "fallback": "frozen_four_b_recovered_direct",
        }
    ):
        raise ValueError("conditional majority contract differs")
    return Config(path=config_path, raw=raw)


def select_cases(config: Config) -> dict[str, Any]:
    raw = config.raw
    source_path = config.resolve(raw["source_dataset_path"])
    preference_path = config.resolve(raw["preference_dataset_path"])
    if (
        sha256_file(source_path) != raw["source_dataset_sha256"]
        or sha256_file(preference_path) != raw["preference_dataset_sha256"]
    ):
        raise ValueError("conditional majority dataset identity differs")
    sft_path = config.resolve(raw["prior_sft_preregister_path"])
    v3_preregister_path = config.resolve(raw["prior_v3_preregister_path"])
    v3_result_path = config.resolve(raw["prior_v3_result_path"])
    if (
        sha256_file(sft_path) != raw["prior_sft_preregister_sha256"]
        or sha256_file(v3_preregister_path)
        != raw["prior_v3_preregister_sha256"]
        or sha256_file(v3_result_path) != raw["prior_v3_result_sha256"]
    ):
        raise ValueError("conditional majority prior identity differs")
    v3 = json.loads(v3_result_path.read_text(encoding="utf-8"))
    if v3.get("decision", {}).get("candidate_admitted") is not False:
        raise ValueError("conditional majority prior result differs")
    sft = json.loads(sft_path.read_text(encoding="utf-8"))
    excluded = set(sft["selection"]["train_sample_ids"]) | set(
        sft["selection"]["dev_sample_ids"]
    )
    excluded.update(
        row["source_sample_id"] for row in _read_jsonl(preference_path)
    )
    v3_preregister = json.loads(
        v3_preregister_path.read_text(encoding="utf-8")
    )
    excluded.update(v3_preregister["selection"]["case_ids"])
    rows = [
        row
        for row in _read_jsonl(source_path)
        if row["split"] == "dev" and row["sample_id"] not in excluded
    ]
    selected = []
    for stratum in ("short", "medium", "long"):
        ranked = sorted(
            (row for row in rows if row["stratum"] == stratum),
            key=lambda row: (
                _rank(raw["selection_seed"], row["sample_id"]),
                row["sample_id"],
            ),
        )
        selected.extend(ranked[: raw["cases_by_stratum"][stratum]])
    selected.sort(key=lambda row: row["sample_id"])
    ids = [row["sample_id"] for row in selected]
    if len(selected) != 96 or len(set(ids)) != 96 or set(ids) & excluded:
        raise ValueError("conditional majority selection differs")
    return {
        "cases": selected,
        "case_ids": ids,
        "case_ids_sha256": _sha256_lines(ids),
        "excluded_source_ids_sha256": _sha256_lines(sorted(excluded)),
    }


def conditional_consensus(
    predictions: list[str | None],
    direct_prediction: str | None,
    *,
    direct_strict_parseable: bool,
    parse_failure_minimum_votes: int,
    parseable_minimum_votes: int,
) -> tuple[str | None, dict[str, Any]]:
    normalized = [
        str(_number(prediction))
        for prediction in predictions
        if prediction is not None and _number(prediction) is not None
    ]
    from collections import Counter

    counts = Counter(normalized)
    minimum_votes = (
        parseable_minimum_votes
        if direct_strict_parseable
        else parse_failure_minimum_votes
    )
    if counts:
        value, votes = counts.most_common(1)[0]
        if votes >= minimum_votes:
            chosen = next(
                prediction
                for prediction in predictions
                if prediction is not None and str(_number(prediction)) == value
            )
            return chosen, {
                "override": _number(chosen) != _number(direct_prediction or ""),
                "consensus_votes": votes,
                "minimum_votes": minimum_votes,
                "parseable_replicas": len(normalized),
                "direct_strict_parseable": direct_strict_parseable,
                "fallback": False,
            }
    return direct_prediction, {
        "override": False,
        "consensus_votes": max(counts.values(), default=0),
        "minimum_votes": minimum_votes,
        "parseable_replicas": len(normalized),
        "direct_strict_parseable": direct_strict_parseable,
        "fallback": True,
    }


def run(config: Config) -> dict[str, Any]:
    selection = select_cases(config)
    raw = config.raw
    output_root = config.resolve(raw["output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)
    clients = {
        "four": OpenAI(
            api_key="local-vllm",
            base_url=raw["four_b"]["base_url"],
            timeout=240,
            max_retries=0,
        ),
        "nine": OpenAI(
            api_key="local-vllm",
            base_url=raw["nine_b"]["base_url"],
            timeout=240,
            max_retries=0,
        ),
    }
    arms = {"four_direct": [], "candidate": [], "nine_direct": []}
    receipts = []
    for case_index, row in enumerate(selection["cases"]):
        messages = row["messages"][:-1]
        four_direct = _request(
            clients["four"],
            model=raw["four_b"]["model"],
            messages=messages,
            seed=raw["candidate"]["seed_base"] + case_index * 10,
            parser=parse_recovered_final,
            **raw["direct"],
        )
        nine_direct = _request(
            clients["nine"],
            model=raw["nine_b"]["model"],
            messages=messages,
            seed=raw["candidate"]["seed_base"] + case_index * 10 + 9,
            parser=parse_recovered_final,
            **raw["direct"],
        )
        replicas = [
            _request(
                clients["four"],
                model=raw["four_b"]["model"],
                messages=messages,
                temperature=raw["candidate"]["temperature"],
                top_p=raw["candidate"]["top_p"],
                max_tokens=raw["candidate"]["max_tokens"],
                seed=raw["candidate"]["seed_base"] + case_index * 10 + replica,
                parser=parse_recovered_final,
            )
            for replica in range(raw["candidate"]["replicas"])
        ]
        strict_direct = parse_final(four_direct["output"])
        candidate_prediction, receipt = conditional_consensus(
            [reply["prediction"] for reply in replicas],
            four_direct["prediction"],
            direct_strict_parseable=strict_direct is not None,
            parse_failure_minimum_votes=raw["override_rule"][
                "direct_strict_parse_failure_minimum_votes"
            ],
            parseable_minimum_votes=raw["override_rule"][
                "direct_strict_parseable_minimum_votes"
            ],
        )
        expected = str(row["numeric_answer"])
        base = {
            "case_id": row["sample_id"],
            "stratum": row["stratum"],
            "expected": expected,
        }
        arms["four_direct"].append(
            {
                **base,
                **four_direct,
                "strict_prediction": strict_direct,
                "correct": score_prediction(
                    four_direct["prediction"], expected
                ),
            }
        )
        arms["nine_direct"].append(
            {
                **base,
                **nine_direct,
                "strict_prediction": parse_final(nine_direct["output"]),
                "correct": score_prediction(
                    nine_direct["prediction"], expected
                ),
            }
        )
        candidate_output = (
            next(
                reply["output"]
                for reply in replicas
                if reply["prediction"] == candidate_prediction
            )
            if not receipt["fallback"]
            else four_direct["output"]
        )
        arms["candidate"].append(
            {
                **base,
                "output": candidate_output,
                "prediction": candidate_prediction,
                "correct": score_prediction(candidate_prediction, expected),
                "replica_usage": [reply["usage"] for reply in replicas],
            }
        )
        receipts.append(
            {
                "case_id": row["sample_id"],
                **receipt,
            }
        )
    for name, rows in arms.items():
        with (output_root / f"{name}.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    (output_root / "receipts.json").write_text(
        json.dumps(receipts, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "schema_version": "nano_harness_orca_conditional_majority_raw_v4",
        "experiment_id": raw["experiment_id"],
        "selection": {
            "cases": len(selection["cases"]),
            "case_ids_sha256": selection["case_ids_sha256"],
            "excluded_source_ids_sha256": selection[
                "excluded_source_ids_sha256"
            ],
        },
        "raw": {
            name: {
                "path": str(output_root / f"{name}.jsonl"),
                "sha256": sha256_file(output_root / f"{name}.jsonl"),
            }
            for name in arms
        },
        "receipts_sha256": sha256_file(output_root / "receipts.json"),
        "generation_boundary": {
            "expected_used_during_generation": False,
            "scoring_after_generation": True,
        },
    }
