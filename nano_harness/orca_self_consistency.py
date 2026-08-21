from __future__ import annotations

import hashlib
import json
import random
import re
import time
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Any

from openai import OpenAI

from nano_harness.baseline import sha256_file


CONFIG_SCHEMA = "nano_harness_orca_self_consistency_v1"
FINAL_PATTERN = re.compile(
    r"^FINAL: ([-+]?(?:[0-9]+(?:\.[0-9]+)?|[0-9]+/[0-9]+))$"
)


@dataclass(frozen=True)
class Config:
    path: Path
    raw: dict[str, Any]

    @property
    def root(self) -> Path:
        return self.path.parents[2]

    def resolve(self, value: str) -> Path:
        return (self.root / value).resolve()


def load_config(path: str | Path) -> Config:
    config_path = Path(path).resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("unsupported self-consistency schema")
    if (
        raw["experiment_id"] != "orca-math-self-consistency-v1"
        or raw["cases_by_stratum"]
        != {"short": 24, "medium": 48, "long": 24}
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
            "bootstrap_seed": 20260821,
            "alpha": 0.05,
            "minimum_candidate_only_wins": 6,
        }
    ):
        raise ValueError("self-consistency contract differs")
    return Config(path=config_path, raw=raw)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _rank(seed: str, sample_id: str) -> str:
    return hashlib.sha256(f"{seed}\n{sample_id}".encode()).hexdigest()


def _sha256_lines(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def select_cases(config: Config) -> dict[str, Any]:
    raw = config.raw
    dataset_path = config.resolve(raw["dataset_path"])
    if sha256_file(dataset_path) != raw["dataset_sha256"]:
        raise ValueError("self-consistency dataset identity differs")
    prior_paths = [
        config.resolve(raw["prior_dpo_v1_preregister_path"]),
        config.resolve(raw["prior_dpo_v2_preregister_path"]),
    ]
    prior_hashes = [
        raw["prior_dpo_v1_preregister_sha256"],
        raw["prior_dpo_v2_preregister_sha256"],
    ]
    for path, expected in zip(prior_paths, prior_hashes):
        if sha256_file(path) != expected:
            raise ValueError("self-consistency prior DPO identity differs")
    result_path = config.resolve(raw["prior_dpo_v2_result_path"])
    if sha256_file(result_path) != raw["prior_dpo_v2_result_sha256"]:
        raise ValueError("self-consistency prior result identity differs")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("decision", {}).get("candidate_admitted") is not False:
        raise ValueError("self-consistency prior DPO boundary differs")
    prior_ids = set()
    for path in prior_paths:
        receipt = json.loads(path.read_text(encoding="utf-8"))
        prior_ids.update(receipt["selection"]["train_ids"])
        prior_ids.update(receipt["selection"]["dev_ids"])
    rows = [
        row
        for row in _read_jsonl(dataset_path)
        if row["sample_id"] not in prior_ids and row["split"] == "train"
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
    ids = [row["sample_id"] for row in selected]
    if len(selected) != 96 or len(set(ids)) != 96 or set(ids) & prior_ids:
        raise ValueError("self-consistency selection differs")
    return {
        "cases": selected,
        "case_ids": ids,
        "case_ids_sha256": _sha256_lines(ids),
        "prior_ids_sha256": _sha256_lines(sorted(prior_ids)),
    }


def parse_final(output: str) -> str | None:
    lines = [line.strip() for line in output.strip().splitlines() if line.strip()]
    if not lines:
        return None
    match = FINAL_PATTERN.fullmatch(lines[-1])
    return match.group(1) if match else None


def _number(value: str) -> Fraction | None:
    try:
        if "/" in value:
            numerator, denominator = value.split("/", 1)
            return Fraction(Decimal(numerator)) / Fraction(Decimal(denominator))
        return Fraction(Decimal(value))
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None


def score_prediction(prediction: str | None, expected: str) -> bool:
    return (
        prediction is not None
        and _number(prediction) is not None
        and _number(prediction) == _number(expected)
    )


def consensus_prediction(
    predictions: list[str | None],
    direct_prediction: str | None,
    *,
    minimum_agreement: int,
) -> tuple[str | None, dict[str, Any]]:
    normalized = [
        str(_number(prediction))
        for prediction in predictions
        if prediction is not None and _number(prediction) is not None
    ]
    counts = Counter(normalized)
    if counts:
        value, votes = counts.most_common(1)[0]
        if votes >= minimum_agreement:
            chosen = next(
                prediction
                for prediction in predictions
                if prediction is not None and str(_number(prediction)) == value
            )
            return chosen, {
                "override": _number(chosen) != _number(direct_prediction or ""),
                "consensus_votes": votes,
                "parseable_replicas": len(normalized),
                "fallback": False,
            }
    return direct_prediction, {
        "override": False,
        "consensus_votes": max(counts.values(), default=0),
        "parseable_replicas": len(normalized),
        "fallback": True,
    }


def _messages(row: dict[str, Any]) -> list[dict[str, str]]:
    return row["prompt_messages"]


def _request(
    client: OpenAI,
    *,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    top_p: float,
    max_tokens: int,
    seed: int | None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "extra_body": {
            "chat_template_kwargs": {"enable_thinking": False}
        },
    }
    if seed is not None:
        kwargs["seed"] = seed
    started = time.time()
    response = client.chat.completions.create(**kwargs)
    output = response.choices[0].message.content or ""
    return {
        "output": output,
        "prediction": parse_final(output),
        "finish_reason": response.choices[0].finish_reason,
        "usage": (
            response.usage.model_dump(exclude_none=True)
            if response.usage
            else {}
        ),
        "latency_seconds": time.time() - started,
    }


def run_selection(
    config: Config,
    selection: dict[str, Any],
) -> dict[str, Any]:
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
        messages = _messages(row)
        four_direct = _request(
            clients["four"],
            model=raw["four_b"]["model"],
            messages=messages,
            seed=raw["candidate"]["seed_base"] + case_index * 10,
            **raw["direct"],
        )
        nine_direct = _request(
            clients["nine"],
            model=raw["nine_b"]["model"],
            messages=messages,
            seed=raw["candidate"]["seed_base"] + case_index * 10 + 9,
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
            )
            for replica in range(raw["candidate"]["replicas"])
        ]
        candidate_prediction, receipt = consensus_prediction(
            [reply["prediction"] for reply in replicas],
            four_direct["prediction"],
            minimum_agreement=raw["candidate"]["minimum_agreement"],
        )
        expected = str(row["expected"])
        base = {
            "case_id": row["sample_id"],
            "stratum": row["stratum"],
            "expected": expected,
        }
        arms["four_direct"].append(
            {
                **base,
                **four_direct,
                "correct": score_prediction(
                    four_direct["prediction"], expected
                ),
            }
        )
        arms["nine_direct"].append(
            {
                **base,
                **nine_direct,
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
                "replica_output_sha256": [
                    hashlib.sha256(reply["output"].encode()).hexdigest()
                    for reply in replicas
                ],
            }
        )
    for name, rows in arms.items():
        path = output_root / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    (output_root / "receipts.json").write_text(
        json.dumps(receipts, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "schema_version": "nano_harness_orca_self_consistency_raw_v1",
        "experiment_id": raw["experiment_id"],
        "selection": {
            "case_ids_sha256": selection["case_ids_sha256"],
            "prior_ids_sha256": selection["prior_ids_sha256"],
            "cases": len(selection["cases"]),
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


def run(config: Config) -> dict[str, Any]:
    return run_selection(config, select_cases(config))
