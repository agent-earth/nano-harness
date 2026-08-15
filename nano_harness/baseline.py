from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

from nano_harness.client import OpenRouterClient
from nano_harness.config import ModelConfig


FINAL_PATTERN = re.compile(r"(?im)^\s*FINAL\s*:\s*(.+?)\s*$")
NUMBER_PATTERN = re.compile(r"[-+]?(?:\d[\d,]*\.?\d*|\.\d+)")
LETTER_PATTERN = re.compile(r"\b([A-J])\b", re.IGNORECASE)


@dataclass(frozen=True)
class BaselineCase:
    case_id: str
    benchmark: str
    prompt: str
    expected: str
    scorer: str
    source_index: int
    metadata: dict[str, Any]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    path: str
    sha256: str
    scorer: str
    limit: int
    max_prompt_chars: int | None = None


@dataclass(frozen=True)
class SuiteManifest:
    schema_version: str
    suite_id: str
    selection_seed: str
    system_prompt: str
    max_tokens: int
    temperature: float
    datasets: tuple[DatasetSpec, ...]


def load_manifest(path: str | Path) -> SuiteManifest:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    expected_keys = {
        "schema_version",
        "suite_id",
        "selection_seed",
        "system_prompt",
        "max_tokens",
        "temperature",
        "datasets",
    }
    unknown = set(raw) - expected_keys
    if unknown:
        raise ValueError(f"unknown manifest fields: {sorted(unknown)}")
    if raw["schema_version"] != "nano_harness_baseline_suite_v1":
        raise ValueError("unsupported baseline suite schema")
    datasets = tuple(DatasetSpec(**item) for item in raw["datasets"])
    if len(datasets) < 3:
        raise ValueError("baseline suite requires at least three datasets")
    if len({item.name for item in datasets}) != len(datasets):
        raise ValueError("dataset names must be unique")
    return SuiteManifest(
        schema_version=raw["schema_version"],
        suite_id=raw["suite_id"],
        selection_seed=raw["selection_seed"],
        system_prompt=raw["system_prompt"],
        max_tokens=int(raw["max_tokens"]),
        temperature=float(raw["temperature"]),
        datasets=datasets,
    )


def resolve_dataset_path(dataset_root: Path, relative_path: str) -> Path:
    root = dataset_root.resolve()
    path = (root / relative_path).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"dataset path escapes root: {relative_path}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_cases(manifest: SuiteManifest, dataset_root: Path) -> list[BaselineCase]:
    from datasets import Dataset

    selected: list[BaselineCase] = []
    for spec in manifest.datasets:
        path = resolve_dataset_path(dataset_root, spec.path)
        actual_sha256 = sha256_file(path)
        if actual_sha256 != spec.sha256:
            raise ValueError(
                f"{spec.name} sha256 mismatch: expected {spec.sha256}, got {actual_sha256}"
            )
        records = Dataset.from_parquet(str(path))
        cases = [
            build_case(spec.name, spec.scorer, index, record)
            for index, record in enumerate(records)
        ]
        if spec.max_prompt_chars is not None:
            cases = [
                case for case in cases if len(case.prompt) <= spec.max_prompt_chars
            ]
        if len(cases) < spec.limit:
            raise ValueError(
                f"{spec.name} has only {len(cases)} eligible cases for limit {spec.limit}"
            )
        cases.sort(
            key=lambda case: hashlib.sha256(
                f"{manifest.selection_seed}\0{case.benchmark}\0{case.case_id}".encode()
            ).hexdigest()
        )
        selected.extend(cases[: spec.limit])
    return selected


def build_case(
    benchmark: str,
    scorer: str,
    source_index: int,
    record: dict[str, Any],
) -> BaselineCase:
    if benchmark == "gsm8k":
        question = str(record["question"]).strip()
        expected = extract_gsm8k_reference(str(record["answer"]))
        prompt = (
            "Solve the following math problem. Show concise reasoning, then end with "
            "exactly one line in the form FINAL: <number>.\n\n"
            f"Problem: {question}"
        )
        metadata: dict[str, Any] = {}
    elif benchmark == "mmlu":
        question = str(record["question"]).strip()
        choices = [str(choice) for choice in record["choices"]]
        expected = chr(ord("A") + int(record["answer"]))
        prompt = format_multiple_choice_prompt(question, choices)
        metadata = {"subject": str(record["subject"])}
    elif benchmark == "gpqa_diamond":
        question = str(record["question"]).strip()
        expected = str(record["answer"]).strip().upper()
        prompt = (
            "Answer the following multiple-choice science question. Reason concisely, "
            "then end with exactly one line in the form FINAL: <letter>.\n\n"
            f"{question}"
        )
        metadata = {}
    else:
        raise ValueError(f"unsupported baseline benchmark: {benchmark}")

    case_digest = hashlib.sha256(
        f"{benchmark}\0{question}".encode("utf-8")
    ).hexdigest()[:16]
    return BaselineCase(
        case_id=f"{benchmark}-{case_digest}",
        benchmark=benchmark,
        prompt=prompt,
        expected=expected,
        scorer=scorer,
        source_index=source_index,
        metadata=metadata,
    )


def format_multiple_choice_prompt(question: str, choices: list[str]) -> str:
    if not 2 <= len(choices) <= 10:
        raise ValueError(f"unsupported number of choices: {len(choices)}")
    rendered = "\n".join(
        f"{chr(ord('A') + index)}. {choice}"
        for index, choice in enumerate(choices)
    )
    return (
        "Answer the following multiple-choice question. Reason concisely, then end "
        "with exactly one line in the form FINAL: <letter>.\n\n"
        f"Question: {question}\n\nChoices:\n{rendered}"
    )


def extract_gsm8k_reference(answer: str) -> str:
    marker = answer.rsplit("####", 1)
    if len(marker) != 2:
        raise ValueError("GSM8K answer does not contain ####")
    normalized = normalize_number(marker[1])
    if normalized is None:
        raise ValueError("GSM8K reference does not contain a number")
    return normalized


def extract_prediction(output: str, scorer: str) -> str | None:
    matches = FINAL_PATTERN.findall(output)
    candidate = matches[-1].strip() if matches else ""
    if scorer == "numeric_exact":
        return normalize_number(candidate)
    if scorer == "choice_exact":
        match = LETTER_PATTERN.search(candidate)
        return match.group(1).upper() if match else None
    raise ValueError(f"unsupported scorer: {scorer}")


def normalize_number(value: str) -> str | None:
    match = NUMBER_PATTERN.search(value.replace("$", ""))
    if not match:
        return None
    token = match.group(0).replace(",", "")
    try:
        number = float(token)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    if number == 0:
        number = 0.0
    if number.is_integer():
        return str(int(number))
    return format(number, ".12g")


def score_output(output: str, expected: str, scorer: str) -> tuple[float, str | None]:
    prediction = extract_prediction(output, scorer)
    return float(prediction == expected), prediction


def run_suite(
    manifest: SuiteManifest,
    dataset_root: Path,
    model: ModelConfig,
    output_path: Path,
) -> dict[str, Any]:
    cases = load_cases(manifest, dataset_root)
    client = OpenRouterClient(model)
    completed = _completed_case_ids(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    attempted = 0
    for case in cases:
        if case.case_id in completed:
            continue
        attempted += 1
        started = time.perf_counter()
        try:
            reply = client.complete(
                [
                    {"role": "system", "content": manifest.system_prompt},
                    {"role": "user", "content": case.prompt},
                ]
            )
        except Exception as exc:
            latency_seconds = time.perf_counter() - started
            record = {
                "schema_version": "nano_harness_baseline_case_v1",
                "suite_id": manifest.suite_id,
                "case_id": case.case_id,
                "benchmark": case.benchmark,
                "model": model.name,
                "source_index": case.source_index,
                "expected": case.expected,
                "prediction": None,
                "score": 0.0,
                "status": "error",
                "failure_type": "model_api_error",
                "error": f"{type(exc).__name__}: {exc}",
                "latency_seconds": round(latency_seconds, 6),
                "usage": {},
                "output": "",
                "metadata": case.metadata,
            }
            _append_jsonl(output_path, record)
            continue
        latency_seconds = time.perf_counter() - started
        score, prediction = score_output(reply.content, case.expected, case.scorer)
        record = {
            "schema_version": "nano_harness_baseline_case_v1",
            "suite_id": manifest.suite_id,
            "case_id": case.case_id,
            "benchmark": case.benchmark,
            "model": model.name,
            "source_index": case.source_index,
            "expected": case.expected,
            "prediction": prediction,
            "score": score,
            "status": "completed",
            "latency_seconds": round(latency_seconds, 6),
            "usage": reply.usage,
            "output": reply.content,
            "metadata": case.metadata,
        }
        _append_jsonl(output_path, record)
    summary = summarize_baseline(output_path)
    summary["attempted_this_invocation"] = attempted
    summary["output"] = str(output_path)
    return summary


def summarize_baseline(path: Path) -> dict[str, Any]:
    attempts = list(_iter_jsonl(path))
    latest_by_case: dict[str, dict[str, Any]] = {}
    for record in attempts:
        latest_by_case[str(record["case_id"])] = record
    records = list(latest_by_case.values())
    by_benchmark: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_benchmark.setdefault(str(record["benchmark"]), []).append(record)
    benchmark_metrics = {
        name: _aggregate_records(items) for name, items in sorted(by_benchmark.items())
    }
    accuracies = [item["accuracy"] for item in benchmark_metrics.values()]
    return {
        "schema_version": "nano_harness_baseline_summary_v1",
        "total_attempts": len(attempts),
        "total_cases": len(records),
        "completed_cases": sum(record.get("status") == "completed" for record in records),
        "error_cases": sum(record.get("status") == "error" for record in records),
        "macro_accuracy": sum(accuracies) / len(accuracies) if accuracies else None,
        "benchmarks": benchmark_metrics,
    }


def _aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(records)
    completed = sum(record.get("status") == "completed" for record in records)
    correct = sum(float(record["score"]) for record in records)
    latencies = [float(record["latency_seconds"]) for record in records]
    prompt_tokens = [
        int(record["usage"]["prompt_tokens"])
        for record in records
        if isinstance(record.get("usage", {}).get("prompt_tokens"), int)
    ]
    completion_tokens = [
        int(record["usage"]["completion_tokens"])
        for record in records
        if isinstance(record.get("usage", {}).get("completion_tokens"), int)
    ]
    return {
        "cases": count,
        "completed": completed,
        "errors": count - completed,
        "correct": int(correct),
        "accuracy": correct / count if count else None,
        "mean_latency_seconds": sum(latencies) / count if count else None,
        "prompt_tokens": sum(prompt_tokens),
        "completion_tokens": sum(completion_tokens),
        "parse_failures": sum(record.get("prediction") is None for record in records),
    }


def _completed_case_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        str(record["case_id"])
        for record in _iter_jsonl(path)
        if record.get("status") == "completed"
    }


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os_fsync(handle)


def os_fsync(handle: Any) -> None:
    import os

    os.fsync(handle.fileno())


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def case_manifest(cases: list[BaselineCase]) -> list[dict[str, Any]]:
    return [
        {
            "case_id": case.case_id,
            "benchmark": case.benchmark,
            "source_index": case.source_index,
            "expected": case.expected,
            "scorer": case.scorer,
            "metadata": case.metadata,
        }
        for case in cases
    ]
