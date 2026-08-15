from __future__ import annotations

import hashlib
import json
import math
import random
import re
import time
from dataclasses import asdict, dataclass, replace
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
    source_chars: int
    system_prompt: str
    max_tokens: int
    metadata: dict[str, Any]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    path: str
    sha256: str
    scorer: str
    limit: int
    max_source_chars: int | None = None
    answer_only: bool = False
    max_tokens: int | None = None
    system_prompt: str | None = None


@dataclass(frozen=True)
class SuiteManifest:
    schema_version: str
    suite_id: str
    selection_seed: str
    system_prompt: str
    max_tokens: int
    temperature: float
    chat_template_kwargs: dict[str, Any]
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
        "chat_template_kwargs",
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
        chat_template_kwargs=dict(raw.get("chat_template_kwargs", {})),
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
            build_case(
                spec.name,
                spec.scorer,
                index,
                record,
                answer_only=spec.answer_only,
                system_prompt=spec.system_prompt or manifest.system_prompt,
                max_tokens=spec.max_tokens or manifest.max_tokens,
            )
            for index, record in enumerate(records)
        ]
        if spec.max_source_chars is not None:
            cases = [
                case for case in cases if case.source_chars <= spec.max_source_chars
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
    *,
    answer_only: bool = False,
    system_prompt: str = "",
    max_tokens: int = 0,
) -> BaselineCase:
    if benchmark == "gsm8k":
        question = str(record["question"]).strip()
        expected = extract_gsm8k_reference(str(record["answer"]))
        prefix = (
            "Return only one line in the form FINAL: <number>. Do not show reasoning.\n\n"
            if answer_only
            else "Solve the following math problem. Show concise reasoning, then end with "
            "exactly one line in the form FINAL: <number>.\n\n"
        )
        prompt = prefix + f"Problem: {question}"
        metadata: dict[str, Any] = {}
    elif benchmark == "mmlu":
        question = str(record["question"]).strip()
        choices = [str(choice) for choice in record["choices"]]
        expected = chr(ord("A") + int(record["answer"]))
        prompt = format_multiple_choice_prompt(
            question,
            choices,
            answer_only=answer_only,
        )
        metadata = {"subject": str(record["subject"])}
    elif benchmark == "gpqa_diamond":
        question = str(record["question"]).strip()
        expected = str(record["answer"]).strip().upper()
        prefix = (
            "Return only one line in the form FINAL: <letter>. Do not show reasoning.\n\n"
            if answer_only
            else "Answer the following multiple-choice science question. Reason concisely, "
            "then end with exactly one line in the form FINAL: <letter>.\n\n"
        )
        prompt = prefix + question
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
        source_chars=len(question),
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        metadata=metadata,
    )


def format_multiple_choice_prompt(
    question: str,
    choices: list[str],
    *,
    answer_only: bool = False,
) -> str:
    if not 2 <= len(choices) <= 10:
        raise ValueError(f"unsupported number of choices: {len(choices)}")
    rendered = "\n".join(
        f"{chr(ord('A') + index)}. {choice}"
        for index, choice in enumerate(choices)
    )
    prefix = (
        "Return only one line in the form FINAL: <letter>. Do not show reasoning.\n\n"
        if answer_only
        else "Answer the following multiple-choice question. Reason concisely, then end "
        "with exactly one line in the form FINAL: <letter>.\n\n"
    )
    return prefix + f"Question: {question}\n\nChoices:\n{rendered}"


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
    clients: dict[int, OpenRouterClient] = {}
    completed = _completed_case_ids(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    attempted = 0
    for case in cases:
        if case.case_id in completed:
            continue
        attempted += 1
        started = time.perf_counter()
        try:
            if case.max_tokens not in clients:
                clients[case.max_tokens] = OpenRouterClient(
                    replace(model, max_tokens=case.max_tokens)
                )
            client = clients[case.max_tokens]
            reply = client.complete(
                [
                    {"role": "system", "content": case.system_prompt},
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
                "max_tokens": case.max_tokens,
                "prompt_sha256": hashlib.sha256(case.prompt.encode()).hexdigest(),
                "system_prompt_sha256": hashlib.sha256(
                    case.system_prompt.encode()
                ).hexdigest(),
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
            "max_tokens": case.max_tokens,
            "prompt_sha256": hashlib.sha256(case.prompt.encode()).hexdigest(),
            "system_prompt_sha256": hashlib.sha256(
                case.system_prompt.encode()
            ).hexdigest(),
            "expected": case.expected,
            "prediction": prediction,
            "score": score,
            "status": "completed",
            "finish_reason": _finish_reason(reply.raw),
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


def compare_baselines(
    candidate_path: Path,
    baseline_path: Path,
    *,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 35,
) -> dict[str, Any]:
    candidate = _latest_records_by_case(candidate_path)
    baseline = _latest_records_by_case(baseline_path)
    if set(candidate) != set(baseline):
        missing_candidate = sorted(set(baseline) - set(candidate))
        missing_baseline = sorted(set(candidate) - set(baseline))
        raise ValueError(
            "case id mismatch: "
            f"missing_candidate={missing_candidate[:5]}, "
            f"missing_baseline={missing_baseline[:5]}"
        )
    if not candidate:
        raise ValueError("cannot compare empty baseline files")

    benchmarks = sorted({str(record["benchmark"]) for record in candidate.values()})
    by_benchmark: dict[str, dict[str, Any]] = {}
    for benchmark in benchmarks:
        case_ids = sorted(
            case_id
            for case_id, record in candidate.items()
            if record["benchmark"] == benchmark
        )
        by_benchmark[benchmark] = _paired_metrics(
            [candidate[case_id] for case_id in case_ids],
            [baseline[case_id] for case_id in case_ids],
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=f"{bootstrap_seed}:{benchmark}",
        )

    all_case_ids = sorted(candidate)
    overall = _paired_metrics(
        [candidate[case_id] for case_id in all_case_ids],
        [baseline[case_id] for case_id in all_case_ids],
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=f"{bootstrap_seed}:overall",
    )
    candidate_macro = sum(
        item["candidate_accuracy"] for item in by_benchmark.values()
    ) / len(by_benchmark)
    baseline_macro = sum(
        item["baseline_accuracy"] for item in by_benchmark.values()
    ) / len(by_benchmark)
    return {
        "schema_version": "nano_harness_baseline_comparison_v1",
        "candidate_model": _single_model(candidate.values()),
        "baseline_model": _single_model(baseline.values()),
        "cases": len(all_case_ids),
        "candidate_macro_accuracy": candidate_macro,
        "baseline_macro_accuracy": baseline_macro,
        "macro_delta": candidate_macro - baseline_macro,
        "overall_micro": overall,
        "benchmarks": by_benchmark,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
    }


def _latest_records_by_case(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for record in _iter_jsonl(path):
        records[str(record["case_id"])] = record
    return records


def _single_model(records: Iterable[dict[str, Any]]) -> str:
    models = {str(record["model"]) for record in records}
    if len(models) != 1:
        raise ValueError(f"expected one model per result file, got {sorted(models)}")
    return models.pop()


def _paired_metrics(
    candidate: list[dict[str, Any]],
    baseline: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: str,
) -> dict[str, Any]:
    if len(candidate) != len(baseline):
        raise ValueError("paired metric inputs differ in length")
    pairs = [
        (float(candidate_row["score"]), float(baseline_row["score"]))
        for candidate_row, baseline_row in zip(candidate, baseline)
    ]
    count = len(pairs)
    candidate_correct = sum(pair[0] for pair in pairs)
    baseline_correct = sum(pair[1] for pair in pairs)
    candidate_only = sum(left == 1.0 and right == 0.0 for left, right in pairs)
    baseline_only = sum(left == 0.0 and right == 1.0 for left, right in pairs)
    both_correct = sum(left == 1.0 and right == 1.0 for left, right in pairs)
    both_wrong = sum(left == 0.0 and right == 0.0 for left, right in pairs)
    deltas = [left - right for left, right in pairs]
    confidence_interval = _bootstrap_mean_interval(
        deltas,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    return {
        "cases": count,
        "candidate_correct": int(candidate_correct),
        "baseline_correct": int(baseline_correct),
        "candidate_accuracy": candidate_correct / count,
        "baseline_accuracy": baseline_correct / count,
        "delta": sum(deltas) / count,
        "paired_counts": {
            "both_correct": both_correct,
            "candidate_only": candidate_only,
            "baseline_only": baseline_only,
            "both_wrong": both_wrong,
        },
        "mcnemar_exact_p": _mcnemar_exact_p(candidate_only, baseline_only),
        "paired_bootstrap_95_ci": confidence_interval,
        "candidate_only_cases": [
            row["case_id"]
            for row, pair in zip(candidate, pairs)
            if pair == (1.0, 0.0)
        ],
        "baseline_only_cases": [
            row["case_id"]
            for row, pair in zip(candidate, pairs)
            if pair == (0.0, 1.0)
        ],
        "candidate_parse_failures": [
            row["case_id"] for row in candidate if row.get("prediction") is None
        ],
        "baseline_parse_failures": [
            row["case_id"] for row in baseline if row.get("prediction") is None
        ],
    }


def _bootstrap_mean_interval(
    values: list[float],
    *,
    samples: int,
    seed: str,
) -> list[float]:
    if not values:
        raise ValueError("cannot bootstrap an empty list")
    rng = random.Random(seed)
    count = len(values)
    estimates = sorted(
        sum(values[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(samples)
    )
    lower = estimates[int(samples * 0.025)]
    upper = estimates[min(samples - 1, int(samples * 0.975))]
    return [lower, upper]


def _mcnemar_exact_p(candidate_only: int, baseline_only: int) -> float:
    discordant = candidate_only + baseline_only
    if discordant == 0:
        return 1.0
    tail = min(candidate_only, baseline_only)
    probability = sum(math.comb(discordant, index) for index in range(tail + 1))
    return min(1.0, 2.0 * probability / (2**discordant))


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
        "length_truncations": sum(
            record.get("finish_reason") == "length" for record in records
        ),
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


def _finish_reason(raw: dict[str, Any]) -> str | None:
    choices = raw.get("choices", [])
    if not choices or not isinstance(choices[0], dict):
        return None
    value = choices[0].get("finish_reason")
    return str(value) if value is not None else None


def case_manifest(cases: list[BaselineCase]) -> list[dict[str, Any]]:
    return [
        {
            "case_id": case.case_id,
            "benchmark": case.benchmark,
            "source_index": case.source_index,
            "source_chars": case.source_chars,
            "max_tokens": case.max_tokens,
            "prompt_sha256": hashlib.sha256(case.prompt.encode()).hexdigest(),
            "system_prompt_sha256": hashlib.sha256(
                case.system_prompt.encode()
            ).hexdigest(),
            "expected": case.expected,
            "scorer": case.scorer,
            "metadata": case.metadata,
        }
        for case in cases
    ]
