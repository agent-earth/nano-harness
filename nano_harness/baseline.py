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
NUMERIC_FINAL_REGEX = r"FINAL: [-+]?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)"


@dataclass(frozen=True)
class BaselineCase:
    case_id: str
    benchmark: str
    prompt: str
    draft_prompt: str
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
    start: int = 0
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
    strategy: str
    benchmark_routing: dict[str, str]
    draft_max_tokens: int
    critique_max_tokens: int
    second_solve_max_tokens: int
    option_evidence_max_tokens: int
    verifier_max_tokens: int
    normalize_bare_choice: bool
    fallback_to_protected_on_parse_failure: bool
    min_task_groups: int
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
        "strategy",
        "benchmark_routing",
        "draft_max_tokens",
        "critique_max_tokens",
        "second_solve_max_tokens",
        "option_evidence_max_tokens",
        "verifier_max_tokens",
        "normalize_bare_choice",
        "fallback_to_protected_on_parse_failure",
        "min_task_groups",
        "datasets",
    }
    unknown = set(raw) - expected_keys
    if unknown:
        raise ValueError(f"unknown manifest fields: {sorted(unknown)}")
    if raw["schema_version"] != "nano_harness_baseline_suite_v1":
        raise ValueError("unsupported baseline suite schema")
    datasets = tuple(DatasetSpec(**item) for item in raw["datasets"])
    min_task_groups = int(raw.get("min_task_groups", 3))
    if min_task_groups < 1:
        raise ValueError("min_task_groups must be at least one")
    if len(datasets) < min_task_groups:
        raise ValueError(
            f"suite requires at least {min_task_groups} dataset task groups"
        )
    if len({item.name for item in datasets}) != len(datasets):
        raise ValueError("dataset names must be unique")
    strategy = str(raw.get("strategy", "direct"))
    supported_strategies = {
        "direct",
        "draft_verify",
        "draft_critique_verify",
        "dual_solve_verify",
        "option_evidence_verify",
        "option_evidence_arbiter",
        "protected_math_arbiter",
        "protected_math_gate",
        "protected_math_majority",
        "protected_math_recovery",
        "protected_math_short_recovery",
        "protected_math_constrained_recovery",
    }
    if strategy not in supported_strategies | {"benchmark_routing"}:
        raise ValueError(f"unsupported baseline strategy: {strategy}")
    benchmark_routing = {
        str(name): str(route)
        for name, route in raw.get("benchmark_routing", {}).items()
    }
    if strategy == "benchmark_routing":
        dataset_names = {item.name for item in datasets}
        if set(benchmark_routing) != dataset_names:
            raise ValueError(
                "benchmark_routing must cover every dataset exactly: "
                f"expected={sorted(dataset_names)}, "
                f"actual={sorted(benchmark_routing)}"
            )
        invalid_routes = {
            name: route
            for name, route in benchmark_routing.items()
            if route not in supported_strategies
        }
        if invalid_routes:
            raise ValueError(f"unsupported benchmark routes: {invalid_routes}")
    elif benchmark_routing:
        raise ValueError(
            "benchmark_routing is only allowed with strategy=benchmark_routing"
        )
    return SuiteManifest(
        schema_version=raw["schema_version"],
        suite_id=raw["suite_id"],
        selection_seed=raw["selection_seed"],
        system_prompt=raw["system_prompt"],
        max_tokens=int(raw["max_tokens"]),
        temperature=float(raw["temperature"]),
        chat_template_kwargs=dict(raw.get("chat_template_kwargs", {})),
        strategy=strategy,
        benchmark_routing=benchmark_routing,
        draft_max_tokens=int(raw.get("draft_max_tokens", 384)),
        critique_max_tokens=int(raw.get("critique_max_tokens", 192)),
        second_solve_max_tokens=int(raw.get("second_solve_max_tokens", 384)),
        option_evidence_max_tokens=int(raw.get("option_evidence_max_tokens", 96)),
        verifier_max_tokens=int(raw.get("verifier_max_tokens", 32)),
        normalize_bare_choice=bool(raw.get("normalize_bare_choice", False)),
        fallback_to_protected_on_parse_failure=bool(
            raw.get("fallback_to_protected_on_parse_failure", False)
        ),
        min_task_groups=min_task_groups,
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
        if spec.start < 0:
            raise ValueError(f"{spec.name} start must be non-negative")
        required = spec.start + spec.limit
        if len(cases) < required:
            raise ValueError(
                f"{spec.name} has only {len(cases)} eligible cases for "
                f"start {spec.start} and limit {spec.limit}"
            )
        cases.sort(
            key=lambda case: hashlib.sha256(
                f"{manifest.selection_seed}\0{case.benchmark}\0{case.case_id}".encode()
            ).hexdigest()
        )
        selected.extend(cases[spec.start : required])
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
        reasoning_prefix = (
            "Solve the following math problem. Show concise reasoning, then end with "
            "exactly one line in the form FINAL: <number>.\n\n"
        )
        prefix = (
            "Return only one line in the form FINAL: <number>. Do not show reasoning.\n\n"
            if answer_only
            else reasoning_prefix
        )
        prompt = prefix + f"Problem: {question}"
        draft_prompt = reasoning_prefix + f"Problem: {question}"
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
        draft_prompt = format_multiple_choice_prompt(
            question,
            choices,
            answer_only=False,
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
        reasoning_prefix = (
            "Answer the following multiple-choice science question. Reason concisely, "
            "then end with exactly one line in the form FINAL: <letter>.\n\n"
        )
        prompt = prefix + question
        draft_prompt = reasoning_prefix + question
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
        draft_prompt=draft_prompt,
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
        selected_strategy = _strategy_for_case(manifest, case)
        try:
            if selected_strategy == "direct":
                reply, stage_metadata = _run_direct_case(
                    case,
                    model,
                    clients,
                )
            elif selected_strategy == "draft_verify":
                reply, stage_metadata = _run_draft_verify_case(
                    case,
                    manifest,
                    model,
                    clients,
                )
            elif selected_strategy == "draft_critique_verify":
                reply, stage_metadata = _run_draft_critique_verify_case(
                    case,
                    manifest,
                    model,
                    clients,
                )
            elif selected_strategy == "dual_solve_verify":
                reply, stage_metadata = _run_dual_solve_verify_case(
                    case,
                    manifest,
                    model,
                    clients,
                )
            elif selected_strategy == "option_evidence_verify":
                reply, stage_metadata = _run_option_evidence_verify_case(
                    case,
                    manifest,
                    model,
                    clients,
                )
            elif selected_strategy == "option_evidence_arbiter":
                reply, stage_metadata = _run_option_evidence_arbiter_case(
                    case,
                    manifest,
                    model,
                    clients,
                )
            elif selected_strategy == "protected_math_arbiter":
                reply, stage_metadata = _run_protected_math_arbiter_case(
                    case,
                    manifest,
                    model,
                    clients,
                )
            elif selected_strategy == "protected_math_gate":
                reply, stage_metadata = _run_protected_math_gate_case(
                    case,
                    manifest,
                    model,
                    clients,
                )
            elif selected_strategy == "protected_math_majority":
                reply, stage_metadata = _run_protected_math_majority_case(
                    case,
                    manifest,
                    model,
                    clients,
                )
            elif selected_strategy == "protected_math_recovery":
                reply, stage_metadata = _run_protected_math_recovery_case(
                    case,
                    manifest,
                    model,
                    clients,
                )
            elif selected_strategy == "protected_math_short_recovery":
                reply, stage_metadata = _run_protected_math_short_recovery_case(
                    case,
                    manifest,
                    model,
                    clients,
                )
            else:
                reply, stage_metadata = _run_protected_math_constrained_recovery_case(
                    case,
                    manifest,
                    model,
                    clients,
                )
        except Exception as exc:
            latency_seconds = time.perf_counter() - started
            record = {
                "schema_version": "nano_harness_baseline_case_v1",
                "suite_id": manifest.suite_id,
                "case_id": case.case_id,
                "benchmark": case.benchmark,
                "model": model.name,
                "strategy": manifest.strategy,
                "selected_strategy": selected_strategy,
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
            "strategy": manifest.strategy,
            "selected_strategy": selected_strategy,
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
            "stages": stage_metadata,
            "metadata": case.metadata,
        }
        _append_jsonl(output_path, record)
    summary = summarize_baseline(output_path)
    summary["attempted_this_invocation"] = attempted
    summary["output"] = str(output_path)
    return summary


def _strategy_for_case(
    manifest: SuiteManifest,
    case: BaselineCase,
) -> str:
    if manifest.strategy == "benchmark_routing":
        return manifest.benchmark_routing[case.benchmark]
    return manifest.strategy


def _run_direct_case(
    case: BaselineCase,
    model: ModelConfig,
    clients: dict[int, OpenRouterClient],
) -> tuple[Any, dict[str, Any]]:
    client = _client_for_budget(clients, model, case.max_tokens)
    reply = client.complete(
        [
            {"role": "system", "content": case.system_prompt},
            {"role": "user", "content": case.prompt},
        ]
    )
    return reply, {
        "direct": {
            "max_tokens": case.max_tokens,
            "input_sha256": hashlib.sha256(case.prompt.encode()).hexdigest(),
            "finish_reason": _finish_reason(reply.raw),
            "usage": reply.usage,
        }
    }


def _run_draft_verify_case(
    case: BaselineCase,
    manifest: SuiteManifest,
    model: ModelConfig,
    clients: dict[int, OpenRouterClient],
) -> tuple[Any, dict[str, Any]]:
    draft_client = _client_for_budget(clients, model, manifest.draft_max_tokens)
    draft = draft_client.complete(
        [
            {
                "role": "system",
                "content": (
                    "Solve the task carefully. Produce a compact candidate analysis "
                    "and candidate answer for a separate verifier. Do not use tools."
                ),
            },
            {"role": "user", "content": case.draft_prompt},
        ]
    )
    verifier_client = _client_for_budget(
        clients,
        model,
        manifest.verifier_max_tokens,
    )
    verifier = verifier_client.complete(
        [
            {
                "role": "system",
                "content": (
                    "You are the final verifier. Check the candidate against the "
                    "original task, correct it if needed, and return only one FINAL "
                    "line in the exact format requested by the task. Do not explain."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"<original_task>\n{case.prompt}\n</original_task>\n\n"
                    f"<candidate>\n{draft.content}\n</candidate>"
                ),
            },
        ]
    )
    verifier_usage = dict(verifier.usage)
    combined_usage = _sum_usage(draft.usage, verifier_usage)
    verifier.usage = combined_usage
    return verifier, {
        "draft": {
            "max_tokens": manifest.draft_max_tokens,
            "input_sha256": hashlib.sha256(
                case.draft_prompt.encode()
            ).hexdigest(),
            "finish_reason": _finish_reason(draft.raw),
            "usage": draft.usage,
            "output": draft.content,
            "output_sha256": hashlib.sha256(draft.content.encode()).hexdigest(),
        },
        "verifier": {
            "max_tokens": manifest.verifier_max_tokens,
            "finish_reason": _finish_reason(verifier.raw),
            "usage": verifier_usage,
        },
    }


def _run_draft_critique_verify_case(
    case: BaselineCase,
    manifest: SuiteManifest,
    model: ModelConfig,
    clients: dict[int, OpenRouterClient],
) -> tuple[Any, dict[str, Any]]:
    draft_client = _client_for_budget(clients, model, manifest.draft_max_tokens)
    draft = draft_client.complete(
        [
            {
                "role": "system",
                "content": (
                    "Solve the task carefully. Produce a compact candidate analysis "
                    "and candidate answer for an independent critic. Do not use tools."
                ),
            },
            {"role": "user", "content": case.draft_prompt},
        ]
    )
    critique_client = _client_for_budget(
        clients,
        model,
        manifest.critique_max_tokens,
    )
    critique = critique_client.complete(
        [
            {
                "role": "system",
                "content": (
                    "Act as an independent critic. Re-solve the original task, identify "
                    "any error in the candidate, and provide a corrected candidate. "
                    "Keep the critique compact and do not use tools."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"<original_task>\n{case.draft_prompt}\n</original_task>\n\n"
                    f"<candidate>\n{draft.content}\n</candidate>"
                ),
            },
        ]
    )
    verifier_client = _client_for_budget(
        clients,
        model,
        manifest.verifier_max_tokens,
    )
    verifier = verifier_client.complete(
        [
            {
                "role": "system",
                "content": (
                    "You are the final formatter. Use the original task, candidate, and "
                    "critique to return only one FINAL line in the exact format "
                    "requested by the task. Do not explain."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"<original_task>\n{case.prompt}\n</original_task>\n\n"
                    f"<candidate>\n{draft.content}\n</candidate>\n\n"
                    f"<critique>\n{critique.content}\n</critique>"
                ),
            },
        ]
    )
    verifier_usage = dict(verifier.usage)
    verifier.usage = _sum_usage(draft.usage, critique.usage, verifier_usage)
    return verifier, {
        "draft": {
            "max_tokens": manifest.draft_max_tokens,
            "finish_reason": _finish_reason(draft.raw),
            "usage": draft.usage,
            "output": draft.content,
            "output_sha256": hashlib.sha256(draft.content.encode()).hexdigest(),
        },
        "critique": {
            "max_tokens": manifest.critique_max_tokens,
            "finish_reason": _finish_reason(critique.raw),
            "usage": critique.usage,
            "output": critique.content,
            "output_sha256": hashlib.sha256(
                critique.content.encode()
            ).hexdigest(),
        },
        "verifier": {
            "max_tokens": manifest.verifier_max_tokens,
            "finish_reason": _finish_reason(verifier.raw),
            "usage": verifier_usage,
        },
    }


def _run_dual_solve_verify_case(
    case: BaselineCase,
    manifest: SuiteManifest,
    model: ModelConfig,
    clients: dict[int, OpenRouterClient],
) -> tuple[Any, dict[str, Any]]:
    draft_client = _client_for_budget(clients, model, manifest.draft_max_tokens)
    draft = draft_client.complete(
        [
            {
                "role": "system",
                "content": (
                    "Solve the math task carefully. Produce concise reasoning and a "
                    "candidate answer. Do not use tools."
                ),
            },
            {"role": "user", "content": case.draft_prompt},
        ]
    )
    second_client = _client_for_budget(
        clients,
        model,
        manifest.second_solve_max_tokens,
    )
    second = second_client.complete(
        [
            {
                "role": "system",
                "content": (
                    "Independently solve the math task from scratch. Check units, time "
                    "periods, totals, and all requested quantities. Produce concise "
                    "reasoning and a candidate answer. Do not use tools."
                ),
            },
            {"role": "user", "content": case.draft_prompt},
        ]
    )
    verifier_client = _client_for_budget(
        clients,
        model,
        manifest.verifier_max_tokens,
    )
    verifier = verifier_client.complete(
        [
            {
                "role": "system",
                "content": (
                    "You are the final selector. Reconcile the two independent "
                    "solutions against the original task. Return only one FINAL line "
                    "in the exact format requested by the task. Do not explain."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"<original_task>\n{case.prompt}\n</original_task>\n\n"
                    f"<solution_a>\n{draft.content}\n</solution_a>\n\n"
                    f"<solution_b>\n{second.content}\n</solution_b>"
                ),
            },
        ]
    )
    verifier_usage = dict(verifier.usage)
    verifier.usage = _sum_usage(draft.usage, second.usage, verifier_usage)
    return verifier, {
        "draft": {
            "max_tokens": manifest.draft_max_tokens,
            "finish_reason": _finish_reason(draft.raw),
            "usage": draft.usage,
            "output": draft.content,
            "output_sha256": hashlib.sha256(draft.content.encode()).hexdigest(),
        },
        "second_solve": {
            "max_tokens": manifest.second_solve_max_tokens,
            "finish_reason": _finish_reason(second.raw),
            "usage": second.usage,
            "output": second.content,
            "output_sha256": hashlib.sha256(second.content.encode()).hexdigest(),
        },
        "verifier": {
            "max_tokens": manifest.verifier_max_tokens,
            "finish_reason": _finish_reason(verifier.raw),
            "usage": verifier_usage,
        },
    }


def _run_option_evidence_verify_case(
    case: BaselineCase,
    manifest: SuiteManifest,
    model: ModelConfig,
    clients: dict[int, OpenRouterClient],
) -> tuple[Any, dict[str, Any]]:
    if case.scorer != "choice_exact":
        raise ValueError("option_evidence_verify requires a choice_exact case")
    option_client = _client_for_budget(
        clients,
        model,
        manifest.option_evidence_max_tokens,
    )
    evidence = []
    option_stages: dict[str, dict[str, Any]] = {}
    for letter in ("A", "B", "C", "D"):
        prompt = (
            f"<original_task>\n{case.draft_prompt}\n</original_task>\n\n"
            f"Evaluate option {letter} independently. State the strongest evidence "
            "for or against it, check the relevant facts or calculation, and end "
            f"with VERDICT {letter}: SUPPORT or VERDICT {letter}: REJECT. Do not "
            "compare against another option's analysis and do not use tools."
        )
        reply = option_client.complete(
            [
                {
                    "role": "system",
                    "content": (
                        "Act as an independent option evaluator. Focus only on the "
                        "assigned option and produce compact, falsifiable evidence."
                    ),
                },
                {"role": "user", "content": prompt},
            ]
        )
        evidence.append(f"<option_{letter}>\n{reply.content}\n</option_{letter}>")
        option_stages[letter] = {
            "max_tokens": manifest.option_evidence_max_tokens,
            "input_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "finish_reason": _finish_reason(reply.raw),
            "usage": reply.usage,
            "output": reply.content,
            "output_sha256": hashlib.sha256(reply.content.encode()).hexdigest(),
        }

    selector_client = _client_for_budget(
        clients,
        model,
        manifest.verifier_max_tokens,
    )
    selector_prompt = (
        f"<original_task>\n{case.prompt}\n</original_task>\n\n"
        + "\n\n".join(evidence)
    )
    selector = selector_client.complete(
        [
            {
                "role": "system",
                "content": (
                    "Select the best answer from the independent option evidence. "
                    "Resolve contradictions against the original task and return only "
                    "one FINAL: <letter> line. Do not explain."
                ),
            },
            {"role": "user", "content": selector_prompt},
        ]
    )
    selector_usage = dict(selector.usage)
    raw_selector_content = selector.content
    normalized_bare_choice = False
    stripped_selector = raw_selector_content.strip()
    if (
        manifest.normalize_bare_choice
        and re.fullmatch(r"[A-Da-d]", stripped_selector)
    ):
        selector.content = f"FINAL: {stripped_selector.upper()}"
        normalized_bare_choice = True
    selector.usage = _sum_usage(
        *(stage["usage"] for stage in option_stages.values()),
        selector_usage,
    )
    return selector, {
        "option_evidence": option_stages,
        "selector": {
            "max_tokens": manifest.verifier_max_tokens,
            "input_sha256": hashlib.sha256(
                selector_prompt.encode()
            ).hexdigest(),
            "finish_reason": _finish_reason(selector.raw),
            "usage": selector_usage,
            "raw_output_sha256": hashlib.sha256(
                raw_selector_content.encode()
            ).hexdigest(),
            "normalized_bare_choice": normalized_bare_choice,
        },
    }


def _run_option_evidence_arbiter_case(
    case: BaselineCase,
    manifest: SuiteManifest,
    model: ModelConfig,
    clients: dict[int, OpenRouterClient],
) -> tuple[Any, dict[str, Any]]:
    if case.scorer != "choice_exact":
        raise ValueError("option_evidence_arbiter requires a choice_exact case")

    direct_client = _client_for_budget(clients, model, case.max_tokens)
    direct = direct_client.complete(
        [
            {"role": "system", "content": case.system_prompt},
            {"role": "user", "content": case.prompt},
        ]
    )
    option_client = _client_for_budget(
        clients,
        model,
        manifest.option_evidence_max_tokens,
    )
    evidence = []
    option_stages: dict[str, dict[str, Any]] = {}
    for letter in ("A", "B", "C", "D"):
        prompt = (
            f"<original_task>\n{case.draft_prompt}\n</original_task>\n\n"
            f"Evaluate option {letter} independently. State the strongest evidence "
            "for or against it, check the relevant facts or calculation, and end "
            f"with VERDICT {letter}: SUPPORT or VERDICT {letter}: REJECT. Do not "
            "compare against another option's analysis and do not use tools."
        )
        option = option_client.complete(
            [
                {
                    "role": "system",
                    "content": (
                        "Act as an independent option evaluator. Focus only on the "
                        "assigned option and produce compact, falsifiable evidence."
                    ),
                },
                {"role": "user", "content": prompt},
            ]
        )
        evidence.append(f"<option_{letter}>\n{option.content}\n</option_{letter}>")
        option_stages[letter] = {
            "max_tokens": manifest.option_evidence_max_tokens,
            "input_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "finish_reason": _finish_reason(option.raw),
            "usage": option.usage,
            "output": option.content,
            "output_sha256": hashlib.sha256(option.content.encode()).hexdigest(),
        }

    arbiter_client = _client_for_budget(
        clients,
        model,
        manifest.verifier_max_tokens,
    )
    arbiter_prompt = (
        f"<original_task>\n{case.prompt}\n</original_task>\n\n"
        f"<protected_direct_candidate>\n{direct.content}"
        "\n</protected_direct_candidate>\n\n"
        + "\n\n".join(evidence)
    )
    arbiter = arbiter_client.complete(
        [
            {
                "role": "system",
                "content": (
                    "Act as a conservative final arbiter. Preserve the protected "
                    "direct candidate by default. Override it only when the option "
                    "evidence identifies a specific contradiction in that candidate "
                    "and a different option has clearly stronger task-grounded "
                    "evidence. Return only one FINAL: <letter> line. Do not explain."
                ),
            },
            {"role": "user", "content": arbiter_prompt},
        ]
    )
    arbiter_usage = dict(arbiter.usage)
    raw_arbiter_content = arbiter.content
    normalized_bare_choice = False
    stripped_arbiter = raw_arbiter_content.strip()
    if (
        manifest.normalize_bare_choice
        and re.fullmatch(r"[A-Da-d]", stripped_arbiter)
    ):
        arbiter.content = f"FINAL: {stripped_arbiter.upper()}"
        normalized_bare_choice = True
    arbiter.usage = _sum_usage(
        direct.usage,
        *(stage["usage"] for stage in option_stages.values()),
        arbiter_usage,
    )
    return arbiter, {
        "protected_direct": {
            "max_tokens": case.max_tokens,
            "input_sha256": hashlib.sha256(case.prompt.encode()).hexdigest(),
            "finish_reason": _finish_reason(direct.raw),
            "usage": direct.usage,
            "output": direct.content,
            "output_sha256": hashlib.sha256(direct.content.encode()).hexdigest(),
        },
        "option_evidence": option_stages,
        "arbiter": {
            "max_tokens": manifest.verifier_max_tokens,
            "input_sha256": hashlib.sha256(arbiter_prompt.encode()).hexdigest(),
            "finish_reason": _finish_reason(arbiter.raw),
            "usage": arbiter_usage,
            "raw_output_sha256": hashlib.sha256(
                raw_arbiter_content.encode()
            ).hexdigest(),
            "normalized_bare_choice": normalized_bare_choice,
        },
    }


def _run_protected_math_arbiter_case(
    case: BaselineCase,
    manifest: SuiteManifest,
    model: ModelConfig,
    clients: dict[int, OpenRouterClient],
) -> tuple[Any, dict[str, Any]]:
    if case.scorer != "numeric_exact":
        raise ValueError("protected_math_arbiter requires a numeric_exact case")

    direct_client = _client_for_budget(clients, model, case.max_tokens)
    direct = direct_client.complete(
        [
            {"role": "system", "content": case.system_prompt},
            {"role": "user", "content": case.prompt},
        ]
    )
    direct_prediction = extract_prediction(direct.content, case.scorer)

    resolve_client = _client_for_budget(
        clients,
        model,
        manifest.second_solve_max_tokens,
    )
    resolve_prompt = (
        f"<original_task>\n{case.draft_prompt}\n</original_task>\n\n"
        "Independently solve this math problem from scratch. Check units, rates, "
        "time periods, totals, and exactly what quantity is requested. Produce "
        "compact calculations and end with FINAL: <number>. Do not use tools."
    )
    resolve = resolve_client.complete(
        [
            {
                "role": "system",
                "content": (
                    "Act as an independent math solver. Do not rely on another "
                    "solution and make every arithmetic dependency explicit."
                ),
            },
            {"role": "user", "content": resolve_prompt},
        ]
    )
    resolve_prediction = extract_prediction(resolve.content, case.scorer)

    arbiter_client = _client_for_budget(
        clients,
        model,
        manifest.verifier_max_tokens,
    )
    arbiter_prompt = (
        f"<original_task>\n{case.prompt}\n</original_task>\n\n"
        f"<protected_direct_answer>{direct_prediction}</protected_direct_answer>\n"
        f"<independent_resolve>\n{resolve.content}\n</independent_resolve>"
    )
    arbiter = arbiter_client.complete(
        [
            {
                "role": "system",
                "content": (
                    "Act as a conservative math arbiter. Preserve the protected "
                    "direct answer by default. Override it only if the independent "
                    "re-solve identifies a specific arithmetic, unit, rate, or "
                    "question-interpretation contradiction and provides a clearly "
                    "verified replacement. Return only one FINAL: <number> line."
                ),
            },
            {"role": "user", "content": arbiter_prompt},
        ]
    )
    arbiter_usage = dict(arbiter.usage)
    raw_arbiter_content = arbiter.content
    fallback_applied = False
    if (
        manifest.fallback_to_protected_on_parse_failure
        and extract_prediction(arbiter.content, case.scorer) is None
        and direct_prediction is not None
    ):
        arbiter.content = f"FINAL: {direct_prediction}"
        fallback_applied = True
    arbiter.usage = _sum_usage(direct.usage, resolve.usage, arbiter_usage)
    return arbiter, {
        "protected_direct": {
            "max_tokens": case.max_tokens,
            "input_sha256": hashlib.sha256(case.prompt.encode()).hexdigest(),
            "finish_reason": _finish_reason(direct.raw),
            "usage": direct.usage,
            "output": direct.content,
            "output_sha256": hashlib.sha256(direct.content.encode()).hexdigest(),
            "prediction": direct_prediction,
        },
        "independent_resolve": {
            "max_tokens": manifest.second_solve_max_tokens,
            "input_sha256": hashlib.sha256(resolve_prompt.encode()).hexdigest(),
            "finish_reason": _finish_reason(resolve.raw),
            "usage": resolve.usage,
            "output": resolve.content,
            "output_sha256": hashlib.sha256(resolve.content.encode()).hexdigest(),
            "prediction": resolve_prediction,
        },
        "arbiter": {
            "max_tokens": manifest.verifier_max_tokens,
            "input_sha256": hashlib.sha256(arbiter_prompt.encode()).hexdigest(),
            "finish_reason": _finish_reason(arbiter.raw),
            "usage": arbiter_usage,
            "raw_output_sha256": hashlib.sha256(
                raw_arbiter_content.encode()
            ).hexdigest(),
            "raw_output": raw_arbiter_content,
            "fallback_to_protected_applied": fallback_applied,
        },
    }


def _run_protected_math_gate_case(
    case: BaselineCase,
    manifest: SuiteManifest,
    model: ModelConfig,
    clients: dict[int, OpenRouterClient],
) -> tuple[Any, dict[str, Any]]:
    if case.scorer != "numeric_exact":
        raise ValueError("protected_math_gate requires a numeric_exact case")

    direct_client = _client_for_budget(clients, model, case.max_tokens)
    direct = direct_client.complete(
        [
            {"role": "system", "content": case.system_prompt},
            {"role": "user", "content": case.prompt},
        ]
    )
    direct_prediction = extract_prediction(direct.content, case.scorer)

    resolve_client = _client_for_budget(
        clients,
        model,
        manifest.second_solve_max_tokens,
    )
    resolve_prompt = (
        f"<original_task>\n{case.draft_prompt}\n</original_task>\n\n"
        "Independently solve this math problem from scratch. Check units, rates, "
        "time periods, totals, and exactly what quantity is requested. Produce "
        "compact calculations and end with FINAL: <number>. Do not use tools."
    )
    resolve = resolve_client.complete(
        [
            {
                "role": "system",
                "content": (
                    "Act as an independent math solver. Do not rely on another "
                    "solution and make every arithmetic dependency explicit."
                ),
            },
            {"role": "user", "content": resolve_prompt},
        ]
    )
    resolve_prediction = extract_prediction(resolve.content, case.scorer)

    gate_client = _client_for_budget(clients, model, manifest.verifier_max_tokens)
    gate_prompt = (
        f"<original_task>\n{case.prompt}\n</original_task>\n\n"
        f"<protected_direct_answer>{direct_prediction}</protected_direct_answer>\n"
        f"<independent_resolve>\n{resolve.content}\n</independent_resolve>"
    )
    gate = gate_client.complete(
        [
            {
                "role": "system",
                "content": (
                    "Decide whether the independent re-solve proves a specific "
                    "arithmetic, unit, rate, or interpretation error in the protected "
                    "direct answer. Return exactly KEEP or USE_RESOLVE. Default to "
                    "KEEP unless the contradiction and replacement are both clear."
                ),
            },
            {"role": "user", "content": gate_prompt},
        ]
    )
    raw_gate = gate.content.strip()
    decision = (
        "USE_RESOLVE"
        if raw_gate == "USE_RESOLVE" and resolve_prediction is not None
        else "KEEP"
    )
    selected_prediction = (
        resolve_prediction if decision == "USE_RESOLVE" else direct_prediction
    )
    gate_usage = dict(gate.usage)
    gate.content = (
        f"FINAL: {selected_prediction}" if selected_prediction is not None else ""
    )
    gate.usage = _sum_usage(direct.usage, resolve.usage, gate_usage)
    return gate, {
        "protected_direct": {
            "max_tokens": case.max_tokens,
            "input_sha256": hashlib.sha256(case.prompt.encode()).hexdigest(),
            "finish_reason": _finish_reason(direct.raw),
            "usage": direct.usage,
            "output": direct.content,
            "output_sha256": hashlib.sha256(direct.content.encode()).hexdigest(),
            "prediction": direct_prediction,
        },
        "independent_resolve": {
            "max_tokens": manifest.second_solve_max_tokens,
            "input_sha256": hashlib.sha256(resolve_prompt.encode()).hexdigest(),
            "finish_reason": _finish_reason(resolve.raw),
            "usage": resolve.usage,
            "output": resolve.content,
            "output_sha256": hashlib.sha256(resolve.content.encode()).hexdigest(),
            "prediction": resolve_prediction,
        },
        "decision_gate": {
            "max_tokens": manifest.verifier_max_tokens,
            "input_sha256": hashlib.sha256(gate_prompt.encode()).hexdigest(),
            "finish_reason": _finish_reason(gate.raw),
            "usage": gate_usage,
            "raw_output": raw_gate,
            "raw_output_sha256": hashlib.sha256(raw_gate.encode()).hexdigest(),
            "decision": decision,
            "selected_prediction": selected_prediction,
        },
    }


def _run_protected_math_majority_case(
    case: BaselineCase,
    manifest: SuiteManifest,
    model: ModelConfig,
    clients: dict[int, OpenRouterClient],
) -> tuple[Any, dict[str, Any]]:
    if case.scorer != "numeric_exact":
        raise ValueError("protected_math_majority requires a numeric_exact case")

    direct_client = _client_for_budget(clients, model, case.max_tokens)
    direct = direct_client.complete(
        [
            {"role": "system", "content": case.system_prompt},
            {"role": "user", "content": case.prompt},
        ]
    )
    direct_usage = dict(direct.usage)
    direct_content = direct.content
    direct_prediction = extract_prediction(direct.content, case.scorer)

    resolve_client = _client_for_budget(
        clients,
        model,
        manifest.second_solve_max_tokens,
    )
    prompts = {
        "resolve_a": (
            f"<original_task>\n{case.draft_prompt}\n</original_task>\n\n"
            "Independently solve this math problem from scratch. Track each "
            "quantity and arithmetic dependency in forward order. Check units, "
            "rates, time periods, and the requested quantity. End with FINAL: "
            "<number>. Do not use tools."
        ),
        "resolve_b": (
            f"<original_task>\n{case.draft_prompt}\n</original_task>\n\n"
            "Independently solve this math problem using a verification-first "
            "approach. Derive the result, then check it by inverse calculation, "
            "estimation, unit analysis, and rereading exactly what is requested. "
            "End with FINAL: <number>. Do not use tools."
        ),
    }
    systems = {
        "resolve_a": (
            "Act as an independent forward math solver. Do not rely on another "
            "solution and make arithmetic dependencies explicit."
        ),
        "resolve_b": (
            "Act as an independent verification-oriented math solver. Use a "
            "different derivation and actively test the result for contradictions."
        ),
    }
    resolves = {}
    stages = {}
    for name in ("resolve_a", "resolve_b"):
        reply = resolve_client.complete(
            [
                {"role": "system", "content": systems[name]},
                {"role": "user", "content": prompts[name]},
            ]
        )
        prediction = extract_prediction(reply.content, case.scorer)
        resolves[name] = (reply, prediction)
        stages[name] = {
            "max_tokens": manifest.second_solve_max_tokens,
            "input_sha256": hashlib.sha256(prompts[name].encode()).hexdigest(),
            "finish_reason": _finish_reason(reply.raw),
            "usage": reply.usage,
            "output": reply.content,
            "output_sha256": hashlib.sha256(reply.content.encode()).hexdigest(),
            "prediction": prediction,
        }

    predictions = [
        direct_prediction,
        resolves["resolve_a"][1],
        resolves["resolve_b"][1],
    ]
    counts: dict[str, int] = {}
    for prediction in predictions:
        if prediction is not None:
            counts[prediction] = counts.get(prediction, 0) + 1
    majority = next(
        (prediction for prediction, count in counts.items() if count >= 2),
        None,
    )
    selected_prediction = majority if majority is not None else direct_prediction
    selection_reason = "numeric_majority" if majority is not None else "no_majority_keep_direct"

    direct.content = (
        f"FINAL: {selected_prediction}" if selected_prediction is not None else ""
    )
    direct.usage = _sum_usage(
        direct_usage,
        resolves["resolve_a"][0].usage,
        resolves["resolve_b"][0].usage,
    )
    return direct, {
        "protected_direct": {
            "max_tokens": case.max_tokens,
            "input_sha256": hashlib.sha256(case.prompt.encode()).hexdigest(),
            "finish_reason": _finish_reason(direct.raw),
            "usage": direct_usage,
            "output": direct_content,
            "output_sha256": hashlib.sha256(
                direct_content.encode()
            ).hexdigest(),
            "prediction": direct_prediction,
        },
        "independent_resolves": stages,
        "deterministic_vote": {
            "predictions": predictions,
            "counts": counts,
            "majority_prediction": majority,
            "selected_prediction": selected_prediction,
            "selection_reason": selection_reason,
        },
    }


def _run_protected_math_recovery_case(
    case: BaselineCase,
    manifest: SuiteManifest,
    model: ModelConfig,
    clients: dict[int, OpenRouterClient],
) -> tuple[Any, dict[str, Any]]:
    if case.scorer != "numeric_exact":
        raise ValueError("protected_math_recovery requires a numeric_exact case")

    direct_client = _client_for_budget(clients, model, case.max_tokens)
    direct = direct_client.complete(
        [
            {"role": "system", "content": case.system_prompt},
            {"role": "user", "content": case.prompt},
        ]
    )
    direct_usage = dict(direct.usage)
    direct_content = direct.content
    direct_prediction = extract_prediction(direct.content, case.scorer)
    recovery_stage = None
    selected_prediction = direct_prediction
    if direct_prediction is None:
        recovery_client = _client_for_budget(
            clients,
            model,
            manifest.second_solve_max_tokens,
        )
        recovery_prompt = (
            f"<original_task>\n{case.draft_prompt}\n</original_task>\n\n"
            "The first attempt did not produce a parseable final number. Solve the "
            "problem independently from scratch. Check units, rates, time periods, "
            "totals, and exactly what quantity is requested. End with exactly one "
            "line FINAL: <number>. Do not use tools."
        )
        recovery = recovery_client.complete(
            [
                {
                    "role": "system",
                    "content": (
                        "Act as a recovery math solver. Produce a compact verified "
                        "derivation and a strict numeric final."
                    ),
                },
                {"role": "user", "content": recovery_prompt},
            ]
        )
        selected_prediction = extract_prediction(recovery.content, case.scorer)
        direct.usage = _sum_usage(direct_usage, recovery.usage)
        recovery_stage = {
            "max_tokens": manifest.second_solve_max_tokens,
            "input_sha256": hashlib.sha256(recovery_prompt.encode()).hexdigest(),
            "finish_reason": _finish_reason(recovery.raw),
            "usage": recovery.usage,
            "output": recovery.content,
            "output_sha256": hashlib.sha256(recovery.content.encode()).hexdigest(),
            "prediction": selected_prediction,
        }
    direct.content = (
        f"FINAL: {selected_prediction}" if selected_prediction is not None else ""
    )
    return direct, {
        "protected_direct": {
            "max_tokens": case.max_tokens,
            "input_sha256": hashlib.sha256(case.prompt.encode()).hexdigest(),
            "finish_reason": _finish_reason(direct.raw),
            "usage": direct_usage,
            "output": direct_content,
            "output_sha256": hashlib.sha256(direct_content.encode()).hexdigest(),
            "prediction": direct_prediction,
        },
        "conditional_recovery": recovery_stage,
        "deterministic_selection": {
            "recovery_triggered": direct_prediction is None,
            "selected_prediction": selected_prediction,
            "selection_reason": (
                "recovery_prediction"
                if direct_prediction is None and selected_prediction is not None
                else "direct_prediction"
                if direct_prediction is not None
                else "recovery_unparseable"
            ),
        },
    }


def _run_protected_math_short_recovery_case(
    case: BaselineCase,
    manifest: SuiteManifest,
    model: ModelConfig,
    clients: dict[int, OpenRouterClient],
) -> tuple[Any, dict[str, Any]]:
    if case.scorer != "numeric_exact":
        raise ValueError("protected_math_short_recovery requires a numeric_exact case")

    direct_client = _client_for_budget(clients, model, case.max_tokens)
    direct = direct_client.complete(
        [
            {"role": "system", "content": case.system_prompt},
            {"role": "user", "content": case.prompt},
        ]
    )
    direct_usage = dict(direct.usage)
    direct_content = direct.content
    direct_prediction = extract_prediction(direct.content, case.scorer)
    recovery_stage = None
    selected_prediction = direct_prediction
    if direct_prediction is None:
        recovery_client = _client_for_budget(
            clients,
            model,
            manifest.second_solve_max_tokens,
        )
        recovery_prompt = (
            f"<original_task>\n{case.draft_prompt}\n</original_task>\n\n"
            "Solve the problem internally from scratch. Return only one line "
            "FINAL: <number>. Do not show reasoning and do not use tools."
        )
        recovery = recovery_client.complete(
            [
                {
                    "role": "system",
                    "content": (
                        "Recover the missing numeric final. Solve internally and "
                        "output only the exact FINAL line."
                    ),
                },
                {"role": "user", "content": recovery_prompt},
            ]
        )
        selected_prediction = extract_prediction(recovery.content, case.scorer)
        direct.usage = _sum_usage(direct_usage, recovery.usage)
        recovery_stage = {
            "max_tokens": manifest.second_solve_max_tokens,
            "input_sha256": hashlib.sha256(recovery_prompt.encode()).hexdigest(),
            "finish_reason": _finish_reason(recovery.raw),
            "usage": recovery.usage,
            "output": recovery.content,
            "output_sha256": hashlib.sha256(recovery.content.encode()).hexdigest(),
            "prediction": selected_prediction,
        }
    direct.content = (
        f"FINAL: {selected_prediction}" if selected_prediction is not None else ""
    )
    return direct, {
        "protected_direct": {
            "max_tokens": case.max_tokens,
            "input_sha256": hashlib.sha256(case.prompt.encode()).hexdigest(),
            "finish_reason": _finish_reason(direct.raw),
            "usage": direct_usage,
            "output": direct_content,
            "output_sha256": hashlib.sha256(direct_content.encode()).hexdigest(),
            "prediction": direct_prediction,
        },
        "conditional_recovery": recovery_stage,
        "deterministic_selection": {
            "recovery_triggered": direct_prediction is None,
            "selected_prediction": selected_prediction,
            "selection_reason": (
                "short_recovery_prediction"
                if direct_prediction is None and selected_prediction is not None
                else "direct_prediction"
                if direct_prediction is not None
                else "short_recovery_unparseable"
            ),
        },
    }


def _run_protected_math_constrained_recovery_case(
    case: BaselineCase,
    manifest: SuiteManifest,
    model: ModelConfig,
    clients: dict[int, OpenRouterClient],
) -> tuple[Any, dict[str, Any]]:
    if case.scorer != "numeric_exact":
        raise ValueError(
            "protected_math_constrained_recovery requires a numeric_exact case"
        )

    direct_client = _client_for_budget(clients, model, case.max_tokens)
    direct = direct_client.complete(
        [
            {"role": "system", "content": case.system_prompt},
            {"role": "user", "content": case.prompt},
        ]
    )
    direct_usage = dict(direct.usage)
    direct_content = direct.content
    direct_prediction = extract_prediction(direct.content, case.scorer)
    recovery_stage = None
    selected_prediction = direct_prediction
    if direct_prediction is None:
        recovery_client = _client_for_budget(
            clients,
            model,
            manifest.second_solve_max_tokens,
        )
        recovery_prompt = (
            f"<original_task>\n{case.draft_prompt}\n</original_task>\n\n"
            "Solve the problem internally from scratch and provide the numeric final."
        )
        structured_outputs = {"regex": NUMERIC_FINAL_REGEX}
        recovery = recovery_client.complete(
            [
                {
                    "role": "system",
                    "content": (
                        "Recover the missing numeric final under the enforced output "
                        "grammar."
                    ),
                },
                {"role": "user", "content": recovery_prompt},
            ],
            extra_body={"structured_outputs": structured_outputs},
        )
        selected_prediction = extract_prediction(recovery.content, case.scorer)
        direct.usage = _sum_usage(direct_usage, recovery.usage)
        recovery_stage = {
            "max_tokens": manifest.second_solve_max_tokens,
            "input_sha256": hashlib.sha256(recovery_prompt.encode()).hexdigest(),
            "finish_reason": _finish_reason(recovery.raw),
            "usage": recovery.usage,
            "output": recovery.content,
            "output_sha256": hashlib.sha256(recovery.content.encode()).hexdigest(),
            "prediction": selected_prediction,
            "structured_outputs": structured_outputs,
        }
    direct.content = (
        f"FINAL: {selected_prediction}" if selected_prediction is not None else ""
    )
    return direct, {
        "protected_direct": {
            "max_tokens": case.max_tokens,
            "input_sha256": hashlib.sha256(case.prompt.encode()).hexdigest(),
            "finish_reason": _finish_reason(direct.raw),
            "usage": direct_usage,
            "output": direct_content,
            "output_sha256": hashlib.sha256(direct_content.encode()).hexdigest(),
            "prediction": direct_prediction,
        },
        "conditional_recovery": recovery_stage,
        "deterministic_selection": {
            "recovery_triggered": direct_prediction is None,
            "selected_prediction": selected_prediction,
            "selection_reason": (
                "constrained_recovery_prediction"
                if direct_prediction is None and selected_prediction is not None
                else "direct_prediction"
                if direct_prediction is not None
                else "constrained_recovery_unparseable"
            ),
        },
    }


def _client_for_budget(
    clients: dict[int, OpenRouterClient],
    model: ModelConfig,
    max_tokens: int,
) -> OpenRouterClient:
    if max_tokens not in clients:
        clients[max_tokens] = OpenRouterClient(replace(model, max_tokens=max_tokens))
    return clients[max_tokens]


def _sum_usage(*usages: dict[str, Any]) -> dict[str, int]:
    output: dict[str, int] = {}
    for usage in usages:
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage.get(key)
            if isinstance(value, int):
                output[key] = output.get(key, 0) + value
    return output


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
