from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nano_harness.baseline import sha256_file
from nano_harness.client import OpenRouterClient
from nano_harness.config import ModelConfig


CONFIG_SCHEMA = "nano_harness_verified_tool_execution_v1"
RESULT_SCHEMA = "nano_harness_verified_tool_execution_result_v1"
FAMILIES = (
    "box_total",
    "remaining_stock",
    "paired_average",
    "labor_total",
)
TOOL_FIELDS = {
    "box_total": ("boxes", "items_per_box", "loose_items"),
    "remaining_stock": (
        "starting_units",
        "batches_used",
        "units_per_batch",
    ),
    "paired_average": ("first_total", "second_total"),
    "labor_total": ("hourly_rate", "regular_hours", "bonus"),
}
DIRECT_REGEX = r"FINAL: -?[0-9]+"
PLAN_REGEX = (
    r"TOOL: (?:"
    r'box_total \{"boxes":-?[0-9]+,"items_per_box":-?[0-9]+,'
    r'"loose_items":-?[0-9]+\}|'
    r'remaining_stock \{"starting_units":-?[0-9]+,"batches_used":'
    r'-?[0-9]+,"units_per_batch":-?[0-9]+\}|'
    r'paired_average \{"first_total":-?[0-9]+,"second_total":'
    r'-?[0-9]+\}|'
    r'labor_total \{"hourly_rate":-?[0-9]+,"regular_hours":'
    r'-?[0-9]+,"bonus":-?[0-9]+\})'
)
FINAL_PATTERN = re.compile(r"^FINAL: (-?[0-9]+)$")
PLAN_PATTERN = re.compile(r"^TOOL: ([a-z_]+) (\{.*\})$")


@dataclass(frozen=True)
class VerifiedToolExecutionConfig:
    schema_version: str
    experiment_id: str
    four_b_model: str
    four_b_base_url: str
    four_b_model_path: str
    four_b_model_config_sha256: str
    four_b_model_index_sha256: str
    four_b_weight_shards: tuple[dict[str, Any], ...]
    nine_b_model: str
    nine_b_base_url: str
    nine_b_model_path: str
    nine_b_model_config_sha256: str
    nine_b_model_index_sha256: str
    nine_b_weight_shards: tuple[dict[str, Any], ...]
    vllm_version: str
    serving_dtype: str
    max_model_len: int
    gpu_memory_utilization: float
    enforce_eager: bool
    max_num_batched_tokens: int
    max_num_seqs: int
    triton_libcuda_path: str
    triton_libcuda_sha256: str
    service_receipt_path: str
    output_path: str
    case_seed: int
    cases_per_family: int
    value_offset: int
    temperature: float
    chat_template_kwargs: dict[str, Any]
    direct_max_tokens: int
    plan_max_tokens: int
    final_max_tokens: int
    plan_retry_limit: int
    direct_structured_output_regex: str
    plan_structured_output_regex: str
    bootstrap_samples: int
    bootstrap_seed: str
    significance_alpha: float
    minimum_harness_wins: int
    maximum_harness_losses: int
    prior_surfaces: tuple[dict[str, str], ...]
    benchmark_sources: tuple[dict[str, str], ...]
    policy: dict[str, bool]


def load_config(path: str | Path) -> VerifiedToolExecutionConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(raw) != set(VerifiedToolExecutionConfig.__dataclass_fields__):
        raise ValueError("verified tool execution config fields differ")
    raw["four_b_weight_shards"] = tuple(raw["four_b_weight_shards"])
    raw["nine_b_weight_shards"] = tuple(raw["nine_b_weight_shards"])
    raw["prior_surfaces"] = tuple(raw["prior_surfaces"])
    raw["benchmark_sources"] = tuple(raw["benchmark_sources"])
    config = VerifiedToolExecutionConfig(**raw)
    validate_config(config)
    return config


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def validate_config(config: VerifiedToolExecutionConfig) -> None:
    if config.schema_version != CONFIG_SCHEMA:
        raise ValueError("unsupported verified tool execution schema")
    expected = {
        "experiment_id": "qwen35-verified-tool-execution-v1",
        "four_b_model": "qwen3.5-4b",
        "four_b_base_url": "http://127.0.0.1:8000/v1",
        "nine_b_model": "qwen3.5-9b",
        "nine_b_base_url": "http://127.0.0.1:8001/v1",
        "vllm_version": "0.19.1",
        "serving_dtype": "float16",
        "max_model_len": 4096,
        "gpu_memory_utilization": 0.85,
        "enforce_eager": True,
        "max_num_batched_tokens": 4096,
        "max_num_seqs": 1,
        "triton_libcuda_path": "/usr/lib/x86_64-linux-gnu",
        "triton_libcuda_sha256": (
            "f2470aa637fa72422534edeaeb9de19afe3d0646388baf7ffb5337b2"
            "edafc59e"
        ),
        "service_receipt_path": (
            "docs/experiments/"
            "qwen35_verified_tool_execution_services_v1.public.json"
        ),
        "output_path": (
            "results/harness/qwen35-verified-tool-execution-v1/result.json"
        ),
        "case_seed": 20260820,
        "cases_per_family": 64,
        "value_offset": 80000,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
        "direct_max_tokens": 32,
        "plan_max_tokens": 96,
        "final_max_tokens": 32,
        "plan_retry_limit": 1,
        "direct_structured_output_regex": DIRECT_REGEX,
        "plan_structured_output_regex": PLAN_REGEX,
        "bootstrap_samples": 10_000,
        "bootstrap_seed": "qwen35-verified-tool-execution-v1",
        "significance_alpha": 0.05,
        "minimum_harness_wins": 12,
        "maximum_harness_losses": 0,
    }
    for field, expected_value in expected.items():
        if getattr(config, field) != expected_value:
            raise ValueError(
                f"verified tool execution freezes {field}={expected_value}"
            )
    if (
        not _is_sha256(config.four_b_model_config_sha256)
        or not _is_sha256(config.four_b_model_index_sha256)
        or not _is_sha256(config.nine_b_model_config_sha256)
        or not _is_sha256(config.nine_b_model_index_sha256)
        or len(config.four_b_weight_shards) != 2
        or len(config.nine_b_weight_shards) != 4
    ):
        raise ValueError("verified tool model identity differs")
    for row in (
        *config.four_b_weight_shards,
        *config.nine_b_weight_shards,
    ):
        if (
            set(row) != {"name", "bytes", "sha256"}
            or int(row["bytes"]) <= 0
            or not _is_sha256(row["sha256"])
        ):
            raise ValueError("verified tool shard identity differs")
    if len(config.prior_surfaces) != 3:
        raise ValueError("verified tool prior surfaces differ")
    if len(config.benchmark_sources) != 3:
        raise ValueError("verified tool benchmark sources differ")
    for source in (*config.prior_surfaces, *config.benchmark_sources):
        if not _is_sha256(source["sha256"]):
            raise ValueError("verified tool source identity differs")
    required_policy = {
        "evaluation_only": True,
        "training_eligible": False,
        "contains_benchmark_rows": False,
        "contains_benchmark_outputs": False,
        "contains_canary_rows": False,
        "contains_holdout_rows": False,
        "uses_observed_quality_outputs": False,
        "executor_uses_expected_answer": False,
        "executor_uses_case_correctness": False,
        "post_observation_prompt_parser_budget_search": False,
    }
    if config.policy != required_policy:
        raise ValueError("verified tool policy differs")


def build_cases(config: VerifiedToolExecutionConfig) -> list[dict[str, Any]]:
    rows = []
    for family in FAMILIES:
        for index in range(config.cases_per_family):
            value = config.value_offset + index
            if family == "box_total":
                facts = {
                    "boxes": value * 7 + 101,
                    "items_per_box": value * 5 + 97,
                    "loose_items": value * 3 + 41,
                }
                prompt = (
                    "A warehouse inventory record states: "
                    f"boxes={facts['boxes']}, "
                    f"items_per_box={facts['items_per_box']}, "
                    f"loose_items={facts['loose_items']}. "
                    "Compute the exact total number of items."
                )
            elif family == "remaining_stock":
                batches_used = value * 3 + 67
                units_per_batch = value * 4 + 89
                remaining = value * 9 + 503
                facts = {
                    "starting_units": (
                        batches_used * units_per_batch + remaining
                    ),
                    "batches_used": batches_used,
                    "units_per_batch": units_per_batch,
                }
                prompt = (
                    "A stock ledger states: "
                    f"starting_units={facts['starting_units']}, "
                    f"batches_used={facts['batches_used']}, "
                    f"units_per_batch={facts['units_per_batch']}. "
                    "Compute the exact number of units remaining."
                )
            elif family == "paired_average":
                first = value * value + 3 * value + 1001
                second = (value + 2) * (value + 4) + 701
                if (first + second) % 2:
                    second += 1
                facts = {
                    "first_total": first,
                    "second_total": second,
                }
                prompt = (
                    "Two audited totals are recorded as "
                    f"first_total={facts['first_total']} and "
                    f"second_total={facts['second_total']}. "
                    "Compute their exact arithmetic mean."
                )
            elif family == "labor_total":
                facts = {
                    "hourly_rate": value * 6 + 113,
                    "regular_hours": value * 2 + 53,
                    "bonus": value * 11 + 307,
                }
                prompt = (
                    "A payroll record states: "
                    f"hourly_rate={facts['hourly_rate']}, "
                    f"regular_hours={facts['regular_hours']}, "
                    f"bonus={facts['bonus']}. "
                    "Compute exact total pay as hourly_rate times "
                    "regular_hours plus bonus."
                )
            else:
                raise ValueError(f"unsupported tool family: {family}")
            expected = execute_verified_tool(family, facts)
            digest = hashlib.sha256(
                f"{family}\0{json.dumps(facts, sort_keys=True)}".encode()
            ).hexdigest()
            rows.append(
                {
                    "case_id": f"verified-tool-{family}-{digest[:16]}",
                    "family": family,
                    "prompt": prompt,
                    "source_facts": facts,
                    "expected": expected,
                }
            )
    rows.sort(
        key=lambda row: hashlib.sha256(
            f"{config.case_seed}\0{row['case_id']}".encode()
        ).hexdigest()
    )
    return rows


def public_case_contract(cases: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [
        {
            "case_id": row["case_id"],
            "family": row["family"],
            "prompt_sha256": hashlib.sha256(
                row["prompt"].encode()
            ).hexdigest(),
            "source_facts_sha256": hashlib.sha256(
                json.dumps(
                    row["source_facts"],
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
        }
        for row in cases
    ]
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return {
        "schema_version": "nano_harness_verified_tool_contract_v1",
        "cases": rows,
        "case_count": len(rows),
        "case_contract_sha256": hashlib.sha256(
            canonical.encode()
        ).hexdigest(),
    }


def contamination_audit(
    config: VerifiedToolExecutionConfig,
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    import pyarrow.parquet as parquet

    normalize = lambda value: " ".join(str(value).casefold().split())
    prompt_hashes = {
        hashlib.sha256(normalize(row["prompt"]).encode()).hexdigest()
        for row in cases
    }
    prior_overlap = {}
    for source in config.prior_surfaces:
        path = Path(source["path"])
        if sha256_file(path) != source["sha256"]:
            raise ValueError(
                f"verified tool prior surface mismatch: {source['name']}"
            )
        document = json.loads(path.read_text(encoding="utf-8"))
        prior_hashes = {
            hashlib.sha256(
                normalize(row["prompt"]).encode()
            ).hexdigest()
            for row in document["cases"]
        }
        prior_overlap[source["name"]] = len(prompt_hashes & prior_hashes)
    benchmark_overlap = {}
    benchmark_rows = {}
    for source in config.benchmark_sources:
        path = Path(source["path"])
        if sha256_file(path) != source["sha256"]:
            raise ValueError(
                f"verified tool benchmark mismatch: {source['name']}"
            )
        values = parquet.read_table(
            path,
            columns=[source["prompt_column"]],
        )[source["prompt_column"]].to_pylist()
        benchmark_rows[source["name"]] = len(values)
        hashes = {
            hashlib.sha256(normalize(value).encode()).hexdigest()
            for value in values
        }
        benchmark_overlap[source["name"]] = len(prompt_hashes & hashes)
    return {
        "prior_surface_prompt_overlap": prior_overlap,
        "benchmark_prompt_overlap": benchmark_overlap,
        "benchmark_rows_hashed": benchmark_rows,
        "benchmark_outputs_loaded": False,
        "canary_or_holdout_loaded": False,
        "passed": (
            not any(prior_overlap.values())
            and not any(benchmark_overlap.values())
        ),
    }


def execute_verified_tool(name: str, arguments: dict[str, int]) -> int:
    if name == "box_total":
        return (
            arguments["boxes"] * arguments["items_per_box"]
            + arguments["loose_items"]
        )
    if name == "remaining_stock":
        return (
            arguments["starting_units"]
            - arguments["batches_used"] * arguments["units_per_batch"]
        )
    if name == "paired_average":
        total = arguments["first_total"] + arguments["second_total"]
        if total % 2:
            raise ValueError("paired average is not integral")
        return total // 2
    if name == "labor_total":
        return (
            arguments["hourly_rate"] * arguments["regular_hours"]
            + arguments["bonus"]
        )
    raise ValueError(f"unsupported verified tool: {name}")


def parse_and_execute_plan(
    text: str,
    *,
    expected_tool: str,
    source_facts: dict[str, int],
) -> dict[str, Any]:
    base = {
        "schema_version": "nano_harness_verified_tool_receipt_v1",
        "eligible": False,
        "executed": False,
        "reason": "",
        "expected_tool": expected_tool,
        "executor_uses_expected_answer": False,
        "executor_uses_case_correctness": False,
    }
    match = PLAN_PATTERN.fullmatch(text.strip())
    if not match:
        return {**base, "reason": "plan_parse_failure"}
    tool_name = match.group(1)
    try:
        arguments = json.loads(match.group(2))
    except json.JSONDecodeError:
        return {**base, "reason": "arguments_json_failure"}
    if tool_name != expected_tool:
        return {
            **base,
            "reason": "tool_name_mismatch",
            "tool_name": tool_name,
        }
    expected_fields = set(TOOL_FIELDS[tool_name])
    if set(arguments) != expected_fields:
        return {
            **base,
            "reason": "argument_fields_mismatch",
            "tool_name": tool_name,
        }
    if (
        any(type(value) is not int for value in arguments.values())
        or arguments != source_facts
    ):
        return {
            **base,
            "reason": "source_facts_mismatch",
            "tool_name": tool_name,
            "arguments_sha256": hashlib.sha256(
                json.dumps(
                    arguments,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
        }
    result = execute_verified_tool(tool_name, arguments)
    return {
        **base,
        "eligible": True,
        "executed": True,
        "reason": "verified_execution",
        "tool_name": tool_name,
        "arguments": arguments,
        "result": result,
    }


def _verify_model(
    path_value: str,
    config_sha256: str,
    index_sha256: str,
    shards: tuple[dict[str, Any], ...],
) -> None:
    path = Path(path_value)
    if (
        sha256_file(path / "config.json") != config_sha256
        or sha256_file(path / "model.safetensors.index.json")
        != index_sha256
    ):
        raise ValueError("verified tool model metadata mismatch")
    for shard in shards:
        shard_path = path / shard["name"]
        if (
            shard_path.stat().st_size != shard["bytes"]
            or sha256_file(shard_path) != shard["sha256"]
        ):
            raise ValueError("verified tool model shard mismatch")


def verify_inputs(config: VerifiedToolExecutionConfig) -> dict[str, Any]:
    _verify_model(
        config.four_b_model_path,
        config.four_b_model_config_sha256,
        config.four_b_model_index_sha256,
        config.four_b_weight_shards,
    )
    _verify_model(
        config.nine_b_model_path,
        config.nine_b_model_config_sha256,
        config.nine_b_model_index_sha256,
        config.nine_b_weight_shards,
    )
    receipt = json.loads(
        Path(config.service_receipt_path).read_text(encoding="utf-8")
    )
    if (
        receipt.get("schema_version")
        != "nano_harness_verified_tool_services_v1"
        or receipt.get("generation_started") is not False
        or receipt.get("models", {}).get("qwen3.5-4b", {}).get(
            "served_model"
        )
        != config.four_b_model
        or receipt.get("models", {}).get("qwen3.5-9b", {}).get(
            "served_model"
        )
        != config.nine_b_model
        or receipt.get("serving", {}).get("vllm_version") != "0.19.1"
        or receipt.get("serving", {}).get("dtype") != config.serving_dtype
        or receipt.get("serving", {}).get("max_model_len")
        != config.max_model_len
        or receipt.get("serving", {}).get("gpu_memory_utilization")
        != config.gpu_memory_utilization
        or receipt.get("serving", {}).get("enforce_eager")
        != config.enforce_eager
        or receipt.get("serving", {}).get("max_num_batched_tokens")
        != config.max_num_batched_tokens
        or receipt.get("serving", {}).get("max_num_seqs")
        != config.max_num_seqs
        or receipt.get("serving", {}).get("triton_libcuda_sha256")
        != config.triton_libcuda_sha256
    ):
        raise ValueError("verified tool service receipt differs")
    return receipt


def _client(
    config: VerifiedToolExecutionConfig,
    *,
    four_b: bool,
    max_tokens: int,
) -> OpenRouterClient:
    return OpenRouterClient(
        ModelConfig(
            name=config.four_b_model if four_b else config.nine_b_model,
            base_url=(
                config.four_b_base_url
                if four_b
                else config.nine_b_base_url
            ),
            api_key_env="NANO_HARNESS_API_KEY",
            temperature=config.temperature,
            max_tokens=max_tokens,
            timeout_seconds=180.0,
            max_retries=3,
            chat_template_kwargs=config.chat_template_kwargs,
        )
    )


def _direct_row(
    case: dict[str, Any],
    client: Any,
    config: VerifiedToolExecutionConfig,
    *,
    model: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    reply = client.complete(
        [
            {
                "role": "system",
                "content": (
                    "Solve the arithmetic task exactly. Return only one line "
                    "in the form FINAL: <integer>."
                ),
            },
            {"role": "user", "content": case["prompt"]},
        ],
        extra_body={
            "structured_outputs": {
                "regex": config.direct_structured_output_regex
            }
        },
    )
    output = reply.content.strip()
    match = FINAL_PATTERN.fullmatch(output)
    prediction = int(match.group(1)) if match else None
    return {
        "case_id": case["case_id"],
        "family": case["family"],
        "model": model,
        "route": "direct",
        "output": output,
        "prediction": prediction,
        "parseable": prediction is not None,
        "correct": prediction == case["expected"],
        "usage": reply.usage,
        "latency_seconds": time.perf_counter() - started,
    }


def _sum_usage(*usages: dict[str, Any]) -> dict[str, int]:
    output: dict[str, int] = {}
    for usage in usages:
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage.get(key)
            if isinstance(value, int):
                output[key] = output.get(key, 0) + value
    return output


def _harness_row(
    case: dict[str, Any],
    direct: dict[str, Any],
    plan_client: Any,
    final_client: Any,
    config: VerifiedToolExecutionConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.perf_counter()
    plan_messages = [
        {
            "role": "system",
            "content": (
                "Select the one typed arithmetic tool matching the task and "
                "copy every labeled source fact exactly. Return only the TOOL "
                "line required by the structured contract. Do not calculate."
            ),
        },
        {"role": "user", "content": case["prompt"]},
    ]
    plan_attempts = []
    receipt = None
    plan_usage: list[dict[str, Any]] = []
    for attempt in range(config.plan_retry_limit + 1):
        reply = plan_client.complete(
            plan_messages,
            extra_body={
                "structured_outputs": {
                    "regex": config.plan_structured_output_regex
                }
            },
        )
        plan_usage.append(reply.usage)
        receipt = parse_and_execute_plan(
            reply.content,
            expected_tool=case["family"],
            source_facts=case["source_facts"],
        )
        plan_attempts.append(
            {
                "attempt": attempt + 1,
                "output": reply.content,
                "output_sha256": hashlib.sha256(
                    reply.content.encode()
                ).hexdigest(),
                "reason": receipt["reason"],
                "executed": receipt["executed"],
            }
        )
        if receipt["executed"]:
            break
        if attempt < config.plan_retry_limit:
            plan_messages.append(
                {"role": "assistant", "content": reply.content}
            )
            plan_messages.append(
                {
                    "role": "user",
                    "content": (
                        "The plan was rejected by the strict source-fact "
                        f"validator: {receipt['reason']}. Retry from the "
                        "original labeled facts and return only one TOOL line."
                    ),
                }
            )
    assert receipt is not None
    if not receipt["executed"]:
        return (
            {
                **direct,
                "model": f"{config.four_b_model}+verified-tool-v1",
                "route": "direct_fallback_after_invalid_plan",
                "usage": _sum_usage(direct["usage"], *plan_usage),
                "latency_seconds": time.perf_counter() - started,
            },
            {
                "plan_attempts": plan_attempts,
                "receipt": receipt,
                "final_feedback_sent": False,
                "fallback_used": True,
            },
        )
    feedback = (
        f"<original_task>\n{case['prompt']}\n</original_task>\n\n"
        f"<verified_tool>\nname={receipt['tool_name']}\n"
        f"arguments={json.dumps(receipt['arguments'], sort_keys=True)}\n"
        f"result={receipt['result']}\n</verified_tool>\n\n"
        "Use the verified result. Return only FINAL: <integer>."
    )
    final = final_client.complete(
        [
            {
                "role": "system",
                "content": (
                    "Use the verified tool result as authoritative and return "
                    "only one FINAL: <integer> line."
                ),
            },
            {"role": "user", "content": feedback},
        ],
        extra_body={
            "structured_outputs": {
                "regex": config.direct_structured_output_regex
            }
        },
    )
    output = final.content.strip()
    match = FINAL_PATTERN.fullmatch(output)
    prediction = int(match.group(1)) if match else None
    return (
        {
            "case_id": case["case_id"],
            "family": case["family"],
            "model": f"{config.four_b_model}+verified-tool-v1",
            "route": "verified_tool_feedback",
            "output": output,
            "prediction": prediction,
            "parseable": prediction is not None,
            "correct": prediction == case["expected"],
            "usage": _sum_usage(*plan_usage, final.usage),
            "latency_seconds": time.perf_counter() - started,
        },
        {
            "plan_attempts": plan_attempts,
            "receipt": receipt,
            "final_feedback_sent": True,
            "final_feedback_sha256": hashlib.sha256(
                feedback.encode()
            ).hexdigest(),
            "fallback_used": False,
        },
    )


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_family = {}
    for family in FAMILIES:
        selected = [row for row in rows if row["family"] == family]
        by_family[family] = {
            "cases": len(selected),
            "correct": sum(row["correct"] for row in selected),
            "parseable": sum(row["parseable"] for row in selected),
        }
    return {
        "cases": len(rows),
        "correct": sum(row["correct"] for row in rows),
        "accuracy": sum(row["correct"] for row in rows) / len(rows),
        "parseable": sum(row["parseable"] for row in rows),
        "by_family": by_family,
    }


def run(config: VerifiedToolExecutionConfig) -> dict[str, Any]:
    receipt = verify_inputs(config)
    cases = build_cases(config)
    four_direct_client = _client(
        config,
        four_b=True,
        max_tokens=config.direct_max_tokens,
    )
    nine_direct_client = _client(
        config,
        four_b=False,
        max_tokens=config.direct_max_tokens,
    )
    plan_client = _client(
        config,
        four_b=True,
        max_tokens=config.plan_max_tokens,
    )
    final_client = _client(
        config,
        four_b=True,
        max_tokens=config.final_max_tokens,
    )
    four_rows = []
    nine_rows = []
    harness_rows = []
    harness_receipts = {}
    for case in cases:
        four = _direct_row(
            case,
            four_direct_client,
            config,
            model=config.four_b_model,
        )
        nine = _direct_row(
            case,
            nine_direct_client,
            config,
            model=config.nine_b_model,
        )
        harness, harness_receipt = _harness_row(
            case,
            four,
            plan_client,
            final_client,
            config,
        )
        four_rows.append(four)
        nine_rows.append(nine)
        harness_rows.append(harness)
        harness_receipts[case["case_id"]] = harness_receipt
    result = {
        "schema_version": RESULT_SCHEMA,
        "experiment_id": config.experiment_id,
        "config": config.__dict__,
        "identity": {
            "service_receipt_sha256": sha256_file(
                Path(config.service_receipt_path)
            ),
            "case_contract": public_case_contract(cases),
        },
        "arms": {
            "four_b_direct": summarize_rows(four_rows),
            "nine_b_direct": summarize_rows(nine_rows),
            "four_b_verified_tool": summarize_rows(harness_rows),
        },
        "four_b_rows": four_rows,
        "nine_b_rows": nine_rows,
        "harness_rows": harness_rows,
        "harness_receipts": harness_receipts,
        "routing": {
            "verified_executions": sum(
                row["receipt"]["executed"]
                for row in harness_receipts.values()
            ),
            "plan_retries": sum(
                len(row["plan_attempts"]) - 1
                for row in harness_receipts.values()
            ),
            "fallbacks": sum(
                row["fallback_used"] for row in harness_receipts.values()
            ),
            "final_feedback_calls": sum(
                row["final_feedback_sent"]
                for row in harness_receipts.values()
            ),
        },
        "service_receipt": receipt,
        "evaluation_boundary": {
            "training_eligible_cases": 0,
            "executor_uses_expected_answer": False,
            "executor_uses_case_correctness": False,
            "benchmark_rows_loaded": False,
            "benchmark_outputs_loaded": False,
            "canary_rows_loaded": False,
            "independent_holdout_rows_loaded": False,
        },
    }
    output = Path(config.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result
