from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from nano_harness.baseline import sha256_file
from nano_harness.semantic_model_router import (
    ALL_FAMILIES,
    NEGATIVE_FAMILIES,
    POSITIVE_FAMILIES,
    _model_selected_tool_row,
    summarize_rows,
)
from nano_harness.semantic_skill_execution import (
    execute_semantic_tool,
    load_config as load_mechanism_config,
    parent_config as load_parent_runtime,
)
from nano_harness.verified_tool_execution import (
    _client,
    _direct_row,
    _sum_usage,
    public_case_contract,
    verify_inputs,
)


CONFIG_SCHEMA = "nano_harness_router_adapter_integration_v1"
RESULT_SCHEMA = "nano_harness_router_adapter_integration_result_v1"
ROUTE_PATTERN = re.compile(r"^FINAL: ([A-C])$")
LABEL_TO_ROUTE = {
    "A": "implicit_scale_total",
    "B": "first_strict_profit_period",
    "C": "NONE",
}
ROUTER_SYSTEM = (
    "Classify the task for a semantic tool router. Return exactly one line: "
    "FINAL: A for implicit rectangular scale totals, FINAL: B for first "
    "strictly profitable whole periods, or FINAL: C for every unsupported task."
)


@dataclass(frozen=True)
class RouterAdapterIntegrationConfig:
    schema_version: str
    experiment_id: str
    mechanism_config_path: str
    mechanism_config_sha256: str
    sft_report_path: str
    sft_report_sha256: str
    router_training_data_path: str
    router_training_data_sha256: str
    prior_router_config_path: str
    prior_router_config_sha256: str
    prior_binary_config_path: str
    prior_binary_config_sha256: str
    adapter_path: str
    adapter_tree_sha256: str
    adapter_config_sha256: str
    adapter_model_sha256: str
    served_adapter_name: str
    service_launch: dict[str, Any]
    service_receipt_path: str
    output_path: str
    case_seed: int
    cases_per_family: int
    route_structured_output_regex: str
    route_max_tokens: int
    direct_max_tokens: int
    plan_max_tokens: int
    final_max_tokens: int
    plan_retry_limit: int
    bootstrap_samples: int
    bootstrap_seed: str
    significance_alpha: float
    minimum_harness_wins: int
    maximum_harness_losses: int
    policy: dict[str, bool]
    execution_boundary: dict[str, bool]


def _sha256_tree(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(path.rglob("*")):
        if not item.is_file():
            continue
        relative = item.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(item).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def load_config(path: str | Path) -> RouterAdapterIntegrationConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(raw) != set(RouterAdapterIntegrationConfig.__dataclass_fields__):
        raise ValueError("router adapter integration config fields differ")
    config = RouterAdapterIntegrationConfig(**raw)
    if config.schema_version != CONFIG_SCHEMA:
        raise ValueError("unsupported router adapter integration schema")
    frozen = {
        "experiment_id": "qwen35-router-adapter-integration-v1",
        "mechanism_config_path": (
            "configs/harness/qwen35_semantic_skill_execution_v1.json"
        ),
        "mechanism_config_sha256": "4e5785b9b4fc9dd5b3a76a95b438c2f2c3d505a2fae6231b81cfbf53ebbd6ad9",
        "sft_report_path": (
            "../nano-train-fullstack-traex-03/docs/results/"
            "qwen35_router_classification_sft_v1.public.json"
        ),
        "sft_report_sha256": "c8af17cfa2fb77b594a9b34deaeccf27273491da6c350c3c5deb1435a9336c69",
        "router_training_data_path": (
            "../nano-data-pipeline-fullstack-traex-03/datasets/"
            "qwen35_router_classification_v1.json"
        ),
        "router_training_data_sha256": (
            "dacd3663639fe9ddc054865b87afdd0c918f0fddb12c8c9355819d4bbce95d65"
        ),
        "prior_router_config_path": (
            "configs/harness/qwen35_semantic_model_router_v1.json"
        ),
        "prior_router_config_sha256": (
            "8c6f0215cb2a0f805e04fe6c00a28fc3b5847d0c2a14575a90b3ed2a6586ebce"
        ),
        "prior_binary_config_path": (
            "configs/harness/qwen35_semantic_binary_detectors_v1.json"
        ),
        "prior_binary_config_sha256": (
            "32c1d877a1cecdb9041fc88226fa2e52890390712f449a8552be0a1202d88748"
        ),
        "adapter_path": (
            "../nano-train-fullstack-traex-03/artifacts/"
            "qwen35-router-classification-sft-smoke-v1/adapter"
        ),
        "adapter_tree_sha256": "48d72666cf269f64825791801c71aad5ae1d9e97cbc70574c216b12974e46e63",
        "adapter_config_sha256": "21be3fad7bb20d9c475ac6c0f317813c2160b3419433b40ca2d3d6c387ea3f49",
        "adapter_model_sha256": "8dfe3a89bf20325a50b9a4c168c4e9039c5f7e84721dd8d5f8b21e3ad829b9ec",
        "served_adapter_name": "qwen3.5-router-v1",
        "service_launch": {
            "gpu_index": 0,
            "host": "127.0.0.1",
            "port": 8000,
            "base_url": "http://127.0.0.1:8000/v1",
            "served_base_model": "qwen3.5-4b",
            "vllm_version": "0.19.1",
            "dtype": "float16",
            "max_model_len": 4096,
            "gpu_memory_utilization": 0.85,
            "enforce_eager": True,
            "max_num_batched_tokens": 4096,
            "max_num_seqs": 1,
            "enable_lora": True,
            "max_loras": 1,
            "max_lora_rank": 8,
            "triton_libcuda_path": "/usr/lib/x86_64-linux-gnu",
        },
        "service_receipt_path": (
            "docs/experiments/qwen35_router_adapter_service_v1.public.json"
        ),
        "output_path": (
            "results/harness/qwen35-router-adapter-integration-v1/result.json"
        ),
        "case_seed": 20260825,
        "cases_per_family": 32,
        "route_structured_output_regex": r"FINAL: [A-C]",
        "route_max_tokens": 8,
        "direct_max_tokens": 32,
        "plan_max_tokens": 96,
        "final_max_tokens": 32,
        "plan_retry_limit": 1,
        "bootstrap_samples": 10_000,
        "bootstrap_seed": "qwen35-router-adapter-integration-v1",
        "significance_alpha": 0.05,
        "minimum_harness_wins": 12,
        "maximum_harness_losses": 0,
        "policy": {
            "evaluation_only": True,
            "training_eligible": False,
            "contains_benchmark_rows": False,
            "contains_benchmark_outputs": False,
            "contains_canary_rows": False,
            "contains_holdout_rows": False,
            "router_uses_case_metadata": False,
            "router_uses_expected_answer": False,
            "router_uses_case_correctness": False,
            "executor_uses_expected_answer": False,
            "executor_uses_case_correctness": False,
            "post_observation_prompt_parser_budget_search": False,
        },
        "execution_boundary": {
            "adapter_service_started": False,
            "model_generation_started": False,
            "evaluation_started": False,
            "benchmark_accessed": False,
            "canary_accessed": False,
            "holdout_accessed": False,
            "training_started": False,
            "this_commit_only_preregisters": True,
        },
    }
    for field, expected in frozen.items():
        if getattr(config, field) != expected:
            raise ValueError(
                f"router adapter integration freezes {field}={expected}"
            )
    for source, digest in (
        (config.mechanism_config_path, config.mechanism_config_sha256),
        (config.sft_report_path, config.sft_report_sha256),
        (
            config.router_training_data_path,
            config.router_training_data_sha256,
        ),
        (config.prior_router_config_path, config.prior_router_config_sha256),
        (config.prior_binary_config_path, config.prior_binary_config_sha256),
    ):
        if sha256_file(Path(source)) != digest:
            raise ValueError("router adapter source evidence differs")
    adapter = Path(config.adapter_path)
    if (
        _sha256_tree(adapter) != config.adapter_tree_sha256
        or sha256_file(adapter / "adapter_config.json")
        != config.adapter_config_sha256
        or sha256_file(adapter / "adapter_model.safetensors")
        != config.adapter_model_sha256
    ):
        raise ValueError("router adapter identity differs")
    sft = json.loads(Path(config.sft_report_path).read_text(encoding="utf-8"))
    if (
        sft.get("decision", {}).get("router_sft_smoke_admitted") is not True
        or sft.get("decision", {}).get(
            "fresh_router_integration_preregistration_allowed"
        )
        is not True
        or sft.get("identity", {}).get("adapter_sha256")
        != config.adapter_tree_sha256
    ):
        raise ValueError("router adapter SFT decision differs")
    return config


def build_cases(config: RouterAdapterIntegrationConfig) -> list[dict[str, Any]]:
    rows = []
    for family in ALL_FAMILIES:
        for index in range(config.cases_per_family):
            value = 11_000 + index
            if family == "implicit_scale_total":
                scale_word = "double" if index % 2 == 0 else "triple"
                facts: dict[str, Any] = {
                    "rows": value * 3 + 17,
                    "columns": value * 2 + 19,
                    "extra": value * 5 + 23,
                    "scale_word": scale_word,
                }
                scale = "twofold" if scale_word == "double" else "threefold"
                prompt = (
                    f"A hall records {facts['rows']} bands with "
                    f"{facts['columns']} places per band and "
                    f"{facts['extra']} reserve places. The request is the "
                    f"{scale} rectangular capacity plus the reserve. Choose "
                    "the semantic route."
                )
                expected = execute_semantic_tool(family, facts)
                label = "A"
            elif family == "first_strict_profit_period":
                units = value * 2 + 11
                price = value * 3 + 13
                net = value * 4 + 31
                recurring = units * price - net
                threshold = value * 2 + 29
                facts = {
                    "setup_cost": net * threshold,
                    "units_per_period": units,
                    "price_per_unit": price,
                    "recurring_cost": recurring,
                }
                prompt = (
                    f"A program pays setup_cost={facts['setup_cost']}, sells "
                    f"units_per_period={facts['units_per_period']} at "
                    f"price_per_unit={facts['price_per_unit']}, and pays "
                    f"recurring_cost={facts['recurring_cost']}. Choose the "
                    "route for its earliest strictly profitable whole cycle."
                )
                expected = execute_semantic_tool(family, facts)
                label = "B"
            elif family == "box_total":
                facts = {
                    "boxes": value * 2 + 3,
                    "items_per_box": value * 3 + 5,
                    "loose_items": value * 7 + 11,
                }
                expected = (
                    facts["boxes"] * facts["items_per_box"]
                    + facts["loose_items"]
                )
                prompt = (
                    f"A depot has boxes={facts['boxes']}, items_per_box="
                    f"{facts['items_per_box']}, and loose_items="
                    f"{facts['loose_items']}. Choose the router class for "
                    "finding the exact inventory total."
                )
                label = "C"
            else:
                batches = value * 2 + 7
                units = value * 3 + 13
                remaining = value * 5 + 17
                facts = {
                    "starting_units": batches * units + remaining,
                    "batches_used": batches,
                    "units_per_batch": units,
                }
                expected = remaining
                prompt = (
                    f"A ledger has starting_units={facts['starting_units']}, "
                    f"batches_used={facts['batches_used']}, and "
                    f"units_per_batch={facts['units_per_batch']}. Choose the "
                    "router class for remaining stock."
                )
                label = "C"
            digest = hashlib.sha256(
                f"{family}\0{json.dumps(facts, sort_keys=True)}".encode()
            ).hexdigest()
            rows.append(
                {
                    "case_id": f"router-adapter-{family}-{digest[:16]}",
                    "family": family,
                    "prompt": prompt,
                    "source_facts": facts,
                    "expected": expected,
                    "expected_label": label,
                    "expected_route": LABEL_TO_ROUTE[label],
                    "positive": family in POSITIVE_FAMILIES,
                }
            )
    rows.sort(
        key=lambda row: hashlib.sha256(
            f"{config.case_seed}\0{row['case_id']}".encode()
        ).hexdigest()
    )
    return rows


def parent_config(config: RouterAdapterIntegrationConfig):
    mechanism = load_mechanism_config(config.mechanism_config_path)
    parent = load_parent_runtime(mechanism)
    return mechanism, parent


def parse_route(text: str) -> str | None:
    match = ROUTE_PATTERN.fullmatch(text.strip())
    return match.group(1) if match else None


def _candidate_row(
    case: dict[str, Any],
    direct: dict[str, Any],
    router: Any,
    planner: Any,
    final: Any,
    config: RouterAdapterIntegrationConfig,
    mechanism: Any,
    parent: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.perf_counter()
    reply = router.complete(
        [
            {"role": "system", "content": ROUTER_SYSTEM},
            {"role": "user", "content": case["prompt"]},
        ],
        extra_body={
            "structured_outputs": {
                "regex": config.route_structured_output_regex
            }
        },
    )
    label = parse_route(reply.content)
    selected = LABEL_TO_ROUTE.get(label) if label is not None else None
    router_receipt = {
        "output": reply.content,
        "output_sha256": hashlib.sha256(reply.content.encode()).hexdigest(),
        "label": label,
        "selected_route": selected,
        "expected_label": case["expected_label"],
        "expected_route": case["expected_route"],
        "correct": selected == case["expected_route"],
        "model": config.served_adapter_name,
        "adapter_sha256": config.adapter_tree_sha256,
        "uses_case_metadata": False,
        "uses_expected_answer": False,
        "uses_case_correctness": False,
    }
    if selected not in POSITIVE_FAMILIES:
        return (
            {
                **direct,
                "model": f"{parent.four_b_model}+router-adapter-v1",
                "route": "direct_preserve_after_router_c",
                "usage": _sum_usage(direct["usage"], reply.usage),
                "latency_seconds": time.perf_counter() - started,
            },
            {
                "router": router_receipt,
                "route": None,
                "exposed_tools": [],
                "plan_attempts": [],
                "receipt": None,
                "final_feedback_sent": False,
                "fallback_used": False,
            },
        )
    routed_case = {**case, "family": selected}
    candidate, tool_receipt = _model_selected_tool_row(
        routed_case,
        direct,
        selected,
        planner,
        final,
        mechanism,
        parent,
    )
    candidate["family"] = case["family"]
    candidate["model"] = f"{parent.four_b_model}+router-adapter-v1"
    candidate["usage"] = _sum_usage(reply.usage, candidate["usage"])
    candidate["latency_seconds"] = time.perf_counter() - started
    return candidate, {"router": router_receipt, **tool_receipt}


def run(config: RouterAdapterIntegrationConfig) -> dict[str, Any]:
    mechanism, parent = parent_config(config)
    service = verify_inputs(parent)
    adapter_receipt = json.loads(
        Path(config.service_receipt_path).read_text(encoding="utf-8")
    )
    if (
        adapter_receipt.get("schema_version")
        != "nano_harness_router_adapter_service_v1"
        or adapter_receipt.get("adapter_sha256") != config.adapter_tree_sha256
        or adapter_receipt.get("served_adapter_name")
        != config.served_adapter_name
        or adapter_receipt.get("healthy") is not True
    ):
        raise ValueError("router adapter service receipt differs")
    cases = build_cases(config)
    four = _client(parent, four_b=True, max_tokens=config.direct_max_tokens)
    nine = _client(parent, four_b=False, max_tokens=config.direct_max_tokens)
    router = _client(parent, four_b=True, max_tokens=config.route_max_tokens)
    router.config = replace(
        router.config,
        name=config.served_adapter_name,
        max_tokens=config.route_max_tokens,
    )
    planner = _client(parent, four_b=True, max_tokens=config.plan_max_tokens)
    final = _client(parent, four_b=True, max_tokens=config.final_max_tokens)
    four_rows = []
    nine_rows = []
    candidate_rows = []
    receipts = {}
    for case in cases:
        direct = _direct_row(case, four, parent, model=parent.four_b_model)
        baseline_nine = _direct_row(
            case, nine, parent, model=parent.nine_b_model
        )
        candidate, receipt = _candidate_row(
            case,
            direct,
            router,
            planner,
            final,
            config,
            mechanism,
            parent,
        )
        four_rows.append(direct)
        nine_rows.append(baseline_nine)
        candidate_rows.append(candidate)
        receipts[case["case_id"]] = receipt
    result = {
        "schema_version": RESULT_SCHEMA,
        "experiment_id": config.experiment_id,
        "config": config.__dict__,
        "identity": {
            "mechanism_config_sha256": config.mechanism_config_sha256,
            "sft_report_sha256": config.sft_report_sha256,
            "router_training_data_sha256": (
                config.router_training_data_sha256
            ),
            "adapter_sha256": config.adapter_tree_sha256,
            "case_contract": public_case_contract(cases),
        },
        "arms": {
            "four_b_direct": summarize_rows(four_rows),
            "nine_b_direct": summarize_rows(nine_rows),
            "four_b_router_adapter": summarize_rows(candidate_rows),
        },
        "four_b_rows": four_rows,
        "nine_b_rows": nine_rows,
        "candidate_rows": candidate_rows,
        "receipts": receipts,
        "routing": {
            "cases": len(cases),
            "correct": sum(
                receipt["router"]["correct"] for receipt in receipts.values()
            ),
            "positive_cases": sum(case["positive"] for case in cases),
            "positive_correct": sum(
                case["positive"]
                and receipts[case["case_id"]]["router"]["correct"]
                for case in cases
            ),
            "negative_cases": sum(not case["positive"] for case in cases),
            "negative_c_correct": sum(
                not case["positive"]
                and receipts[case["case_id"]]["router"]["label"] == "C"
                for case in cases
            ),
            "negative_false_positive_routes": sum(
                not case["positive"]
                and receipts[case["case_id"]]["router"]["label"] != "C"
                for case in cases
            ),
            "verified_executions": sum(
                bool(receipt["receipt"] and receipt["receipt"]["executed"])
                for receipt in receipts.values()
            ),
            "fallbacks": sum(
                receipt["fallback_used"] for receipt in receipts.values()
            ),
        },
        "base_service_receipt": service,
        "adapter_service_receipt": adapter_receipt,
        "evaluation_boundary": {
            "training_eligible_cases": 0,
            "router_uses_case_metadata": False,
            "router_uses_expected_answer": False,
            "router_uses_case_correctness": False,
            "executor_uses_expected_answer": False,
            "executor_uses_case_correctness": False,
            "benchmark_rows_loaded": False,
            "canary_rows_loaded": False,
            "holdout_rows_loaded": False,
        },
    }
    output = Path(config.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result
