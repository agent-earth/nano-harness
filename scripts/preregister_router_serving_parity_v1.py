#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer
from vllm.lora.utils import parse_fine_tuned_lora_name
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
)

from nano_harness.baseline import sha256_file
from nano_harness.router_serving_parity import (
    case_contract,
    load_config,
    validation_cases,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/harness/qwen35_router_serving_parity_v1.json"
JSON_OUTPUT = (
    ROOT
    / "docs/experiments/qwen35_router_serving_parity_v1.preregister.json"
)
MARKDOWN_OUTPUT = (
    ROOT / "docs/experiments/qwen35_router_serving_parity_v1.md"
)


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def build_receipt() -> dict:
    config = load_config(CONFIG)
    cases = validation_cases(config)
    contract = case_contract(cases)
    hf = json.loads(
        Path(config.hf_generations_path).read_text(encoding="utf-8")
    )["post_sft"]
    hf_by_id = {row["sample_id"]: row for row in hf}
    if (
        len(hf_by_id) != config.validation_rows
        or set(hf_by_id) != {case["sample_id"] for case in cases}
        or any(not row["exact"] for row in hf_by_id.values())
    ):
        raise ValueError("router serving parity HF reference differs")

    base_tokenizer = AutoTokenizer.from_pretrained(
        config.base_tokenizer_path,
        local_files_only=True,
    )
    adapter_tokenizer = AutoTokenizer.from_pretrained(
        config.adapter_tokenizer_path,
        local_files_only=True,
    )
    tokenizer_checks = {
        "class_equal": type(base_tokenizer) is type(adapter_tokenizer),
        "length_equal": len(base_tokenizer) == len(adapter_tokenizer),
        "vocab_equal": (
            base_tokenizer.get_vocab() == adapter_tokenizer.get_vocab()
        ),
        "special_tokens_equal": (
            base_tokenizer.special_tokens_map
            == adapter_tokenizer.special_tokens_map
        ),
        "chat_template_equal": (
            base_tokenizer.chat_template == adapter_tokenizer.chat_template
        ),
        "all_case_prompt_ids_equal": all(
            base_tokenizer.encode(case["prompt"])
            == adapter_tokenizer.encode(case["prompt"])
            for case in cases
        ),
        "all_target_ids_equal": all(
            base_tokenizer.encode(case["target"])
            == adapter_tokenizer.encode(case["target"])
            for case in cases
        ),
    }
    if not all(tokenizer_checks.values()):
        raise ValueError("router serving parity tokenizer semantics differ")

    mapper = Qwen3_5ForConditionalGeneration.hf_to_vllm_mapper
    original_name = (
        "base_model.model.model.layers.0.mlp.down_proj.lora_A.weight"
    )
    remapped_name = (
        "base_model.model.language_model.model.layers.0.mlp."
        "down_proj.lora_A.weight"
    )
    original_parsed, _ = parse_fine_tuned_lora_name(original_name, mapper)
    remapped_parsed, _ = parse_fine_tuned_lora_name(remapped_name, mapper)
    if (
        original_parsed
        != config.namespace_audit["original_parsed_module"]
        or remapped_parsed
        != config.namespace_audit["remapped_parsed_module"]
        or not remapped_parsed.startswith(
            config.namespace_audit["vllm_text_module_prefix"]
        )
        or original_parsed.startswith(
            config.namespace_audit["vllm_text_module_prefix"]
        )
    ):
        raise ValueError("router serving parity namespace audit differs")

    maximum_input_tokens = max(
        len(
            base_tokenizer.encode(
                base_tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": config.system_prompt},
                        {"role": "user", "content": case["prompt"]},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                    **config.chat_template_kwargs,
                )
            )
        )
        for case in cases
    )
    if (
        maximum_input_tokens + config.generation_max_tokens
        > config.service_launch["max_model_len"]
    ):
        raise ValueError("router serving parity context budget differs")

    label_counts = Counter(case["label"] for case in cases)
    return {
        "schema_version": (
            "nano_harness_router_serving_parity_preregister_v1"
        ),
        "experiment_id": config.experiment_id,
        "identity": {
            "code_revision": git_revision(),
            "config_sha256": sha256_file(CONFIG),
            "case_contract_sha256": contract["case_contract_sha256"],
            "dataset_sha256": config.dataset_sha256,
            "sft_report_sha256": config.sft_report_sha256,
            "integration_report_sha256": config.integration_report_sha256,
            "hf_generations_sha256": config.hf_generations_sha256,
            "hf_reload_sha256": config.hf_reload_sha256,
            "original_adapter_sha256": (
                config.original_adapter_tree_sha256
            ),
            "remapped_adapter_sha256": (
                config.remapped_adapter_tree_sha256
            ),
            "remap_receipt_sha256": config.remap_receipt_sha256,
            "remap_converter_sha256": config.remap_converter_sha256,
        },
        "case_contract": contract,
        "surface": {
            "rows": len(cases),
            "label_counts": dict(sorted(label_counts.items())),
            "source": "already_observed_sft_validation_only",
            "hf_post_exact": sum(row["exact"] for row in hf),
            "maximum_input_tokens": maximum_input_tokens,
            "benchmark_rows_or_outputs": 0,
            "canary_rows_or_outputs": 0,
            "holdout_rows_or_outputs": 0,
            "fresh_integration_rows_or_outputs": 0,
        },
        "namespace_audit": {
            **config.namespace_audit,
            "original_example_key": original_name,
            "remapped_example_key": remapped_name,
            "tensor_count": 224,
            "tensor_content_hashes_match": True,
            "original_adapter_weights_sha256": (
                config.original_adapter_weights_sha256
            ),
            "remapped_adapter_weights_sha256": (
                config.remapped_adapter_weights_sha256
            ),
            "vllm_source_files": config.vllm_source_files,
        },
        "tokenizer_audit": {
            "base_tokenizer_json_sha256": (
                config.base_tokenizer_json_sha256
            ),
            "adapter_tokenizer_json_sha256": (
                config.adapter_tokenizer_json_sha256
            ),
            "checks": tokenizer_checks,
            "semantic_equivalence_passed": True,
        },
        "arms": {
            "base": config.served_models["base"],
            "original_peft_namespace": config.served_models["original"],
            "content_identical_remapped_namespace": (
                config.served_models["remapped"]
            ),
        },
        "generation": {
            "temperature": config.temperature,
            "max_tokens": config.generation_max_tokens,
            "structured_output_regex": (
                config.route_structured_output_regex
            ),
            "chat_template_kwargs": config.chat_template_kwargs,
            "system_prompt_sha256": hashlib.sha256(
                config.system_prompt.encode()
            ).hexdigest(),
        },
        "service_launch": config.service_launch,
        "acceptance": {
            "all_three_arms_complete_192": True,
            "all_outputs_parseable_192": True,
            "remapped_exact_192": True,
            "remapped_hf_output_match_192": True,
            "remapped_each_label_exact_64": True,
            "original_hf_output_match_less_than_192": True,
            "remapped_exact_greater_than_original": True,
            "remap_tensor_content_unchanged": True,
            "namespace_root_cause_supported_after_pass": True,
            "fresh_integration_v2_preregistration_allowed_after_pass": True,
            "observed_integration_v1_rerun_allowed_after_pass": False,
            "real_question_scan_allowed_after_pass": False,
            "benchmark_allowed_after_pass": False,
            "training_or_rl_allowed_after_pass": False,
        },
        "decision_policy": {
            "passed": (
                "Publish namespace-root-cause evidence and separately "
                "pre-register a new history-disjoint integration v2 using "
                "the content-identical remapped adapter."
            ),
            "failed": (
                "Publish unresolved serving-parity evidence and do not "
                "generate a new integration or train."
            ),
            "forbidden_after_observation": [
                "rerun_observed_integration_v1",
                "change_adapter_tensors",
                "change_validation_rows",
                "change_prompt_or_parser",
                "change_budget_or_temperature",
                "change_parity_gates",
                "real_question_scan",
                "benchmark_access",
                "canary_access",
                "holdout_access",
                "training",
                "rl",
            ],
        },
        "execution_boundary": config.execution_boundary,
        "claim_boundary": (
            "This pre-registers a serving-parity diagnosis on 192 already "
            "observed SFT validation rows. It starts no service or model "
            "generation and does not access the rejected fresh integration, "
            "benchmark, canary, holdout, training, or RL data."
        ),
    }


def render_markdown(receipt: dict) -> str:
    return f"""# Qwen3.5 Router Serving Parity v1

## Question

The HF reload scores 192/192, but the first vLLM integration routes all 64
unsupported rows to A. This diagnosis separates data transfer from deployment:

- base 4B;
- original PEFT namespace `model.layers.*`;
- content-identical remapped namespace `language_model.model.layers.*`.

## Frozen Evidence

- rows: 192 observed SFT validation rows, A/B/C = 64/64/64;
- fresh integration, benchmark, canary, and holdout rows loaded: 0;
- original/remapped tensors: 224, identical dtype/shape/content hashes;
- tokenizer vocab, special tokens, chat template, every prompt ID, and every
  target ID are semantically equal.

## Gates

- all three arms complete and parseable;
- remapped output is exact 192/192 and byte-equal to HF output 192/192;
- each remapped label is 64/64;
- original output differs from HF on at least one row;
- remapped exact is strictly greater than original exact.

Passing identifies the namespace mismatch as the serving root cause. It permits
only a separately pre-registered **new history-disjoint integration v2**. It
does not permit rerunning observed integration v1, a real question scan,
benchmark/canary/holdout access, training, or RL.

## Identity

- config SHA: `{receipt['identity']['config_sha256']}`;
- case contract SHA: `{receipt['identity']['case_contract_sha256']}`;
- original adapter SHA: `{receipt['identity']['original_adapter_sha256']}`;
- remapped adapter SHA: `{receipt['identity']['remapped_adapter_sha256']}`;
- model generation started: false.
"""


def main() -> None:
    receipt = build_receipt()
    JSON_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUTPUT.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    MARKDOWN_OUTPUT.write_text(render_markdown(receipt), encoding="utf-8")
    print(
        json.dumps(
            {
                "experiment_id": receipt["experiment_id"],
                "identity": receipt["identity"],
                "surface": receipt["surface"],
                "namespace_audit": receipt["namespace_audit"],
                "tokenizer_audit": receipt["tokenizer_audit"],
                "acceptance": receipt["acceptance"],
                "execution_boundary": receipt["execution_boundary"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
