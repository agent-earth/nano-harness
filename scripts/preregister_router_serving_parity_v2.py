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
from nano_harness.router_serving_parity_v2 import (
    case_contract,
    load_config,
    validation_cases,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/harness/qwen35_router_serving_parity_v2.json"
JSON_OUTPUT = (
    ROOT
    / "docs/experiments/qwen35_router_serving_parity_v2.preregister.json"
)
MARKDOWN_OUTPUT = (
    ROOT / "docs/experiments/qwen35_router_serving_parity_v2.md"
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
        raise ValueError("router serving parity v2 HF reference differs")

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
            base_tokenizer.encode(str(case["prompt"]))
            == adapter_tokenizer.encode(str(case["prompt"]))
            for case in cases
        ),
        "all_target_ids_equal": all(
            base_tokenizer.encode(str(case["target"]))
            == adapter_tokenizer.encode(str(case["target"]))
            for case in cases
        ),
    }
    if not all(tokenizer_checks.values()):
        raise ValueError("router serving parity v2 tokenizer semantics differ")

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
        raise ValueError("router serving parity v2 namespace audit differs")

    input_lengths = [
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
    ]
    maximum_input_tokens = max(input_lengths)
    if (
        maximum_input_tokens + config.generation_max_tokens
        > config.service_launch["max_model_len"]
    ):
        raise ValueError("router serving parity v2 context budget differs")

    label_counts = Counter(str(case["label"]) for case in cases)
    subtype_counts = Counter(
        str(case["negative_subtype"])
        for case in cases
        if case["label"] == "C"
    )
    return {
        "schema_version": (
            "nano_harness_router_serving_parity_preregister_v2"
        ),
        "experiment_id": config.experiment_id,
        "identity": {
            "code_revision": git_revision(),
            "config_sha256": sha256_file(CONFIG),
            "case_contract_sha256": contract["case_contract_sha256"],
            "dataset_sha256": config.dataset_sha256,
            "sft_report_sha256": config.sft_report_sha256,
            "namespace_root_cause_report_sha256": (
                config.namespace_root_cause_report_sha256
            ),
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
            "c_subtype_counts": dict(sorted(subtype_counts.items())),
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
            "root_cause_already_established_by": (
                config.namespace_root_cause_report_path
            ),
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
            "content_identical_remapped_namespace": (
                config.served_models["remapped"]
            ),
            "base_service_parent": config.served_models["base"],
            "reason_original_arm_not_repeated": (
                "v1 already established that the original PEFT namespace is "
                "inert and the content-identical remap restores HF parity"
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
            "remapped_complete_1536": True,
            "all_outputs_parseable_1536": True,
            "remapped_exact_1536": True,
            "remapped_hf_output_match_1536": True,
            "each_label_exact_and_hf_match_512": True,
            "each_c_subtype_exact_and_hf_match_64": True,
            "remap_tensor_content_unchanged": True,
            "fresh_integration_v3_preregistration_allowed_after_pass": True,
            "fresh_integration_generation_allowed_after_pass": False,
            "observed_integration_v1_or_v2_rerun_allowed_after_pass": False,
            "benchmark_allowed_after_pass": False,
            "training_or_rl_allowed_after_pass": False,
        },
        "decision_policy": {
            "passed": (
                "Publish exact vLLM/HF serving parity and separately "
                "pre-register one new history-disjoint integration v3."
            ),
            "failed": (
                "Publish serving-parity failure and stop this adapter. Do not "
                "change namespace, prompt, parser, budget, or adapter weight."
            ),
            "forbidden_after_observation": [
                "rerun_observed_integration_v1_or_v2",
                "change_adapter_tensors_or_weight",
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
            "This pre-registers one serving-parity check on 1,536 already "
            "observed SFT validation rows. It starts no service or model "
            "generation and accesses no fresh integration, benchmark, "
            "canary, holdout, training, or RL data."
        ),
    }


def render_markdown(receipt: dict) -> str:
    return f"""# Qwen3.5 Router Serving Parity v2

## Question

The new negative-diversity adapter reaches 1,536/1,536 under an independent HF
reload. V1 already proved that Qwen3.5 vLLM requires a content-identical
namespace remap. This experiment asks only whether the new remapped adapter
reproduces all frozen HF outputs under vLLM.

## Frozen Evidence

- rows: 1,536 observed SFT validation rows, A/B/C = 512/512/512;
- C subtypes: 8 x 64;
- fresh integration, benchmark, canary, and holdout rows loaded: 0;
- remap: 224 tensors with identical dtype, shape, and content hashes;
- tokenizer vocab, special tokens, chat template, prompt IDs, and target IDs:
  equal.

## Gates

- all 1,536 outputs complete and parseable;
- all 1,536 outputs exactly match both target and HF output;
- every label is 512/512;
- every C subtype is 64/64.

Passing permits only a separately pre-registered, new history-disjoint
integration v3. It does not permit rerunning integration v1/v2, benchmark,
canary, holdout, training, or RL.

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
