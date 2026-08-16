#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

from openai import OpenAI
from transformers import AutoTokenizer

from nano_train.data import semantic_output_valid, tokenize_samples


KNOWN_CASE_ID = "synthetic-0e6f6130c187e1ba9b94"
BASE_MODEL = "qwen3.5-4b-base-v6-host"
ADAPTER_MODEL = "qwen3.5-4b-process-v6"
EXPECTED_SOURCE_WEIGHTS_SHA256 = (
    "1b2065129f368f6d3b72bbf875bbd0a2"
    "d2b7b97ab8b7c4ec7ca10c8155f343ea"
)
EXPECTED_SERVING_WEIGHTS_SHA256 = (
    "057ec7aa2214e5fb35d8bb6afd88ec71"
    "b3d9e468dc12bd69ccc7a9e5c1c43d4d"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8003/v1",
    )
    parser.add_argument(
        "--receipt",
        default="results/serving/qwen35-v6-vllm-adapter.receipt.json",
    )
    parser.add_argument(
        "--output",
        default="results/serving/qwen35-v6-serving-parity.json",
    )
    args = parser.parse_args()

    receipt = json.loads(Path(args.receipt).read_text(encoding="utf-8"))
    if (
        receipt["source_adapter_weights_sha256"]
        != EXPECTED_SOURCE_WEIGHTS_SHA256
        or receipt["serving_adapter_weights_sha256"]
        != EXPECTED_SERVING_WEIGHTS_SHA256
        or receipt["tensor_count"] != 224
        or receipt["tensor_content_hashes_match"] is not True
        or receipt["remapped_key_count"] != 224
    ):
        raise SystemExit("serving adapter namespace receipt mismatch")

    client = OpenAI(
        api_key="local-vllm",
        base_url=args.base_url,
        timeout=180,
        max_retries=0,
    )
    models = {model.id: model for model in client.models.list().data}
    if {BASE_MODEL, ADAPTER_MODEL} - set(models):
        raise SystemExit(f"serving models missing: {sorted(models)}")
    adapter_description = models[ADAPTER_MODEL].model_dump(exclude_none=True)
    if adapter_description.get("parent") != BASE_MODEL:
        raise SystemExit("adapter model parent differs from frozen base")

    logits = {}
    for model in (BASE_MODEL, ADAPTER_MODEL):
        response = client.completions.create(
            model=model,
            prompt="Test",
            temperature=0,
            max_tokens=1,
            logprobs=10,
        )
        choice = response.choices[0]
        logits[model] = {
            "token": choice.text,
            "token_logprob": choice.logprobs.token_logprobs[0],
            "top_logprobs": choice.logprobs.top_logprobs[0],
        }
    if logits[BASE_MODEL] == logits[ADAPTER_MODEL]:
        raise SystemExit("adapter logits are identical to base logits")

    dataset = json.loads(
        Path(
            "../nano-data-pipeline/"
            "datasets/verified_arithmetic_process_traces_v4.json"
        ).read_text(encoding="utf-8")
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "../../models/Qwen3.5-4B",
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    samples = {
        sample.sample_id: sample
        for sample in tokenize_samples(dataset, tokenizer, max_length=192)
    }
    sample = samples[KNOWN_CASE_ID]
    raw = next(
        row for row in dataset["samples"] if row["sample_id"] == KNOWN_CASE_ID
    )
    parity = {}
    for model in (BASE_MODEL, ADAPTER_MODEL):
        response = client.chat.completions.create(
            model=model,
            messages=raw["messages"][:-1],
            temperature=0,
            max_tokens=80,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False}
            },
        )
        output = response.choices[0].message.content or ""
        parity[model] = {
            "output": output,
            "exact": output.strip() == sample.target,
            "semantic_valid": semantic_output_valid(sample, output),
            "finish_reason": response.choices[0].finish_reason,
        }
    if parity[BASE_MODEL]["semantic_valid"] is not False:
        raise SystemExit("known case unexpectedly passes under base")
    if (
        parity[ADAPTER_MODEL]["exact"] is not True
        or parity[ADAPTER_MODEL]["semantic_valid"] is not True
    ):
        raise SystemExit("known case does not reproduce adapter behavior")

    result = {
        "schema_version": "qwen35_v6_serving_parity_v1",
        "base_url": args.base_url,
        "base_model": BASE_MODEL,
        "adapter_model": ADAPTER_MODEL,
        "adapter_parent_matches": True,
        "namespace_receipt": {
            "source_adapter_weights_sha256": (
                receipt["source_adapter_weights_sha256"]
            ),
            "serving_adapter_weights_sha256": (
                receipt["serving_adapter_weights_sha256"]
            ),
            "tensor_count": receipt["tensor_count"],
            "tensor_content_hashes_match": True,
        },
        "logits_differ": True,
        "logits": logits,
        "known_case_id": KNOWN_CASE_ID,
        "known_case": parity,
        "full_benchmark_allowed": True,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "schema_version": result["schema_version"],
                "adapter_parent_matches": True,
                "logits_differ": True,
                "base_known_case_semantic": parity[BASE_MODEL][
                    "semantic_valid"
                ],
                "adapter_known_case_exact": parity[ADAPTER_MODEL]["exact"],
                "adapter_known_case_semantic": parity[ADAPTER_MODEL][
                    "semantic_valid"
                ],
                "full_benchmark_allowed": True,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
