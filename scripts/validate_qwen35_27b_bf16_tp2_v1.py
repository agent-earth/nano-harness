#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from openai import OpenAI

from nano_harness.baseline import sha256_file


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/serving/qwen35_27b_bf16_tp2_v1.json"
RAW = ROOT / "results/serving/qwen35-27b-bf16-tp2-v1.json"
PUBLIC = ROOT / "docs/results/qwen35_27b_bf16_tp2_v1.public.json"
MARKDOWN = ROOT / "docs/results/qwen35_27b_bf16_tp2_v1.md"


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if (
        config.get("schema_version")
        != "nano_harness_qwen35_27b_serving_v1"
        or config.get("policy")
        != {
            "benchmark_generation_started": False,
            "serving_smoke_only": True,
            "gptq_outputs_scoreable": False,
            "raw_server_logs_committed": False,
        }
        or config.get("rejected_gptq", {}).get("parity_eligible") is not False
    ):
        raise ValueError("Qwen3.5-27B serving contract differs")
    return config


def validate() -> dict[str, Any]:
    config = load_config()
    model_root = Path(config["model"]["path"])
    if (
        sha256_file(model_root / "config.json")
        != config["model"]["config_sha256"]
        or sha256_file(model_root / "model.safetensors.index.json")
        != config["model"]["index_sha256"]
    ):
        raise ValueError("Qwen3.5-27B BF16 identity differs")
    index = json.loads(
        (model_root / "model.safetensors.index.json").read_text(
            encoding="utf-8"
        )
    )
    shards = sorted(set(index["weight_map"].values()))
    if (
        len(shards) != config["model"]["weight_shards"]
        or any(not (model_root / shard).is_file() for shard in shards)
    ):
        raise ValueError("Qwen3.5-27B BF16 shards differ")
    with urlopen(config["service"]["health_url"], timeout=30) as response:
        if response.status != 200:
            raise ValueError("Qwen3.5-27B health check failed")
    with urlopen(config["service"]["base_url"] + "/models", timeout=30) as response:
        models = json.loads(response.read().decode("utf-8"))
    rows = models.get("data", [])
    if (
        len(rows) != 1
        or rows[0].get("id") != config["service"]["served_model_name"]
        or rows[0].get("max_model_len")
        != config["service"]["max_model_len"]
        or rows[0].get("owned_by") != "vllm"
    ):
        raise ValueError("Qwen3.5-27B service identity differs")
    client = OpenAI(
        base_url=config["service"]["base_url"],
        api_key="local-vllm",
        timeout=300,
        max_retries=0,
    )
    probes = []
    for index, probe in enumerate(config["smoke"]["probes"]):
        response = client.chat.completions.create(
            model=config["service"]["served_model_name"],
            messages=[
                {"role": "system", "content": config["smoke"]["system"]},
                {"role": "user", "content": probe["prompt"]},
            ],
            temperature=config["smoke"]["temperature"],
            top_p=config["smoke"]["top_p"],
            max_tokens=config["smoke"]["max_tokens"],
            seed=config["smoke"]["seed_base"] + index,
            extra_body={
                "chat_template_kwargs": config["smoke"][
                    "chat_template_kwargs"
                ]
            },
        )
        output = response.choices[0].message.content or ""
        probes.append(
            {
                "id": probe["id"],
                "expected": probe["expected"],
                "output": output,
                "exact": output.strip() == probe["expected"],
                "finish_reason": response.choices[0].finish_reason,
                "usage": response.usage.model_dump(),
            }
        )
    if not all(row["exact"] for row in probes):
        raise ValueError("Qwen3.5-27B deterministic smoke failed")
    RAW.parent.mkdir(parents=True, exist_ok=True)
    raw = {
        "schema_version": "nano_harness_qwen35_27b_serving_raw_v1",
        "experiment_id": config["experiment_id"],
        "models_response_sha256": hashlib.sha256(
            json.dumps(models, sort_keys=True).encode()
        ).hexdigest(),
        "probes": probes,
    }
    RAW.write_text(
        json.dumps(raw, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = {
        "schema_version": "nano_harness_qwen35_27b_serving_public_v1",
        "experiment_id": config["experiment_id"],
        "identity": {
            "result_revision": git_revision(),
            "config_sha256": sha256_file(CONFIG),
            "model_config_sha256": config["model"]["config_sha256"],
            "model_index_sha256": config["model"]["index_sha256"],
            "raw_result_sha256": sha256_file(RAW),
        },
        "service": config["service"],
        "download": config["download"],
        "smoke": {
            "passed": True,
            "passed_probes": len(probes),
            "total_probes": len(probes),
            "results": probes,
        },
        "rejected_gptq": config["rejected_gptq"],
        "decision": {
            "bf16_tp2_service_ready": True,
            "gptq_service_rejected": True,
            "parity_preregistration_allowed": True,
            "benchmark_score_established": False,
        },
    }
    PUBLIC.parent.mkdir(parents=True, exist_ok=True)
    PUBLIC.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    MARKDOWN.write_text(
        "# Qwen3.5-27B BF16 TP=2 Serving v1\n\n"
        "The two-GPU BF16 service passed 3/3 deterministic smoke probes at "
        "1024 context. The GPTQ-Int4 service is rejected because vLLM 0.19.1 "
        "warns that its 4-bit GPTQ GEMM is buggy and the observed outputs "
        "degenerated to punctuation; Marlin has no supported quantization "
        "type on the V100 compute capability 7.0 GPUs.\n\n"
        "This is serving evidence only and establishes no benchmark score.\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    print(json.dumps(validate(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
