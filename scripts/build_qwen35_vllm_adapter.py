#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


SOURCE_PREFIX = "base_model.model.model.layers."
TARGET_PREFIX = "base_model.model.language_model.model.layers."
PASSTHROUGH_FILES = (
    "adapter_config.json",
    "README.md",
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer_config.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(value.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args()

    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    receipt_path = Path(args.receipt).resolve()
    source_weights = source / "adapter_model.safetensors"
    if not source_weights.is_file():
        raise SystemExit(f"missing source adapter: {source_weights}")
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    tensors = {}
    source_hashes = {}
    remapped = []
    with safe_open(source_weights, framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
        for source_key in handle.keys():
            if not source_key.startswith(SOURCE_PREFIX):
                raise SystemExit(f"unexpected source adapter key: {source_key}")
            target_key = TARGET_PREFIX + source_key.removeprefix(SOURCE_PREFIX)
            if target_key in tensors:
                raise SystemExit(f"duplicate remapped adapter key: {target_key}")
            tensor = handle.get_tensor(source_key)
            tensors[target_key] = tensor
            source_hashes[source_key] = tensor_sha256(tensor)
            remapped.append({"source": source_key, "target": target_key})
    save_file(
        tensors,
        output / "adapter_model.safetensors",
        metadata=metadata,
    )
    copied = []
    for name in PASSTHROUGH_FILES:
        source_file = source / name
        if source_file.is_file():
            shutil.copy2(source_file, output / name)
            copied.append(name)

    target_hashes = {}
    with safe_open(
        output / "adapter_model.safetensors",
        framework="pt",
        device="cpu",
    ) as handle:
        for target_key in handle.keys():
            source_key = SOURCE_PREFIX + target_key.removeprefix(TARGET_PREFIX)
            target_hashes[source_key] = tensor_sha256(handle.get_tensor(target_key))
    if source_hashes != target_hashes:
        raise SystemExit("serving adapter tensor content changed during remap")

    receipt = {
        "schema_version": "qwen35_vllm_adapter_namespace_receipt_v1",
        "source_prefix": SOURCE_PREFIX,
        "target_prefix": TARGET_PREFIX,
        "source_adapter_weights_sha256": sha256_file(source_weights),
        "serving_adapter_weights_sha256": sha256_file(
            output / "adapter_model.safetensors"
        ),
        "tensor_count": len(tensors),
        "tensor_content_hashes_match": True,
        "remapped_key_count": len(remapped),
        "copied_files": copied,
        "remapped_keys": remapped,
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "schema_version": receipt["schema_version"],
                "tensor_count": receipt["tensor_count"],
                "tensor_content_hashes_match": True,
                "source_adapter_weights_sha256": receipt[
                    "source_adapter_weights_sha256"
                ],
                "serving_adapter_weights_sha256": receipt[
                    "serving_adapter_weights_sha256"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
