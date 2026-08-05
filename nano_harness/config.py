from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


MODELS = {
    "nano-9b": "nvidia/nemotron-nano-9b-v2:free",
    "nano-30b": "nvidia/nemotron-3-nano-30b-a3b:free",
    "super-120b": "nvidia/nemotron-3-super-120b-a12b:free",
    "ultra-550b": "nvidia/nemotron-3-ultra-550b-a55b:free",
}


@dataclass(frozen=True)
class ModelConfig:
    name: str
    base_url: str = "https://openrouter.ai/api/v1"
    api_key_env: str = "OPENROUTER_API_KEY"
    temperature: float = 0.0
    max_tokens: int = 4096
    timeout_seconds: float = 180.0
    max_retries: int = 5


@dataclass(frozen=True)
class HarnessConfig:
    strategy: str = "base"
    max_steps: int = 24
    audit_passes: int = 1
    max_context_chars: int = 90000
    reserve_chars: int = 12000
    scratchpad_chars: int = 12000
    max_tool_errors: int = 4
    require_verification: bool = True


@dataclass(frozen=True)
class BenchmarkConfig:
    name: str
    source: str
    split: str = "test"
    limit: int | None = None
    start: int = 0
    num_shards: int = 1
    shard_id: int = 0
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunConfig:
    model: ModelConfig
    harness: HarnessConfig
    benchmark: BenchmarkConfig
    output_dir: Path
    run_id: str


def _strict_dataclass(cls: type, raw: dict[str, Any]):
    allowed = set(cls.__dataclass_fields__)
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {sorted(unknown)}")
    return cls(**raw)


def load_run_config(path: str | Path) -> RunConfig:
    config_path = Path(path).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    allowed = {"model", "harness", "benchmark", "output_dir", "run_id"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"Unknown RunConfig fields: {sorted(unknown)}")
    model_raw = dict(raw["model"])
    if model_raw.get("name") in MODELS:
        model_raw["name"] = MODELS[model_raw["name"]]
    output_dir = Path(raw.get("output_dir", "results"))
    if not output_dir.is_absolute():
        output_dir = (config_path.parent / output_dir).resolve()
    return RunConfig(
        model=_strict_dataclass(ModelConfig, model_raw),
        harness=_strict_dataclass(HarnessConfig, dict(raw.get("harness", {}))),
        benchmark=_strict_dataclass(BenchmarkConfig, dict(raw["benchmark"])),
        output_dir=output_dir,
        run_id=str(raw.get("run_id", config_path.stem)),
    )
