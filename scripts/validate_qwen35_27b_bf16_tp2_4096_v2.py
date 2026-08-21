#!/usr/bin/env python3

from pathlib import Path

from scripts.validate_qwen35_27b_bf16_tp2_v1 import validate


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/serving/qwen35_27b_bf16_tp2_4096_v2.json"
RAW = ROOT / "results/serving/qwen35-27b-bf16-tp2-4096-v2.json"
PUBLIC = ROOT / "docs/results/qwen35_27b_bf16_tp2_4096_v2.public.json"
MARKDOWN = ROOT / "docs/results/qwen35_27b_bf16_tp2_4096_v2.md"


if __name__ == "__main__":
    import json

    print(
        json.dumps(
            validate(CONFIG, RAW, PUBLIC, MARKDOWN),
            indent=2,
            sort_keys=True,
        )
    )
