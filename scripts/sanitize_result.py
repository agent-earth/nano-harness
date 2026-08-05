#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REDACTED_KEYS = {"repository_path"}


def sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: (
                "<local-repository-checkout>"
                if key in REDACTED_KEYS and item
                else sanitize(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, str) and value.startswith("/"):
        return "<local-path>"
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source = Path(args.input)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(sanitize(json.loads(line)))
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
