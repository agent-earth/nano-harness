#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_score(path: str) -> float:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    score = data.get("score")
    if not isinstance(score, (int, float)):
        raise ValueError(f"{path} has no numeric score")
    return float(score)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimized", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--base-30b")
    parser.add_argument("--base-120b")
    parser.add_argument("--base-550b")
    args = parser.parse_args()

    optimized = load_score(args.optimized)
    base = load_score(args.base)
    comparison = {
        "optimized": optimized,
        "base": base,
        "absolute_improvement": optimized - base,
        "relative_improvement_percent": (
            ((optimized - base) / base) * 100 if base else None
        ),
    }
    for label, path in (
        ("30b", args.base_30b),
        ("120b", args.base_120b),
        ("550b", args.base_550b),
    ):
        if path:
            other = load_score(path)
            comparison[f"base_{label}"] = other
            comparison[f"gap_to_{label}"] = optimized - other
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
