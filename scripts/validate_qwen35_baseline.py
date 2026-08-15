#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from nano_harness.baseline import case_manifest, load_cases, load_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="configs/baselines/qwen35_local_v4.yaml",
    )
    parser.add_argument("--dataset-root", default="../../datasets")
    parser.add_argument("--tokenizer", default="../../models/Qwen3.5-4B")
    parser.add_argument("--context-limit", type=int, default=1024)
    parser.add_argument(
        "--case-manifest",
        default="configs/generated/qwen35_local_v4_cases.json",
    )
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    cases = load_cases(manifest, Path(args.dataset_root))
    expected_cases = json.loads(Path(args.case_manifest).read_text(encoding="utf-8"))
    actual_cases = case_manifest(cases)
    if actual_cases != expected_cases:
        raise SystemExit("selected cases differ from the committed case manifest")

    counts = Counter(case.benchmark for case in cases)
    expected_counts = {spec.name: spec.limit for spec in manifest.datasets}
    if dict(counts) != expected_counts:
        raise SystemExit(f"case counts differ: {dict(counts)} != {expected_counts}")
    if len({case.case_id for case in cases}) != len(cases):
        raise SystemExit("case ids are not unique")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    lengths: list[int] = []
    by_benchmark: dict[str, list[int]] = {}
    for case in cases:
        text = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": manifest.system_prompt},
                {"role": "user", "content": case.prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
            **manifest.chat_template_kwargs,
        )
        length = len(tokenizer.encode(text))
        lengths.append(length)
        by_benchmark.setdefault(case.benchmark, []).append(length)

    lengths.sort()
    report = {
        "schema_version": "nano_harness_baseline_validation_v1",
        "suite_id": manifest.suite_id,
        "cases": len(cases),
        "counts": dict(sorted(counts.items())),
        "input_tokens": {
            "min": min(lengths),
            "p50": lengths[len(lengths) // 2],
            "p95": lengths[int(len(lengths) * 0.95) - 1],
            "max": max(lengths),
        },
        "max_output_tokens": manifest.max_tokens,
        "chat_template_kwargs": manifest.chat_template_kwargs,
        "max_total_tokens": max(lengths) + manifest.max_tokens,
        "context_limit": args.context_limit,
        "by_benchmark": {
            name: {"min": min(values), "max": max(values)}
            for name, values in sorted(by_benchmark.items())
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["max_total_tokens"] > args.context_limit:
        raise SystemExit("suite exceeds the configured context limit")


if __name__ == "__main__":
    main()
