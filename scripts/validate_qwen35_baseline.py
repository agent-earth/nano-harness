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
        default="configs/baselines/qwen35_local_v5.yaml",
    )
    parser.add_argument("--dataset-root", default="../../datasets")
    parser.add_argument("--tokenizer", default="../../models/Qwen3.5-4B")
    parser.add_argument("--context-limit", type=int, default=1024)
    parser.add_argument(
        "--case-manifest",
        default="configs/generated/qwen35_local_v5_cases.json",
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
    totals: list[int] = []
    by_benchmark: dict[str, dict[str, list[int]]] = {}
    for case in cases:
        if manifest.strategy == "direct":
            text = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": case.system_prompt},
                    {"role": "user", "content": case.prompt},
                ],
                tokenize=False,
                add_generation_prompt=True,
                **manifest.chat_template_kwargs,
            )
            length = len(tokenizer.encode(text))
            output_tokens = case.max_tokens
            total = length + output_tokens
        elif manifest.strategy == "draft_verify":
            draft_text = tokenizer.apply_chat_template(
                [
                    {
                        "role": "system",
                        "content": (
                            "Solve the task carefully. Produce a compact candidate "
                            "analysis and candidate answer for a separate verifier. "
                            "Do not use tools."
                        ),
                    },
                    {"role": "user", "content": case.draft_prompt},
                ],
                tokenize=False,
                add_generation_prompt=True,
                **manifest.chat_template_kwargs,
            )
            draft_input = len(tokenizer.encode(draft_text))
            candidate_placeholder = "x " * manifest.draft_max_tokens
            verifier_text = tokenizer.apply_chat_template(
                [
                    {
                        "role": "system",
                        "content": (
                            "You are the final verifier. Check the candidate against "
                            "the original task, correct it if needed, and return only "
                            "one FINAL line in the exact format requested by the task. "
                            "Do not explain."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"<original_task>\n{case.prompt}\n</original_task>\n\n"
                            f"<candidate>\n{candidate_placeholder}</candidate>"
                        ),
                    },
                ],
                tokenize=False,
                add_generation_prompt=True,
                **manifest.chat_template_kwargs,
            )
            verifier_input = len(tokenizer.encode(verifier_text))
            length = max(draft_input, verifier_input)
            output_tokens = max(
                manifest.draft_max_tokens,
                manifest.verifier_max_tokens,
            )
            total = max(
                draft_input + manifest.draft_max_tokens,
                verifier_input + manifest.verifier_max_tokens,
            )
        elif manifest.strategy == "draft_critique_verify":
            draft_text = tokenizer.apply_chat_template(
                [
                    {
                        "role": "system",
                        "content": (
                            "Solve the task carefully. Produce a compact candidate "
                            "analysis and candidate answer for an independent critic. "
                            "Do not use tools."
                        ),
                    },
                    {"role": "user", "content": case.draft_prompt},
                ],
                tokenize=False,
                add_generation_prompt=True,
                **manifest.chat_template_kwargs,
            )
            draft_input = len(tokenizer.encode(draft_text))
            draft_placeholder = "x " * manifest.draft_max_tokens
            critique_text = tokenizer.apply_chat_template(
                [
                    {
                        "role": "system",
                        "content": (
                            "Act as an independent critic. Re-solve the original task, "
                            "identify any error in the candidate, and provide a corrected "
                            "candidate. Keep the critique compact and do not use tools."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"<original_task>\n{case.draft_prompt}\n</original_task>\n\n"
                            f"<candidate>\n{draft_placeholder}</candidate>"
                        ),
                    },
                ],
                tokenize=False,
                add_generation_prompt=True,
                **manifest.chat_template_kwargs,
            )
            critique_input = len(tokenizer.encode(critique_text))
            critique_placeholder = "y " * manifest.critique_max_tokens
            verifier_text = tokenizer.apply_chat_template(
                [
                    {
                        "role": "system",
                        "content": (
                            "You are the final formatter. Use the original task, "
                            "candidate, and critique to return only one FINAL line in "
                            "the exact format requested by the task. Do not explain."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"<original_task>\n{case.prompt}\n</original_task>\n\n"
                            f"<candidate>\n{draft_placeholder}</candidate>\n\n"
                            f"<critique>\n{critique_placeholder}</critique>"
                        ),
                    },
                ],
                tokenize=False,
                add_generation_prompt=True,
                **manifest.chat_template_kwargs,
            )
            verifier_input = len(tokenizer.encode(verifier_text))
            length = max(draft_input, critique_input, verifier_input)
            output_tokens = max(
                manifest.draft_max_tokens,
                manifest.critique_max_tokens,
                manifest.verifier_max_tokens,
            )
            total = max(
                draft_input + manifest.draft_max_tokens,
                critique_input + manifest.critique_max_tokens,
                verifier_input + manifest.verifier_max_tokens,
            )
        else:
            draft_text = tokenizer.apply_chat_template(
                [
                    {
                        "role": "system",
                        "content": (
                            "Solve the math task carefully. Produce concise reasoning "
                            "and a candidate answer. Do not use tools."
                        ),
                    },
                    {"role": "user", "content": case.draft_prompt},
                ],
                tokenize=False,
                add_generation_prompt=True,
                **manifest.chat_template_kwargs,
            )
            second_text = tokenizer.apply_chat_template(
                [
                    {
                        "role": "system",
                        "content": (
                            "Independently solve the math task from scratch. Check "
                            "units, time periods, totals, and all requested quantities. "
                            "Produce concise reasoning and a candidate answer. Do not "
                            "use tools."
                        ),
                    },
                    {"role": "user", "content": case.draft_prompt},
                ],
                tokenize=False,
                add_generation_prompt=True,
                **manifest.chat_template_kwargs,
            )
            draft_input = len(tokenizer.encode(draft_text))
            second_input = len(tokenizer.encode(second_text))
            draft_placeholder = "x " * manifest.draft_max_tokens
            second_placeholder = "y " * manifest.second_solve_max_tokens
            verifier_text = tokenizer.apply_chat_template(
                [
                    {
                        "role": "system",
                        "content": (
                            "You are the final selector. Reconcile the two independent "
                            "solutions against the original task. Return only one FINAL "
                            "line in the exact format requested by the task. Do not "
                            "explain."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"<original_task>\n{case.prompt}\n</original_task>\n\n"
                            f"<solution_a>\n{draft_placeholder}</solution_a>\n\n"
                            f"<solution_b>\n{second_placeholder}</solution_b>"
                        ),
                    },
                ],
                tokenize=False,
                add_generation_prompt=True,
                **manifest.chat_template_kwargs,
            )
            verifier_input = len(tokenizer.encode(verifier_text))
            length = max(draft_input, second_input, verifier_input)
            output_tokens = max(
                manifest.draft_max_tokens,
                manifest.second_solve_max_tokens,
                manifest.verifier_max_tokens,
            )
            total = max(
                draft_input + manifest.draft_max_tokens,
                second_input + manifest.second_solve_max_tokens,
                verifier_input + manifest.verifier_max_tokens,
            )
        totals.append(total)
        metrics = by_benchmark.setdefault(
            case.benchmark,
            {"input_tokens": [], "output_tokens": [], "total_tokens": []},
        )
        metrics["input_tokens"].append(length)
        metrics["output_tokens"].append(output_tokens)
        metrics["total_tokens"].append(total)

    input_lengths = sorted(
        length
        for metrics in by_benchmark.values()
        for length in metrics["input_tokens"]
    )
    report = {
        "schema_version": "nano_harness_baseline_validation_v1",
        "suite_id": manifest.suite_id,
        "cases": len(cases),
        "counts": dict(sorted(counts.items())),
        "input_tokens": {
            "min": min(input_lengths),
            "p50": input_lengths[len(input_lengths) // 2],
            "p95": input_lengths[int(len(input_lengths) * 0.95) - 1],
            "max": max(input_lengths),
        },
        "chat_template_kwargs": manifest.chat_template_kwargs,
        "max_total_tokens": max(totals),
        "context_limit": args.context_limit,
        "by_benchmark": {
            name: {
                "input_min": min(metrics["input_tokens"]),
                "input_max": max(metrics["input_tokens"]),
                "output_max": max(metrics["output_tokens"]),
                "total_max": max(metrics["total_tokens"]),
            }
            for name, metrics in sorted(by_benchmark.items())
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["max_total_tokens"] > args.context_limit:
        raise SystemExit("suite exceeds the configured context limit")


if __name__ == "__main__":
    main()
