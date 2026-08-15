from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from nano_harness.baseline import (
    case_manifest,
    load_cases,
    load_manifest,
    run_suite,
    summarize_baseline,
)
from nano_harness.config import ModelConfig
from nano_harness.config import load_run_config
from nano_harness.runner import merge_paths, run_config, summarize_paths


def main() -> None:
    parser = argparse.ArgumentParser(prog="nano-harness")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--config", required=True)

    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--input", action="append", required=True)
    merge_parser.add_argument("--output", required=True)

    summarize_parser = subparsers.add_parser("summarize")
    summarize_parser.add_argument("paths", nargs="+")

    baseline_parser = subparsers.add_parser("baseline")
    baseline_parser.add_argument("--manifest", required=True)
    baseline_parser.add_argument("--dataset-root", required=True)
    baseline_parser.add_argument("--model", required=True)
    baseline_parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    baseline_parser.add_argument("--api-key-env", default="NANO_HARNESS_API_KEY")
    baseline_parser.add_argument("--output", required=True)

    cases_parser = subparsers.add_parser("baseline-cases")
    cases_parser.add_argument("--manifest", required=True)
    cases_parser.add_argument("--dataset-root", required=True)
    cases_parser.add_argument("--output", required=True)

    baseline_summary_parser = subparsers.add_parser("baseline-summary")
    baseline_summary_parser.add_argument("path")

    args = parser.parse_args()
    if args.command == "run":
        summary = run_config(load_run_config(args.config))
    elif args.command == "merge":
        summary = merge_paths(
            [Path(pattern) for item in args.input for pattern in sorted(Path().glob(item))],
            Path(args.output),
        )
    elif args.command == "baseline":
        manifest = load_manifest(args.manifest)
        if not os.getenv(args.api_key_env):
            os.environ[args.api_key_env] = "local-vllm"
        summary = run_suite(
            manifest,
            Path(args.dataset_root),
            ModelConfig(
                name=args.model,
                base_url=args.base_url,
                api_key_env=args.api_key_env,
                temperature=manifest.temperature,
                max_tokens=manifest.max_tokens,
                timeout_seconds=180.0,
                max_retries=3,
            ),
            Path(args.output),
        )
    elif args.command == "baseline-cases":
        manifest = load_manifest(args.manifest)
        cases = load_cases(manifest, Path(args.dataset_root))
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(case_manifest(cases), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        summary = {
            "suite_id": manifest.suite_id,
            "case_count": len(cases),
            "output": str(output),
        }
    elif args.command == "baseline-summary":
        summary = summarize_baseline(Path(args.path))
    else:
        summary = summarize_paths([Path(path) for path in args.paths])
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
