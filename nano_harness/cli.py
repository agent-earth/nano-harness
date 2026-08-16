from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from nano_harness.analog_contract import load_config as load_analog_contract_config
from nano_harness.analog_contract import run as run_analog_contract
from nano_harness.baseline import (
    case_manifest,
    compare_baselines,
    load_cases,
    load_manifest,
    run_suite,
    summarize_baseline,
)
from nano_harness.config import ModelConfig
from nano_harness.config import load_run_config
from nano_harness.choice_matrix_eval import (
    load_config as load_choice_matrix_eval_config,
)
from nano_harness.choice_matrix_eval import run as run_choice_matrix_eval
from nano_harness.choice_matrix_eval_v2 import (
    load_config as load_choice_matrix_eval_v2_config,
)
from nano_harness.choice_matrix_eval_v2 import run as run_choice_matrix_eval_v2
from nano_harness.choice_verifier_matrix_eval_v2 import (
    load_config as load_choice_verifier_matrix_eval_config,
)
from nano_harness.choice_verifier_matrix_eval_v2 import (
    run as run_choice_verifier_matrix_eval,
)
from nano_harness.runner import merge_paths, run_config, summarize_paths
from nano_harness.verified_choice import load_config as load_verified_choice_config
from nano_harness.verified_choice import run as run_verified_choice
from nano_harness.verified_choice_canary import (
    load_config as load_verified_choice_canary_config,
)
from nano_harness.verified_choice_canary import run as run_verified_choice_canary
from nano_harness.verified_choice_full import (
    load_config as load_verified_choice_full_config,
)
from nano_harness.verified_choice_full import run as run_verified_choice_full


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

    compare_parser = subparsers.add_parser("baseline-compare")
    compare_parser.add_argument("--candidate", required=True)
    compare_parser.add_argument("--baseline", required=True)
    compare_parser.add_argument("--output", default=None)
    compare_parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    compare_parser.add_argument("--bootstrap-seed", type=int, default=35)

    analog_contract_parser = subparsers.add_parser("analog-contract")
    analog_contract_parser.add_argument("--config", required=True)

    verified_choice_parser = subparsers.add_parser("verified-choice")
    verified_choice_parser.add_argument("--config", required=True)

    verified_choice_canary_parser = subparsers.add_parser(
        "verified-choice-canary"
    )
    verified_choice_canary_parser.add_argument("--config", required=True)

    verified_choice_full_parser = subparsers.add_parser("verified-choice-full")
    verified_choice_full_parser.add_argument("--config", required=True)

    choice_matrix_eval_parser = subparsers.add_parser("choice-matrix-eval")
    choice_matrix_eval_parser.add_argument("--config", required=True)

    choice_matrix_eval_v2_parser = subparsers.add_parser(
        "choice-matrix-eval-v2"
    )
    choice_matrix_eval_v2_parser.add_argument("--config", required=True)

    choice_verifier_matrix_parser = subparsers.add_parser(
        "choice-verifier-matrix-eval"
    )
    choice_verifier_matrix_parser.add_argument("--config", required=True)

    args = parser.parse_args()
    if args.command == "run":
        summary = run_config(load_run_config(args.config))
    elif args.command == "analog-contract":
        if not os.getenv("NANO_HARNESS_API_KEY"):
            os.environ["NANO_HARNESS_API_KEY"] = "local-vllm"
        summary = run_analog_contract(
            load_analog_contract_config(args.config)
        )
    elif args.command == "verified-choice":
        summary = run_verified_choice(
            load_verified_choice_config(args.config)
        )
    elif args.command == "verified-choice-canary":
        summary = run_verified_choice_canary(
            load_verified_choice_canary_config(args.config)
        )
    elif args.command == "verified-choice-full":
        summary = run_verified_choice_full(
            load_verified_choice_full_config(args.config)
        )
    elif args.command == "choice-matrix-eval":
        if not os.getenv("NANO_HARNESS_API_KEY"):
            os.environ["NANO_HARNESS_API_KEY"] = "local-vllm"
        summary = run_choice_matrix_eval(
            load_choice_matrix_eval_config(args.config)
        )
    elif args.command == "choice-matrix-eval-v2":
        if not os.getenv("NANO_HARNESS_API_KEY"):
            os.environ["NANO_HARNESS_API_KEY"] = "local-vllm"
        summary = run_choice_matrix_eval_v2(
            load_choice_matrix_eval_v2_config(args.config)
        )
    elif args.command == "choice-verifier-matrix-eval":
        if not os.getenv("NANO_HARNESS_API_KEY"):
            os.environ["NANO_HARNESS_API_KEY"] = "local-vllm"
        summary = run_choice_verifier_matrix_eval(
            load_choice_verifier_matrix_eval_config(args.config)
        )
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
                chat_template_kwargs=manifest.chat_template_kwargs,
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
    elif args.command == "baseline-compare":
        summary = compare_baselines(
            Path(args.candidate),
            Path(args.baseline),
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        )
        if args.output:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    else:
        summary = summarize_paths([Path(path) for path in args.paths])
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
