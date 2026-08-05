from __future__ import annotations

import argparse
import json
from pathlib import Path

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

    args = parser.parse_args()
    if args.command == "run":
        summary = run_config(load_run_config(args.config))
    elif args.command == "merge":
        summary = merge_paths(
            [Path(pattern) for item in args.input for pattern in sorted(Path().glob(item))],
            Path(args.output),
        )
    else:
        summary = summarize_paths([Path(path) for path in args.paths])
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
