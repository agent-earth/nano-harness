#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--strategy", choices=["base", "optimized"], required=True)
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    template_path = Path(args.template).resolve()
    output_path = Path(args.output).resolve()
    config = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    config["model"]["name"] = args.model
    config["harness"]["strategy"] = args.strategy
    config["benchmark"]["shard_id"] = args.shard_id
    config["benchmark"]["num_shards"] = args.num_shards
    config["run_id"] = args.run_id
    config["output_dir"] = str((template_path.parents[2] / "results/full").resolve())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
