#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json

from nano_harness.mbpp_iterative_repair import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=(
            "configs/campaign/mbpp_sanitized_iterative_repair_train_v2.json"
        ),
    )
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--shard-id", type=int, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.config,
                num_shards=args.num_shards,
                shard_id=args.shard_id,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
