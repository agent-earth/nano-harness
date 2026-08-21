#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json

from nano_harness.complete_conditional_majority import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=(
            "configs/campaign/"
            "qwen35_complete_conditional_majority_v1.json"
        ),
    )
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
