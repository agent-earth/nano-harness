#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json

from nano_harness.v5_complete_treatment import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/campaign/qwen35_v5_complete_treatment_v1.json",
    )
    args = parser.parse_args()
    result = run(args.config)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
