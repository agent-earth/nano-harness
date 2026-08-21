#!/usr/bin/env python3

from __future__ import annotations

import json

from nano_harness.orca_conditional_majority import load_config, run


def main() -> None:
    result = run(
        load_config("configs/campaign/orca_math_conditional_majority_v4.json")
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
