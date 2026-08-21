#!/usr/bin/env python3

from __future__ import annotations

import json

from nano_harness.orca_recovered_self_consistency import (
    load_config,
    select_cases,
)
from nano_harness.orca_self_consistency import build_raw_result


def main() -> None:
    config = load_config(
        "configs/campaign/orca_math_recovered_self_consistency_v3.json"
    )
    result = build_raw_result(config, select_cases(config))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
