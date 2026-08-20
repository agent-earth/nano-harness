#!/usr/bin/env python3

from __future__ import annotations

import json

from nano_harness.orca_self_consistency import load_config, run


def main() -> None:
    result = run(
        load_config("configs/campaign/orca_math_self_consistency_v1.json")
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
