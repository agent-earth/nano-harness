from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def select_shard(
    records: Iterable[dict[str, Any]],
    start: int,
    limit: int | None,
    shard_id: int,
    num_shards: int,
) -> Iterable[tuple[int, dict[str, Any]]]:
    if num_shards < 1 or not 0 <= shard_id < num_shards:
        raise ValueError("shard_id must be in [0, num_shards)")
    emitted = 0
    for index, record in enumerate(records):
        if index < start or index % num_shards != shard_id:
            continue
        if limit is not None and emitted >= limit:
            break
        emitted += 1
        yield index, record
