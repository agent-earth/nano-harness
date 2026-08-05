from __future__ import annotations

from pathlib import Path
from typing import Iterable

from nano_harness.adapters.common import iter_jsonl, select_shard
from nano_harness.config import BenchmarkConfig
from nano_harness.types import Task, TaskResult


class CLBenchAdapter:
    name = "clbench"

    def load(self, config: BenchmarkConfig) -> Iterable[Task]:
        records = _load_records(config.source, config.split)
        for index, record in select_shard(
            records,
            config.start,
            config.limit,
            config.shard_id,
            config.num_shards,
        ):
            metadata = dict(record.get("metadata", {}))
            task_id = str(metadata.get("task_id", record.get("idx", index)))
            messages = list(record.get("messages", []))
            if messages and messages[-1].get("role") == "assistant":
                messages = messages[:-1]
            yield Task(
                task_id=task_id,
                benchmark=self.name,
                messages=messages,
                metadata={
                    **metadata,
                    "rubrics": record.get("rubrics", []),
                    "source_record": record,
                    "audit_policy": "always",
                    "constraints": [
                        "Use only the supplied context for newly defined knowledge.",
                        "Satisfy every format, content, ordering, and style constraint.",
                        "Return only the final answer requested by the task.",
                    ],
                },
            )

    def serialize(self, result: TaskResult) -> dict:
        source = dict(result.metadata.get("source_record", {}))
        source.update(
            {
                "idx": result.task_id,
                "model_output": result.output,
                "nano_harness": {
                    key: value
                    for key, value in result.to_dict().items()
                    if key not in {"metadata", "output"}
                },
            }
        )
        return source


def _load_records(source: str, split: str):
    path = Path(source)
    if path.exists():
        return iter_jsonl(path)
    from datasets import load_dataset

    return load_dataset(source, split=split)
