from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from nano_harness.adapters import CLBenchAdapter, SWEBenchAdapter, SyntheticAdapter
from nano_harness.client import OpenRouterClient
from nano_harness.coding_tools import CodingToolExecutor
from nano_harness.config import RunConfig
from nano_harness.harness import AgentHarness
from nano_harness.skill_system import SkillRegistry


ADAPTERS = {
    "swebench": SWEBenchAdapter,
    "clbench": CLBenchAdapter,
    "synthetic": SyntheticAdapter,
}


class SyntheticToolExecutor:
    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        if name != "lookup_account":
            raise ValueError(f"unknown tool: {name}")
        account_id = arguments.get("account_id")
        if account_id != "A-17":
            raise ValueError("account not found")
        return json.dumps({"account_id": "A-17", "status": "active"})


def run_config(config: RunConfig, client: Any | None = None) -> dict[str, Any]:
    if config.benchmark.name not in ADAPTERS:
        raise ValueError(
            f"{config.benchmark.name} uses a dedicated runner; "
            f"available generic adapters: {sorted(ADAPTERS)}"
        )
    adapter = ADAPTERS[config.benchmark.name]()
    client = client or OpenRouterClient(config.model)
    skill_registry = None
    if config.harness.strategy == "skill_routed":
        if not config.harness.skill_registry_path:
            raise ValueError(
                "skill_routed strategy requires harness.skill_registry_path"
            )
        skill_registry = SkillRegistry.from_manifest(
            config.harness.skill_registry_path
        )
    harness = AgentHarness(
        client,
        config.model.name,
        config.harness,
        skill_registry=skill_registry,
    )
    run_dir = config.output_dir / config.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / (
        f"shard-{config.benchmark.shard_id:03d}-of-"
        f"{config.benchmark.num_shards:03d}.jsonl"
    )
    completed = completed_task_ids(output_path)
    attempted = 0
    written = 0
    for task in adapter.load(config.benchmark):
        if task.task_id in completed:
            continue
        attempted += 1
        if task.tools and task.benchmark == "synthetic":
            executor = SyntheticToolExecutor()
        elif task.tools and task.benchmark == "swebench":
            repository = task.metadata.get("repository_path")
            if not repository:
                result = harness.run(task, None)
                result.status = "blocked"
                result.failure_type = "missing_repository_checkout"
                result.error = (
                    "SWE-bench agent mode requires metadata.repository_path; "
                    "use the SWE task preparation runner."
                )
                append_jsonl_atomic(output_path, adapter.serialize(result))
                completed.add(task.task_id)
                written += 1
                continue
            executor = CodingToolExecutor(repository)
        else:
            executor = None
        result = harness.run(task, executor)
        if (
            task.benchmark == "swebench"
            and isinstance(executor, CodingToolExecutor)
            and result.status not in {"error", "blocked"}
        ):
            actual_diff = executor.diff()
            result.metadata["model_reported_output"] = result.output
            result.output = actual_diff
            if not actual_diff.strip():
                result.status = "invalid_final"
                result.failure_type = "empty_repository_diff"
                result.error = "The agent finished without a repository diff."
        append_jsonl_atomic(output_path, adapter.serialize(result))
        completed.add(task.task_id)
        written += 1
        if result.failure_type == "provider_daily_quota":
            break
    summary = summarize_paths([output_path])
    summary.update(
        {
            "run_id": config.run_id,
            "output": str(output_path),
            "attempted_this_invocation": attempted,
            "written_this_invocation": written,
        }
    )
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def completed_task_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            task_id = record.get("instance_id", record.get("task_id", record.get("idx")))
            payload = record.get("nano_harness", record)
            status = payload.get("status")
            retryable = status in {"error", "blocked"} or payload.get(
                "failure_type"
            ) in {
                "model_api_error",
                "provider_daily_quota",
                "missing_repository_checkout",
            }
            if task_id is not None and not retryable:
                completed.add(str(task_id))
    return completed


def append_jsonl_atomic(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(descriptor, encoded.encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def merge_paths(paths: list[Path], output: Path) -> dict[str, Any]:
    records: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                task_id = record.get(
                    "instance_id", record.get("task_id", record.get("idx"))
                )
                if task_id is None:
                    raise ValueError(f"record in {path} lacks a task id")
                records[str(task_id)] = record
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for task_id in sorted(records):
            handle.write(json.dumps(records[task_id], ensure_ascii=False) + "\n")
    return summarize_paths([output])


def summarize_paths(paths: list[Path]) -> dict[str, Any]:
    total = 0
    scored = 0
    score_sum = 0.0
    statuses: dict[str, int] = {}
    failures: dict[str, int] = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                payload = record.get("nano_harness", record)
                total += 1
                status = str(payload.get("status", "unknown"))
                statuses[status] = statuses.get(status, 0) + 1
                failure = payload.get("failure_type")
                if failure:
                    failures[str(failure)] = failures.get(str(failure), 0) + 1
                score = record.get("score", payload.get("score"))
                if isinstance(score, (int, float)):
                    scored += 1
                    score_sum += float(score)
    return {
        "total": total,
        "scored": scored,
        "score": score_sum / scored if scored else None,
        "statuses": statuses,
        "failure_types": failures,
    }
