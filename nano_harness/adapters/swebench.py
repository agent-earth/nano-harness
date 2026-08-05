from __future__ import annotations

import re
import shutil
import subprocess
import tarfile
import time
import urllib.request
from pathlib import Path
from typing import Iterable

from nano_harness.adapters.common import iter_jsonl, select_shard
from nano_harness.coding_tools import CODING_TOOLS
from nano_harness.config import BenchmarkConfig
from nano_harness.types import Task, TaskResult


class SWEBenchAdapter:
    name = "swebench"

    def load(self, config: BenchmarkConfig) -> Iterable[Task]:
        records = _load_records(config.source, config.split)
        for index, record in select_shard(
            records,
            config.start,
            config.limit,
            config.shard_id,
            config.num_shards,
        ):
            instance_id = str(record.get("instance_id", index))
            prompt = _build_prompt(record)
            repository_path = None
            cache_root = config.options.get("repository_cache")
            if cache_root:
                repository_path = prepare_repository(
                    repo=str(record["repo"]),
                    base_commit=str(record["base_commit"]),
                    instance_id=instance_id,
                    cache_root=Path(cache_root),
                )
            yield Task(
                task_id=instance_id,
                benchmark=self.name,
                messages=[{"role": "user", "content": prompt}],
                metadata={
                    "repo": record.get("repo"),
                    "base_commit": record.get("base_commit"),
                    "problem_statement": record.get("problem_statement", ""),
                    "repository_path": str(repository_path) if repository_path else None,
                    "audit_policy": "missing_patch",
                    "constraints": [
                        "Return a real unified diff between PATCH markers.",
                        "Do not modify tests merely to hide the reported failure.",
                        "Do not claim tests passed without observed test output.",
                    ],
                },
                tools=CODING_TOOLS,
            )

    def serialize(self, result: TaskResult) -> dict:
        return {
            "instance_id": result.task_id,
            "model_name_or_path": result.model,
            "model_patch": extract_patch(result.output),
            "nano_harness": result.to_dict(),
        }


def _load_records(source: str, split: str):
    path = Path(source)
    if path.exists():
        return iter_jsonl(path)
    from datasets import load_dataset

    return load_dataset(source, split=split)


def _build_prompt(record: dict) -> str:
    return (
        f"Repository: {record.get('repo', 'unknown')}\n"
        f"Base commit: {record.get('base_commit', 'unknown')}\n\n"
        f"Issue:\n{record.get('problem_statement', '')}\n\n"
        "Work in the checked-out repository. Inspect relevant code and tests, make "
        "the smallest correct fix, run focused validation, inspect the final diff, "
        "then return the patch enclosed by <<PATCH>> and <<END_PATCH>>."
    )


def extract_patch(output: str) -> str:
    marked = re.search(r"<<PATCH>>\s*(.*?)\s*<<END_PATCH>>", output, re.DOTALL)
    if marked:
        return marked.group(1).strip()
    fenced = re.search(r"```diff\s*(.*?)```", output, re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    diff_start = output.find("diff --git ")
    return output[diff_start:].strip() if diff_start >= 0 else ""


def prepare_repository(
    repo: str,
    base_commit: str,
    instance_id: str,
    cache_root: Path,
) -> Path:
    cache_root = cache_root.resolve()
    snapshot = cache_root / "snapshots" / f"{repo.replace('/', '__')}__{base_commit}"
    checkout = cache_root / "worktrees" / _safe_name(instance_id)
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    checkout.parent.mkdir(parents=True, exist_ok=True)
    if not snapshot.exists():
        _download_snapshot(repo, base_commit, snapshot)
    if checkout.exists():
        shutil.rmtree(checkout)
    shutil.copytree(snapshot, checkout, copy_function=_copy_or_link)
    _run_git(["-C", str(checkout), "init", "-b", "main"])
    _run_git(["-C", str(checkout), "config", "user.name", "Nano Harness"])
    _run_git(["-C", str(checkout), "config", "user.email", "noreply@example.com"])
    _run_git(["-C", str(checkout), "add", "."])
    _run_git(["-C", str(checkout), "commit", "-m", f"baseline {base_commit}"])
    return checkout


def _run_git(arguments: list[str]) -> None:
    process = subprocess.run(
        ["git", *arguments],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=600,
        check=False,
    )
    if process.returncode:
        raise RuntimeError(process.stdout.strip() or f"git {' '.join(arguments)} failed")


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def _download_snapshot(repo: str, commit: str, destination: Path) -> None:
    archive = destination.parent / f"{destination.name}.tar.gz"
    temporary = destination.parent / f".{destination.name}.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    url = f"https://codeload.github.com/{repo}/tar.gz/{commit}"
    last_error: Exception | None = None
    for attempt in range(5):
        archive.unlink(missing_ok=True)
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "nano-harness/0.1"}
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                with archive.open("wb") as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
            with tarfile.open(archive, "r:gz") as validation:
                validation.getmembers()
            last_error = None
            break
        except Exception as exc:
            last_error = exc
            archive.unlink(missing_ok=True)
            time.sleep(min(16, 2**attempt))
    if last_error is not None:
        shutil.rmtree(temporary, ignore_errors=True)
        raise RuntimeError(f"failed to download repository snapshot: {last_error}")
    with tarfile.open(archive, "r:gz") as handle:
        members = handle.getmembers()
        root = members[0].name.split("/", 1)[0] if members else ""
        for member in members:
            relative = member.name[len(root) :].lstrip("/")
            if not relative:
                continue
            target = (temporary / relative).resolve()
            if temporary.resolve() not in target.parents:
                raise ValueError("unsafe path in repository archive")
            member.name = relative
            handle.extract(member, temporary, filter="data")
    archive.unlink(missing_ok=True)
    temporary.rename(destination)


def _copy_or_link(source: str, destination: str) -> str:
    try:
        Path(destination).hardlink_to(source)
        return destination
    except OSError:
        return shutil.copy2(source, destination)
