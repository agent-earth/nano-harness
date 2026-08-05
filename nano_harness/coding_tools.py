from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


CODING_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List repository files under a relative directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "max_entries": {"type": "integer", "minimum": 1, "maximum": 500},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 repository file with line numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer", "minimum": 1},
                    "end_line": {"type": "integer", "minimum": 1},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search repository text with ripgrep.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "path": {"type": "string"},
                    "max_matches": {"type": "integer", "minimum": 1, "maximum": 300},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_patch",
            "description": "Apply a unified patch to the checked-out repository.",
            "parameters": {
                "type": "object",
                "properties": {"patch": {"type": "string"}},
                "required": ["patch"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run an allowlisted repository inspection or test command.",
            "parameters": {
                "type": "object",
                "properties": {
                    "argv": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": 20,
                    },
                    "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 600},
                },
                "required": ["argv"],
                "additionalProperties": False,
            },
        },
    },
]


class CodingToolExecutor:
    ALLOWED_COMMANDS = {
        "git",
        "pytest",
        "python",
        "python3",
        "tox",
        "npm",
        "pnpm",
        "yarn",
        "cargo",
        "go",
        "make",
        "ruff",
        "mypy",
    }

    def __init__(self, repository: str | Path):
        self.root = Path(repository).resolve()
        if not self.root.is_dir():
            raise ValueError(f"repository does not exist: {self.root}")

    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        method = getattr(self, f"_tool_{name}", None)
        if method is None:
            raise ValueError(f"unknown coding tool: {name}")
        return method(**arguments)

    def diff(self) -> str:
        process = subprocess.run(
            ["git", "diff", "--binary"],
            cwd=self.root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
            check=False,
        )
        if process.returncode:
            raise RuntimeError(process.stdout.strip() or "git diff failed")
        return process.stdout

    def _tool_list_files(self, path: str, max_entries: int = 200) -> str:
        target = self._resolve(path)
        entries = []
        for child in sorted(target.iterdir(), key=lambda item: (not item.is_dir(), item.name)):
            relative = child.relative_to(self.root)
            entries.append(f"{relative}/" if child.is_dir() else str(relative))
            if len(entries) >= max_entries:
                break
        return "\n".join(entries)

    def _tool_read_file(
        self, path: str, start_line: int = 1, end_line: int = 400
    ) -> str:
        target = self._resolve(path)
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        end_line = min(end_line, start_line + 999, len(lines))
        return "\n".join(
            f"{index}: {lines[index - 1]}"
            for index in range(start_line, end_line + 1)
        )

    def _tool_search(
        self, query: str, path: str = ".", max_matches: int = 100
    ) -> str:
        target = self._resolve(path)
        process = subprocess.run(
            ["rg", "-n", "--hidden", "--glob", "!.git", query, str(target)],
            cwd=self.root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
            check=False,
        )
        return "\n".join(process.stdout.splitlines()[:max_matches])

    def _tool_apply_patch(self, patch: str) -> str:
        process = subprocess.run(
            ["git", "apply", "--whitespace=nowarn", "-"],
            cwd=self.root,
            input=patch,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=60,
            check=False,
        )
        if process.returncode:
            raise RuntimeError(process.stdout.strip() or "git apply failed")
        status = subprocess.run(
            ["git", "diff", "--stat"],
            cwd=self.root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
            check=False,
        )
        return status.stdout.strip() or "patch applied; diff stat is empty"

    def _tool_run_command(
        self, argv: list[str], timeout_seconds: int = 120
    ) -> str:
        if not argv or argv[0] not in self.ALLOWED_COMMANDS:
            raise ValueError(
                f"command must start with an allowlisted executable: "
                f"{sorted(self.ALLOWED_COMMANDS)}"
            )
        if any("\n" in item or "\x00" in item for item in argv):
            raise ValueError("command arguments contain forbidden characters")
        process = subprocess.run(
            argv,
            cwd=self.root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_seconds,
            check=False,
        )
        payload = {
            "argv": argv,
            "exit_code": process.returncode,
            "output": process.stdout[-20000:],
        }
        return json.dumps(payload, ensure_ascii=False)

    def _resolve(self, relative: str) -> Path:
        target = (self.root / relative).resolve()
        if target != self.root and self.root not in target.parents:
            raise ValueError("path escapes repository root")
        return target
