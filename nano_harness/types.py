from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol


@dataclass
class Task:
    task_id: str
    benchmark: str
    messages: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)
    tools: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ModelReply:
    content: str
    reasoning: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    task_id: str
    benchmark: str
    model: str
    harness: str
    status: str
    output: str
    score: float | None = None
    error: str | None = None
    failure_type: str | None = None
    steps: int = 0
    tool_calls: int = 0
    usage: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    trajectory: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ToolExecutor(Protocol):
    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        ...
