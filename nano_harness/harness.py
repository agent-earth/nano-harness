from __future__ import annotations

import json
from typing import Any

from nano_harness.client import ProviderQuotaError
from nano_harness.config import HarnessConfig
from nano_harness.contracts import should_audit
from nano_harness.prompts import AUDIT_SYSTEM, system_prompt
from nano_harness.state import StateLedger, compact_messages
from nano_harness.types import Task, TaskResult, ToolExecutor


class AgentHarness:
    def __init__(self, client: Any, model_name: str, config: HarnessConfig):
        self.client = client
        self.model_name = model_name
        self.config = config

    def run(self, task: Task, tool_executor: ToolExecutor | None = None) -> TaskResult:
        messages = [
            {"role": "system", "content": system_prompt(self.config.strategy, task.benchmark)},
            *task.messages,
        ]
        ledger = StateLedger(
            objective=_last_user_content(task.messages),
            constraints=list(task.metadata.get("constraints", [])),
            pending=["Solve the task and produce directly verified output."],
        )
        trajectory: list[dict[str, Any]] = []
        usage: dict[str, int] = {}
        tool_errors = 0
        tool_count = 0

        for step in range(1, self.config.max_steps + 1):
            request_messages = messages
            if self.config.strategy == "optimized":
                request_messages = compact_messages(
                    messages,
                    ledger,
                    self.config.max_context_chars,
                    self.config.reserve_chars,
                    self.config.scratchpad_chars,
                )
            try:
                reply = self.client.complete(request_messages, task.tools or None)
            except Exception as exc:
                quota_error = isinstance(exc, ProviderQuotaError)
                return TaskResult(
                    task_id=task.task_id,
                    benchmark=task.benchmark,
                    model=self.model_name,
                    harness=self.config.strategy,
                    status="error",
                    output="",
                    error=str(exc),
                    failure_type=(
                        "provider_daily_quota" if quota_error else "model_api_error"
                    ),
                    steps=step,
                    tool_calls=tool_count,
                    usage=usage,
                    metadata={
                        **task.metadata,
                        **(
                            {"provider_quota_reset_at": exc.reset_at}
                            if quota_error
                            else {}
                        ),
                    },
                    trajectory=trajectory,
                )
            _merge_usage(usage, reply.usage)
            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": reply.content,
            }
            if reply.tool_calls:
                assistant_message["tool_calls"] = reply.tool_calls
            messages.append(assistant_message)
            trajectory.append(
                {
                    "step": step,
                    "kind": "model",
                    "content": reply.content,
                    "reasoning": reply.reasoning,
                    "tool_calls": reply.tool_calls,
                }
            )

            if not reply.tool_calls:
                if task.benchmark == "taubench" and hasattr(tool_executor, "respond"):
                    observation = tool_executor.respond(reply.content)
                    trajectory.append(
                        {
                            "step": step,
                            "kind": "user_environment",
                            "observation": observation,
                            "done": bool(getattr(tool_executor, "done", False)),
                        }
                    )
                    if getattr(tool_executor, "done", False):
                        return TaskResult(
                            task_id=task.task_id,
                            benchmark=task.benchmark,
                            model=self.model_name,
                            harness=self.config.strategy,
                            status="completed",
                            output=reply.content,
                            steps=step,
                            tool_calls=tool_count,
                            usage=usage,
                            metadata=task.metadata,
                            trajectory=trajectory,
                        )
                    messages.append({"role": "user", "content": observation})
                    ledger.facts.append(
                        f"User/environment observation: {observation[:800]}"
                    )
                    ledger.pending = [
                        "Continue the conversation until the environment reports done."
                    ]
                    continue
                gate_errors = _completion_gate_errors(task, trajectory)
                if gate_errors:
                    correction = (
                        "COMPLETION GATE FAILED:\n"
                        + "\n".join(f"- {error}" for error in gate_errors)
                        + "\nContinue working. Use the required tools and do not present "
                        "a final answer until every gate is backed by observations."
                    )
                    messages.append({"role": "user", "content": correction})
                    ledger.failures.extend(gate_errors)
                    ledger.pending = gate_errors
                    trajectory.append(
                        {
                            "step": step,
                            "kind": "completion_gate",
                            "errors": gate_errors,
                        }
                    )
                    continue
                final_content = reply.content
                audit_needed, contract_errors = should_audit(
                    final_content, task.metadata
                )
                if (
                    self.config.strategy == "optimized"
                    and self.config.audit_passes > 0
                    and audit_needed
                ):
                    try:
                        final_content, audit_trajectory, audit_usage = self._audit_final(
                            task,
                            messages,
                            ledger,
                            reply.content,
                            contract_errors,
                        )
                        trajectory.extend(audit_trajectory)
                        _merge_usage(usage, audit_usage)
                    except Exception as exc:
                        trajectory.append(
                            {
                                "step": step,
                                "kind": "audit_error",
                                "error": str(exc),
                                "candidate_preserved": True,
                            }
                        )
                failure_type = _final_failure_type(
                    final_content, self.config, task.benchmark
                )
                status = "completed" if failure_type is None else "invalid_final"
                return TaskResult(
                    task_id=task.task_id,
                    benchmark=task.benchmark,
                    model=self.model_name,
                    harness=self.config.strategy,
                    status=status,
                    output=final_content,
                    failure_type=failure_type,
                    steps=step,
                    tool_calls=tool_count,
                    usage=usage,
                    metadata=task.metadata,
                    trajectory=trajectory,
                )

            if tool_executor is None:
                return TaskResult(
                    task_id=task.task_id,
                    benchmark=task.benchmark,
                    model=self.model_name,
                    harness=self.config.strategy,
                    status="error",
                    output="",
                    error="task requested tools but no tool executor was configured",
                    failure_type="missing_tool_executor",
                    steps=step,
                    tool_calls=tool_count,
                    usage=usage,
                    metadata=task.metadata,
                    trajectory=trajectory,
                )

            for call in reply.tool_calls[:1]:
                tool_count += 1
                call_id = call.get("id", f"call-{step}")
                function = call.get("function", {})
                name = function.get("name", "")
                arguments: dict[str, Any] = {}
                try:
                    arguments = _parse_arguments(function.get("arguments", {}))
                    observation = tool_executor.execute(name, arguments)
                    ledger.completed.append(f"Called {name} with schema-valid arguments.")
                    ledger.facts.append(f"{name} observation: {observation[:800]}")
                except Exception as exc:
                    tool_errors += 1
                    observation = f"TOOL_ERROR: {type(exc).__name__}: {exc}"
                    ledger.failures.append(f"{name}: {observation}")
                tool_message = {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": name,
                    "content": observation,
                }
                messages.append(tool_message)
                trajectory.append(
                    {
                        "step": step,
                        "kind": "tool",
                        "name": name,
                        "arguments": arguments,
                        "observation": observation,
                    }
                )
                if task.benchmark == "taubench" and getattr(
                    tool_executor, "done", False
                ):
                    return TaskResult(
                        task_id=task.task_id,
                        benchmark=task.benchmark,
                        model=self.model_name,
                        harness=self.config.strategy,
                        status="completed",
                        output=observation,
                        steps=step,
                        tool_calls=tool_count,
                        usage=usage,
                        metadata=task.metadata,
                        trajectory=trajectory,
                    )
                if tool_errors > self.config.max_tool_errors:
                    return TaskResult(
                        task_id=task.task_id,
                        benchmark=task.benchmark,
                        model=self.model_name,
                        harness=self.config.strategy,
                        status="error",
                        output="",
                        error="tool error budget exceeded",
                        failure_type="tool_error_budget",
                        steps=step,
                        tool_calls=tool_count,
                        usage=usage,
                        metadata=task.metadata,
                        trajectory=trajectory,
                    )

        return TaskResult(
            task_id=task.task_id,
            benchmark=task.benchmark,
            model=self.model_name,
            harness=self.config.strategy,
            status="max_steps",
            output="",
            failure_type="early_stop_guard",
            steps=self.config.max_steps,
            tool_calls=tool_count,
            usage=usage,
            metadata=task.metadata,
            trajectory=trajectory,
        )

    def _audit_final(
        self,
        task: Task,
        messages: list[dict[str, Any]],
        ledger: StateLedger,
        candidate: str,
        detected_errors: list[str],
    ) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
        audit_messages = [
            {"role": "system", "content": AUDIT_SYSTEM},
            *task.messages,
            {"role": "system", "content": ledger.render(self.config.scratchpad_chars)},
            {
                "role": "user",
                "content": (
                    "Constraints:\n"
                    + "\n".join(
                        f"- {constraint}"
                        for constraint in task.metadata.get("constraints", [])
                    )
                    + "\n\nMachine-detected contract errors:\n"
                    + (
                        "\n".join(f"- {error}" for error in detected_errors)
                        if detected_errors
                        else "- none"
                    )
                    + f"\n\nCandidate answer:\n{candidate}"
                ),
            },
        ]
        reply = self.client.complete(audit_messages, None)
        corrected = reply.content.strip() or candidate
        return (
            corrected,
            [
                {
                    "step": len(messages),
                    "kind": "audit",
                    "candidate": candidate,
                    "content": corrected,
                    "reasoning": reply.reasoning,
                }
            ],
            reply.usage,
        )


def _last_user_content(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content", ""))
    return "Complete the benchmark task."


def _parse_arguments(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    parsed = json.loads(arguments or "{}")
    if not isinstance(parsed, dict):
        raise ValueError("tool arguments must be a JSON object")
    return parsed


def _merge_usage(total: dict[str, int], current: dict[str, Any]) -> None:
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = current.get(key)
        if isinstance(value, int):
            total[key] = total.get(key, 0) + value


def _final_failure_type(
    content: str, config: HarnessConfig, benchmark: str
) -> str | None:
    if not content.strip():
        return "empty_output"
    if (
        benchmark == "swebench"
        and config.strategy == "optimized"
        and config.require_verification
    ):
        weak_markers = ("unable to make", "no changes", "probably fixed")
        lowered = content.lower()
        if any(marker in lowered for marker in weak_markers):
            return "unverified_or_empty_change"
    return None


def _completion_gate_errors(
    task: Task, trajectory: list[dict[str, Any]]
) -> list[str]:
    if task.benchmark != "swebench":
        return []
    applied_indexes = [
        index
        for index, item in enumerate(trajectory)
        if item.get("kind") == "tool"
        and item.get("name") == "apply_patch"
        and not str(item.get("observation", "")).startswith("TOOL_ERROR:")
    ]
    if not applied_indexes:
        return ["No patch was applied to the repository with apply_patch."]
    last_apply = applied_indexes[-1]
    validations = [
        item
        for item in trajectory[last_apply + 1 :]
        if item.get("kind") == "tool" and item.get("name") == "run_command"
    ]
    if not validations:
        return ["No validation command was run after the latest patch."]
    successful = False
    for validation in validations:
        try:
            payload = json.loads(str(validation.get("observation", "")))
        except json.JSONDecodeError:
            continue
        if payload.get("exit_code") == 0:
            successful = True
            break
    if not successful:
        return ["No post-patch validation command completed with exit code 0."]
    return []
