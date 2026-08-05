from __future__ import annotations

import json
from typing import Any

from nano_harness.types import TaskResult


class TauEnvExecutor:
    """Bridge a tau-bench Env to NanoHarness's one-call-at-a-time executor."""

    def __init__(self, env: Any):
        self.env = env
        self.last_response = None

    @property
    def done(self) -> bool:
        return bool(self.last_response and self.last_response.done)

    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        from tau_bench.types import Action

        response = self.env.step(Action(name=name, kwargs=arguments))
        self.last_response = response
        payload = {
            "observation": response.observation,
            "reward": response.reward,
            "done": response.done,
            "info": response.info.model_dump(),
        }
        return json.dumps(payload, ensure_ascii=False)

    def respond(self, content: str) -> str:
        from tau_bench.types import Action, RESPOND_ACTION_NAME

        response = self.env.step(
            Action(name=RESPOND_ACTION_NAME, kwargs={"content": content})
        )
        self.last_response = response
        return response.observation


def make_tau_task(env: Any, task_index: int, wiki: str, tools_info: list[dict]) -> tuple:
    reset = env.reset(task_index=task_index)
    task = {
        "task_id": str(task_index),
        "benchmark": "taubench",
        "messages": [{"role": "user", "content": reset.observation}],
        "metadata": {
            "initial_info": reset.info.model_dump(),
            "constraints": [
                "Follow the domain policy exactly.",
                "Verify prerequisites before mutating state.",
                "Never report success without a confirming tool observation.",
            ],
        },
        "tools": tools_info,
    }
    from nano_harness.types import Task

    return Task(**task), TauEnvExecutor(env)


def serialize_tau_result(result: TaskResult, executor: TauEnvExecutor) -> dict:
    response = executor.last_response
    reward = response.reward if response is not None else 0.0
    info = response.info.model_dump() if response is not None else {}
    return {
        "task_id": int(result.task_id),
        "reward": reward,
        "info": info,
        "traj": result.trajectory,
        "trial": int(result.metadata.get("trial", 0)),
        "nano_harness": result.to_dict(),
    }
