from __future__ import annotations

import json
import os
import random
import time
from typing import Any

from openai import OpenAI

from nano_harness.config import ModelConfig
from nano_harness.types import ModelReply


class OpenRouterClient:
    def __init__(self, config: ModelConfig):
        api_key = os.getenv(config.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"{config.api_key_env} is not set; export it before running inference"
            )
        self.config = config
        self.client = OpenAI(
            api_key=api_key,
            base_url=config.base_url,
            timeout=config.timeout_seconds,
        )

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> ModelReply:
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries):
            try:
                kwargs: dict[str, Any] = {
                    "model": self.config.name,
                    "messages": messages,
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                }
                if tools:
                    kwargs["tools"] = tools
                    kwargs["tool_choice"] = "auto"
                response = self.client.chat.completions.create(**kwargs)
                message = response.choices[0].message
                raw_message = message.model_dump(exclude_none=True)
                tool_calls = [
                    call.model_dump(exclude_none=True)
                    for call in (message.tool_calls or [])
                ]
                reasoning = getattr(message, "reasoning", None) or raw_message.get(
                    "reasoning", ""
                )
                return ModelReply(
                    content=message.content or "",
                    reasoning=reasoning or "",
                    tool_calls=tool_calls,
                    usage=(
                        response.usage.model_dump(exclude_none=True)
                        if response.usage
                        else {}
                    ),
                    raw=response.model_dump(exclude_none=True),
                )
            except Exception as exc:
                last_error = exc
                if attempt + 1 >= self.config.max_retries:
                    break
                delay = min(30.0, (2**attempt) + random.random())
                time.sleep(delay)
        raise RuntimeError(f"model request failed: {last_error}") from last_error


class ScriptedClient:
    """Deterministic client used by unit tests and offline smoke runs."""

    def __init__(self, replies: list[ModelReply | dict[str, Any]]):
        self.replies = [
            reply if isinstance(reply, ModelReply) else ModelReply(**reply)
            for reply in replies
        ]
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> ModelReply:
        self.calls.append(
            {
                "messages": json.loads(json.dumps(messages)),
                "tools": json.loads(json.dumps(tools)) if tools else None,
            }
        )
        if not self.replies:
            raise RuntimeError("ScriptedClient exhausted")
        return self.replies.pop(0)
