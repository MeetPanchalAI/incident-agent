"""Adapter over the OpenAI Chat Completions API, plus a scripted fake.

The fake is what lets every loop and guardrail test run deterministically,
with no API key and no network.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Protocol

from .config import Settings
from .tools.executor import ToolCall


@dataclass
class LLMReply:
    message: dict
    tool_calls: list[ToolCall]
    usage: dict = field(default_factory=dict)


class LLMClient(Protocol):
    def chat(self, messages: list[dict], tools: list[dict], force: str | None = None) -> LLMReply: ...


def _parse(call_id: str, name: str, raw_arguments: str) -> ToolCall:
    try:
        arguments = json.loads(raw_arguments or "{}")
        if not isinstance(arguments, dict):
            raise ValueError("arguments must be a JSON object")
        return ToolCall(id=call_id, name=name, arguments=arguments)
    except (json.JSONDecodeError, ValueError) as exc:
        return ToolCall(id=call_id, name=name, arguments={}, arguments_error=f"arguments were not valid JSON: {exc}")


class OpenAIClient:
    """Thin wrapper. `force` names the one tool the model must call."""

    def __init__(self, settings: Settings) -> None:
        from openai import OpenAI

        if not settings.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in.")
        self._client = OpenAI(api_key=settings.api_key)
        self.settings = settings
        self.model = settings.model

    def chat(self, messages: list[dict], tools: list[dict], force: str | None = None) -> LLMReply:
        # reasoning_effort is only sent when set, so non-reasoning models are unaffected.
        extra = {"reasoning_effort": self.settings.reasoning_effort} if self.settings.reasoning_effort else {}
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice={"type": "function", "function": {"name": force}} if force else "required",
            parallel_tool_calls=force is None,
            temperature=self.settings.temperature,
            **extra,
        )
        message = response.choices[0].message
        calls = [_parse(c.id, c.function.name, c.function.arguments) for c in (message.tool_calls or [])]
        usage = response.usage.model_dump() if response.usage else {}
        return LLMReply(message=message.model_dump(exclude_none=True), tool_calls=calls, usage=usage)


class FakeLLM:
    """Replays a script. Each step is a list of (tool_name, arguments) pairs."""

    def __init__(self, script: list[list[tuple[str, dict]]]) -> None:
        self.script = list(script)
        self.calls: list[list[dict]] = []

    def chat(self, messages: list[dict], tools: list[dict], force: str | None = None) -> LLMReply:
        if not self.script:
            raise AssertionError("FakeLLM script is exhausted: the agent made more calls than expected.")
        step = self.script.pop(0)
        self.calls.append(messages)
        tool_calls, payload = [], []
        for index, (name, arguments) in enumerate(step):
            call_id = f"call_{len(self.calls)}_{index}"
            tool_calls.append(ToolCall(id=call_id, name=name, arguments=arguments))
            payload.append({
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments, default=str)},
            })
        return LLMReply(message={"role": "assistant", "content": None, "tool_calls": payload}, tool_calls=tool_calls)
