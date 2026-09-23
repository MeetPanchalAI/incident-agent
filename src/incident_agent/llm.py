"""Adapter over the OpenAI Responses API, plus a scripted fake.

The Responses API rather than Chat Completions, because the model this agent
targets refuses function tools and reasoning together on Chat Completions:

    Function tools with reasoning_effort are not supported for gpt-5.6-luna in
    /v1/chat/completions. To use function tools, use /v1/responses or set
    reasoning_effort to 'none'.

Turning reasoning off was the alternative, and a poor one: `tool_choice` is
"required", so every reply is tool calls with no text content, and reasoning is
the only place this agent can deliberate between steps. See DESIGN.md.

The fake is what lets every loop and guardrail test run deterministically, with
no API key and no network.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Protocol

from .config import Settings
from .tools.executor import ToolCall


@dataclass
class LLMReply:
    """One model turn: the items to append to the conversation, and the calls to run."""

    items: list[dict]
    tool_calls: list[ToolCall]
    usage: dict = field(default_factory=dict)


class LLMClient(Protocol):
    def chat(self, items: list[dict], tools: list[dict], force: str | None = None) -> LLMReply: ...


def tool_result_item(call_id: str, payload: dict) -> dict:
    """The conversation item that answers one tool call."""
    return {"type": "function_call_output", "call_id": call_id, "output": json.dumps(payload, default=str)}


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

    def chat(self, items: list[dict], tools: list[dict], force: str | None = None) -> LLMReply:
        # reasoning is only sent when an effort is configured, so non-reasoning
        # models are unaffected. store=False keeps nothing server side, which is
        # why the reasoning items have to travel in the conversation instead.
        extra: dict = {}
        if self.settings.reasoning_effort:
            extra["reasoning"] = {"effort": self.settings.reasoning_effort}
            extra["include"] = ["reasoning.encrypted_content"]

        response = self._client.responses.create(
            model=self.model,
            input=items,
            tools=tools,
            tool_choice={"type": "function", "name": force} if force else "required",
            parallel_tool_calls=force is None,
            temperature=self.settings.temperature,
            store=False,
            **extra,
        )
        calls = [
            _parse(item.call_id, item.name, item.arguments)
            for item in response.output
            if item.type == "function_call"
        ]
        return LLMReply(
            items=[item.model_dump(exclude_none=True) for item in response.output],
            tool_calls=calls,
            usage=response.usage.model_dump() if response.usage else {},
        )


class FakeLLM:
    """Replays a script. Each step is a list of (tool_name, arguments) pairs."""

    def __init__(self, script: list[list[tuple[str, dict]]]) -> None:
        self.script = list(script)
        self.calls: list[list[dict]] = []

    def chat(self, items: list[dict], tools: list[dict], force: str | None = None) -> LLMReply:
        if not self.script:
            raise AssertionError("FakeLLM script is exhausted: the agent made more calls than expected.")
        step = self.script.pop(0)
        self.calls.append(items)
        tool_calls, emitted = [], []
        for index, (name, arguments) in enumerate(step):
            call_id = f"call_{len(self.calls)}_{index}"
            tool_calls.append(ToolCall(id=call_id, name=name, arguments=arguments))
            emitted.append({
                "type": "function_call",
                "call_id": call_id,
                "name": name,
                "arguments": json.dumps(arguments, default=str),
            })
        return LLMReply(items=emitted, tool_calls=tool_calls)
