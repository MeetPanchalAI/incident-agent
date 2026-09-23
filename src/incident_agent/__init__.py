"""An AI agent that investigates production incidents using tools."""

from __future__ import annotations

from .agent import AgentService, TurnResult
from .config import Settings, load_settings
from .llm import FakeLLM, LLMClient, OpenAIClient
from .session import Session
from .tools.mock_backend import MockBackend

__all__ = [
    "AgentService",
    "TurnResult",
    "Settings",
    "load_settings",
    "Session",
    "MockBackend",
    "FakeLLM",
    "LLMClient",
    "OpenAIClient",
    "build_service",
]


def build_service(
    settings: Settings,
    world: str | None = None,
    faults: dict[str, str] | None = None,
    llm: LLMClient | None = None,
) -> AgentService:
    """Assemble an agent. Pass `llm` to use a fake client instead of the real one."""
    backend = MockBackend(world or settings.world, faults, now=settings.now)
    return AgentService(settings, llm or OpenAIClient(settings), backend)
