"""An AI agent that investigates production incidents using tools."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from .agent import AgentService, TurnResult
from .config import Settings, load_settings
from .llm import FakeLLM, LLMClient, OpenAIClient
from .session import Session
from .tools.store import Store

__all__ = [
    "AgentService", "TurnResult", "Settings", "load_settings", "Session",
    "Store", "FakeLLM", "LLMClient", "OpenAIClient", "build_service",
]


def build_service(
    settings: Settings,
    db_path: str | Path | None = None,
    faults: dict[str, str] | None = None,
    llm: LLMClient | None = None,
) -> AgentService:
    """Open the store and assemble an agent.

    `settings.now` of None means "the last event in the dataset"; it is filled
    in here, once the store is open. Pass `llm` to use a scripted fake.
    """
    store = Store(db_path or settings.db_path, faults)
    if settings.now is None:
        settings = replace(settings, now=store.now() or datetime.now(timezone.utc))
    return AgentService(settings, llm or OpenAIClient(settings), store)
