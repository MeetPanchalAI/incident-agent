from __future__ import annotations

import pytest

from incident_agent import AgentService, FakeLLM
from incident_agent.config import Budgets, Settings, parse_iso
from incident_agent.session import Session
from incident_agent.tools.executor import Batch, Budget, ToolCall, ToolExecutor
from incident_agent.tools.mock_backend import MockBackend

NOW = parse_iso("2026-09-23T10:00:00Z")
WINDOW = {"start_time": "2026-09-22T14:00:00Z", "end_time": "2026-09-22T16:00:00Z"}


@pytest.fixture
def settings() -> Settings:
    return Settings(now=NOW)


@pytest.fixture
def session() -> Session:
    created = Session()
    created.turn = 1
    return created


@pytest.fixture
def executor(settings: Settings) -> ToolExecutor:
    return ToolExecutor(settings, MockBackend("incident"))


@pytest.fixture
def budget(settings: Settings) -> Budget:
    return Budget(settings.budgets)


def call(executor, session, name, args, budget=None, batch=None):
    """Run one tool call and return the resulting observation."""
    budget = budget or Budget(executor.settings.budgets)
    return executor.run(ToolCall(f"c{len(session.observations)}", name, args), session, budget, batch or Batch())


def service(settings: Settings, script, world: str = "incident", faults=None, budgets: Budgets | None = None):
    """An AgentService driven by a scripted fake model."""
    if budgets is not None:
        settings = Settings(now=settings.now, budgets=budgets)
    return AgentService(settings, FakeLLM(script), MockBackend(world, faults, now=settings.now))


def submit(**overrides) -> dict:
    """A minimal valid submit_response payload."""
    return {"response_type": "answer", "message": "done", **overrides}
