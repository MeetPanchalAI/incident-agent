from __future__ import annotations

import pytest

from incident_agent import AgentService, FakeLLM
from incident_agent.config import Budgets, Settings, parse_iso
from incident_agent.session import Session
from incident_agent.tools.executor import Batch, Budget, ToolCall, ToolExecutor
from incident_agent.tools.ingest import ingest
from incident_agent.tools.store import Store
from tests.sample_data import lines

# The sample dataset runs 14:00 to 15:59:55 on 2026-09-22, so "now" sits just
# after its last event, as it would if the data had only just been collected.
NOW = parse_iso("2026-09-22T16:00:00Z")
WINDOW = {"start_time": "2026-09-22T14:00:00Z", "end_time": "2026-09-22T16:00:00Z"}


@pytest.fixture(scope="session")
def dataset(tmp_path_factory) -> str:
    """One ingested database, built once and shared by every test."""
    path = tmp_path_factory.mktemp("data") / "test.db"
    ingest(lines(), path, "sample.jsonl")
    return str(path)


@pytest.fixture
def settings(dataset: str) -> Settings:
    return Settings(now=NOW, db_path=dataset)


@pytest.fixture
def store(dataset: str) -> Store:
    return Store(dataset)


@pytest.fixture
def session() -> Session:
    created = Session()
    created.turn = 1
    return created


@pytest.fixture
def executor(settings: Settings, store: Store) -> ToolExecutor:
    return ToolExecutor(settings, store)


@pytest.fixture
def budget(settings: Settings) -> Budget:
    return Budget(settings.budgets)


def call(executor, session, name, args, budget=None, batch=None):
    """Run one tool call and return the resulting observation."""
    budget = budget or Budget(executor.settings.budgets)
    return executor.run(ToolCall(f"c{len(session.observations)}", name, args), session, budget, batch or Batch())


def service(settings: Settings, script, faults=None, budgets: Budgets | None = None):
    """An AgentService driven by a scripted fake model."""
    if budgets is not None:
        settings = Settings(now=settings.now, db_path=settings.db_path, budgets=budgets)
    return AgentService(settings, FakeLLM(script), Store(settings.db_path, faults))


def submit(**overrides) -> dict:
    """A minimal valid submit_response payload."""
    return {"response_type": "answer", "message": "done", **overrides}
