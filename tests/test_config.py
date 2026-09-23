"""Configuration: everything tunable comes from the environment, and the
agent's clock comes from the dataset unless it is pinned."""

from __future__ import annotations

import os

import pytest

from incident_agent import build_service
from incident_agent.config import Settings, format_iso, load_settings, resolve_now
from incident_agent.llm import FakeLLM
from incident_agent.prompts import Prompts
from incident_agent.tools.store import Store
from tests.conftest import NOW


@pytest.fixture
def env(monkeypatch):
    for name in list(os.environ):
        if name.startswith("AGENT_"):
            monkeypatch.delenv(name, raising=False)
    return monkeypatch


# -- environment -----------------------------------------------------------


def test_every_tunable_parameter_is_read_from_the_environment(env):
    for name, value in {
        "AGENT_NOW": "2025-01-02T03:04:05Z",
        "AGENT_MODEL": "some-model",
        "AGENT_TEMPERATURE": "0.3",
        "AGENT_REASONING_EFFORT": "high",
        "AGENT_DB_PATH": "somewhere/else.db",
        "AGENT_MAX_LLM_STEPS": "4",
        "AGENT_MAX_TOOL_CALLS": "5",
        "AGENT_STUCK_THRESHOLD": "2",
        "AGENT_TOOL_TIMEOUT_S": "0.5",
        "AGENT_TOOL_RETRIES": "3",
        "AGENT_REPAIR_ATTEMPTS": "2",
        "AGENT_MAX_ROWS": "7",
        "AGENT_MAX_WINDOW_DAYS": "2",
        "AGENT_SPIKE_MULTIPLIER": "1.5",
        "AGENT_MIN_METRIC_POINTS": "9",
    }.items():
        env.setenv(name, value)

    settings = load_settings()
    assert format_iso(settings.now) == "2025-01-02T03:04:05Z"
    assert (settings.model, settings.temperature, settings.reasoning_effort) == ("some-model", 0.3, "high")
    assert settings.db_path.as_posix().endswith("somewhere/else.db")
    assert (settings.max_rows, settings.max_window_days) == (7, 2)
    assert settings.budgets == type(settings.budgets)(4, 5, 2, 0.5, 3, 2)
    assert (settings.detection.spike_multiplier, settings.detection.min_metric_points) == (1.5, 9)


def test_defaults_apply_when_a_variable_is_unset_or_blank(env):
    env.setenv("AGENT_MAX_LLM_STEPS", "   ")
    settings = load_settings()
    assert settings.budgets.max_llm_steps == 10
    assert settings.reasoning_effort is None  # not sent to the API unless set


# -- the clock -------------------------------------------------------------


def test_a_pinned_now_is_used_verbatim():
    assert format_iso(resolve_now("2026-09-23T10:00:00Z")) == "2026-09-23T10:00:00Z"


@pytest.mark.parametrize("value", ["data", "", "  "])
def test_data_means_take_the_clock_from_the_dataset(value):
    assert resolve_now(value) is None


def test_the_clock_defaults_to_the_last_event_in_the_dataset(dataset):
    agent = build_service(Settings(db_path=dataset), llm=FakeLLM([]))
    assert format_iso(agent.settings.now) == "2026-09-22T15:59:55Z"
    assert format_iso(Store(dataset).now()) == "2026-09-22T15:59:55Z"


def test_a_pinned_clock_overrides_the_dataset(dataset):
    agent = build_service(Settings(now=NOW, db_path=dataset), llm=FakeLLM([]))
    assert format_iso(agent.settings.now) == "2026-09-22T16:00:00Z"


# -- prompts ---------------------------------------------------------------


def test_the_system_prompt_describes_the_dataset_it_can_actually_query(dataset):
    agent = build_service(Settings(now=NOW, db_path=dataset), llm=FakeLLM([]))
    assert "Current time: 2026-09-22T16:00:00Z." in agent.prompts.system
    assert "checkout-api" in agent.prompts.system
    assert "sample.jsonl" in agent.prompts.system
    assert "$now" not in agent.prompts.system and "$dataset" not in agent.prompts.system


def test_the_system_prompt_says_when_there_is_no_data(tmp_path):
    agent = build_service(Settings(now=NOW, db_path=tmp_path / "empty.db"), llm=FakeLLM([]))
    assert "none ingested yet" in agent.prompts.system
    assert "no data has been ingested yet" in agent.prompts.system


def test_prompts_can_be_pointed_somewhere_else(tmp_path, dataset):
    for name in ("system", "submit_not_alone", "submit_invalid", "force_final", "stopped_early", "judge"):
        (tmp_path / f"{name}.md").write_text(f"{name} for $now", encoding="utf-8")
    prompts = Prompts(Settings(now=NOW, prompts_dir=tmp_path), ["checkout-api"], "sample")
    assert prompts.system == "system for 2026-09-22T16:00:00Z"
    assert prompts.stopped_early == "stopped_early for $now"


def test_a_missing_prompt_file_names_what_is_expected(tmp_path):
    with pytest.raises(FileNotFoundError, match="system.md"):
        Prompts(Settings(now=NOW, prompts_dir=tmp_path), [], "sample")


def test_every_prompt_the_code_asks_for_exists_as_a_file():
    """Prompts live in files, all of them; none is a literal in the code."""
    from incident_agent.config import PROMPTS_DIR
    from incident_agent.prompts import FILES, load

    assert set(FILES) == {p.stem for p in PROMPTS_DIR.glob("*.md")}
    assert all(load(PROMPTS_DIR, name) for name in FILES)
