"""Configuration: everything tunable comes from the environment, and the
mock worlds follow whatever `AGENT_NOW` is set to."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pytest

from incident_agent.config import Settings, format_iso, load_settings, resolve_now
from incident_agent.prompts import Prompts
from incident_agent.session import Session
from incident_agent.tools.executor import Batch, Budget, ToolCall, ToolExecutor
from incident_agent.tools.mock_backend import MockBackend, shift_for
from incident_agent.tools.time_resolver import resolve
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
        "AGENT_WORLD": "healthy",
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
    assert settings.world == "healthy"
    assert (settings.max_rows, settings.max_window_days) == (7, 2)
    assert settings.budgets == type(settings.budgets)(4, 5, 2, 0.5, 3, 2)
    assert (settings.detection.spike_multiplier, settings.detection.min_metric_points) == (1.5, 9)


def test_defaults_apply_when_a_variable_is_unset_or_blank(env):
    env.setenv("AGENT_MAX_LLM_STEPS", "   ")
    settings = load_settings()
    assert settings.budgets.max_llm_steps == 10
    assert settings.reasoning_effort is None  # not sent to the API unless set


def test_the_cost_bound_follows_the_step_budget():
    assert Settings(now=NOW).budgets.max_llm_calls == 11


# -- the clock -------------------------------------------------------------


def test_a_pinned_now_is_used_verbatim():
    assert format_iso(resolve_now("2026-09-23T10:00:00Z")) == "2026-09-23T10:00:00Z"


def test_auto_means_the_system_clock():
    assert abs(resolve_now("auto") - datetime.now(timezone.utc)) < timedelta(seconds=5)


def test_a_pinned_now_leaves_the_fixtures_exactly_as_written():
    assert shift_for("incident", NOW) == timedelta(0)


def test_the_world_follows_the_clock_so_yesterday_always_finds_the_incident():
    """The point of rebasing: AGENT_NOW=auto must still hit the fixture data."""
    now = datetime.now(timezone.utc).replace(hour=10, minute=0, second=0, microsecond=0)
    settings = Settings(now=now)
    window = resolve("yesterday between 2 PM and 4 PM", now)

    session = Session()
    session.turn = 1
    executor = ToolExecutor(settings, MockBackend("incident", now=now))
    observation = executor.run(
        ToolCall("c", "get_metrics", {
            "service": "checkout-api", "metric": "error_rate",
            "start_time": format_iso(window.start), "end_time": format_iso(window.end)}),
        session, Budget(settings.budgets), Batch())

    assert observation.status == "ok"
    assert "spike starts" in observation.summary
    assert "peak 0.1720 (17.20%)" in observation.summary  # identical numbers to the pinned run


def test_a_rebased_deployment_keeps_its_clock_time():
    now = datetime.now(timezone.utc)
    backend = MockBackend("incident", now=now)
    deployed = backend.world["deployments"][-1]["deployed_at"]
    assert deployed.endswith("T14:32:00Z")
    assert deployed.startswith((now - timedelta(days=1)).date().isoformat())


# -- prompts ---------------------------------------------------------------


def test_prompts_are_read_from_files_and_filled_in():
    prompts = Prompts(Settings(now=NOW))
    assert "Current time: 2026-09-23T10:00:00Z." in prompts.system
    assert "checkout-api" in prompts.system
    assert "$now" not in prompts.system
    assert "obs_404" in prompts.submit_invalid(["obs_404 does not exist"])


def test_prompts_can_be_pointed_somewhere_else(tmp_path):
    for name in ("system", "submit_not_alone", "submit_invalid", "force_final", "stopped_early"):
        (tmp_path / f"{name}.md").write_text(f"{name} for $now", encoding="utf-8")
    prompts = Prompts(Settings(now=NOW, prompts_dir=tmp_path))
    assert prompts.system == "system for 2026-09-23T10:00:00Z"
    assert prompts.stopped_early == "stopped_early for $now"


def test_a_missing_prompt_file_names_what_is_expected(tmp_path):
    with pytest.raises(FileNotFoundError, match="system.md"):
        Prompts(Settings(now=NOW, prompts_dir=tmp_path))
