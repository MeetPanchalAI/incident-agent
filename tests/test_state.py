"""Run logs: every workflow records what it did, and nothing more."""

from __future__ import annotations

import json

import pytest

from incident_agent.state import Recorder, recent_runs, run_logs
from incident_agent.tools.ingest import IngestError, ingest
from tests.conftest import WINDOW, service, submit

METRICS = ("get_metrics", {"service": "checkout-api", "metric": "error_rate", **WINDOW})
BAD = ("get_deployments", {"service": "not-a-service", **WINDOW})
DONE = ("submit_response", submit())


@pytest.fixture
def db(tmp_path):
    return str(tmp_path / "logs.db")


def events(db, run_id) -> list[str]:
    return [line["event"] for line in run_logs(db, run_id)]


# -- the recorder ----------------------------------------------------------


def test_a_run_records_its_events_in_order(db):
    log = Recorder(db, "turn", "why did checkout-api fail?")
    log.event("model.step", "get_metrics", step=1)
    log.event("tool.call", "get_metrics(checkout-api) -> ok", observation="obs_001")
    log.finish("answer · 1 tool call")

    [run] = recent_runs(db)
    assert (run["kind"], run["status"], run["label"]) == ("turn", "ok", "why did checkout-api fail?")
    assert run["duration_ms"] is not None
    assert [line["seq"] for line in run_logs(db, log.id)] == [1, 2]
    assert run_logs(db, log.id)[1]["data"] == {"observation": "obs_001"}


def test_runs_come_back_newest_first_and_can_be_filtered(db):
    Recorder(db, "ingest", "a.jsonl").finish("done")
    Recorder(db, "turn", "q").finish("done")
    assert [r["kind"] for r in recent_runs(db)] == ["turn", "ingest"]
    assert [r["kind"] for r in recent_runs(db, kind="ingest")] == ["ingest"]


def test_a_failed_run_keeps_its_status(db):
    Recorder(db, "ingest", "bad.jsonl").finish("refused", status="error")
    assert recent_runs(db)[0]["status"] == "error"


# -- ingest ----------------------------------------------------------------


def _line(**overrides) -> str:
    return json.dumps({"ts": "2026-09-22T14:00:00Z", "service": "api", "level": "INFO",
                       "message": "ok", **overrides})


def test_an_ingest_records_what_it_parsed_and_derived(db, tmp_path):
    ingest([_line(), _line(target="db"), "{not json"], tmp_path / "d.db", "logs.jsonl", db)
    [run] = recent_runs(db)
    assert run["kind"] == "ingest" and run["status"] == "ok"
    assert "2 events, 1 skipped" in run["summary"]
    assert events(db, run["id"]) == ["ingest.start", "ingest.parsed", "ingest.derived"]
    parsed = run_logs(db, run["id"])[1]
    assert parsed["level"] == "warn"  # because lines were skipped
    assert parsed["data"]["problems"]


def test_a_refused_ingest_is_recorded_as_an_error(db, tmp_path):
    with pytest.raises(IngestError):
        ingest(["{not json"], tmp_path / "d.db", "bad.jsonl", db)
    [run] = recent_runs(db)
    assert run["status"] == "error"
    assert "ingest.failed" in events(db, run["id"])


def test_ingest_without_a_log_database_still_works(tmp_path):
    report = ingest([_line()], tmp_path / "d.db", "logs.jsonl")
    assert report.events == 1


# -- agent turns -----------------------------------------------------------


def test_a_turn_records_each_step_and_each_tool_call(settings, state_db):
    agent = service(settings, [[METRICS], [DONE]])
    agent.run_turn(agent.new_session(), "why did checkout-api fail?")

    run = recent_runs(state_db, kind="turn")[0]
    assert run["status"] == "ok"
    assert events(state_db, run["id"]) == ["turn.start", "model.step", "tool.call", "model.step", "turn.done"]
    tool = next(line for line in run_logs(state_db, run["id"]) if line["event"] == "tool.call")
    assert tool["message"].startswith("#1 get_metrics(checkout-api, error_rate, 14:00-16:00) -> ok")
    assert "spike starts" in tool["message"]  # what the call actually found
    assert (tool["data"]["order"], tool["data"]["step"], tool["data"]["observation"]) == (1, 1, "obs_001")
    assert tool["data"]["ms"] >= 0


def test_model_steps_say_how_many_calls_were_asked_for(settings, state_db):
    from tests.conftest import WINDOW as W
    deploys = ("get_deployments", {"service": "checkout-api", **W})
    agent = service(settings, [[METRICS, deploys], [DONE]])
    agent.run_turn(agent.new_session(), "investigate")
    run = recent_runs(state_db, kind="turn")[0]
    step = next(line for line in run_logs(state_db, run["id"]) if line["event"] == "model.step")
    assert step["message"] == "step 1/10: 2 call(s) - get_metrics, get_deployments"
    orders = [line["data"]["order"] for line in run_logs(state_db, run["id"]) if line["event"] == "tool.call"]
    assert orders == [1, 2]


def test_a_failing_tool_call_is_logged_as_a_warning(settings, state_db):
    agent = service(settings, [[BAD], [DONE]])
    agent.run_turn(agent.new_session(), "check a service that is not there")
    run = recent_runs(state_db, kind="turn")[0]
    tool = next(line for line in run_logs(state_db, run["id"]) if line["event"] == "tool.call")
    assert tool["level"] == "warn"
    assert "invalid_arguments" in tool["message"]


def test_stopping_early_says_why(settings, state_db):
    from incident_agent.config import Budgets

    agent = service(settings, [[METRICS], [METRICS], [DONE]], budgets=Budgets(max_llm_steps=2))
    agent.run_turn(agent.new_session(), "keep going")
    run = recent_runs(state_db, kind="turn")[0]
    forced = next(line for line in run_logs(state_db, run["id"]) if line["event"] == "turn.forced_final")
    assert "step budget spent" in forced["message"]


def test_the_summary_says_what_the_turn_produced(settings, state_db):
    agent = service(settings, [[METRICS], [DONE]])
    agent.run_turn(agent.new_session(), "anything wrong?")
    run = recent_runs(state_db, kind="turn")[0]
    assert run["summary"] == "answer · 1 tool calls · 2 model calls"


def test_a_crash_inside_a_turn_is_recorded_before_it_propagates(settings, state_db):
    agent = service(settings, [])  # an exhausted script raises
    with pytest.raises(AssertionError):
        agent.run_turn(agent.new_session(), "boom")
    run = recent_runs(state_db, kind="turn")[0]
    assert run["status"] == "error"
    assert "turn.failed" in events(state_db, run["id"])


def test_an_unexpected_ingest_failure_is_recorded_too(db, tmp_path, monkeypatch):
    """Only a refusal used to close the run; anything else left it open."""
    import incident_agent.tools.ingest as module

    monkeypatch.setattr(module, "_derive", lambda db: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError):
        ingest([_line()], tmp_path / "d.db", "logs.jsonl", db)
    [run] = recent_runs(db)
    assert run["status"] == "error" and run["ended_at"] is not None
    assert "failed: RuntimeError" in run["summary"]
