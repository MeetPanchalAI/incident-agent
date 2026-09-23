"""The HTTP layer. It carries no logic of its own, so these tests are thin."""

from __future__ import annotations

import json
import pathlib

import pytest
from fastapi.testclient import TestClient

from incident_agent import api
from incident_agent.config import Settings
from incident_agent.llm import FakeLLM
from tests.conftest import NOW, WINDOW, submit
from tests.sample_data import lines

METRICS = ("get_metrics", {"service": "checkout-api", "metric": "error_rate", **WINDOW})
DONE = ("submit_response", submit(message="checkout-api looks normal."))


@pytest.fixture
def client(monkeypatch, tmp_path, dataset):
    # A private copy: these tests ingest, and must not disturb the shared dataset.
    own = tmp_path / "api.db"
    own.write_bytes(pathlib.Path(dataset).read_bytes())
    monkeypatch.setattr(api, "settings", Settings(now=NOW, db_path=own, state_db_path=tmp_path / "logs.db"))
    monkeypatch.setattr(api, "LLM_OVERRIDE", FakeLLM([[METRICS], [DONE], [DONE], [DONE]]))
    api._reload()
    return TestClient(api.app)


@pytest.fixture
def blank(monkeypatch, tmp_path):
    """A client with no dataset ingested."""
    monkeypatch.setattr(api, "settings",
                        Settings(now=NOW, db_path=tmp_path / "empty.db", state_db_path=tmp_path / "logs.db"))
    monkeypatch.setattr(api, "LLM_OVERRIDE", FakeLLM([]))
    api._reload()
    return TestClient(api.app)


def test_health_describes_the_loaded_dataset(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["dataset"]["filename"] == "sample.jsonl"
    assert "checkout-api" in body["services"]
    assert body["busiest"][0] == "checkout-api"  # the service with the most events


def test_health_reports_no_dataset_when_nothing_is_loaded(blank):
    body = blank.get("/health").json()
    assert body["dataset"] is None
    assert body["services"] == [] and body["busiest"] == []


def test_the_page_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Incident Agent" in response.text


def test_chat_returns_the_response_and_the_tool_trace(client):
    body = client.post("/api/chat", json={"message": "anything wrong with checkout-api?"}).json()
    assert body["response"]["message"] == "checkout-api looks normal."
    assert body["response"]["unverified"] is False
    assert [t["tool"] for t in body["trace"]] == ["get_metrics"]
    assert body["llm_calls"] == 2
    assert "sample.jsonl" in body["dataset"]


def test_chat_is_refused_before_any_data_is_ingested(blank):
    response = blank.post("/api/chat", json={"message": "why did checkout-api fail?"})
    assert response.status_code == 409
    assert "ingested" in response.json()["detail"]


def test_a_session_keeps_its_evidence_across_requests(client):
    first = client.post("/api/chat", json={"message": "first"}).json()
    second = client.post("/api/chat", json={"message": "second", "session_id": first["session_id"]}).json()
    assert second["session_id"] == first["session_id"]
    assert second["trace"] == []  # answered from the ledger, no new tool calls


def test_reset_starts_a_new_session(client):
    first = client.post("/api/chat", json={"message": "first"}).json()
    assert client.post("/api/reset", json={}).json()["session_id"] != first["session_id"]


def test_an_empty_message_is_refused(client):
    assert client.post("/api/chat", json={"message": ""}).status_code == 422


# -- ingest ----------------------------------------------------------------


def _upload(client, text: str, name: str = "logs.jsonl"):
    return client.post("/api/ingest", files={"file": (name, text.encode("utf-8"), "application/json")})


def test_uploading_a_file_loads_it_and_reports_what_happened(blank):
    response = _upload(blank, "\n".join(lines()))
    assert response.status_code == 200
    report = response.json()
    assert report["skipped"] == 0
    assert "checkout-api" in report["services"]
    assert report["deployments"] == 1
    assert blank.get("/health").json()["dataset"]["filename"] == "logs.jsonl"


def test_a_bad_line_is_reported_rather_than_hidden(blank):
    good = json.dumps({"ts": "2026-09-22T14:00:00Z", "service": "api", "level": "INFO", "message": "ok"})
    report = _upload(blank, f"{good}\n{{not json").json()
    assert (report["events"], report["skipped"]) == (1, 1)
    assert report["problems"]


def test_an_unusable_file_is_refused(blank):
    response = _upload(blank, "not json at all\nstill not json")
    assert response.status_code == 400
    assert "No usable events" in response.json()["detail"]


def test_uploading_replaces_the_dataset_and_clears_sessions(client):
    first = client.post("/api/chat", json={"message": "first"}).json()
    good = json.dumps({"ts": "2026-09-22T14:00:00Z", "service": "api", "level": "INFO", "message": "ok"})
    assert _upload(client, good).status_code == 200
    assert client.get("/health").json()["services"] == ["api"]
    assert first["session_id"] not in api._sessions


def test_the_monitoring_endpoints_report_what_ran(client):
    client.post("/api/chat", json={"message": "anything wrong with checkout-api?"})
    runs = client.get("/api/runs").json()["runs"]
    assert [r["kind"] for r in runs] == ["turn"]
    logs = client.get(f"/api/runs/{runs[0]['id']}").json()["logs"]
    assert [line["event"] for line in logs][0] == "turn.start"
    assert any(line["event"] == "tool.call" for line in logs)


def test_an_upload_is_recorded_as_an_ingest_run(blank):
    good = json.dumps({"ts": "2026-09-22T14:00:00Z", "service": "api", "level": "INFO", "message": "ok"})
    _upload(blank, good)
    assert [r["kind"] for r in blank.get("/api/runs?kind=ingest").json()["runs"]] == ["ingest"]


# -- persistence -----------------------------------------------------------


def test_a_conversation_survives_a_server_restart(client):
    first = client.post("/api/chat", json={"message": "anything wrong with checkout-api?"}).json()
    api._sessions.clear()  # what a restart looks like from here

    second = client.post("/api/chat", json={"message": "and now?", "session_id": first["session_id"]}).json()
    assert second["session_id"] == first["session_id"]
    assert second["trace"] == []  # the evidence ledger came back with it


def test_a_conversation_can_be_replayed_from_the_database(client):
    first = client.post("/api/chat", json={"message": "what happened?"}).json()
    stored = client.get(f"/api/sessions/{first['session_id']}").json()
    assert [t["question"] for t in stored["turns"]] == ["what happened?"]
    assert stored["turns"][0]["response"]["message"] == "checkout-api looks normal."
    assert stored["dataset"] == "sample.jsonl"


def test_conversations_are_listed_newest_first(client):
    client.post("/api/chat", json={"message": "first question"})
    client.post("/api/reset", json={})
    client.post("/api/chat", json={"message": "second question"})
    titles = [s["title"] for s in client.get("/api/sessions").json()["sessions"]]
    assert titles[:2] == ["second question", "first question"]


def test_an_unknown_conversation_is_a_404(client):
    assert client.get("/api/sessions/nope").status_code == 404


def test_a_conversation_from_another_dataset_is_not_resumed(client):
    first = client.post("/api/chat", json={"message": "about the old data"}).json()
    good = json.dumps({"ts": "2026-09-22T14:00:00Z", "service": "api", "level": "INFO", "message": "ok"})
    _upload(client, good)
    second = client.post("/api/chat", json={"message": "about the new data",
                                            "session_id": first["session_id"]}).json()
    assert second["session_id"] != first["session_id"]


# -- evaluation endpoints --------------------------------------------------


def test_the_scenarios_are_listed_for_the_ui(client):
    scenarios = client.get("/api/evaluations/scenarios").json()["scenarios"]
    assert len(scenarios) == 12
    assert scenarios[0]["id"] == "E01" and scenarios[0]["question"]


def test_the_status_endpoint_reports_an_idle_suite(client):
    status = client.get("/api/evaluations/status").json()
    assert status["running"] is False and status["done"] == 0


def test_two_evaluations_cannot_run_at_once(client, monkeypatch):
    monkeypatch.setitem(api._evaluation, "running", True)
    assert client.post("/api/evaluations/run", json={}).status_code == 409


def test_past_evaluation_runs_are_listed(client):
    from incident_agent.state import save_eval_run

    save_eval_run(api.settings.state_db_path, "run1", "sample.jsonl", "m", True,
                  [{"scenario_id": "E01", "pass": True, "critical_error": False}], "2026-09-23T10:00:00Z")
    runs = client.get("/api/evaluations/runs").json()["runs"]
    assert runs[0]["id"] == "run1" and runs[0]["passed"] == 1
    assert client.get("/api/evaluations/runs/run1").json()["results"][0]["scenario_id"] == "E01"
