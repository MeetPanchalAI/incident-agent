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
    monkeypatch.setattr(api, "settings", Settings(now=NOW, db_path=own))
    monkeypatch.setattr(api, "LLM_OVERRIDE", FakeLLM([[METRICS], [DONE], [DONE], [DONE]]))
    api._reload()
    return TestClient(api.app)


@pytest.fixture
def blank(monkeypatch, tmp_path):
    """A client with no dataset ingested."""
    monkeypatch.setattr(api, "settings", Settings(now=NOW, db_path=tmp_path / "empty.db"))
    monkeypatch.setattr(api, "LLM_OVERRIDE", FakeLLM([]))
    api._reload()
    return TestClient(api.app)


def test_health_describes_the_loaded_dataset(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["dataset"]["filename"] == "sample.jsonl"
    assert "checkout-api" in body["services"]


def test_health_reports_no_dataset_when_nothing_is_loaded(blank):
    body = blank.get("/health").json()
    assert body["dataset"] is None
    assert body["services"] == []


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
