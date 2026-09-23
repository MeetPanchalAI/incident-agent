"""The HTTP layer. It carries no logic of its own, so these tests are thin."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from incident_agent import api
from incident_agent.config import Settings
from incident_agent.llm import FakeLLM
from tests.conftest import NOW, WINDOW, submit

METRICS = ("get_metrics", {"service": "checkout-api", "metric": "error_rate", **WINDOW})
DONE = ("submit_response", submit(message="checkout-api looks normal."))


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api, "settings", Settings(now=NOW))
    monkeypatch.setattr(api, "LLM_OVERRIDE", FakeLLM([[METRICS], [DONE], [DONE], [DONE]]))
    api._sessions.clear()
    return TestClient(api.app)


def test_health_lists_the_available_worlds(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert "incident" in body["worlds"]


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
    assert body["session_id"]


def test_a_session_keeps_its_evidence_across_requests(client):
    first = client.post("/api/chat", json={"message": "first"}).json()
    second = client.post("/api/chat", json={"message": "second", "session_id": first["session_id"]}).json()
    assert second["session_id"] == first["session_id"]
    assert second["trace"] == []  # answered from the ledger, no new tool calls


def test_reset_starts_a_new_session(client):
    first = client.post("/api/chat", json={"message": "first"}).json()
    fresh = client.post("/api/reset", json={}).json()
    assert fresh["session_id"] != first["session_id"]


def test_an_unknown_world_is_refused(client):
    response = client.post("/api/reset", json={"world": "atlantis"})
    assert response.status_code == 400
    assert "atlantis" in response.json()["detail"]


def test_an_empty_message_is_refused(client):
    assert client.post("/api/chat", json={"message": ""}).status_code == 422
