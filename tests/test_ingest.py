"""Ingest: what it accepts, what it rejects, and what it derives."""

from __future__ import annotations

import json

import pytest

from incident_agent.config import parse_iso
from incident_agent.tools.ingest import IngestError, ingest
from incident_agent.tools.store import Store
from tests.sample_data import lines

START, END = parse_iso("2026-09-22T14:00:00Z"), parse_iso("2026-09-22T16:00:00Z")


def load(tmp_path, records) -> tuple[Store, object]:
    text = [r if isinstance(r, str) else json.dumps(r) for r in records]
    report = ingest(text, tmp_path / "t.db", "test.jsonl")
    return Store(tmp_path / "t.db"), report


def event(**overrides) -> dict:
    return {"ts": "2026-09-22T14:00:00Z", "service": "api", "level": "INFO",
            "message": "ok", **overrides}


# -- parsing ---------------------------------------------------------------


def test_a_well_formed_file_is_fully_ingested(tmp_path):
    _, report = load(tmp_path, [event(), event(ts="2026-09-22T14:01:00Z")])
    assert (report.events, report.skipped) == (2, 0)
    assert (report.first_ts, report.last_ts) == ("2026-09-22T14:00:00Z", "2026-09-22T14:01:00Z")


@pytest.mark.parametrize("bad,reason", [
    ("{not json", "expecting"),
    ('["a list"]', "not a JSON object"),
    (json.dumps({"service": "api", "level": "INFO", "message": "m"}), "missing ts"),
    (json.dumps({"ts": "2026-09-22T14:00:00Z", "level": "INFO", "message": "m"}), "missing service"),
    (json.dumps(event(level="SHOUTING")), "unknown level"),
])
def test_a_bad_line_is_skipped_counted_and_explained(tmp_path, bad, reason):
    _, report = load(tmp_path, [json.dumps(event()), bad])
    assert (report.events, report.skipped) == (1, 1)
    assert reason.lower() in report.problems[0].lower()


def test_blank_lines_are_ignored_not_counted_as_skipped(tmp_path):
    _, report = load(tmp_path, [json.dumps(event()), "", "   "])
    assert (report.events, report.skipped) == (1, 0)


def test_a_file_with_nothing_usable_is_refused(tmp_path):
    with pytest.raises(IngestError, match="No usable events"):
        load(tmp_path, ["{not json", "also not json"])


def test_services_and_levels_are_normalised(tmp_path):
    store, _ = load(tmp_path, [event(service="  Checkout-API  ", level="warning")])
    assert store.services() == ["checkout-api"]
    assert store.search_logs("checkout-api", START, END, "")[0]["level"] == "WARN"


def test_ingesting_again_replaces_the_previous_dataset(tmp_path):
    load(tmp_path, [event(service="old")])
    store, report = load(tmp_path, [event(service="new")])
    assert store.services() == ["new"]
    assert report.events == 1


# -- derivation ------------------------------------------------------------


def test_request_rate_counts_events_per_minute(tmp_path):
    store, _ = load(tmp_path, [event(ts=f"2026-09-22T14:00:{s:02d}Z") for s in range(5)])
    assert store.get_metrics("api", "request_rate", START, END) == [
        {"timestamp": "2026-09-22T14:00:00Z", "value": 5.0}]


def test_error_rate_is_errors_over_events_per_minute(tmp_path):
    records = [event(ts="2026-09-22T14:00:00Z", level="ERROR")] + [
        event(ts=f"2026-09-22T14:00:{s:02d}Z") for s in range(1, 4)]
    store, _ = load(tmp_path, records)
    assert store.get_metrics("api", "error_rate", START, END)[0]["value"] == 0.25


def test_a_service_that_never_reports_latency_has_no_latency_metric(tmp_path):
    store, report = load(tmp_path, [event(), event(service="other", latency_ms=42)])
    assert store.get_metrics("api", "latency_p95_ms", START, END) == []
    assert "latency_p95_ms" in report.metrics  # some service has it


def test_latency_p95_takes_the_top_of_the_minute(tmp_path):
    store, _ = load(tmp_path, [event(ts=f"2026-09-22T14:00:{s:02d}Z", latency_ms=v)
                               for s, v in enumerate(range(1, 21))])
    assert store.get_metrics("api", "latency_p95_ms", START, END)[0]["value"] == 19.0


def test_a_deployment_event_becomes_a_release_and_stays_a_log_line(tmp_path):
    store, report = load(tmp_path, [event(event_type="deployment", version="v9", status="success",
                                          message="deployment complete")])
    assert report.deployments == 1
    assert store.get_deployments("api", START, END) == [
        {"service": "api", "version": "v9", "deployed_at": "2026-09-22T14:00:00Z", "status": "success"}]
    assert len(store.search_logs("api", START, END, "deployment")) == 1


def test_a_call_to_another_service_becomes_a_dependency_edge(tmp_path):
    store, report = load(tmp_path, [event(target="db"), event(target="db"), event(service="db")])
    assert report.dependencies == 1
    assert store.get_service_dependencies("api")["upstream"] == ["db"]
    assert store.get_service_dependencies("db")["downstream"] == ["api"]


def test_a_service_only_ever_called_still_exists(tmp_path):
    store, _ = load(tmp_path, [event(target="never-logs-anything")])
    assert "never-logs-anything" in store.services()
    assert store.get_metrics("never-logs-anything", "request_rate", START, END) == []


# -- the sample dataset ----------------------------------------------------


def test_the_sample_dataset_contains_what_the_tests_rely_on(tmp_path):
    store, report = load(tmp_path, lines())
    assert report.skipped == 0
    assert set(report.services) >= {"checkout-api", "orders-db", "payment-service"}
    assert report.deployments == 1
    assert store.get_metrics("checkout-api", "error_rate", START, END)
    assert store.get_metrics("orders-db", "latency_p95_ms", START, END)
