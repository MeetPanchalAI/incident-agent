"""Argument validation. Nothing reaches the backend until it passes."""

from __future__ import annotations

import pytest

from tests.conftest import WINDOW, call


def test_unknown_service_is_rejected_and_the_error_lists_the_valid_names(executor, session):
    observation = call(executor, session, "get_deployments", {"service": "billing-api", **WINDOW})
    assert observation.status == "invalid_arguments"
    assert "checkout-api" in observation.summary


def test_unknown_metric_is_rejected_and_the_error_lists_the_valid_names(executor, session):
    observation = call(executor, session, "get_metrics",
                       {"service": "checkout-api", "metric": "cpu", **WINDOW})
    assert observation.status == "invalid_arguments"
    assert "error_rate" in observation.summary


def test_service_names_are_case_insensitive(executor, session):
    observation = call(executor, session, "get_service_dependencies", {"service": "Checkout-API"})
    assert observation.status == "ok"
    assert observation.args["service"] == "checkout-api"


@pytest.mark.parametrize(
    "start,end",
    [
        ("2026-09-22T16:00:00Z", "2026-09-22T14:00:00Z"),  # reversed
        ("2026-09-23T11:00:00Z", "2026-09-23T12:00:00Z"),  # in the future
        ("2026-09-01T00:00:00Z", "2026-09-22T00:00:00Z"),  # longer than seven days
        ("2026-09-22T14:00:00Z", "2026-09-22T14:00:00Z"),  # empty range
    ],
)
def test_invalid_time_ranges_are_rejected(executor, session, start, end):
    observation = call(executor, session, "get_deployments",
                       {"service": "checkout-api", "start_time": start, "end_time": end})
    assert observation.status == "invalid_arguments"
    assert "time range" in observation.summary.lower()


def test_unparseable_timestamp_is_rejected(executor, session):
    observation = call(executor, session, "get_deployments",
                       {"service": "checkout-api", "start_time": "yesterday", "end_time": "today"})
    assert observation.status == "invalid_arguments"


def test_over_long_query_is_rejected(executor, session):
    observation = call(executor, session, "search_logs",
                       {"service": "checkout-api", "query": "x" * 201, **WINDOW})
    assert observation.status == "invalid_arguments"


def test_unexpected_argument_is_rejected(executor, session):
    observation = call(executor, session, "get_service_dependencies",
                       {"service": "checkout-api", "depth": 2})
    assert observation.status == "invalid_arguments"


def test_unknown_tool_is_rejected(executor, session):
    observation = call(executor, session, "delete_everything", {})
    assert observation.status == "invalid_arguments"


def test_malformed_json_arguments_are_rejected(executor, session):
    from incident_agent.tools.executor import Batch, Budget, ToolCall

    bad = ToolCall("c1", "get_service_dependencies", {}, arguments_error="arguments were not valid JSON")
    observation = executor.run(bad, session, Budget(executor.settings.budgets), Batch())
    assert observation.status == "invalid_arguments"
    assert "JSON" in observation.summary


def test_a_rejected_call_still_costs_budget(executor, session, budget):
    call(executor, session, "get_deployments", {"service": "nope", **WINDOW}, budget=budget)
    assert budget.tool_calls_used == 1
