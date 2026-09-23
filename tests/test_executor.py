"""The guardrail pipeline: failures, retries, de-duplication, budgets, policy."""

from __future__ import annotations

import time

from incident_agent.config import Budgets, Settings
from incident_agent.tools.executor import Batch, Budget, ToolExecutor
from incident_agent.tools.mock_backend import MockBackend
from tests.conftest import WINDOW, call


def _executor(settings, world="incident", faults=None, **budget_overrides):
    tuned = Settings(now=settings.now, budgets=Budgets(**budget_overrides)) if budget_overrides else settings
    return ToolExecutor(tuned, MockBackend(world, faults))


# -- failures --------------------------------------------------------------


def test_timeout_is_retried_once_then_reported_and_cannot_be_cited(settings, session):
    executor = _executor(settings, faults={"get_metrics": "timeout"})
    observation = call(executor, session, "get_metrics",
                       {"service": "checkout-api", "metric": "error_rate", **WINDOW})
    assert observation.status == "timeout"
    assert observation.attempts == 2
    assert observation.citable is False


def test_a_transient_failure_succeeds_on_the_retry(settings, session):
    executor = _executor(settings, faults={"get_deployments": "transient"})
    observation = call(executor, session, "get_deployments", {"service": "checkout-api", **WINDOW})
    assert (observation.status, observation.attempts) == ("ok", 2)


def test_a_slow_backend_hits_the_real_timeout(settings, session, monkeypatch):
    executor = _executor(settings, tool_timeout_s=0.05, tool_retries=0)
    monkeypatch.setattr(executor.backend, "get_deployments", lambda *a: time.sleep(5))
    observation = call(executor, session, "get_deployments", {"service": "checkout-api", **WINDOW})
    assert observation.status == "timeout"


def test_a_malformed_payload_is_contained_and_never_reaches_the_model(settings, session):
    executor = _executor(settings, faults={"get_deployments": "malformed"})
    observation = call(executor, session, "get_deployments", {"service": "checkout-api", **WINDOW})
    assert observation.status == "error"
    assert observation.error["type"] == "malformed_response"
    assert observation.data is None
    assert observation.citable is False


def test_an_empty_result_is_a_fact_about_the_query_not_about_the_world(settings, session):
    executor = _executor(settings, faults={"search_logs": "empty"})
    observation = call(executor, session, "search_logs", {"service": "checkout-api", "query": "x", **WINDOW})
    assert observation.status == "empty"
    assert observation.citable is True
    assert "no matching records" in observation.summary
    assert "not proof that nothing happened" in observation.summary


def test_empty_and_error_are_different_statuses(settings, session):
    args = {"service": "checkout-api", "query": "x", **WINDOW}
    empty = call(_executor(settings, faults={"search_logs": "empty"}), session, "search_logs", args)
    session.turn = 2
    failed = call(_executor(settings, faults={"search_logs": "timeout"}), session, "search_logs",
                  {**args, "query": "y"})
    assert (empty.status, failed.status) == ("empty", "timeout")
    assert (empty.citable, failed.citable) == (True, False)


# -- de-duplication --------------------------------------------------------


def test_a_repeated_successful_call_is_not_run_again(executor, session):
    first = call(executor, session, "get_service_dependencies", {"service": "checkout-api"})
    second = call(executor, session, "get_service_dependencies", {"service": "checkout-api"})
    assert second.status == "duplicate"
    assert first.id in second.summary


def test_de_duplication_normalises_arguments(executor, session):
    call(executor, session, "search_logs", {"service": "checkout-api", "query": "Timeout", **WINDOW})
    repeat = call(executor, session, "search_logs", {"service": "CHECKOUT-API", "query": " timeout ", **WINDOW})
    assert repeat.status == "duplicate"


def test_a_different_window_is_not_a_duplicate(executor, session):
    call(executor, session, "get_deployments", {"service": "checkout-api", **WINDOW})
    narrowed = call(executor, session, "get_deployments",
                    {"service": "checkout-api",
                     "start_time": "2026-09-22T14:30:00Z", "end_time": "2026-09-22T14:45:00Z"})
    assert narrowed.status == "ok"


def test_a_failed_call_stays_retryable_on_a_later_turn(settings, session):
    executor = _executor(settings, faults={"get_metrics": "timeout"})
    args = {"service": "checkout-api", "metric": "error_rate", **WINDOW}
    first = call(executor, session, "get_metrics", args)
    session.turn = 2
    second = call(executor, session, "get_metrics", args)
    assert (first.status, second.status) == ("timeout", "timeout")
    assert second.status != "duplicate"


def test_the_same_call_twice_in_one_step_is_caught(executor, session):
    batch = Batch()
    args = {"service": "checkout-api", **WINDOW}
    call(executor, session, "get_deployments", args, batch=batch)
    assert call(executor, session, "get_deployments", args, batch=batch).status == "duplicate"


# -- budgets ---------------------------------------------------------------


def test_calls_past_the_budget_are_refused_not_executed(settings, session):
    executor = _executor(settings, max_tool_calls=2)
    budget = Budget(executor.settings.budgets)
    for index in range(4):
        observation = call(executor, session, "get_metrics",
                           {"service": "checkout-api", "metric": "error_rate",
                            "start_time": f"2026-09-22T1{index}:00:00Z", "end_time": "2026-09-22T16:00:00Z"},
                           budget=budget)
    assert observation.status == "budget_exceeded"
    assert budget.tool_calls_used == 2


def test_unproductive_results_lead_to_stuck(executor, session, budget):
    for _ in range(3):
        call(executor, session, "get_deployments", {"service": "nope", **WINDOW}, budget=budget)
    assert budget.stuck() is True


def test_a_productive_result_clears_the_stuck_counter(executor, session, budget):
    call(executor, session, "get_deployments", {"service": "nope", **WINDOW}, budget=budget)
    call(executor, session, "get_service_dependencies", {"service": "checkout-api"}, budget=budget)
    assert budget.unproductive_streak == 0


# -- note policy -----------------------------------------------------------


def _note(executor, session, evidence, budget=None):
    return call(executor, session, "create_incident_note",
                {"title": "t", "summary": "s", "evidence": evidence, "recommended_actions": []},
                budget=budget)


def test_a_note_may_only_cite_observations_that_succeeded(executor, session):
    ok = call(executor, session, "get_service_dependencies", {"service": "checkout-api"})
    assert _note(executor, session, [ok.id]).status == "ok"


def test_a_note_citing_a_failed_observation_is_refused(settings, session):
    executor = _executor(settings, faults={"get_metrics": "timeout"})
    failed = call(executor, session, "get_metrics",
                  {"service": "checkout-api", "metric": "error_rate", **WINDOW})
    refused = _note(executor, session, [failed.id])
    assert refused.status == "invalid_arguments"
    assert failed.id in refused.summary


def test_a_note_citing_an_unknown_observation_is_refused(executor, session):
    assert _note(executor, session, ["obs_999"]).status == "invalid_arguments"


def test_only_one_note_per_turn(executor, session):
    ok = call(executor, session, "get_service_dependencies", {"service": "checkout-api"})
    assert _note(executor, session, [ok.id]).status == "ok"
    second = _note(executor, session, [ok.id])
    assert second.status == "invalid_arguments"
    assert "already been created" in second.summary


def test_the_note_limit_resets_on_the_next_turn(executor, session):
    ok = call(executor, session, "get_service_dependencies", {"service": "checkout-api"})
    _note(executor, session, [ok.id])
    session.turn = 2
    assert _note(executor, session, [ok.id]).status == "ok"
