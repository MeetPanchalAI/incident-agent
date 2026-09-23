"""The evaluation harness itself, so a broken check cannot pass silently."""

from __future__ import annotations

from evals.scenarios import SCENARIOS, Run, data_calls, no_failed_call_cited, notes, used_a_prior_result
from tests.conftest import WINDOW, service, submit

GATEWAY_WINDOW = {"service": "payment-gateway", "metric": "latency_p95_ms", **WINDOW}


def make_run(settings, script, world="incident", faults=None, prompt="why?") -> Run:
    agent = service(settings, script, world=world, faults=faults)
    session = agent.new_session()
    return Run(session=session, result=agent.run_turn(session, prompt))


def test_every_scenario_is_distinct_and_covers_the_brief():
    assert len(SCENARIOS) >= 10
    assert len({s.id for s in SCENARIOS}) == len(SCENARIOS)
    assert all(s.prompts and s.expected and s.checks for s in SCENARIOS)


def test_the_chaining_check_sees_a_service_learnt_from_dependencies(settings):
    script = [
        [("get_service_dependencies", {"service": "payment-service"})],
        [("get_metrics", GATEWAY_WINDOW)],
        [("submit_response", submit())],
    ]
    run = make_run(settings, script, world="upstream")
    assert used_a_prior_result()[1](run) is True


def test_the_chaining_check_sees_a_narrowed_window(settings):
    script = [
        [("get_metrics", {"service": "checkout-api", "metric": "error_rate", **WINDOW})],
        [("search_logs", {"service": "checkout-api", "query": "timeout",
                          "start_time": "2026-09-22T14:30:00Z", "end_time": "2026-09-22T15:00:00Z"})],
        [("submit_response", submit())],
    ]
    assert used_a_prior_result()[1](make_run(settings, script)) is True


def test_the_chaining_check_rejects_two_unrelated_calls(settings):
    script = [
        [("get_metrics", {"service": "checkout-api", "metric": "error_rate", **WINDOW}),
         ("get_deployments", {"service": "checkout-api", **WINDOW})],
        [("submit_response", submit())],
    ]
    assert used_a_prior_result()[1](make_run(settings, script)) is False


def test_the_citation_check_catches_a_cited_failure(session):
    """The loop should never produce this, so the check is tested directly."""
    from incident_agent.agent import TurnResult
    from incident_agent.report import FinalOutcome, FinalResponse
    from incident_agent.session import Observation

    session.record(Observation(id="obs_001", turn=1, tool="get_metrics", category="data",
                               args={}, status="timeout", summary="failed"))
    response = FinalResponse.model_validate(
        {"response_type": "answer", "message": "m",
         "observed_facts": [{"statement": "x", "evidence_ids": ["obs_001"]}]})
    run = Run(session=session, result=TurnResult(outcome=FinalOutcome(response=response)))
    assert no_failed_call_cited()[1](run) is False
    assert data_calls(1)[1](run) is True


def test_the_counting_checks_use_tool_categories(settings):
    script = [
        [("resolve_time_range", {"expression": "yesterday"}),
         ("get_service_dependencies", {"service": "checkout-api"})],
        [("submit_response", submit())],
    ]
    run = make_run(settings, script)
    assert data_calls(1)[1](run) is True  # resolve_time_range is internal, not data
    assert notes(0)[1](run) is True
