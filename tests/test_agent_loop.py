"""The agent loop, driven by a scripted model. No API key, no network."""

from __future__ import annotations

import pytest

from incident_agent.config import Budgets
from incident_agent.prompts import STOPPED_EARLY_GAP
from tests.conftest import WINDOW, service, submit

DEPLOYS = ("get_deployments", {"service": "checkout-api", **WINDOW})
DEPS = ("get_service_dependencies", {"service": "checkout-api"})
METRICS = ("get_metrics", {"service": "checkout-api", "metric": "error_rate", **WINDOW})
BAD = ("get_deployments", {"service": "nope", **WINDOW})
DONE = ("submit_response", submit())


def unanswered(session) -> list[str]:
    """Tool call ids the loop failed to reply to."""
    requested = {c["id"] for m in session.messages for c in (m.get("tool_calls") or [])}
    return sorted(requested - {m["tool_call_id"] for m in session.messages if m["role"] == "tool"})


def run(settings, script, **kwargs):
    agent = service(settings, script, **kwargs)
    session = agent.new_session()
    return agent, session, agent.run_turn(session, "why did checkout-api fail?")


# -- the contract with the API --------------------------------------------


def test_every_tool_call_is_answered(settings):
    _, session, _ = run(settings, [[METRICS, DEPLOYS], [BAD], [DONE]])
    assert unanswered(session) == []


def test_every_tool_call_is_answered_even_when_rejected(settings):
    _, session, _ = run(settings, [[METRICS, DONE], [DONE]])
    assert unanswered(session) == []


# -- parallel calls in one step -------------------------------------------


def test_independent_calls_in_one_step_all_run_in_order(settings):
    _, _, result = run(settings, [[METRICS, DEPLOYS, DEPS], [DONE]])
    assert [t["observation_id"] for t in result.trace] == ["obs_001", "obs_002", "obs_003"]
    assert [t["tool"] for t in result.trace] == ["get_metrics", "get_deployments", "get_service_dependencies"]
    assert result.steps == 2


def test_the_tool_budget_cuts_a_batch_off_in_emission_order(settings):
    script = [[METRICS, DEPLOYS, DEPS], [DONE]]
    _, _, result = run(settings, script, budgets=Budgets(max_tool_calls=2))
    assert [t["status"] for t in result.trace] == ["ok", "ok", "budget_exceeded"]


# -- submit_response ------------------------------------------------------


def test_submit_alongside_other_tools_is_rejected_and_the_loop_continues(settings):
    _, session, result = run(settings, [[METRICS, DONE], [DONE]])
    assert result.steps == 2
    assert any("only call in a step" in m["content"] for m in session.messages if m["role"] == "tool")


def test_submit_is_never_blocked_by_the_tool_budget(settings):
    script = [[METRICS], [DEPLOYS], [DONE]]
    _, _, result = run(settings, script, budgets=Budgets(max_tool_calls=1))
    assert result.outcome.unverified is False
    assert result.stopped_early is False


# -- budgets and stopping -------------------------------------------------


def test_running_out_of_steps_still_produces_an_answer(settings):
    script = [[METRICS], [DEPLOYS], [DONE]]
    _, _, result = run(settings, script, budgets=Budgets(max_llm_steps=2))
    assert result.stopped_early is True
    assert STOPPED_EARLY_GAP in result.outcome.response.gaps
    assert result.llm_calls == 3  # two loop steps plus the forced final call


def test_three_unproductive_results_stop_the_loop_early(settings):
    _, _, result = run(settings, [[BAD], [BAD], [BAD], [DONE]])
    assert result.stopped_early is True
    assert result.steps == 3


def test_a_reply_with_no_tool_calls_counts_as_unproductive(settings):
    _, _, result = run(settings, [[], [], [], [DONE]])
    assert result.stopped_early is True


def test_the_number_of_model_calls_is_bounded(settings):
    limit = 3
    script = [[METRICS]] * 20 + [[DONE]]
    _, _, result = run(settings, script, budgets=Budgets(max_llm_steps=limit))
    assert result.llm_calls == limit + 1


# -- repair and the fallbacks ---------------------------------------------

INVENTED = ("submit_response", submit(observed_facts=[{"statement": "x", "evidence_ids": ["obs_404"]}]))


def test_an_invalid_final_response_gets_one_repair_attempt(settings):
    _, session, result = run(settings, [[INVENTED], [DONE]])
    assert result.outcome.unverified is False
    assert any("obs_404 does not exist" in m["content"] for m in session.messages if m["role"] == "tool")


def test_a_second_invalid_response_is_returned_unverified(settings):
    _, _, result = run(settings, [[INVENTED], [INVENTED]])
    assert result.outcome.unverified is True
    assert result.outcome.response.observed_facts == []


def test_an_unparseable_final_response_falls_back_to_the_ledger(settings):
    broken = ("submit_response", {"response_type": "nonsense"})
    _, _, result = run(settings, [[METRICS], [broken], [broken]])
    assert result.outcome.unverified is True
    assert [f.evidence_ids for f in result.outcome.response.observed_facts] == [["obs_001"]]


def test_a_forced_final_response_is_validated_too(settings):
    """The forced call gets no second chance: an invalid payload is returned unverified."""
    script = [[METRICS], [INVENTED], [INVENTED]]
    _, _, result = run(settings, script, budgets=Budgets(max_llm_steps=2))
    assert result.llm_calls == 3
    assert result.outcome.unverified is True
    assert STOPPED_EARLY_GAP in result.outcome.response.gaps


# -- conversation state ---------------------------------------------------


def test_a_follow_up_can_answer_from_the_existing_ledger(settings):
    agent = service(settings, [[METRICS, DEPLOYS], [DONE], [DONE]])
    session = agent.new_session()
    agent.run_turn(session, "why did checkout-api fail yesterday afternoon?")
    follow_up = agent.run_turn(session, "was there a deployment around then?")
    assert follow_up.trace == []
    assert follow_up.outcome.unverified is False


def test_a_follow_up_can_cite_an_observation_from_the_previous_turn(settings):
    cite = ("submit_response", submit(observed_facts=[{"statement": "v142 at 14:32", "evidence_ids": ["obs_002"]}]))
    agent = service(settings, [[METRICS, DEPLOYS], [DONE], [cite]])
    session = agent.new_session()
    agent.run_turn(session, "why did checkout-api fail yesterday afternoon?")
    follow_up = agent.run_turn(session, "was there a deployment around then?")
    assert follow_up.outcome.unverified is False
    assert follow_up.outcome.response.observed_facts[0].evidence_ids == ["obs_002"]


def test_budgets_reset_each_turn_but_the_ledger_persists(settings):
    agent = service(settings, [[METRICS], [DONE], [DEPLOYS], [DONE]])
    session = agent.new_session()
    agent.run_turn(session, "first")
    second = agent.run_turn(session, "second")
    assert second.steps == 2
    assert len(session.observations) == 2
    assert [t["observation_id"] for t in second.trace] == ["obs_002"]


def test_the_script_is_exhausted_if_the_loop_runs_longer_than_expected(settings):
    with pytest.raises(AssertionError):
        run(settings, [[METRICS]])
