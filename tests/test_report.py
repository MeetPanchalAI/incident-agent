"""The final response: citation checks, confidence limits, and the fallbacks."""

from __future__ import annotations

from incident_agent.report import (
    FinalResponse,
    apply_confidence_limits,
    build_from_ledger,
    finalize,
    finalize_unverified,
    validate,
)
from incident_agent.session import Observation


def obs(session, tool="get_metrics", status="ok", category="data", data=None) -> Observation:
    return session.record(Observation(
        id=session.next_observation_id(), turn=session.turn, tool=tool, category=category,
        args={"n": len(session.observations)}, status=status, summary=f"{tool} summary", data=data,
    ))


def payload(**overrides) -> dict:
    return {"response_type": "investigation_report", "message": "m", **overrides}


def hypothesis(supporting, confidence="high", **overrides) -> dict:
    return {"statement": "s", "supporting_evidence_ids": supporting, "confidence": confidence, **overrides}


# -- citations -------------------------------------------------------------


def test_a_fact_citing_an_unknown_observation_is_rejected(session):
    _, errors = validate(payload(observed_facts=[{"statement": "x", "evidence_ids": ["obs_404"]}]), session)
    assert any("obs_404 does not exist" in e for e in errors)


def test_a_fact_citing_a_failed_observation_is_rejected(session):
    failed = obs(session, status="timeout")
    _, errors = validate(
        payload(observed_facts=[{"statement": "x", "evidence_ids": [failed.id]}], gaps=["get_metrics timed out"]),
        session,
    )
    assert any("cannot be cited" in e for e in errors)


def test_a_fact_with_no_evidence_is_rejected_by_the_schema(session):
    _, errors = validate(payload(observed_facts=[{"statement": "x", "evidence_ids": []}]), session)
    assert errors


def test_an_empty_observation_may_be_cited(session):
    nothing = obs(session, tool="get_deployments", status="empty")
    response, errors = validate(
        payload(observed_facts=[{"statement": "no deployments found", "evidence_ids": [nothing.id]}]), session)
    assert errors == [] and response is not None


# -- failures must be disclosed --------------------------------------------


def test_an_undisclosed_failure_is_rejected(session):
    obs(session, status="timeout")
    _, errors = validate(payload(), session)
    assert any("get_metrics failed" in e for e in errors)


def test_a_failure_listed_in_gaps_is_accepted(session):
    obs(session, status="timeout")
    _, errors = validate(payload(gaps=["get_metrics timed out and was not retried further"]), session)
    assert errors == []


def test_a_failure_listed_in_missing_evidence_is_accepted(session):
    obs(session, status="timeout")
    _, errors = validate(
        payload(hypotheses=[hypothesis([], confidence="low", missing_evidence=["get_metrics did not respond"])]),
        session,
    )
    assert errors == []


# -- confidence limits -----------------------------------------------------


def capped(session, hypotheses) -> tuple[list[str], list[str]]:
    response = FinalResponse.model_validate(payload(hypotheses=hypotheses, likely_cause="something"))
    notes = apply_confidence_limits(response, session)
    return [h.confidence for h in response.hypotheses], notes


def test_a_hypothesis_with_no_supporting_evidence_is_low(session):
    levels, notes = capped(session, [hypothesis([])])
    assert levels == ["low"]
    assert "no supporting evidence" in notes[0]


def test_one_data_tool_is_not_enough_for_high(session):
    first, second = obs(session, tool="get_metrics"), obs(session, tool="get_metrics")
    levels, notes = capped(session, [hypothesis([first.id, second.id])])
    assert levels == ["medium"]
    assert "fewer than two independent data tools" in notes[0]


def test_two_independent_data_tools_allow_high(session):
    metrics, deploys = obs(session, tool="get_metrics"), obs(session, tool="get_deployments")
    assert capped(session, [hypothesis([metrics.id, deploys.id])])[0] == ["high"]


def test_an_empty_result_does_not_count_towards_high(session):
    metrics = obs(session, tool="get_metrics")
    nothing = obs(session, tool="get_deployments", status="empty")
    assert capped(session, [hypothesis([metrics.id, nothing.id])])[0] == ["medium"]


def test_an_internal_tool_does_not_count_as_a_data_tool(session):
    metrics = obs(session, tool="get_metrics")
    clock = obs(session, tool="resolve_time_range", category="internal")
    assert capped(session, [hypothesis([metrics.id, clock.id])])[0] == ["medium"]


def test_any_failed_call_this_turn_caps_confidence_at_medium(session):
    metrics, deploys = obs(session, tool="get_metrics"), obs(session, tool="get_deployments")
    obs(session, tool="search_logs", status="timeout")
    levels, notes = capped(session, [hypothesis([metrics.id, deploys.id])])
    assert levels == ["medium"]
    assert "a tool call failed this turn (search_logs)" in notes[0]


def test_the_failure_cap_is_read_from_the_ledger_not_from_the_payload(session):
    """Leaving missing_evidence empty must not buy a higher confidence."""
    metrics, deploys = obs(session, tool="get_metrics"), obs(session, tool="get_deployments")
    obs(session, tool="search_logs", status="timeout")
    silent = capped(session, [hypothesis([metrics.id, deploys.id], missing_evidence=[])])[0]
    honest = capped(session, [hypothesis([metrics.id, deploys.id], missing_evidence=["logs"])])[0]
    assert silent == honest == ["medium"]


def test_cited_contradicting_evidence_caps_confidence_at_medium(session):
    metrics, deploys = obs(session, tool="get_metrics"), obs(session, tool="get_deployments")
    against = obs(session, tool="search_logs")
    levels, notes = capped(session, [hypothesis([metrics.id, deploys.id],
                                                contradicting_evidence_ids=[against.id])])
    assert levels == ["medium"]
    assert "contradicting evidence" in notes[0]


def test_two_equally_supported_hypotheses_cannot_both_be_high(session):
    metrics, deploys, logs = (obs(session, tool=t) for t in ("get_metrics", "get_deployments", "search_logs"))
    levels, notes = capped(session, [hypothesis([metrics.id, deploys.id]), hypothesis([metrics.id, logs.id])])
    assert levels == ["medium", "medium"]
    assert any("supported just as broadly" in n for n in notes)


def test_the_better_supported_of_two_hypotheses_may_be_high(session):
    metrics, deploys, logs = (obs(session, tool=t) for t in ("get_metrics", "get_deployments", "search_logs"))
    levels, _ = capped(session, [hypothesis([metrics.id, deploys.id, logs.id]), hypothesis([metrics.id, deploys.id])])
    assert levels == ["high", "medium"]


def test_likely_cause_is_cleared_when_nothing_reaches_medium(session):
    response = FinalResponse.model_validate(payload(hypotheses=[hypothesis([], confidence="high")],
                                                    likely_cause="the deployment"))
    notes = apply_confidence_limits(response, session)
    assert response.likely_cause is None
    assert any("inconclusive" in n for n in notes)


def test_likely_cause_survives_a_medium_hypothesis(session):
    metrics = obs(session, tool="get_metrics")
    response = FinalResponse.model_validate(payload(hypotheses=[hypothesis([metrics.id], confidence="medium")],
                                                    likely_cause="the deployment"))
    apply_confidence_limits(response, session)
    assert response.likely_cause == "the deployment"


# -- assumptions and fallbacks ---------------------------------------------


def test_time_assumptions_are_copied_into_the_report_by_code(session):
    obs(session, tool="resolve_time_range", category="internal",
        data={"assumptions": ["Date not given; assumed 2026-09-22."]})
    outcome, errors = finalize(payload(), session)
    assert errors == []
    assert outcome.response.assumptions == ["Date not given; assumed 2026-09-22."]


def test_an_unverified_response_drops_invalid_citations_and_is_marked(session):
    good = obs(session, tool="get_metrics")
    outcome = finalize_unverified(
        payload(observed_facts=[
            {"statement": "real", "evidence_ids": [good.id]},
            {"statement": "invented", "evidence_ids": ["obs_404"]},
        ]),
        session, ["obs_404 does not exist"],
    )
    assert outcome.unverified is True
    assert [f.statement for f in outcome.response.observed_facts] == ["real"]
    assert any("unverified" in w for w in outcome.warnings)


def test_an_unparseable_payload_falls_back_to_the_ledger(session):
    obs(session, tool="get_metrics")
    obs(session, tool="search_logs", status="timeout")
    outcome = finalize_unverified({"not": "a response"}, session, ["schema error"])
    assert outcome.unverified is True
    assert [f.evidence_ids for f in outcome.response.observed_facts] == [["obs_001"]]
    assert any("search_logs failed" in g for g in outcome.response.gaps)
    assert outcome.response.hypotheses == []


def test_the_ledger_fallback_never_invents_a_cause(session):
    obs(session, tool="get_metrics")
    outcome = build_from_ledger(session, ["no final response"])
    assert outcome.response.likely_cause is None


# -- the answer stays the size of the question -----------------------------


def test_the_schema_caps_how_much_can_be_said(session):
    """An answer is bounded by the schema, not only asked for in the prompt."""
    too_much = payload(
        message="x" * 1300,
        observed_facts=[{"statement": "f", "evidence_ids": ["obs_1"]} for _ in range(9)],
        recommended_actions=["a"] * 4,
        gaps=["g"] * 6,
    )
    _, errors = validate(too_much, session)
    joined = " ".join(errors)
    assert "message" in joined and "observed_facts" in joined
    assert "recommended_actions" in joined and "gaps" in joined


def test_a_short_answer_is_accepted_as_it_is(session):
    observation = obs(session, tool="get_deployments")
    response, errors = validate(payload(
        response_type="answer",
        message="Yes. checkout-api v142 was deployed at 14:32 UTC.",
        observed_facts=[{"statement": "v142 deployed at 14:32", "evidence_ids": [observation.id]}],
    ), session)
    assert errors == []
    assert response.hypotheses == [] and response.recommended_actions == []
