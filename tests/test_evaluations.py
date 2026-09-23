"""The evaluation harness, so a broken check cannot quietly pass a scenario."""

from __future__ import annotations

import pytest

from evaluations.judge import (deterministic, evaluate, parse_required, tool_coverage,
                               used_a_prior_result)
from evaluations.runner import load_scenarios, questions, run_scenario
from tests.conftest import WINDOW, submit

METRICS = ("get_metrics", {"service": "checkout-api", "metric": "error_rate", **WINDOW})
DEPLOYS = ("get_deployments", {"service": "checkout-api", **WINDOW})
DEPS = ("get_service_dependencies", {"service": "checkout-api"})
LOGS = ("search_logs", {"service": "checkout-api", "query": "timeout", **WINDOW})
DONE = ("submit_response", submit())


def observation(tool="get_metrics", status="ok", category="data", identifier="obs_001", **args):
    return {"observation_id": identifier, "tool": tool, "category": category, "status": status,
            "attempts": 1, "args": args, "summary": ""}


def transcript(observations, cited=(), notes=0, error=None, turns=None):
    return {"scenario_id": "T", "observations": list(observations), "cited_ids": list(cited),
            "notes": notes, "error": error,
            "turns": turns or [{"question": "q", "response": submit(), "trace": observations}]}


# -- the scenario file -----------------------------------------------------


def test_the_twelve_scenarios_load_and_are_well_formed():
    scenarios = load_scenarios()
    assert len(scenarios) == 12
    assert [s["id"] for s in scenarios] == [f"E{n:02d}" for n in range(1, 13)]
    assert all(s["question"] and s["expected_answer"] and s["action"] for s in scenarios)


def test_a_two_turn_scenario_is_split_into_turns():
    follow_up = next(s for s in load_scenarios() if s["id"] == "E10")
    assert len(questions(follow_up)) == 2
    assert questions(follow_up)[1].startswith("What evidence")


def test_a_single_turn_scenario_stays_one_question():
    assert len(questions({"question": "Why did checkout-api fail?"})) == 1


def test_the_fault_scenarios_name_a_fault_the_store_understands():
    faults = {s["id"]: s.get("faults") for s in load_scenarios() if s.get("fault_injection")}
    assert faults == {"E08": {"get_metrics": "transient"}, "E09": {"get_metrics": "malformed_once"}}


# -- required tools --------------------------------------------------------


@pytest.mark.parametrize("spec,expected", [
    ("get_metrics(checkout-api, error_rate)", ("get_metrics", ["checkout-api", "error_rate"])),
    ("get_deployments(checkout-api)", ("get_deployments", ["checkout-api"])),
    ("get_deployments(checkout-api, 14:00-15:00 UTC)", ("get_deployments", ["checkout-api"])),
    ("get_metrics(orders-db, ...)", ("get_metrics", ["orders-db"])),
    ("Initial turn: normal checkout investigation tools", None),
])
def test_required_tool_specs_are_parsed_and_time_ranges_ignored(spec, expected):
    assert parse_required(spec) == expected


def test_coverage_counts_a_matching_call_whatever_the_order():
    scenario = {"required_tools": ["get_metrics(checkout-api, error_rate)", "get_deployments(checkout-api)"]}
    covered = tool_coverage(scenario, [
        observation(tool="get_deployments", service="checkout-api"),
        observation(tool="get_metrics", service="checkout-api", metric="error_rate"),
    ])
    assert covered == {"required": 2, "covered": 2, "missing": []}


def test_coverage_names_what_was_missing():
    scenario = {"required_tools": ["get_metrics(checkout-api, latency_p95_ms)"]}
    covered = tool_coverage(scenario, [observation(service="checkout-api", metric="error_rate")])
    assert covered["missing"] == ["get_metrics(checkout-api, latency_p95_ms)"]


# -- deterministic checks --------------------------------------------------


def test_citing_a_failed_call_is_a_critical_error():
    result = deterministic({"id": "E01", "action": "a note is appropriate"},
                           transcript([observation(status="timeout")], cited=["obs_001"]))
    assert result["checks"]["no_failed_output_cited"] is False
    assert "cited failed" in result["critical"][0]


def test_an_invalid_time_range_must_not_reach_the_backend():
    blocked = deterministic({"id": "E11", "action": "No incident note."},
                            transcript([observation(status="invalid_arguments",
                                                    summary="Invalid time range. start must be before end")]))
    assert blocked["checks"]["invalid_range_reached_no_backend"] is True
    assert blocked["critical"] == []  # the guardrail working is not an error

    leaked = deterministic({"id": "E11", "action": "No incident note."},
                           transcript([observation(status="ok")]))
    assert leaked["checks"]["invalid_range_reached_no_backend"] is False


def test_a_transient_timeout_must_be_retried_and_recovered():
    good = transcript([{**observation(status="ok"), "attempts": 2}])
    assert deterministic({"id": "E08", "action": "note"}, good)["checks"] == {
        "no_failed_output_cited": True, "transient_timeout_retried": True,
        "recovered_after_retry": True, "note_policy": True}
    bad = deterministic({"id": "E08", "action": "note"}, transcript([observation(status="ok")]))
    assert bad["checks"]["transient_timeout_retried"] is False


def test_a_note_where_none_was_wanted_is_a_critical_error():
    result = deterministic({"id": "E02", "action": "No incident note."}, transcript([], notes=1))
    assert result["checks"]["note_policy"] is False
    assert "incident note" in result["critical"][0]


def test_a_note_where_one_was_wanted_is_fine():
    scenario = {"id": "E01", "action": "A single incident note is appropriate."}
    assert deterministic(scenario, transcript([], notes=1))["checks"]["note_policy"] is True


def test_wasted_calls_are_counted_not_failed():
    result = deterministic({"id": "E01", "action": "note"}, transcript([
        observation(status="duplicate"), observation(status="budget_exceeded"), observation(status="ok")]))
    assert result["unnecessary_tool_calls"] == 2
    assert result["passed"] is True


def test_a_follow_up_that_repeats_the_investigation_fails_the_check():
    turns = [{"question": "a", "response": submit(), "trace": []},
             {"question": "b", "response": submit(),
              "trace": [observation(identifier="obs_002"), observation(identifier="obs_003")]}]
    result = deterministic({"id": "E10", "action": "Do not create a second note."},
                           transcript([], turns=turns))
    assert result["checks"]["follow_up_reused_state"] is False


# -- combining -------------------------------------------------------------


def test_a_failed_run_is_a_critical_error_without_calling_the_judge(settings):
    scenario = {"id": "E01", "title": "t", "action": "note", "required_tools": []}
    result = evaluate(scenario, transcript([], error="RuntimeError: boom"), settings)
    assert result["pass"] is False and result["critical_error"] is True
    assert result["judge_scores"] == {}


def test_a_clean_run_passes_when_the_judge_is_skipped(settings):
    scenario = {"id": "E03", "title": "t", "action": "No incident note.",
                "required_tools": ["get_deployments(checkout-api)"]}
    result = evaluate(scenario, transcript([observation(tool="get_deployments", service="checkout-api")]),
                      settings, use_judge=False)
    assert result["pass"] is True
    assert result["required_tool_coverage"] == "1/1"


def test_missing_a_required_tool_fails_even_without_a_critical_error(settings):
    scenario = {"id": "E03", "title": "t", "action": "No incident note.",
                "required_tools": ["get_deployments(checkout-api)"]}
    result = evaluate(scenario, transcript([]), settings, use_judge=False)
    assert result["pass"] is False and result["critical_error"] is False
    assert result["missing_tools"] == ["get_deployments(checkout-api)"]


# -- the runner ------------------------------------------------------------


def test_a_scenario_runs_end_to_end_against_a_scripted_model(settings):
    from incident_agent.llm import FakeLLM

    scenario = {"id": "E01", "title": "t", "question": "why did checkout-api fail?",
                "action": "note", "required_tools": ["get_metrics(checkout-api, error_rate)"]}
    cite = ("submit_response", submit(observed_facts=[{"statement": "spike", "evidence_ids": ["obs_001"]}]))
    result = run_scenario(scenario, settings, llm=FakeLLM([[METRICS, DEPLOYS, LOGS], [cite]]))

    assert result["error"] is None
    assert [o["tool"] for o in result["observations"]] == ["get_metrics", "get_deployments", "search_logs"]
    assert result["cited_ids"] == ["obs_001"]
    assert tool_coverage(scenario, result["observations"])["missing"] == []


def test_a_crash_is_captured_rather_than_raised(settings):
    from incident_agent.llm import FakeLLM

    scenario = {"id": "E01", "title": "t", "question": "q", "action": "note", "required_tools": []}
    result = run_scenario(scenario, settings, llm=FakeLLM([]))
    assert "AssertionError" in result["error"]
    assert result["turns"] == []


def test_both_turns_of_a_follow_up_share_one_session(settings):
    from incident_agent.llm import FakeLLM

    scenario = {"id": "E10", "title": "t", "action": "no second note", "required_tools": [],
                "question": "Turn 1: Investigate checkout-api. Turn 2: What implicates the deployment?"}
    result = run_scenario(scenario, settings, llm=FakeLLM([[METRICS], [DONE], [DONE]]))
    assert len(result["turns"]) == 2
    assert result["turns"][1]["trace"] == []  # answered from the first turn's evidence


def test_running_many_scenarios_does_not_accumulate_threads_or_connections(settings):
    """One pool is shared, and each scenario's agent releases its database."""
    import threading

    from incident_agent.llm import FakeLLM

    scenario = {"id": "E12", "title": "t", "question": "what does checkout-api depend on?",
                "action": "No incident note.", "required_tools": []}
    before = threading.active_count()
    for _ in range(6):
        run_scenario(scenario, settings, llm=FakeLLM([[DEPS], [("submit_response", submit())]]))
    assert threading.active_count() <= before + 8  # the shared pool, not six pools


# -- the multi-step requirement --------------------------------------------


def test_chaining_is_seen_when_a_service_came_from_a_dependency_lookup():
    assert used_a_prior_result([
        {**observation(tool="get_service_dependencies", category="data"),
         "summary": "payment-service depends on: payment-gateway."},
        observation(tool="get_metrics", identifier="obs_002", service="payment-gateway"),
    ]) is True


def test_chaining_is_seen_when_a_window_is_narrowed_inside_an_earlier_one():
    wide = observation(service="checkout-api", start_time="2026-09-22T14:00:00Z",
                       end_time="2026-09-22T16:00:00Z")
    narrow = observation(tool="search_logs", identifier="obs_002", service="checkout-api",
                         start_time="2026-09-22T14:30:00Z", end_time="2026-09-22T15:00:00Z")
    assert used_a_prior_result([wide, narrow]) is True


def test_two_independent_calls_are_not_chaining():
    assert used_a_prior_result([
        observation(service="checkout-api", start_time="2026-09-22T14:00:00Z", end_time="2026-09-22T16:00:00Z"),
        observation(tool="get_deployments", identifier="obs_002", service="checkout-api",
                    start_time="2026-09-22T14:00:00Z", end_time="2026-09-22T16:00:00Z"),
    ]) is False


def test_the_upstream_scenario_requires_chaining():
    flat = deterministic({"id": "E04", "action": "a note"}, transcript([observation(service="payment-service")]))
    assert flat["checks"]["used_a_prior_result"] is False
