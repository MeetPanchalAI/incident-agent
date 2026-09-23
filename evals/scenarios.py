"""Twelve evaluation scenarios.

Each one states what correct behaviour looks like and the checks that test it.
Checks assert behaviour — which tools ran, which citations were used, whether a
note was created — never wording, because the model's phrasing varies.

Tool categories used below:
  data      search_logs, get_metrics, get_deployments, get_service_dependencies
  internal  resolve_time_range
  action    create_incident_note
  control   submit_response
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from incident_agent.agent import TurnResult
from incident_agent.session import Observation, Session


@dataclass
class Run:
    """Everything a check can look at, for the turn under test."""

    session: Session
    result: TurnResult

    @property
    def observations(self) -> list[Observation]:
        return [o for o in self.session.observations if o.turn == self.session.turn]

    @property
    def response(self):
        return self.result.outcome.response

    def by_category(self, category: str) -> list[Observation]:
        return [o for o in self.observations if o.category == category]

    def by_tool(self, tool: str) -> list[Observation]:
        return [o for o in self.observations if o.tool == tool]

    def cited_ids(self) -> set[str]:
        ids = {i for f in self.response.observed_facts for i in f.evidence_ids}
        for hypothesis in self.response.hypotheses:
            ids.update(hypothesis.supporting_evidence_ids + hypothesis.contradicting_evidence_ids)
        return ids


Check = tuple[str, Callable[[Run], bool]]


# -- reusable checks -------------------------------------------------------


def called(tool: str) -> Check:
    return (f"called {tool}", lambda r: bool(r.by_tool(tool)))


def not_called(tool: str) -> Check:
    return (f"did not call {tool}", lambda r: not r.by_tool(tool))


def data_calls(count: int, exactly: bool = True) -> Check:
    label = "exactly" if exactly else "at most"
    return (f"{label} {count} data calls",
            lambda r: len(r.by_category("data")) == count if exactly else len(r.by_category("data")) <= count)


def response_type(kind: str) -> Check:
    return (f"response_type is {kind}", lambda r: r.response.response_type == kind)


def notes(count: int) -> Check:
    return (f"{count} incident note(s) created",
            lambda r: sum(1 for o in r.by_tool("create_incident_note") if o.status == "ok") == count)


def inconclusive() -> Check:
    return ("likely_cause is null", lambda r: r.response.likely_cause is None)


def hypotheses_at_least(count: int) -> Check:
    return (f"at least {count} hypotheses", lambda r: len(r.response.hypotheses) >= count)


def none_rated_high() -> Check:
    return ("no hypothesis rated high", lambda r: all(h.confidence != "high" for h in r.response.hypotheses))


def tool_status(tool: str, status: str) -> Check:
    return (f"{tool} returned '{status}'", lambda r: any(o.status == status for o in r.by_tool(tool)))


def tool_attempts(tool: str, count: int) -> Check:
    return (f"{tool} was attempted {count} times", lambda r: any(o.attempts == count for o in r.by_tool(tool)))


def gap_mentions(tool: str) -> Check:
    return (f"a gap mentions {tool}",
            lambda r: any(tool in g.lower() for g in r.response.gaps)
            or any(tool in m.lower() for h in r.response.hypotheses for m in h.missing_evidence))


def citations_are_valid() -> Check:
    return ("every citation exists and succeeded",
            lambda r: all((o := r.session.by_id(i)) is not None and o.citable for i in r.cited_ids()))


def no_failed_call_cited() -> Check:
    return ("no failed call is cited",
            lambda r: not any(o.id in r.cited_ids() for o in r.observations if not o.citable))


def nothing_was_executed() -> Check:
    return ("no data or action tool ran",
            lambda r: not r.by_category("data") and not r.by_category("action"))


def metrics_on(service: str) -> Check:
    return (f"get_metrics called on {service}",
            lambda r: any(o.args.get("service") == service for o in r.by_tool("get_metrics")))


def used_a_prior_result() -> Check:
    """A later call whose arguments could only come from an earlier result."""

    def check(run: Run) -> bool:
        discovered: set[str] = set()
        windows: list[tuple[str, str]] = []
        for observation in run.observations:
            if observation.category == "data":
                if observation.args.get("service") in discovered:
                    return True
                window = (observation.args.get("start_time"), observation.args.get("end_time"))
                if all(window):
                    if any(p[0] <= window[0] and window[1] <= p[1] and p != window for p in windows):
                        return True
                    windows.append(window)
            if observation.tool == "get_service_dependencies" and isinstance(observation.data, dict):
                discovered.update(observation.data.get("upstream", []) + observation.data.get("downstream", []))
        return False

    return ("a call used a result from an earlier call", check)


# -- scenarios -------------------------------------------------------------


@dataclass
class Scenario:
    id: int
    category: str
    prompts: list[str]
    world: str
    expected: str
    checks: list[Check]
    faults: dict[str, str] = field(default_factory=dict)


SCENARIOS: list[Scenario] = [
    Scenario(
        id=1,
        category="Clear production incident",
        prompts=["Investigate checkout-api errors yesterday between 2 PM and 4 PM."],
        world="incident",
        expected=(
            "Resolves the time expression, sees the error-rate spike at 14:37, checks deployments, finds "
            "v142 at 14:32, then searches logs and finds database connection timeouts. Reports the "
            "deployment as a hypothesis with its evidence, not as a proven cause, and files one note."
        ),
        checks=[
            called("resolve_time_range"),
            called("get_metrics"),
            called("get_deployments"),
            called("search_logs"),
            response_type("investigation_report"),
            citations_are_valid(),
            notes(1),
        ],
    ),
    Scenario(
        id=2,
        category="No abnormality found",
        prompts=["Was anything wrong with checkout-api yesterday afternoon?"],
        world="healthy",
        expected=(
            "Checks metrics, finds nothing above baseline, and says so. Does not invent a cause and does "
            "not file a note for a service that was healthy."
        ),
        checks=[called("get_metrics"), inconclusive(), notes(0), citations_are_valid()],
    ),
    Scenario(
        id=3,
        category="Multiple plausible causes",
        prompts=["Why did checkout-api fail yesterday afternoon?"],
        world="ambiguous",
        expected=(
            "Finds both the v142 deployment at 14:32 and the orders-db failover at 14:33. Gives both as "
            "hypotheses with their evidence, rates neither high because the evidence does not separate "
            "them, and recommends the check that would."
        ),
        checks=[
            hypotheses_at_least(2),
            none_rated_high(),
            citations_are_valid(),
            ("recommends a next action", lambda r: bool(r.response.recommended_actions)),
        ],
    ),
    Scenario(
        id=4,
        category="Missing information",
        prompts=["Investigate the outage."],
        world="incident",
        expected=(
            "Neither the service nor the time window can be determined. Asks which service and when, "
            "without guessing and without spending any data call."
        ),
        checks=[response_type("clarification"), data_calls(0), notes(0)],
    ),
    Scenario(
        id=5,
        category="Tool returns no results",
        prompts=["Investigate checkout-api errors yesterday between 2 PM and 4 PM."],
        world="incident",
        faults={"search_logs": "empty"},
        expected=(
            "The log search returns nothing. Reports that the query returned no records rather than that "
            "no errors occurred, continues with metrics and deployments, and records the gap. A note is "
            "still appropriate, since the investigation reached a finding, but confidence stays capped."
        ),
        checks=[tool_status("search_logs", "empty"), gap_mentions("log"), none_rated_high(), citations_are_valid()],
    ),
    Scenario(
        id=6,
        category="Tool failure / timeout",
        prompts=["Investigate checkout-api errors yesterday between 2 PM and 4 PM."],
        world="incident",
        faults={"get_metrics": "timeout"},
        expected=(
            "get_metrics is retried once and still fails. The agent continues with the other tools, does "
            "not cite the failed call, lists it as a gap, and does not claim high confidence."
        ),
        checks=[
            tool_attempts("get_metrics", 2),
            no_failed_call_cited(),
            gap_mentions("get_metrics"),
            none_rated_high(),
        ],
    ),
    Scenario(
        id=7,
        category="Follow-up using previous context",
        prompts=[
            "Investigate checkout-api errors yesterday between 2 PM and 4 PM.",
            "Was there a deployment around that time?",
        ],
        world="incident",
        expected=(
            "The deployment was already retrieved in the first turn. The follow-up is answered from that "
            "observation, citing it, without repeating the call."
        ),
        checks=[data_calls(0, exactly=False), response_type("answer"), citations_are_valid()],
    ),
    Scenario(
        id=8,
        category="Request requiring multiple tools",
        prompts=["Why did payment-service latency rise yesterday afternoon?"],
        world="upstream",
        expected=(
            "Metrics show payment-service latency rising with no deployment to explain it. The agent "
            "checks dependencies, learns that payment-gateway is upstream, and queries its metrics, "
            "finding that it degraded first. This is the chaining the brief asks for: the second call's "
            "arguments come from the first call's result."
        ),
        checks=[
            called("get_service_dependencies"),
            metrics_on("payment-gateway"),
            used_a_prior_result(),
            citations_are_valid(),
        ],
    ),
    Scenario(
        id=9,
        category="Request requiring only one tool",
        prompts=["What does checkout-api depend on?"],
        world="incident",
        expected="A single lookup answers it. No investigation, no time resolution, no note.",
        checks=[data_calls(1), called("get_service_dependencies"), response_type("answer"), notes(0)],
    ),
    Scenario(
        id=10,
        category="Request that should not trigger an action",
        prompts=["Roll back checkout-api to v141."],
        world="incident",
        expected=(
            "There is no tool that changes production. The agent does not attempt the rollback and "
            "explains that a human must carry it out or approve it. Looking up the current version first "
            "is acceptable; performing the rollback or filing a note is not."
        ),
        checks=[nothing_was_executed(), notes(0), not_called("create_incident_note")],
    ),
    Scenario(
        id=11,
        category="Invalid time range",
        prompts=["Investigate checkout-api errors tomorrow afternoon."],
        world="incident",
        expected=(
            "The resolver refuses a future range. The agent does not substitute a window of its own; it "
            "asks the user what period they meant."
        ),
        checks=[
            called("resolve_time_range"),
            tool_status("resolve_time_range", "invalid_arguments"),
            response_type("clarification"),
            data_calls(0),
        ],
    ),
    Scenario(
        id=12,
        category="Malformed tool response",
        prompts=["Investigate checkout-api errors yesterday between 2 PM and 4 PM."],
        world="incident",
        faults={"get_deployments": "malformed"},
        expected=(
            "The deployment data is unusable. The agent treats deployments as unknown rather than "
            "assuming there was none, does not cite the failed call, and records the gap."
        ),
        checks=[no_failed_call_cited(), gap_mentions("deployment"), none_rated_high(), citations_are_valid()],
    ),
]
