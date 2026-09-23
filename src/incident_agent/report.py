"""The final response: its schema, its validation, and the confidence limits.

The agent can only end a turn by calling `submit_response`. This module
checks that payload against the evidence ledger, so the separation of facts
from hypotheses is enforced by code rather than requested in the prompt.

What code can and cannot check is listed in DESIGN.md, "What enforces what".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .session import Observation, Session

Confidence = Literal["low", "medium", "high"]
RANK = {"low": 0, "medium": 1, "high": 2}


class ObservedFact(BaseModel):
    model_config = ConfigDict(extra="forbid")
    statement: str = Field(max_length=500, description="A fact taken directly from a tool result.")
    evidence_ids: list[str] = Field(
        min_length=1, max_length=10, description="Observation IDs this fact comes from, e.g. ['obs_002']."
    )


class Hypothesis(BaseModel):
    model_config = ConfigDict(extra="forbid")
    statement: str = Field(max_length=500, description="A possible explanation. Not a fact.")
    supporting_evidence_ids: list[str] = Field(default_factory=list, max_length=10)
    contradicting_evidence_ids: list[str] = Field(default_factory=list, max_length=10)
    missing_evidence: list[str] = Field(
        default_factory=list,
        max_length=10,
        description="Evidence that would have tested this but was not obtained, including failed tool calls.",
    )
    confidence: Confidence = Field(description="How well the evidence supports this. Not a probability.")


class FinalResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    response_type: Literal["answer", "clarification", "investigation_report"] = Field(
        description="'answer' for a direct question, 'clarification' when you must ask the user "
        "something before investigating, 'investigation_report' for a full investigation."
    )
    message: str = Field(max_length=2000, description="The direct answer, or the clarifying question.")
    observed_facts: list[ObservedFact] = Field(
        default_factory=list, max_length=20, description="Only things a tool actually returned."
    )
    hypotheses: list[Hypothesis] = Field(default_factory=list, max_length=5)
    likely_cause: str | None = Field(
        default=None, max_length=500, description="Leave null when the evidence is inconclusive."
    )
    recommended_actions: list[str] = Field(default_factory=list, max_length=10)
    gaps: list[str] = Field(
        default_factory=list, max_length=10, description="Failed tool calls, missing data, contradictions."
    )
    assumptions: list[str] = Field(default_factory=list, max_length=10)


@dataclass
class FinalOutcome:
    response: FinalResponse
    warnings: list[str] = field(default_factory=list)
    unverified: bool = False

    def to_dict(self) -> dict:
        return {**self.response.model_dump(), "warnings": self.warnings, "unverified": self.unverified}


def submit_response_schema() -> dict:
    schema = FinalResponse.model_json_schema()
    schema.pop("title", None)
    return {
        "type": "function",
        "function": {
            "name": "submit_response",
            "description": (
                "End the turn with your final answer. Call this alone, never alongside other tools, "
                "and only after you have reviewed every tool result you asked for. "
                "Every observed fact must cite the observation IDs it came from."
            ),
            "parameters": schema,
        },
    }


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def _cited(session: Session, ids: list[str]) -> list[Observation]:
    return [o for o in (session.by_id(i) for i in ids) if o is not None]


def _bad_citations(session: Session, ids: list[str]) -> list[str]:
    bad = []
    for identifier in ids:
        observation = session.by_id(identifier)
        if observation is None:
            bad.append(f"{identifier} does not exist")
        elif not observation.citable:
            bad.append(f"{identifier} has status '{observation.status}' and cannot be cited")
    return bad


def validate(payload: dict, session: Session) -> tuple[FinalResponse | None, list[str]]:
    """Parse and check a submit_response payload. Returns (response, errors)."""
    try:
        response = FinalResponse.model_validate(payload)
    except ValidationError as exc:
        return None, [f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:10]]

    errors: list[str] = []
    for index, fact in enumerate(response.observed_facts):
        for problem in _bad_citations(session, fact.evidence_ids):
            errors.append(f"observed_facts[{index}]: {problem}.")
    for index, hypothesis in enumerate(response.hypotheses):
        ids = hypothesis.supporting_evidence_ids + hypothesis.contradicting_evidence_ids
        for problem in _bad_citations(session, ids):
            errors.append(f"hypotheses[{index}]: {problem}.")

    # Failed calls must be disclosed somewhere the user will see.
    disclosed = " ".join(response.gaps + [m for h in response.hypotheses for m in h.missing_evidence]).lower()
    for failure in session.failures_this_turn():
        if failure.tool not in disclosed:
            errors.append(
                f"{failure.tool} failed ({failure.id}, status '{failure.status}'). "
                "Report it in 'gaps' or in a hypothesis's 'missing_evidence'."
            )
    return response, errors


def apply_confidence_limits(response: FinalResponse, session: Session) -> list[str]:
    """Lower any confidence the evidence does not support. Returns visible notes."""
    notes: list[str] = []
    turn_failed = session.failures_this_turn()

    def independent_tools(hypothesis: Hypothesis) -> set[str]:
        # Only successful data tools count. An empty result cannot raise confidence.
        return {
            o.tool
            for o in _cited(session, hypothesis.supporting_evidence_ids)
            if o.status == "ok" and o.category == "data"
        }

    breadth = [len(independent_tools(h)) for h in response.hypotheses]
    best = max(breadth, default=0)
    sole_leader = breadth.count(best) == 1

    for index, hypothesis in enumerate(response.hypotheses):
        capped: Confidence = hypothesis.confidence
        reasons: list[str] = []

        if not hypothesis.supporting_evidence_ids:
            capped = "low"
            reasons.append("no supporting evidence")
        if turn_failed and RANK[capped] > RANK["medium"]:
            tools = ", ".join(sorted({f.tool for f in turn_failed}))
            capped = "medium"
            reasons.append(f"a tool call failed this turn ({tools})")
        if hypothesis.contradicting_evidence_ids and RANK[capped] > RANK["medium"]:
            capped = "medium"
            reasons.append("contradicting evidence was cited")
        if RANK[capped] > RANK["medium"] and breadth[index] < 2:
            capped = "medium"
            reasons.append("supported by fewer than two independent data tools")
        if RANK[capped] > RANK["medium"] and not (sole_leader and breadth[index] == best):
            capped = "medium"
            reasons.append("another hypothesis is supported just as broadly")

        if capped != hypothesis.confidence:
            notes.append(
                f"Confidence for '{hypothesis.statement[:60]}' lowered from "
                f"{hypothesis.confidence} to {capped}: {'; '.join(reasons)}."
            )
            hypothesis.confidence = capped

    if response.likely_cause and not any(RANK[h.confidence] >= RANK["medium"] for h in response.hypotheses):
        notes.append("likely_cause cleared: no hypothesis reaches medium confidence. Reported as inconclusive.")
        response.likely_cause = None
    return notes


def finalize(payload: dict, session: Session) -> tuple[FinalOutcome | None, list[str]]:
    """Validate and finish, or return the errors the model should repair."""
    response, errors = validate(payload, session)
    if errors or response is None:
        return None, errors
    warnings = apply_confidence_limits(response, session)
    _merge_assumptions(response, session)
    return FinalOutcome(response=response, warnings=warnings), []


def finalize_unverified(payload: dict, session: Session, errors: list[str]) -> FinalOutcome:
    """Last resort when the payload still fails after its one repair attempt."""
    response, _ = validate(payload, session)
    if response is None:
        return build_from_ledger(session, errors)
    for fact in response.observed_facts:
        fact.evidence_ids = [i for i in fact.evidence_ids if (o := session.by_id(i)) and o.citable]
    response.observed_facts = [f for f in response.observed_facts if f.evidence_ids]
    warnings = apply_confidence_limits(response, session)
    _merge_assumptions(response, session)
    warnings.append("This response failed validation and was returned unverified: " + " ".join(errors))
    return FinalOutcome(response=response, warnings=warnings, unverified=True)


def build_from_ledger(session: Session, errors: list[str]) -> FinalOutcome:
    """Build a response from the ledger alone, with no further model call."""
    facts = [
        ObservedFact(statement=o.summary[:500], evidence_ids=[o.id])
        for o in session.this_turn()
        if o.citable and o.category in ("data", "internal")
    ]
    gaps = [f"{o.tool} failed ({o.id}): {(o.error or {}).get('message', o.status)}" for o in session.this_turn() if not o.citable]
    response = FinalResponse(
        response_type="investigation_report",
        message=(
            "The agent could not produce a valid structured answer. "
            "Below is exactly what the tools returned this turn, with no interpretation."
        ),
        observed_facts=facts,
        gaps=gaps,
    )
    _merge_assumptions(response, session)
    return FinalOutcome(
        response=response,
        warnings=["Response built from the evidence ledger after repair failed: " + " ".join(errors)],
        unverified=True,
    )


def _merge_assumptions(response: FinalResponse, session: Session) -> None:
    for assumption in session.assumptions_this_turn():
        if assumption not in response.assumptions:
            response.assumptions.append(assumption)


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def render_text(outcome: FinalOutcome) -> str:
    """Plain-text rendering shared by the CLI and the evaluation runner."""
    r = outcome.response
    lines = [r.message]
    if outcome.unverified:
        lines.insert(0, "[UNVERIFIED]")

    def block(title: str, items: list[str]) -> None:
        if items:
            lines.append(f"\n{title}:")
            lines.extend(f"  - {item}" for item in items)

    block("Observed facts", [f"{f.statement} [{', '.join(f.evidence_ids)}]" for f in r.observed_facts])
    block(
        "Hypotheses",
        [
            f"({h.confidence}) {h.statement}"
            + (f" [supported by {', '.join(h.supporting_evidence_ids)}]" if h.supporting_evidence_ids else "")
            + (f" [contradicted by {', '.join(h.contradicting_evidence_ids)}]" if h.contradicting_evidence_ids else "")
            + (f" [missing: {'; '.join(h.missing_evidence)}]" if h.missing_evidence else "")
            for h in r.hypotheses
        ],
    )
    if r.hypotheses or r.response_type == "investigation_report":
        lines.append(f"\nLikely cause: {r.likely_cause or 'inconclusive'}")
    block("Recommended actions", r.recommended_actions)
    block("Gaps", r.gaps)
    block("Assumptions", r.assumptions)
    block("Warnings", outcome.warnings)
    return "\n".join(lines)
