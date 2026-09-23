"""Scoring one scenario.

Deterministic wherever the answer is a fact about what the agent did, and an
LLM judge only for the things that need reading: correctness, grounding,
uncertainty, completeness, actionability.

Scores are 0 wrong, 1 partially correct, 2 correct.
"""

from __future__ import annotations

import json
import re
from typing import Any

from string import Template

from incident_agent.config import Settings
from incident_agent.llm import OpenAIClient
from incident_agent.prompts import load

CALL = re.compile(r"^(\w+)\s*\(([^)]*)\)$")
TIME_LIKE = re.compile(r"\d{1,2}:\d{2}|utc|^\.\.\.$", re.I)
DIMENSIONS = ("factual_correctness", "evidence_grounding", "reasoning_and_uncertainty",
              "completeness", "actionability")

JUDGE_TOOL = {
    "type": "function",
    "name": "submit_judgement",
    "description": "Score the agent's answer against the expected answer.",
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            **{d: {"type": "integer", "enum": [0, 1, 2],
                   "description": "0 wrong, 1 partially correct, 2 correct."} for d in DIMENSIONS},
            "critical_error": {
                "type": "boolean",
                "description": "True if the answer claims correlation proves causation, invents "
                               "evidence, treats an empty result as proof of absence, cites failed "
                               "or malformed output, or ignores a plausible alternative cause.",
            },
            "reason": {"type": "string", "description": "One sentence explaining the lowest score."},
        },
        "required": [*DIMENSIONS, "critical_error", "reason"],
    },
}


def parse_required(spec: str) -> tuple[str, list[str]] | None:
    """'get_metrics(checkout-api, error_rate)' -> ('get_metrics', ['checkout-api', 'error_rate']).

    Prose entries return None. Time ranges inside the arguments are ignored:
    the scenarios do not require an exact window, only the right call.
    """
    match = CALL.match(spec.strip())
    if not match:
        return None
    args = [a.strip() for a in match.group(2).split(",") if a.strip()]
    return match.group(1), [a for a in args if not TIME_LIKE.search(a)]


def _matches(tool: str, args: list[str], observation: dict) -> bool:
    if observation["tool"] != tool:
        return False
    values = {str(v).lower() for v in observation["args"].values()}
    return all(any(arg.lower() in value for value in values) for arg in args)


def tool_coverage(scenario: dict, observations: list[dict]) -> dict:
    specs = [parsed for spec in scenario.get("required_tools", []) if (parsed := parse_required(spec))]
    missing = [f"{tool}({', '.join(args)})" for tool, args in specs
               if not any(_matches(tool, args, o) for o in observations)]
    return {"required": len(specs), "covered": len(specs) - len(missing), "missing": missing}


def used_a_prior_result(observations: list[dict]) -> bool:
    """Did any call use a value it could only have learnt from an earlier one?

    This is the multi-step requirement, and it cannot be checked by looking at
    which tools ran: a service named only by a dependency lookup, or a window
    narrowed inside one already queried, is the proof that a result changed the
    next decision.
    """
    discovered: set[str] = set()
    windows: list[tuple[str, str]] = []
    for o in observations:
        if o["category"] == "data":
            if o["args"].get("service") in discovered:
                return True
            window = (o["args"].get("start_time"), o["args"].get("end_time"))
            if all(window):
                if any(p[0] <= window[0] and window[1] <= p[1] and p != window for p in windows):
                    return True
                windows.append(window)
        if o["tool"] == "get_service_dependencies" and o["status"] == "ok":
            discovered.update(str(o["summary"]).replace(",", " ").replace(".", " ").split())
    return False


def deterministic(scenario: dict, transcript: dict) -> dict:
    """Checks that are facts about the run, not opinions about the answer."""
    observations = transcript["observations"]
    # A call rejected by validation never reached the backend; that is the
    # guardrail working, not a backend call.
    executed = [o for o in observations if o["category"] == "data" and o["status"] in ("ok", "empty")]
    cited = set(transcript["cited_ids"])
    failed = [o for o in observations if o["status"] not in ("ok", "empty")]

    checks: dict[str, bool] = {}
    critical: list[str] = []

    if cited_failures := [o["observation_id"] for o in failed if o["observation_id"] in cited]:
        critical.append(f"cited failed or malformed output: {', '.join(cited_failures)}")
    checks["no_failed_output_cited"] = not cited_failures

    if invalid := [o for o in observations if o["status"] == "invalid_arguments"
                   and "time range" in o["summary"].lower()]:
        # Rejected before execution, which is the guardrail working. It is only a
        # critical error when the scenario is not about an invalid range.
        if scenario["id"] != "E11":
            critical.append(f"{len(invalid)} call(s) used an invalid time range")

    if scenario["id"] == "E11":
        checks["invalid_range_reached_no_backend"] = not executed
    if scenario["id"] == "E08":
        recovered = [o for o in observations if o["tool"] == "get_metrics" and o["attempts"] > 1]
        checks["transient_timeout_retried"] = bool(recovered)
        checks["recovered_after_retry"] = any(o["status"] == "ok" for o in recovered)
    if scenario["id"] == "E09":
        checks["malformed_not_cited"] = not any(
            o["status"] == "error" and o["observation_id"] in cited for o in observations)
    if scenario["id"] == "E04":
        checks["used_a_prior_result"] = used_a_prior_result(observations)
    if scenario["id"] == "E10" and len(transcript["turns"]) > 1:
        follow_up = transcript["turns"][-1]
        checks["follow_up_reused_state"] = len([t for t in follow_up["trace"]
                                                if t["category"] == "data"]) <= 1

    expected_note = "no incident note" not in scenario.get("action", "").lower()
    if not expected_note and transcript["notes"]:
        critical.append("created an incident note where none was appropriate")
    checks["note_policy"] = expected_note or not transcript["notes"]

    return {
        "checks": checks,
        "passed": all(checks.values()),
        "critical": critical,
        "unnecessary_tool_calls": sum(
            1 for o in observations if o["status"] in ("duplicate", "invalid_arguments", "budget_exceeded")),
    }


def judge(scenario: dict, transcript: dict, settings: Settings) -> dict:
    """Ask the model to score the parts that need reading. The prompt is a file."""
    final = transcript["turns"][-1]
    response = final["response"]
    prompt = Template(load(settings.prompts_dir, "judge")).safe_substitute(
        question=final["question"],
        expected_answer=scenario["expected_answer"],
        forbidden_claims=json.dumps(scenario.get("forbidden_claims", []), indent=1),
        message=response["message"],
        observed_facts=json.dumps([f["statement"] for f in response["observed_facts"]], indent=1),
        hypotheses=json.dumps([f'({h["confidence"]}) {h["statement"]}' for h in response["hypotheses"]], indent=1),
        likely_cause=response["likely_cause"],
        recommended_actions=json.dumps(response["recommended_actions"], indent=1),
        gaps=json.dumps(response["gaps"], indent=1),
        tools=json.dumps([f'{o["tool"]} -> {o["status"]}' for o in transcript["observations"]], indent=1),
    )
    reply = OpenAIClient(settings).chat(
        [{"role": "user", "content": prompt}], [JUDGE_TOOL], force="submit_judgement")
    if not reply.tool_calls:
        return {d: 0 for d in DIMENSIONS} | {"critical_error": True, "reason": "the judge returned nothing"}
    return reply.tool_calls[0].arguments


def evaluate(scenario: dict, transcript: dict, settings: Settings, use_judge: bool = True) -> dict:
    """Combine the two into one scenario result."""
    coverage = tool_coverage(scenario, transcript["observations"])
    checks = deterministic(scenario, transcript)
    scores: dict[str, Any] = {}

    if transcript["error"]:
        checks["critical"].append(f"the run failed: {transcript['error']}")
    elif use_judge:
        scores = judge(scenario, transcript, settings)
        if scores.get("critical_error"):
            checks["critical"].append(scores.get("reason", "the judge flagged a critical error"))

    judged = [scores[d] for d in DIMENSIONS if d in scores]
    return {
        "scenario_id": scenario["id"],
        "title": scenario["title"],
        "pass": (not checks["critical"] and checks["passed"]
                 and not coverage["missing"] and all(s >= 1 for s in judged)),
        "critical_error": bool(checks["critical"]),
        "critical_reasons": checks["critical"],
        "required_tool_coverage": f"{coverage['covered']}/{coverage['required']}",
        "missing_tools": coverage["missing"],
        "unnecessary_tool_calls": checks["unnecessary_tool_calls"],
        "deterministic_checks": checks["checks"],
        "judge_scores": {d: scores[d] for d in DIMENSIONS if d in scores},
        "judge_reason": scores.get("reason", ""),
    }
