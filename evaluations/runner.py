"""Run the evaluation scenarios against the agent.

    python -m evaluations.runner                  every scenario
    python -m evaluations.runner --scenario E05   one scenario (repeatable)
    python -m evaluations.runner --no-judge       deterministic checks only, no model grading
    python -m evaluations.runner --out report.json

Needs a dataset to have been ingested, and OPENAI_API_KEY. The run is recorded
in the log database like any other workflow, so it shows in the Monitoring tab.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from typing import Callable

from incident_agent import build_service, load_settings
from incident_agent.config import Settings
from incident_agent.state import Recorder, save_eval_run
from incident_agent.tools.store import Store

from .judge import evaluate


class EvalError(Exception):
    """The suite could not be started."""

SCENARIOS = Path(__file__).parent / "scenarios.json"
TURN_SPLIT = re.compile(r"\s*Turn\s*\d+\s*:\s*")


def load_scenarios(path: Path = SCENARIOS) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["scenarios"]


def questions(scenario: dict) -> list[str]:
    """A scenario is one question, unless it is written as "Turn 1: ... Turn 2: ..."."""
    text = scenario["question"]
    if not TURN_SPLIT.match(text):
        return [text]
    return [part.strip() for part in TURN_SPLIT.split(text) if part.strip()]


def run_scenario(scenario: dict, settings: Settings, llm=None) -> dict:
    """Run every turn of a scenario in one session and capture what happened."""
    agent = build_service(settings, faults=scenario.get("faults"), llm=llm)
    session = agent.new_session()
    turns, error = [], None

    try:
        for question in questions(scenario):
            result = agent.run_turn(session, question)
            turns.append({
                "question": question,
                "response": result.outcome.to_dict(),
                "trace": result.trace,
                "model_calls": result.llm_calls,
            })
    except Exception as failure:  # a broken run is a result, not a crash
        error = f"{type(failure).__name__}: {failure}"

    cited: set[str] = set()
    for turn in turns:
        response = turn["response"]
        cited.update(i for fact in response["observed_facts"] for i in fact["evidence_ids"])
        for hypothesis in response["hypotheses"]:
            cited.update(hypothesis["supporting_evidence_ids"] + hypothesis["contradicting_evidence_ids"])

    return {
        "scenario_id": scenario["id"],
        "turns": turns,
        "observations": [t for turn in turns for t in turn["trace"]],
        "cited_ids": sorted(cited),
        "notes": sum(1 for o in session.observations if o.tool == "create_incident_note" and o.status == "ok"),
        "error": error,
    }


def report_line(result: dict) -> str:
    scores = result["judge_scores"]
    mean = f"{sum(scores.values()) / len(scores):.1f}" if scores else "-"
    flags = []
    if result["critical_error"]:
        flags.append("CRITICAL")
    if failed := [name for name, ok in result["deterministic_checks"].items() if not ok]:
        flags.append("checks: " + ", ".join(failed))
    if result["missing_tools"]:
        flags.append("missing: " + ", ".join(result["missing_tools"]))
    if result["unnecessary_tool_calls"]:
        flags.append(f"{result['unnecessary_tool_calls']} wasted calls")
    return (f"{result['scenario_id']:<5} {'PASS' if result['pass'] else 'FAIL':<5} "
            f"{result['required_tool_coverage']:<6} {mean:<5} {result['title']:<34} {'; '.join(flags)}")


def run_suite(settings: Settings, scenario_ids: list[str] | None = None, use_judge: bool = True,
              on_result: Callable[[dict], None] | None = None) -> dict:
    """Run the scenarios, record the run, persist the results, return the report.

    Results are written after each scenario, so a run that is interrupted still
    leaves everything it managed to score.
    """
    store = Store(settings.db_path)
    dataset = store.info()
    store.close()
    if not dataset:
        raise EvalError("No dataset has been ingested. Load a log file first.")
    if not settings.api_key:
        raise EvalError("OPENAI_API_KEY is not set.")

    chosen = [s for s in load_scenarios() if not scenario_ids or s["id"] in scenario_ids]
    if not chosen:
        raise EvalError("No scenario matched.")

    started_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    log = Recorder(settings.state_db_path, "eval", f"{len(chosen)} scenarios on {dataset['filename']}")
    log.event("eval.start", f"{len(chosen)} scenarios against {dataset['filename']}.",
              model=settings.model, judge=use_judge, dataset=dataset["filename"],
              scenarios=[s["id"] for s in chosen])

    results: list[dict] = []
    for scenario in chosen:
        transcript = run_scenario(scenario, settings)
        result = evaluate(scenario, transcript, settings, use_judge=use_judge)
        results.append(result)
        save_eval_run(settings.state_db_path, log.id, dataset["filename"], settings.model,
                      use_judge, results, started_at)
        log.event(
            "eval.scenario",
            f"{scenario['id']} {'passed' if result['pass'] else 'failed'}: {scenario['title']}"
            + (f" - {'; '.join(result['critical_reasons'])}" if result["critical_reasons"] else ""),
            level="info" if result["pass"] else "warn",
            scenario=scenario["id"], passed=result["pass"], critical=result["critical_error"],
            coverage=result["required_tool_coverage"], wasted=result["unnecessary_tool_calls"],
            scores=result["judge_scores"],
        )
        if on_result:
            on_result(result)

    passed = sum(r["pass"] for r in results)
    critical = sum(r["critical_error"] for r in results)
    log.event("eval.done", f"{passed}/{len(results)} passed, {critical} critical.",
              level="warn" if critical else "info", passed=passed, total=len(results), critical=critical)
    log.finish(f"{passed}/{len(results)} passed, {critical} critical")
    return {"run_id": log.id, "dataset": dataset, "model": settings.model,
            "judged": use_judge, "results": results,
            "passed": passed, "total": len(results), "critical": critical}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Run the incident-agent evaluation scenarios.")
    parser.add_argument("--scenario", action="append", help="run only this scenario id (repeatable)")
    parser.add_argument("--no-judge", action="store_true", help="deterministic checks only")
    parser.add_argument("--out", type=Path, help="write the full report as JSON")
    args = parser.parse_args(argv)

    settings = load_settings()
    print(f"{'id':<5} {'res':<5} {'tools':<6} {'score':<5} {'scenario':<34} notes")
    print("-" * 110)
    try:
        report = run_suite(settings, args.scenario, use_judge=not args.no_judge,
                           on_result=lambda result: print(report_line(result)))
    except EvalError as error:
        print(error)
        return 2

    print("-" * 110)
    print(f"{report['passed']}/{report['total']} passed, {report['critical']} with a critical error.")
    if args.out:
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Report written to {args.out}.")
    return 1 if report["passed"] < report["total"] else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
