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
from pathlib import Path

from incident_agent import build_service, load_settings
from incident_agent.config import Settings
from incident_agent.logs import Recorder
from incident_agent.tools.store import Store

from .judge import evaluate

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


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Run the incident-agent evaluation scenarios.")
    parser.add_argument("--scenario", action="append", help="run only this scenario id (repeatable)")
    parser.add_argument("--no-judge", action="store_true", help="deterministic checks only")
    parser.add_argument("--out", type=Path, help="write the full report as JSON")
    args = parser.parse_args(argv)

    settings = load_settings()
    if not settings.api_key:
        print("OPENAI_API_KEY is not set. Add it to .env and run this again.")
        return 2

    store = Store(settings.db_path)
    dataset = store.info()
    store.close()
    if not dataset:
        print("No dataset has been ingested. Load one first:")
        print("  python -m incident_agent.cli --ingest <file.jsonl>")
        return 2

    chosen = [s for s in load_scenarios() if not args.scenario or s["id"] in args.scenario]
    if not chosen:
        print("No scenario matched.")
        return 2

    log = Recorder(settings.log_db_path, "eval", f"{len(chosen)} scenarios on {dataset['filename']}")
    log.event("eval.start", f"{len(chosen)} scenarios against {dataset['filename']}.",
              model=settings.model, judge=not args.no_judge, dataset=dataset["filename"])

    print(f"{dataset['filename']} | {dataset['event_count']} events | model {settings.model}"
          + ("" if not args.no_judge else " | deterministic checks only") + "\n")
    print(f"{'id':<5} {'res':<5} {'tools':<6} {'score':<5} {'scenario':<34} notes")
    print("-" * 110)

    results = []
    for scenario in chosen:
        transcript = run_scenario(scenario, settings)
        result = evaluate(scenario, transcript, settings, use_judge=not args.no_judge)
        results.append(result)
        print(report_line(result))
        log.event(
            "eval.scenario", f"{scenario['id']} {'passed' if result['pass'] else 'failed'}: {scenario['title']}",
            level="warn" if not result["pass"] else "info",
            scenario=scenario["id"], passed=result["pass"], critical=result["critical_error"],
            coverage=result["required_tool_coverage"], scores=result["judge_scores"],
        )

    passed = sum(r["pass"] for r in results)
    critical = sum(r["critical_error"] for r in results)
    print("-" * 110)
    print(f"{passed}/{len(results)} passed, {critical} with a critical error.")
    log.event("eval.done", f"{passed}/{len(results)} passed, {critical} critical.",
              level="warn" if critical else "info", passed=passed, total=len(results), critical=critical)
    log.finish(f"{passed}/{len(results)} passed, {critical} critical")

    if args.out:
        args.out.write_text(json.dumps({"dataset": dataset, "model": settings.model,
                                        "results": results}, indent=2), encoding="utf-8")
        print(f"Report written to {args.out}.")
    return 1 if passed < len(results) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
