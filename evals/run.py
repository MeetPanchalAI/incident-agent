"""Run the evaluation scenarios against the real model.

    python -m evals.run                 every scenario
    python -m evals.run --scenario 3    one scenario
    python -m evals.run --list          print what correct behaviour is, run nothing
    python -m evals.run --verbose       also print each final report

Needs OPENAI_API_KEY. The automated tests in tests/ do not.
"""

from __future__ import annotations

import argparse
import sys

from incident_agent import build_service, load_settings
from incident_agent.report import render_text

from .scenarios import SCENARIOS, Run, Scenario

PASS, FAIL = "PASS", "FAIL"


def describe() -> None:
    for scenario in SCENARIOS:
        print(f"\n{scenario.id:>2}. {scenario.category}")
        print(f"    world: {scenario.world}" + (f", faults: {scenario.faults}" if scenario.faults else ""))
        for prompt in scenario.prompts:
            print(f'    user: "{prompt}"')
        print(f"    expected: {scenario.expected}")
        print(f"    checks: {', '.join(name for name, _ in scenario.checks)}")


def execute(scenario: Scenario, verbose: bool) -> tuple[str, list[str], str]:
    settings = load_settings()
    agent = build_service(settings, world=scenario.world, faults=scenario.faults)
    session = agent.new_session()

    result = None
    for prompt in scenario.prompts:
        result = agent.run_turn(session, prompt)

    run = Run(session=session, result=result)
    failed = []
    for name, predicate in scenario.checks:
        try:
            if not predicate(run):
                failed.append(name)
        except Exception as error:  # a check must never mask a scenario failure
            failed.append(f"{name} (check raised {type(error).__name__})")

    detail = ""
    if verbose or failed:
        calls = ", ".join(f"{o.tool}:{o.status}" for o in run.observations) or "no tool calls"
        detail = f"      tools: {calls}\n"
        if verbose:
            report = render_text(result.outcome).replace("\n", "\n      ")
            detail += f"      {report}\n"
    return (FAIL if failed else PASS), failed, detail


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Run the incident-agent evaluation scenarios.")
    parser.add_argument("--scenario", type=int, action="append", help="run only this scenario id (repeatable)")
    parser.add_argument("--list", action="store_true", help="print the scenarios and exit")
    parser.add_argument("--verbose", action="store_true", help="print each final report")
    args = parser.parse_args(argv)

    if args.list:
        describe()
        return 0

    selected = [s for s in SCENARIOS if not args.scenario or s.id in args.scenario]
    if not selected:
        print("No scenario matched.")
        return 2

    print(f"{'#':>3}  {'result':<6} {'category':<42} failed checks")
    print("-" * 100)
    failures = 0
    for scenario in selected:
        try:
            status, failed, detail = execute(scenario, args.verbose)
        except Exception as error:
            status, failed, detail = FAIL, [f"raised {type(error).__name__}: {error}"], ""
        failures += status == FAIL
        print(f"{scenario.id:>3}  {status:<6} {scenario.category:<42} {'; '.join(failed)}")
        if detail:
            print(detail, end="")

    print("-" * 100)
    print(f"{len(selected) - failures}/{len(selected)} scenarios passed.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
