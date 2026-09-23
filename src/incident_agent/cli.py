"""Command-line interface.

    python -m incident_agent.cli                 interactive
    python -m incident_agent.cli "question"      one question and exit
"""

from __future__ import annotations

import sys

from . import build_service, load_settings
from .report import render_text
from .tools.mock_backend import available_worlds

HELP = """Commands:
  /world <name>   switch the mock world and start a new session ({worlds})
  /trace          show the tool calls from the last turn
  /reset          start a new session in the same world
  /help           show this
  /quit           exit"""


def _print_trace(trace: list[dict]) -> None:
    if not trace:
        print("  (no tool calls)")
        return
    for row in trace:
        args = ", ".join(f"{k}={v}" for k, v in row["args"].items())
        attempts = f" x{row['attempts']}" if row["attempts"] > 1 else ""
        print(f"  {row['observation_id']}  {row['status']:<17}{attempts} {row['tool']}({args})")


def main(argv: list[str]) -> int:
    settings = load_settings()
    world = settings.world
    try:
        service = build_service(settings, world=world)
    except (RuntimeError, ValueError) as error:
        print(error)
        return 1
    session = service.new_session()
    last_trace: list[dict] = []

    if argv:
        result = service.run_turn(session, " ".join(argv))
        print(render_text(result.outcome))
        return 0

    print(f"Incident agent. World: {world}. Now: {settings.now:%Y-%m-%d %H:%M}Z. Type /help for commands.")
    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            return 0
        if not line:
            continue
        if line in ("/quit", "/exit"):
            return 0
        if line == "/help":
            print(HELP.format(worlds=", ".join(available_worlds())))
            continue
        if line == "/trace":
            _print_trace(last_trace)
            continue
        if line == "/reset":
            session = service.new_session()
            print(f"New session in world '{world}'.")
            continue
        if line.startswith("/world "):
            name = line.split(maxsplit=1)[1].strip()
            if name not in available_worlds():
                print(f"Unknown world. Available: {', '.join(available_worlds())}")
                continue
            world = name
            service = build_service(settings, world=world)
            session = service.new_session()
            print(f"New session in world '{world}'.")
            continue

        result = service.run_turn(session, line)
        last_trace = result.trace
        print()
        print(render_text(result.outcome))
        print(f"\n[{len(result.trace)} tool calls, {result.llm_calls} model calls, {result.steps} steps]")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
