"""Command-line interface.

    python -m incident_agent.cli                     interactive
    python -m incident_agent.cli "question"          one question and exit
    python -m incident_agent.cli --ingest logs.jsonl load a dataset and exit
"""

from __future__ import annotations

import sys

from . import build_service, load_settings
from .agent import describe_dataset
from .report import render_text
from .tools.ingest import IngestError, ingest_file
from .tools.store import Store

HELP = """Commands:
  /ingest <path>  replace the dataset with a JSONL log file
  /data           show the loaded dataset
  /trace          show the tool calls from the last turn
  /reset          start a new session
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


def _load(path: str, db_path, log_db) -> bool:
    try:
        report = ingest_file(path, db_path, log_db)
    except (IngestError, OSError) as error:
        print(error)
        return False
    print(f"Ingested {report.events} events from {report.filename}"
          + (f", skipped {report.skipped}" if report.skipped else "") + ".")
    print(f"  {report.first_ts} to {report.last_ts}")
    print(f"  services: {', '.join(report.services)}")
    print(f"  metrics: {', '.join(report.metrics)} | deployments: {report.deployments}"
          f" | dependency edges: {report.dependencies}")
    for problem in report.problems[:3]:
        print(f"  skipped: {problem}")
    return True


def main(argv: list[str]) -> int:
    settings = load_settings()

    if argv and argv[0] == "--ingest":
        if len(argv) < 2:
            print("Usage: python -m incident_agent.cli --ingest <path>")
            return 2
        return 0 if _load(argv[1], settings.db_path, settings.log_db_path) else 1

    store = Store(settings.db_path)
    empty = store.is_empty()
    summary = describe_dataset(store)
    store.close()
    if empty:
        print("No data has been ingested yet. Load a log file first:")
        print("  python -m incident_agent.cli --ingest <path.jsonl>")
        return 1

    try:
        service = build_service(settings)
    except (RuntimeError, ValueError) as error:
        print(error)
        return 1
    session = service.new_session()
    last_trace: list[dict] = []

    if argv:
        print(render_text(service.run_turn(session, " ".join(argv)).outcome))
        return 0

    print(f"Incident agent. Dataset: {summary}.")
    print(f"Current time: {service.settings.now:%Y-%m-%d %H:%M}Z. Type /help for commands.")
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
            print(HELP)
            continue
        if line == "/data":
            print(describe_dataset(service.store))
            continue
        if line == "/trace":
            _print_trace(last_trace)
            continue
        if line == "/reset":
            session = service.new_session()
            print("New session.")
            continue
        if line.startswith("/ingest "):
            if _load(line.split(maxsplit=1)[1].strip(), settings.db_path, settings.log_db_path):
                service = build_service(settings)
                session = service.new_session()
            continue

        result = service.run_turn(session, line)
        last_trace = result.trace
        print()
        print(render_text(result.outcome))
        print(f"\n[{len(result.trace)} tool calls, {result.llm_calls} model calls, {result.steps} steps]")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
