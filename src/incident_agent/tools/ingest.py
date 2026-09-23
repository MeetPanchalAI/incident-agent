"""Load a log file into the store, and derive the other tool surfaces from it.

The input is JSONL: one JSON object per line. Events are the only thing the
file provides; metrics, deployments and dependencies are computed from them
here, once, so those definitions live in one readable place rather than inside
the tools.

Required per line:  ts, service, level, message
Optional:           latency_ms, status_code, target, event_type, version, status

`latency_ms` is what produces latency_p95_ms. `target` (the service being
called) is what produces the dependency graph. `event_type: "deployment"` with
a `version` is what produces a release record.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator

from ..config import format_iso, parse_iso
from ..state import Recorder
from .store import connect

BATCH = 5000
MAX_PROBLEMS = 10
LEVELS = ("DEBUG", "INFO", "WARN", "WARNING", "ERROR", "FATAL", "CRITICAL")


@dataclass
class IngestReport:
    """What the ingest actually did. Shown in the UI; never rounded up."""

    filename: str
    events: int = 0
    skipped: int = 0
    services: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    deployments: int = 0
    dependencies: int = 0
    first_ts: str | None = None
    last_ts: str | None = None
    problems: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "filename": self.filename, "events": self.events, "skipped": self.skipped,
            "services": self.services, "metrics": self.metrics,
            "deployments": self.deployments, "dependencies": self.dependencies,
            "first_ts": self.first_ts, "last_ts": self.last_ts, "problems": self.problems,
        }


class IngestError(Exception):
    """The file could not be used at all."""


def _parse_line(line: str) -> dict:
    """Return a normalised event, or raise ValueError with the reason."""
    record = json.loads(line)
    if not isinstance(record, dict):
        raise ValueError("line is not a JSON object")
    missing = [f for f in ("ts", "service", "level", "message") if not record.get(f)]
    if missing:
        raise ValueError(f"missing {', '.join(missing)}")
    level = str(record["level"]).strip().upper()
    if level not in LEVELS:
        raise ValueError(f"unknown level '{record['level']}'")
    return {
        "ts": format_iso(parse_iso(str(record["ts"]))),
        "service": str(record["service"]).strip().lower(),
        "level": "WARN" if level == "WARNING" else level,
        "message": str(record["message"]),
        "latency_ms": float(record["latency_ms"]) if record.get("latency_ms") is not None else None,
        "status_code": int(record["status_code"]) if record.get("status_code") is not None else None,
        "target": str(record["target"]).strip().lower() if record.get("target") else None,
        "is_deployment": str(record.get("event_type", "log")).lower() == "deployment",
        "version": str(record["version"]) if record.get("version") else None,
        "status": str(record.get("status", "success")),
    }


def ingest(lines: Iterable[str], db_path: str | Path, filename: str,
           log_db: str | Path | None = None) -> IngestReport:
    """Replace the dataset with the contents of `lines`."""
    report = IngestReport(filename=filename)
    log = Recorder(log_db, "ingest", filename) if log_db else None
    if log:
        log.event("ingest.start", f"Replacing the dataset with {filename}.")
    db = connect(db_path)
    try:
        for table in ("dataset", "event", "metric_point", "deployment", "dependency", "service"):
            db.execute(f"DELETE FROM {table}")

        for batch in _batched(_events(lines, report), BATCH):
            db.executemany(
                "INSERT INTO event (ts, service, level, message, latency_ms, status_code, target) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                [(e["ts"], e["service"], e["level"], e["message"],
                  e["latency_ms"], e["status_code"], e["target"]) for e in batch],
            )
            deployments = [e for e in batch if e["is_deployment"] and e["version"]]
            if deployments:
                db.executemany(
                    "INSERT INTO deployment (ts, service, version, status) VALUES (?, ?, ?, ?)",
                    [(e["ts"], e["service"], e["version"], e["status"]) for e in deployments],
                )
            report.events += len(batch)

        if log:
            log.event("ingest.parsed", f"{report.events} events parsed, {report.skipped} skipped.",
                      level="warn" if report.skipped else "info",
                      events=report.events, skipped=report.skipped, problems=report.problems[:5])
        if report.events == 0:
            raise IngestError(
                f"No usable events in {filename}. {report.skipped} line(s) were skipped. "
                + (f"First problems: {'; '.join(report.problems[:3])}" if report.problems else "")
            )

        _derive(db)
        _summarise(db, report)
        if log:
            log.event("ingest.derived",
                      f"{len(report.services)} services, metrics {', '.join(report.metrics)}, "
                      f"{report.deployments} deployments, {report.dependencies} dependency edges.",
                      services=report.services, metrics=report.metrics,
                      deployments=report.deployments, dependencies=report.dependencies)
        db.execute(
            "INSERT INTO dataset (filename, ingested_at, event_count, skipped, first_ts, last_ts) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (filename, format_iso(datetime.now().astimezone()), report.events,
             report.skipped, report.first_ts, report.last_ts),
        )
        db.commit()
    except IngestError as error:
        if log:
            log.event("ingest.failed", str(error), level="error")
            log.finish("refused: no usable events", status="error")
        raise
    finally:
        db.close()
    if log:
        log.finish(f"{report.events} events, {report.skipped} skipped, "
                   f"{report.first_ts} to {report.last_ts}")
    return report


def ingest_file(path: str | Path, db_path: str | Path, log_db: str | Path | None = None) -> IngestReport:
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        return ingest(handle, db_path, path.name, log_db)


def _events(lines: Iterable[str], report: IngestReport) -> Iterator[dict]:
    for number, line in enumerate(lines, start=1):
        text = line.strip() if isinstance(line, str) else line.decode("utf-8", "replace").strip()
        if not text:
            continue
        try:
            yield _parse_line(text)
        except (ValueError, json.JSONDecodeError) as error:
            report.skipped += 1
            if len(report.problems) < MAX_PROBLEMS:
                report.problems.append(f"line {number}: {error}")


def _batched(items: Iterator[dict], size: int) -> Iterator[list[dict]]:
    batch: list[dict] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def _derive(db) -> None:
    """Compute the other three tool surfaces from the events, in SQL."""
    minute = "substr(ts, 1, 16) || ':00Z'"

    db.execute("INSERT INTO service (name) SELECT DISTINCT service FROM event")
    db.execute(
        "INSERT OR IGNORE INTO service (name) "
        "SELECT DISTINCT target FROM event WHERE target IS NOT NULL AND target <> ''"
    )
    db.execute(
        "INSERT INTO dependency (service, depends_on) "
        "SELECT DISTINCT service, target FROM event "
        "WHERE target IS NOT NULL AND target <> '' AND target <> service"
    )
    db.execute(
        f"INSERT INTO metric_point (ts, service, metric, value) "
        f"SELECT {minute}, service, 'request_rate', COUNT(*) FROM event GROUP BY 1, 2"
    )
    db.execute(
        f"INSERT INTO metric_point (ts, service, metric, value) "
        f"SELECT {minute}, service, 'error_rate', "
        f"CAST(SUM(CASE WHEN level IN ('ERROR', 'FATAL', 'CRITICAL') THEN 1 ELSE 0 END) AS REAL) / COUNT(*) "
        f"FROM event GROUP BY 1, 2"
    )
    # p95 without a percentile function: rank each minute's latencies and take
    # the value at position n - floor(n * 0.05), which is the 95th percentile
    # for every n >= 1. Only minutes that actually recorded a latency appear,
    # so a service that never reports one simply has no rows for this metric.
    db.execute(
        f"INSERT INTO metric_point (ts, service, metric, value) "
        f"WITH ranked AS ("
        f"  SELECT {minute} AS minute, service, latency_ms,"
        f"         ROW_NUMBER() OVER (PARTITION BY {minute}, service ORDER BY latency_ms) AS rn,"
        f"         COUNT(*) OVER (PARTITION BY {minute}, service) AS n"
        f"  FROM event WHERE latency_ms IS NOT NULL"
        f") SELECT minute, service, 'latency_p95_ms', latency_ms FROM ranked "
        f"WHERE rn = n - CAST(n * 0.05 AS INTEGER)"
    )


def _summarise(db, report: IngestReport) -> None:
    row = db.execute("SELECT MIN(ts) AS first, MAX(ts) AS last FROM event").fetchone()
    report.first_ts, report.last_ts = row["first"], row["last"]
    report.services = [r["name"] for r in db.execute("SELECT name FROM service ORDER BY name")]
    report.metrics = [r["metric"] for r in db.execute("SELECT DISTINCT metric FROM metric_point ORDER BY metric")]
    report.deployments = db.execute("SELECT COUNT(*) AS n FROM deployment").fetchone()["n"]
    report.dependencies = db.execute("SELECT COUNT(*) AS n FROM dependency").fetchone()["n"]
