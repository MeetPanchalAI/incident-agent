"""The data store the tools read from.

One SQLite file holding one dataset. Events are the source of truth; metrics,
deployments and dependencies are derived from them at ingest time (see
ingest.py) and stored as rows.

Deriving rather than generating is what makes the agent honest: if a service
never emitted a metric, there are no rows, `get_metrics` returns nothing, and
the agent reports "no matching records" instead of reasoning over an invented
series.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

from ..config import format_iso, parse_iso

# Metric definitions. `min_delta` stops a tiny baseline turning noise into a
# "3x spike"; see DESIGN.md.
METRICS: dict[str, dict] = {
    "error_rate": {"unit": "fraction", "min_delta": 0.01},
    "latency_p95_ms": {"unit": "ms", "min_delta": 100},
    "request_rate": {"unit": "requests/min", "min_delta": 10},
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS dataset (
    filename TEXT, ingested_at TEXT, event_count INTEGER,
    skipped INTEGER, first_ts TEXT, last_ts TEXT
);
CREATE TABLE IF NOT EXISTS event (
    ts TEXT NOT NULL, service TEXT NOT NULL, level TEXT NOT NULL, message TEXT NOT NULL,
    latency_ms REAL, status_code INTEGER, target TEXT
);
CREATE TABLE IF NOT EXISTS metric_point (
    ts TEXT NOT NULL, service TEXT NOT NULL, metric TEXT NOT NULL, value REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS deployment (
    ts TEXT NOT NULL, service TEXT NOT NULL, version TEXT NOT NULL, status TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS dependency (service TEXT NOT NULL, depends_on TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS service (name TEXT PRIMARY KEY);

CREATE INDEX IF NOT EXISTS event_lookup ON event (service, ts);
CREATE INDEX IF NOT EXISTS metric_lookup ON metric_point (service, metric, ts);
CREATE INDEX IF NOT EXISTS deployment_lookup ON deployment (service, ts);
"""


class TransientError(Exception):
    """A failure that may succeed if the call is retried."""


class MalformedResponse(Exception):
    """The store returned something that does not match the expected shape."""


def connect(path: str | Path) -> sqlite3.Connection:
    """Open the database, creating the schema if this is a new file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.executescript(SCHEMA)
    return connection


class Store:
    """Read access for the tools. `faults` injects failures, for tests and evals."""

    def __init__(self, path: str | Path, faults: dict[str, str] | None = None) -> None:
        self.path = Path(path)
        self._db = connect(path)
        self.faults = dict(faults or {})
        self._attempts: dict[str, int] = {}

    def close(self) -> None:
        self._db.close()

    # -- dataset ----------------------------------------------------------

    def info(self) -> dict | None:
        """The ingested dataset, or None if nothing has been loaded yet."""
        row = self._db.execute("SELECT * FROM dataset LIMIT 1").fetchone()
        return dict(row) if row else None

    def is_empty(self) -> bool:
        return self.info() is None

    def now(self) -> datetime | None:
        """The last event in the dataset. This is the agent's "current time"."""
        info = self.info()
        return parse_iso(info["last_ts"]) if info and info["last_ts"] else None

    def services(self) -> list[str]:
        return [r["name"] for r in self._db.execute("SELECT name FROM service ORDER BY name")]

    def busiest_services(self, limit: int = 3) -> list[str]:
        """Services that emit the most events. One discovered only as a call
        target emits none, so it is never suggested as a starting point."""
        return [r["service"] for r in self._db.execute(
            "SELECT service, COUNT(*) AS n FROM event GROUP BY service ORDER BY n DESC LIMIT ?", (limit,))]

    # -- tool surfaces ----------------------------------------------------

    def get_metrics(self, service: str, metric: str, start: datetime, end: datetime) -> list[dict]:
        if (fault := self._apply_fault("get_metrics")) == "empty":
            return []
        if fault == "malformed":
            return [{"timestamp": "not-a-timestamp", "value": None}]
        rows = self._db.execute(
            "SELECT ts, value FROM metric_point WHERE service = ? AND metric = ? AND ts >= ? AND ts < ? ORDER BY ts",
            (service, metric, format_iso(start), format_iso(end)),
        )
        return [{"timestamp": r["ts"], "value": r["value"]} for r in rows]

    def search_logs(self, service: str, start: datetime, end: datetime, query: str) -> list[dict]:
        if (fault := self._apply_fault("search_logs")) == "empty":
            return []
        if fault == "malformed":
            return [{"level": "ERROR"}]
        sql = "SELECT ts, level, message FROM event WHERE service = ? AND ts >= ? AND ts < ?"
        params: list = [service, format_iso(start), format_iso(end)]
        if needle := query.strip():
            sql += " AND (level || ' ' || message) LIKE ?"
            params.append(f"%{needle}%")
        rows = self._db.execute(sql + " ORDER BY ts", params)
        return [{"timestamp": r["ts"], "level": r["level"], "message": r["message"]} for r in rows]

    def get_deployments(self, service: str, start: datetime, end: datetime) -> list[dict]:
        if (fault := self._apply_fault("get_deployments")) == "empty":
            return []
        if fault == "malformed":
            return [{"service": service, "version": None, "deployed_at": "not-a-timestamp"}]
        rows = self._db.execute(
            "SELECT ts, service, version, status FROM deployment WHERE service = ? AND ts >= ? AND ts < ? ORDER BY ts",
            (service, format_iso(start), format_iso(end)),
        )
        return [
            {"service": r["service"], "version": r["version"], "deployed_at": r["ts"], "status": r["status"]}
            for r in rows
        ]

    def get_service_dependencies(self, service: str) -> dict:
        if (fault := self._apply_fault("get_service_dependencies")) == "empty":
            return {"service": service, "upstream": [], "downstream": []}
        if fault == "malformed":
            return {"service": service}
        upstream = [r["depends_on"] for r in self._db.execute(
            "SELECT DISTINCT depends_on FROM dependency WHERE service = ? ORDER BY depends_on", (service,))]
        downstream = [r["service"] for r in self._db.execute(
            "SELECT DISTINCT service FROM dependency WHERE depends_on = ? ORDER BY service", (service,))]
        return {"service": service, "upstream": upstream, "downstream": downstream}

    # -- fault injection --------------------------------------------------

    def _apply_fault(self, tool: str) -> str | None:
        fault = self.faults.get(tool)
        if fault is None:
            return None
        self._attempts[tool] = self._attempts.get(tool, 0) + 1
        if fault == "timeout":
            raise TimeoutError(f"{tool} did not respond in time")
        if fault == "transient" and self._attempts[tool] == 1:
            raise TransientError(f"{tool} is temporarily unavailable")
        if fault == "malformed_once":
            return "malformed" if self._attempts[tool] == 1 else None
        return fault if fault in ("empty", "malformed") else None
