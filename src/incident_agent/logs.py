"""Run logs.

Every workflow — an ingest, an agent turn, an evaluation run — opens a `run`
and writes ordered `log` events to it. One line per thing the system actually
did, with the few fields needed to understand it and nothing else.

Logs live in their own database, because ingesting replaces the dataset and
the record of what happened must survive that.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

LEVELS = ("info", "warn", "error")
KINDS = ("ingest", "turn", "eval")
MAX_DATA = 2000

SCHEMA = """
CREATE TABLE IF NOT EXISTS run (
    id TEXT PRIMARY KEY, kind TEXT NOT NULL, label TEXT,
    started_at TEXT NOT NULL, ended_at TEXT, duration_ms INTEGER,
    status TEXT, summary TEXT
);
CREATE TABLE IF NOT EXISTS log (
    run_id TEXT NOT NULL, seq INTEGER NOT NULL, at TEXT NOT NULL,
    level TEXT NOT NULL, event TEXT NOT NULL, message TEXT NOT NULL, data TEXT
);
CREATE INDEX IF NOT EXISTS log_by_run ON log (run_id, seq);
CREATE INDEX IF NOT EXISTS run_recent ON run (started_at DESC);
"""


def connect(path: str | Path) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.executescript(SCHEMA)
    return connection


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class Recorder:
    """One run. Create it, log events, finish it."""

    db_path: str | Path
    kind: str
    label: str = ""

    def __post_init__(self) -> None:
        self.id = uuid4().hex[:12]
        self._seq = 0
        self._started = time.monotonic()
        self._db = connect(self.db_path)
        self._db.execute(
            "INSERT INTO run (id, kind, label, started_at) VALUES (?, ?, ?, ?)",
            (self.id, self.kind, self.label[:200], _now()),
        )
        self._db.commit()

    def event(self, event: str, message: str, level: str = "info", **data) -> None:
        self._seq += 1
        payload = json.dumps(data, default=str)[:MAX_DATA] if data else None
        self._db.execute(
            "INSERT INTO log (run_id, seq, at, level, event, message, data) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (self.id, self._seq, _now(), level, event, message[:500], payload),
        )
        self._db.commit()

    def finish(self, summary: str, status: str = "ok") -> None:
        self._db.execute(
            "UPDATE run SET ended_at = ?, duration_ms = ?, status = ?, summary = ? WHERE id = ?",
            (_now(), round((time.monotonic() - self._started) * 1000), status, summary[:500], self.id),
        )
        self._db.commit()
        self._db.close()


def recent_runs(db_path: str | Path, limit: int = 50, kind: str | None = None) -> list[dict]:
    db = connect(db_path)
    try:
        sql = "SELECT * FROM run"
        params: list = []
        if kind:
            sql += " WHERE kind = ?"
            params.append(kind)
        rows = db.execute(sql + " ORDER BY started_at DESC, rowid DESC LIMIT ?", [*params, limit])
        return [dict(r) for r in rows]
    finally:
        db.close()


def run_logs(db_path: str | Path, run_id: str) -> list[dict]:
    db = connect(db_path)
    try:
        rows = db.execute("SELECT * FROM log WHERE run_id = ? ORDER BY seq", (run_id,))
        return [{**dict(r), "data": json.loads(r["data"]) if r["data"] else None} for r in rows]
    finally:
        db.close()
