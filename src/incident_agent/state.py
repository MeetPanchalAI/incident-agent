"""Everything the product remembers, in one SQLite file.

The dataset lives in its own database and is replaced whenever someone uploads
a new log file. This one is not: conversations, run logs and evaluation results
have to survive that, and a server restart.

Three groups of tables, each with the smallest shape that serves it:
  run, log                        what every workflow did
  session, turn, message, obs     conversations, replayable and resumable
  eval_run, eval_result           evaluation results, comparable over time
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .session import Observation, Session

MAX_DATA = 2000

SCHEMA = """
CREATE TABLE IF NOT EXISTS run (
    id TEXT PRIMARY KEY, kind TEXT NOT NULL, label TEXT,
    started_at TEXT NOT NULL, ended_at TEXT, duration_ms INTEGER, status TEXT, summary TEXT
);
CREATE TABLE IF NOT EXISTS log (
    run_id TEXT NOT NULL, seq INTEGER NOT NULL, at TEXT NOT NULL,
    level TEXT NOT NULL, event TEXT NOT NULL, message TEXT NOT NULL, data TEXT
);
CREATE TABLE IF NOT EXISTS session (
    id TEXT PRIMARY KEY, dataset TEXT, created_at TEXT, updated_at TEXT, turns INTEGER, title TEXT
);
CREATE TABLE IF NOT EXISTS turn (session_id TEXT NOT NULL, n INTEGER NOT NULL, payload TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS message (session_id TEXT NOT NULL, seq INTEGER NOT NULL, item TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS obs (session_id TEXT NOT NULL, seq INTEGER NOT NULL, payload TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS eval_run (
    id TEXT PRIMARY KEY, started_at TEXT, ended_at TEXT, dataset TEXT, model TEXT,
    judged INTEGER, passed INTEGER, total INTEGER, critical INTEGER
);
CREATE TABLE IF NOT EXISTS eval_result (run_id TEXT NOT NULL, scenario_id TEXT NOT NULL, payload TEXT NOT NULL);

CREATE INDEX IF NOT EXISTS log_by_run ON log (run_id, seq);
CREATE INDEX IF NOT EXISTS run_recent ON run (started_at DESC);
CREATE INDEX IF NOT EXISTS turn_by_session ON turn (session_id, n);
CREATE INDEX IF NOT EXISTS message_by_session ON message (session_id, seq);
CREATE INDEX IF NOT EXISTS obs_by_session ON obs (session_id, seq);
CREATE INDEX IF NOT EXISTS eval_result_by_run ON eval_result (run_id);
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


# --------------------------------------------------------------------------
# Run logs
# --------------------------------------------------------------------------


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
        self._db.execute("INSERT INTO run (id, kind, label, started_at) VALUES (?, ?, ?, ?)",
                         (self.id, self.kind, self.label[:200], _now()))
        self._db.commit()

    def event(self, event: str, message: str, level: str = "info", **data) -> None:
        self._seq += 1
        self._db.execute(
            "INSERT INTO log (run_id, seq, at, level, event, message, data) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (self.id, self._seq, _now(), level, event, message[:600],
             json.dumps(data, default=str)[:MAX_DATA] if data else None))
        self._db.commit()

    def finish(self, summary: str, status: str = "ok") -> None:
        self._db.execute(
            "UPDATE run SET ended_at = ?, duration_ms = ?, status = ?, summary = ? WHERE id = ?",
            (_now(), round((time.monotonic() - self._started) * 1000), status, summary[:500], self.id))
        self._db.commit()
        self._db.close()


def recent_runs(db_path: str | Path, limit: int = 50, kind: str | None = None) -> list[dict]:
    db = connect(db_path)
    try:
        sql, params = "SELECT * FROM run", []
        if kind:
            sql, params = sql + " WHERE kind = ?", [kind]
        return [dict(r) for r in db.execute(sql + " ORDER BY started_at DESC, rowid DESC LIMIT ?",
                                            [*params, limit])]
    finally:
        db.close()


def run_logs(db_path: str | Path, run_id: str) -> list[dict]:
    db = connect(db_path)
    try:
        rows = db.execute("SELECT * FROM log WHERE run_id = ? ORDER BY seq", (run_id,))
        return [{**dict(r), "data": json.loads(r["data"]) if r["data"] else None} for r in rows]
    finally:
        db.close()


# --------------------------------------------------------------------------
# Conversations
# --------------------------------------------------------------------------


@dataclass
class StoredSession:
    session: Session
    turns: list[dict] = field(default_factory=list)
    dataset: str = ""


def save_session(db_path: str | Path, session: Session, turns: list[dict], dataset: str) -> None:
    """Write the whole conversation. Small enough to rewrite after each turn."""
    db = connect(db_path)
    try:
        title = next((t["question"] for t in turns), "")
        db.execute(
            "INSERT INTO session (id, dataset, created_at, updated_at, turns, title) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET updated_at = excluded.updated_at, turns = excluded.turns, "
            "title = excluded.title, dataset = excluded.dataset",
            (session.id, dataset, _now(), _now(), len(turns), title[:200]))
        for table in ("turn", "message", "obs"):
            db.execute(f"DELETE FROM {table} WHERE session_id = ?", (session.id,))
        db.executemany("INSERT INTO turn (session_id, n, payload) VALUES (?, ?, ?)",
                       [(session.id, n, json.dumps(t, default=str)) for n, t in enumerate(turns)])
        db.executemany("INSERT INTO message (session_id, seq, item) VALUES (?, ?, ?)",
                       [(session.id, n, json.dumps(m, default=str)) for n, m in enumerate(session.messages)])
        db.executemany("INSERT INTO obs (session_id, seq, payload) VALUES (?, ?, ?)",
                       [(session.id, n, json.dumps(vars(o), default=str))
                        for n, o in enumerate(session.observations)])
        db.commit()
    finally:
        db.close()


def load_session(db_path: str | Path, session_id: str) -> StoredSession | None:
    """Rebuild a conversation so the agent can continue it and the UI can show it."""
    db = connect(db_path)
    try:
        row = db.execute("SELECT * FROM session WHERE id = ?", (session_id,)).fetchone()
        if row is None:
            return None
        session = Session(id=session_id, turn=row["turns"])
        session.messages = [json.loads(r["item"]) for r in
                            db.execute("SELECT item FROM message WHERE session_id = ? ORDER BY seq", (session_id,))]
        for record in db.execute("SELECT payload FROM obs WHERE session_id = ? ORDER BY seq", (session_id,)):
            session.record(Observation(**json.loads(record["payload"])))
        turns = [json.loads(r["payload"]) for r in
                 db.execute("SELECT payload FROM turn WHERE session_id = ? ORDER BY n", (session_id,))]
        return StoredSession(session=session, turns=turns, dataset=row["dataset"] or "")
    finally:
        db.close()


def recent_sessions(db_path: str | Path, limit: int = 30) -> list[dict]:
    db = connect(db_path)
    try:
        return [dict(r) for r in db.execute(
            "SELECT id, dataset, updated_at, turns, title FROM session "
            "WHERE turns > 0 ORDER BY updated_at DESC, rowid DESC LIMIT ?", (limit,))]
    finally:
        db.close()


# --------------------------------------------------------------------------
# Evaluation results
# --------------------------------------------------------------------------


def save_eval_run(db_path: str | Path, run_id: str, dataset: str, model: str,
                  judged: bool, results: list[dict], started_at: str) -> None:
    db = connect(db_path)
    try:
        db.execute(
            "INSERT INTO eval_run (id, started_at, ended_at, dataset, model, judged, passed, total, critical) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET ended_at = excluded.ended_at, passed = excluded.passed, "
            "total = excluded.total, critical = excluded.critical",
            (run_id, started_at, _now(), dataset, model, int(judged),
             sum(r["pass"] for r in results), len(results), sum(r["critical_error"] for r in results)))
        db.execute("DELETE FROM eval_result WHERE run_id = ?", (run_id,))
        db.executemany("INSERT INTO eval_result (run_id, scenario_id, payload) VALUES (?, ?, ?)",
                       [(run_id, r["scenario_id"], json.dumps(r, default=str)) for r in results])
        db.commit()
    finally:
        db.close()


def recent_eval_runs(db_path: str | Path, limit: int = 20) -> list[dict]:
    db = connect(db_path)
    try:
        return [dict(r) for r in db.execute(
            "SELECT * FROM eval_run ORDER BY started_at DESC, rowid DESC LIMIT ?", (limit,))]
    finally:
        db.close()


def eval_results(db_path: str | Path, run_id: str) -> list[dict]:
    db = connect(db_path)
    try:
        return [json.loads(r["payload"]) for r in
                db.execute("SELECT payload FROM eval_result WHERE run_id = ? ORDER BY rowid", (run_id,))]
    finally:
        db.close()
