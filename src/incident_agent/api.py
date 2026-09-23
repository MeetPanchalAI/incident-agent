"""HTTP layer. Transport only: it holds no logic the CLI does not also use."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import threading

from . import AgentService, build_service, load_settings
from .agent import describe_dataset
from .llm import LLMClient
from .session import Session
from .state import (eval_results, load_session, recent_eval_runs, recent_runs, recent_sessions,
                    run_logs, save_session)
from .tools.ingest import IngestError, ingest
from .tools.store import Store
from evaluations.runner import EvalError, load_scenarios, run_suite

STATIC = Path(__file__).parent / "static"
MAX_UPLOAD_MB = 200

app = FastAPI(title="Incident Agent")
settings = load_settings()

# Tests set this to a scripted client so no API key or network is needed.
LLM_OVERRIDE: LLMClient | None = None

_sessions: dict[str, tuple[Session, list[dict]]] = {}
_agent: AgentService | None = None
_evaluation: dict = {"running": False, "run_id": None, "done": 0, "total": 0,
                     "results": [], "error": None}


def _service() -> AgentService:
    """One agent over the current dataset. Rebuilt whenever the data changes."""
    global _agent
    if _agent is None:
        try:
            _agent = build_service(settings, llm=LLM_OVERRIDE)
        except RuntimeError as error:  # no API key: say so instead of returning a 500
            raise HTTPException(503, str(error)) from error
    return _agent


def _reload() -> None:
    """Forget the agent and the open conversations; the data underneath has changed.

    The conversations stay in the state database as history; they are just not
    resumed against a dataset they were not asked about.
    """
    global _agent
    _agent = None
    _sessions.clear()


def _dataset_name() -> str:
    store = Store(settings.db_path)
    try:
        info = store.info()
        return info["filename"] if info else ""
    finally:
        store.close()


def _conversation(session_id: str | None) -> tuple[Session, list[dict]]:
    """The open conversation, resumed from the database if the server restarted."""
    if session_id and session_id in _sessions:
        return _sessions[session_id]
    dataset = _dataset_name()
    if session_id and (stored := load_session(settings.state_db_path, session_id)):
        if stored.dataset == dataset:
            _sessions[session_id] = (stored.session, stored.turns)
            return _sessions[session_id]
    session = _service().new_session()
    _sessions[session.id] = (session, [])
    return _sessions[session.id]


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: str | None = None


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health() -> dict:
    store = Store(settings.db_path)
    try:
        return {
            "status": "ok",
            "model": settings.model,
            "configured": bool(settings.api_key) or LLM_OVERRIDE is not None,
            "dataset": store.info(),
            "services": store.services(),
            "busiest": store.busiest_services(),
        }
    finally:
        store.close()


@app.post("/api/ingest")
async def ingest_upload(file: UploadFile) -> dict:
    """Replace the dataset with an uploaded JSONL log file."""
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File is larger than {MAX_UPLOAD_MB} MB.")
    try:
        report = ingest(raw.decode("utf-8", "replace").splitlines(),
                        settings.db_path, file.filename or "upload.jsonl", settings.state_db_path)
    except IngestError as error:
        raise HTTPException(400, str(error)) from error
    _reload()
    return report.to_dict()


@app.get("/api/runs")
def runs(limit: int = 50, kind: str | None = None) -> dict:
    return {"runs": recent_runs(settings.state_db_path, limit=min(limit, 200), kind=kind)}


@app.get("/api/runs/{run_id}")
def run_detail(run_id: str) -> dict:
    return {"logs": run_logs(settings.state_db_path, run_id)}


@app.post("/api/reset")
def reset() -> dict:
    session = _service().new_session()
    _sessions[session.id] = (session, [])
    return {"session_id": session.id}


@app.get("/api/sessions")
def sessions() -> dict:
    return {"sessions": recent_sessions(settings.state_db_path)}


@app.get("/api/sessions/{session_id}")
def session_detail(session_id: str) -> dict:
    stored = load_session(settings.state_db_path, session_id)
    if stored is None:
        raise HTTPException(404, "No such conversation.")
    return {"session_id": session_id, "dataset": stored.dataset, "turns": stored.turns}


@app.post("/api/chat")
def chat(request: ChatRequest) -> dict:
    agent = _service()
    if agent.store.is_empty():
        raise HTTPException(409, "No data has been ingested yet. Upload a log file first.")
    session, turns = _conversation(request.session_id)
    result = agent.run_turn(session, request.message)
    payload = {"question": request.message, **result.to_dict()}
    turns.append(payload)
    save_session(settings.state_db_path, session, turns, _dataset_name())
    return {"session_id": session.id, "dataset": describe_dataset(agent.store), **payload}


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


class EvalRequest(BaseModel):
    scenarios: list[str] | None = None
    judge: bool = True


def _evaluate(scenario_ids: list[str] | None, judge: bool) -> None:
    """Runs on a worker thread so the server stays responsive."""
    try:
        report = run_suite(settings, scenario_ids, use_judge=judge,
                           on_result=_evaluation["results"].append)
        _evaluation["run_id"] = report["run_id"]
    except EvalError as error:
        _evaluation["error"] = str(error)
    except Exception as error:  # a broken suite must not leave the UI spinning
        _evaluation["error"] = f"{type(error).__name__}: {error}"
    finally:
        _evaluation["running"] = False


@app.get("/api/evaluations/scenarios")
def evaluation_scenarios() -> dict:
    return {"scenarios": [{"id": s["id"], "title": s["title"], "question": s["question"]}
                          for s in load_scenarios()]}


@app.post("/api/evaluations/run")
def start_evaluation(request: EvalRequest) -> dict:
    if _evaluation["running"]:
        raise HTTPException(409, "An evaluation is already running.")
    total = len(request.scenarios or load_scenarios())
    _evaluation.update(running=True, run_id=None, done=0, total=total, results=[], error=None)
    threading.Thread(target=_evaluate, args=(request.scenarios, request.judge), daemon=True).start()
    return {"running": True, "total": total}


@app.get("/api/evaluations/status")
def evaluation_status() -> dict:
    return {**_evaluation, "done": len(_evaluation["results"])}


@app.get("/api/evaluations/runs")
def evaluation_runs() -> dict:
    return {"runs": recent_eval_runs(settings.state_db_path)}


@app.get("/api/evaluations/runs/{run_id}")
def evaluation_run(run_id: str) -> dict:
    return {"results": eval_results(settings.state_db_path, run_id)}
