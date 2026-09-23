"""HTTP layer. Transport only: it holds no logic the CLI does not also use."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import AgentService, build_service, load_settings
from .agent import describe_dataset
from .llm import LLMClient
from .session import Session
from .tools.ingest import IngestError, ingest
from .tools.store import Store

STATIC = Path(__file__).parent / "static"
MAX_UPLOAD_MB = 200

app = FastAPI(title="Incident Agent")
settings = load_settings()

# Tests set this to a scripted client so no API key or network is needed.
LLM_OVERRIDE: LLMClient | None = None

_sessions: dict[str, Session] = {}
_agent: AgentService | None = None


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
    """Forget the agent and every session; the data underneath them has changed."""
    global _agent
    _agent = None
    _sessions.clear()


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
                        settings.db_path, file.filename or "upload.jsonl")
    except IngestError as error:
        raise HTTPException(400, str(error)) from error
    _reload()
    return report.to_dict()


@app.post("/api/reset")
def reset() -> dict:
    session = _service().new_session()
    _sessions[session.id] = session
    return {"session_id": session.id}


@app.post("/api/chat")
def chat(request: ChatRequest) -> dict:
    agent = _service()
    if agent.store.is_empty():
        raise HTTPException(409, "No data has been ingested yet. Upload a log file first.")
    session = _sessions.get(request.session_id) if request.session_id else None
    if session is None:
        session = agent.new_session()
        _sessions[session.id] = session
    result = agent.run_turn(session, request.message)
    return {"session_id": session.id, "dataset": describe_dataset(agent.store), **result.to_dict()}
