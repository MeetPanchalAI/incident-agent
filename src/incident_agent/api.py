"""HTTP layer. Transport only: it holds no logic the CLI does not also use."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import AgentService, build_service, load_settings
from .llm import LLMClient
from .session import Session
from .tools.mock_backend import available_worlds

STATIC = Path(__file__).parent / "static"

app = FastAPI(title="Incident Agent")
settings = load_settings()

# Tests set this to a scripted client so no API key or network is needed.
LLM_OVERRIDE: LLMClient | None = None

_sessions: dict[str, Session] = {}
_services: dict[str, AgentService] = {}


def _service(world: str) -> AgentService:
    if world not in available_worlds():
        raise HTTPException(400, f"Unknown world '{world}'. Available: {', '.join(available_worlds())}")
    if LLM_OVERRIDE is not None:
        return build_service(settings, world=world, llm=LLM_OVERRIDE)
    if world not in _services:
        try:
            _services[world] = build_service(settings, world=world)
        except RuntimeError as error:  # no API key: say so instead of returning a 500
            raise HTTPException(503, str(error)) from error
    return _services[world]


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: str | None = None
    world: str | None = None


class ResetRequest(BaseModel):
    world: str | None = None


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "now": settings.now.isoformat(),
        "worlds": available_worlds(),
        "model": settings.model,
        "configured": bool(settings.api_key) or LLM_OVERRIDE is not None,
    }


@app.post("/api/reset")
def reset(request: ResetRequest) -> dict:
    world = request.world or settings.world
    session = _service(world).new_session()
    _sessions[session.id] = session
    return {"session_id": session.id, "world": world}


@app.post("/api/chat")
def chat(request: ChatRequest) -> dict:
    session = _sessions.get(request.session_id) if request.session_id else None
    if session is None:
        world = request.world or settings.world
        session = _service(world).new_session()
        _sessions[session.id] = session
    result = _service(session.world).run_turn(session, request.message)
    return {"session_id": session.id, "world": session.world, **result.to_dict()}
