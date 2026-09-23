"""Settings: the fixed clock, the budgets, the model, the active world.

`NOW` is fixed rather than read from the system clock so that mock data,
tests and evaluations stay reproducible.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

DEFAULT_NOW = "2026-09-23T10:00:00Z"
DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_WORLD = "incident"


def parse_iso(value: str) -> datetime:
    """Parse an ISO 8601 timestamp into an aware UTC datetime."""
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def format_iso(value: datetime) -> str:
    """Format an aware datetime as a UTC ISO 8601 string."""
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class Budgets:
    """Per-turn limits. See DESIGN.md section "Budgets"."""

    max_llm_steps: int = 10
    max_tool_calls: int = 12
    stuck_threshold: int = 3
    tool_timeout_s: float = 2.0
    tool_retries: int = 1
    repair_attempts: int = 1


@dataclass(frozen=True)
class Settings:
    now: datetime
    model: str = DEFAULT_MODEL
    world: str = DEFAULT_WORLD
    api_key: str | None = None
    max_rows: int = 20
    max_window_days: int = 7
    budgets: Budgets = field(default_factory=Budgets)


def load_settings() -> Settings:
    """Build settings from the environment."""
    return Settings(
        now=parse_iso(os.getenv("AGENT_NOW", DEFAULT_NOW)),
        model=os.getenv("AGENT_MODEL", DEFAULT_MODEL),
        world=os.getenv("AGENT_WORLD", DEFAULT_WORLD),
        api_key=os.getenv("OPENAI_API_KEY"),
    )
