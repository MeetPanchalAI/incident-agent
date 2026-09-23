"""Settings.

Every tunable parameter is read from the environment, so the agent can be
retuned and re-measured without touching the code. Defaults match the values
the evaluation scenarios were written against.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DEFAULT_NOW = "2026-09-23T10:00:00Z"
DEFAULT_MODEL = "gpt-5.6-luna"
DEFAULT_WORLD = "incident"
PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"


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


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value is None or not value.strip() else value.strip()


def _int(name: str, default: int) -> int:
    return int(_env(name, str(default)))


def _float(name: str, default: float) -> float:
    return float(_env(name, str(default)))


@dataclass(frozen=True)
class Budgets:
    """Per-turn limits. Raising these buys more investigation for more money."""

    max_llm_steps: int = 10
    max_tool_calls: int = 12
    stuck_threshold: int = 3
    tool_timeout_s: float = 2.0
    tool_retries: int = 1
    repair_attempts: int = 1

    @property
    def max_llm_calls(self) -> int:
        """Hard cost bound per turn: the loop steps plus one forced final call."""
        return self.max_llm_steps + 1

    @classmethod
    def from_env(cls) -> Budgets:
        return cls(
            max_llm_steps=_int("AGENT_MAX_LLM_STEPS", 10),
            max_tool_calls=_int("AGENT_MAX_TOOL_CALLS", 12),
            stuck_threshold=_int("AGENT_STUCK_THRESHOLD", 3),
            tool_timeout_s=_float("AGENT_TOOL_TIMEOUT_S", 2.0),
            tool_retries=_int("AGENT_TOOL_RETRIES", 1),
            repair_attempts=_int("AGENT_REPAIR_ATTEMPTS", 1),
        )


@dataclass(frozen=True)
class Detection:
    """Spike detection. See DESIGN.md, "Handling poor or contradictory tool responses"."""

    spike_multiplier: float = 3.0
    min_metric_points: int = 5

    @classmethod
    def from_env(cls) -> Detection:
        return cls(
            spike_multiplier=_float("AGENT_SPIKE_MULTIPLIER", 3.0),
            min_metric_points=_int("AGENT_MIN_METRIC_POINTS", 5),
        )


@dataclass(frozen=True)
class Settings:
    now: datetime
    model: str = DEFAULT_MODEL
    temperature: float = 1.0
    reasoning_effort: str | None = None
    world: str = DEFAULT_WORLD
    api_key: str | None = None
    prompts_dir: Path = PROMPTS_DIR
    max_rows: int = 20
    max_window_days: int = 7
    budgets: Budgets = field(default_factory=Budgets)
    detection: Detection = field(default_factory=Detection)


def resolve_now(value: str) -> datetime:
    """`auto` means the system clock. Anything else is a fixed ISO timestamp.

    Pinning it keeps evaluation runs reproducible. The mock worlds follow it
    either way: their incident day is rebased onto the day before `now`.
    """
    return datetime.now(timezone.utc) if value.lower() == "auto" else parse_iso(value)


def load_settings() -> Settings:
    """Build settings from the environment."""
    return Settings(
        now=resolve_now(_env("AGENT_NOW", DEFAULT_NOW)),
        model=_env("AGENT_MODEL", DEFAULT_MODEL),
        temperature=_float("AGENT_TEMPERATURE", 1.0),
        reasoning_effort=os.getenv("AGENT_REASONING_EFFORT", "").strip() or None,
        world=_env("AGENT_WORLD", DEFAULT_WORLD),
        api_key=os.getenv("OPENAI_API_KEY"),
        prompts_dir=Path(_env("AGENT_PROMPTS_DIR", str(PROMPTS_DIR))),
        max_rows=_int("AGENT_MAX_ROWS", 20),
        max_window_days=_int("AGENT_MAX_WINDOW_DAYS", 7),
        budgets=Budgets.from_env(),
        detection=Detection.from_env(),
    )
