"""Conversation state: the message history and the evidence ledger.

One Session per conversation. It holds everything a follow-up question needs:
the full message history, every observation made so far, and the hashes used
to detect repeated calls.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

CITABLE = ("ok", "empty")


@dataclass
class Observation:
    """One completed tool call, successful or not."""

    id: str
    turn: int
    tool: str
    category: str
    args: dict
    status: str
    summary: str
    data: Any = None
    error: dict | None = None
    attempts: int = 1

    @property
    def citable(self) -> bool:
        return self.status in CITABLE

    @property
    def failed(self) -> bool:
        return self.status in ("error", "timeout")

    def envelope(self) -> dict:
        """The exact payload returned to the model."""
        return {
            "observation_id": self.id,
            "status": self.status,
            "summary": self.summary,
            "data": self.data,
            "error": self.error,
            "attempts": self.attempts,
        }


def call_hash(tool: str, args: dict) -> str:
    """Stable key for a tool call. Arguments are already normalised by validation."""
    return json.dumps({"tool": tool, "args": args}, sort_keys=True, default=str)


@dataclass
class Session:
    id: str = field(default_factory=lambda: uuid4().hex[:12])
    world: str = "incident"
    messages: list[dict] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)
    turn: int = 0
    _succeeded: dict[str, str] = field(default_factory=dict)

    def start_turn(self, user_message: str) -> None:
        self.turn += 1
        self.messages.append({"role": "user", "content": user_message})

    def next_observation_id(self) -> str:
        return f"obs_{len(self.observations) + 1:03d}"

    def record(self, observation: Observation) -> Observation:
        self.observations.append(observation)
        # Only successful queries are remembered for de-duplication: a failed call
        # must stay retryable on a later turn, and an action is governed by its policy.
        if observation.citable and observation.category != "action":
            self._succeeded.setdefault(call_hash(observation.tool, observation.args), observation.id)
        return observation

    def duplicate_of(self, tool: str, args: dict) -> str | None:
        return self._succeeded.get(call_hash(tool, args))

    def by_id(self, observation_id: str) -> Observation | None:
        return next((o for o in self.observations if o.id == observation_id), None)

    def this_turn(self) -> list[Observation]:
        return [o for o in self.observations if o.turn == self.turn]

    def notes_this_turn(self) -> int:
        return sum(1 for o in self.this_turn() if o.tool == "create_incident_note" and o.status == "ok")

    def failures_this_turn(self) -> list[Observation]:
        return [o for o in self.this_turn() if o.failed]

    def assumptions_this_turn(self) -> list[str]:
        found: list[str] = []
        for observation in self.this_turn():
            if observation.tool == "resolve_time_range" and observation.citable:
                for item in (observation.data or {}).get("assumptions", []):
                    if item not in found:
                        found.append(item)
        return found
