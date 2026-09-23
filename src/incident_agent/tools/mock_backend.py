"""Mock observability backend.

Data is generated deterministically from compact JSON "worlds", so the same
question always produces the same evidence and the model is the only source
of variance. Faults are injected per world to exercise the failure paths.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path

from ..config import format_iso, parse_iso

WORLDS_DIR = Path(__file__).parent / "worlds"
MAX_POINTS = 720

FAULTS = ("empty", "timeout", "transient", "malformed")


class TransientError(Exception):
    """A failure that may succeed if the call is retried."""


class MalformedResponse(Exception):
    """The backend returned something that does not match the expected shape."""


@lru_cache(maxsize=1)
def catalog() -> dict:
    return json.loads((WORLDS_DIR / "_catalog.json").read_text(encoding="utf-8"))


def known_services() -> list[str]:
    return sorted(catalog()["services"])


def known_metrics() -> list[str]:
    return sorted(catalog()["metrics"])


def available_worlds() -> list[str]:
    return sorted(p.stem for p in WORLDS_DIR.glob("*.json") if not p.stem.startswith("_"))


@lru_cache(maxsize=8)
def _load_world(name: str) -> dict:
    path = WORLDS_DIR / f"{name}.json"
    if not path.exists():
        raise ValueError(f"Unknown world '{name}'. Available: {', '.join(available_worlds())}")
    world = json.loads(path.read_text(encoding="utf-8"))
    world["logs"] = _expand_logs(world.get("logs", []))
    return world


def _expand_logs(entries: list[dict]) -> list[dict]:
    """Turn `repeat` specs into individual events and sort everything by time."""
    events: list[dict] = []
    for entry in entries:
        common = {"service": entry["service"], "level": entry["level"], "message": entry["message"]}
        if "repeat" in entry:
            spec = entry["repeat"]
            step = timedelta(minutes=spec["every_minutes"])
            at, last = parse_iso(spec["from"]), parse_iso(spec["to"])
            while at <= last:
                events.append({"at": at, **common})
                at += step
        else:
            events.append({"at": parse_iso(entry["at"]), **common})
    return sorted(events, key=lambda e: e["at"])


def _noise(service: str, metric: str, at: datetime, amplitude: float) -> float:
    if amplitude <= 0:
        return 0.0
    key = f"{service}|{metric}|{format_iso(at)}".encode()
    digest = hashlib.blake2b(key, digest_size=8).digest()
    unit = int.from_bytes(digest, "big") / 2**64  # [0, 1)
    return (unit * 2 - 1) * amplitude


def _value_at(spec: dict, at: datetime) -> float:
    base = spec["baseline"]
    anomaly = spec.get("anomaly")
    if not anomaly:
        return base
    start, peak_at, end = (parse_iso(anomaly[k]) for k in ("start", "peak_at", "end"))
    if not start <= at <= end:
        return base
    if at <= peak_at:
        fraction = 1.0 if peak_at == start else (at - start) / (peak_at - start)
    else:
        fraction = 0.0 if end == peak_at else 1 - (at - peak_at) / (end - peak_at)
    return base + (anomaly["peak"] - base) * fraction


class MockBackend:
    """Reads one world and answers tool calls against it."""

    def __init__(self, world: str, faults: dict[str, str] | None = None) -> None:
        self.world_name = world
        self.world = _load_world(world)
        self.faults = dict(faults or {})
        self._attempts: dict[str, int] = {}

    def _apply_fault(self, tool: str) -> str | None:
        """Return "empty" if the caller should return no rows, else raise or return None."""
        fault = self.faults.get(tool)
        if fault is None:
            return None
        self._attempts[tool] = self._attempts.get(tool, 0) + 1
        if fault == "empty":
            return "empty"
        if fault == "timeout":
            raise TimeoutError(f"{tool} did not respond in time")
        if fault == "transient" and self._attempts[tool] == 1:
            raise TransientError(f"{tool} is temporarily unavailable")
        if fault == "malformed":
            return "malformed"
        return None

    def get_metrics(self, service: str, metric: str, start: datetime, end: datetime) -> list[dict]:
        fault = self._apply_fault("get_metrics")
        if fault == "empty":
            return []
        spec = dict(catalog()["metrics"][metric])
        spec.update(self.world.get("metrics", {}).get(service, {}).get(metric, {}))
        if fault == "malformed":
            return [{"timestamp": "not-a-timestamp", "value": None}]

        total_minutes = max(1, int((end - start).total_seconds() // 60))
        step = timedelta(minutes=max(1, math.ceil(total_minutes / MAX_POINTS)))
        digits = 4 if catalog()["metrics"][metric]["unit"] == "fraction" else 1
        points, at = [], start
        while at < end:
            value = _value_at(spec, at) + _noise(service, metric, at, spec.get("noise", 0))
            points.append({"timestamp": format_iso(at), "value": round(max(value, 0.0), digits)})
            at += step
        return points

    def search_logs(self, service: str, start: datetime, end: datetime, query: str) -> list[dict]:
        fault = self._apply_fault("search_logs")
        if fault == "empty":
            return []
        if fault == "malformed":
            return [{"level": "ERROR"}]
        needle = query.strip().lower()
        return [
            {"timestamp": format_iso(e["at"]), "level": e["level"], "message": e["message"]}
            for e in self.world["logs"]
            if e["service"] == service
            and start <= e["at"] < end
            and (not needle or needle in f"{e['level']} {e['message']}".lower())
        ]

    def get_deployments(self, service: str, start: datetime, end: datetime) -> list[dict]:
        fault = self._apply_fault("get_deployments")
        if fault == "empty":
            return []
        if fault == "malformed":
            return [{"service": service, "version": None, "deployed_at": "not-a-timestamp"}]
        return [
            d for d in self.world.get("deployments", [])
            if d["service"] == service and start <= parse_iso(d["deployed_at"]) < end
        ]

    def get_service_dependencies(self, service: str) -> dict:
        fault = self._apply_fault("get_service_dependencies")
        if fault == "empty":
            return {"service": service, "upstream": [], "downstream": []}
        if fault == "malformed":
            return {"service": service}
        entry = catalog()["services"][service]
        return {"service": service, "upstream": list(entry["upstream"]), "downstream": list(entry["downstream"])}
