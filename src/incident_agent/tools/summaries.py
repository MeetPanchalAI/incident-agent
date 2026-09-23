"""Deterministic summaries.

Every tool result is summarised by code, not by the model. Anomaly detection
therefore has one correct answer and can be unit tested, and it costs no
extra model call. The rows sent to the model are capped; the summary always
says when that happened.
"""

from __future__ import annotations

import statistics
from datetime import datetime

from ..config import Detection, format_iso
from .store import METRICS
from .schemas import Dependencies, Deployment, LogEvent, MetricPoint


def _window(start: datetime, end: datetime) -> str:
    return f"{format_iso(start)} to {format_iso(end)}"


def _sample(rows: list, limit: int) -> list:
    """Evenly spaced sample preserving order."""
    if len(rows) <= limit:
        return rows
    step = (len(rows) - 1) / (limit - 1)
    return [rows[round(i * step)] for i in range(limit)]


def _fmt(value: float, metric: str) -> str:
    if METRICS[metric]["unit"] == "fraction":
        return f"{value:.4f} ({value * 100:.2f}%)"
    return f"{value:g}"


def empty_summary(tool: str, service: str, start: datetime | None, end: datetime | None) -> str:
    where = f"{service}" if start is None else f"{service}, {_window(start, end)}"
    return (
        f"{tool} returned no matching records for {where}. "
        "This is a result about the query, not proof that nothing happened."
    )


def summarize_metrics(
    service: str, metric: str, start: datetime, end: datetime, points: list[MetricPoint],
    limit: int, detection: Detection,
) -> tuple[str, list[dict]]:
    values = [p.value for p in points]
    head = f"{metric} on {service}, {_window(start, end)}: {len(points)} points"
    rows = [{"timestamp": format_iso(p.timestamp), "value": p.value} for p in _sample(points, limit)]
    truncated = f" Showing {len(rows)} of {len(points)} points." if len(rows) < len(points) else ""

    if len(points) < detection.min_metric_points:
        return (f"{head}; insufficient metric data for spike detection "
                f"(at least {detection.min_metric_points} needed).{truncated}"), rows

    min_delta = METRICS[metric]["min_delta"]
    baseline = statistics.median(values)
    threshold = max(detection.spike_multiplier * baseline, baseline + min_delta)
    body = f"; baseline (median) {_fmt(baseline, metric)}; threshold {_fmt(threshold, metric)}"

    # The median is only a usable baseline while normal behaviour occupies most of
    # the window. Comparing it against the quietest tenth detects when it does not,
    # which would otherwise report "no spike" and read as an all-clear.
    quietest = statistics.quantiles(values, n=10)[0] if len(values) >= 10 else min(values)
    if baseline > max(detection.spike_multiplier * quietest, quietest + min_delta):
        return (
            f"{head}; quietest tenth {_fmt(quietest, metric)}, median {_fmt(baseline, metric)}: "
            "elevated for most of the window, so the median is not a usable baseline and no spike "
            "start can be given. Query a wider window that includes time before the change."
            f"{truncated}",
            rows,
        )

    above = [p for p in points if p.value > threshold]
    if not above:
        return f"{head}{body}; no spike detected.{truncated}", rows
    peak = max(points, key=lambda p: p.value)
    return (
        f"{head}{body}; spike starts {format_iso(above[0].timestamp)}; "
        f"peak {_fmt(peak.value, metric)} at {format_iso(peak.timestamp)}; "
        f"{len(above)} of {len(points)} points above threshold.{truncated}",
        rows,
    )


def summarize_logs(
    service: str, start: datetime, end: datetime, query: str, events: list[LogEvent], limit: int
) -> tuple[str, list[dict]]:
    if not events:
        # Replaced by empty_summary() once the executor classifies the result.
        return f"search_logs on {service}, {_window(start, end)}: 0 events.", []
    shown = events[:limit]
    rows = [{"timestamp": format_iso(e.timestamp), "level": e.level, "message": e.message} for e in shown]
    levels = {}
    for event in events:
        levels[event.level] = levels.get(event.level, 0) + 1
    breakdown = ", ".join(f"{level} {count}" for level, count in sorted(levels.items()))
    term = f", query '{query}'" if query else ", no query filter"
    truncated = f" Showing the first {len(rows)} of {len(events)} events." if len(rows) < len(events) else ""
    return (
        f"search_logs on {service}, {_window(start, end)}{term}: {len(events)} events ({breakdown}); "
        f"first {format_iso(events[0].timestamp)}, last {format_iso(events[-1].timestamp)}.{truncated}",
        rows,
    )


def summarize_deployments(
    service: str, start: datetime, end: datetime, deployments: list[Deployment], limit: int
) -> tuple[str, list[dict]]:
    if not deployments:
        # Replaced by empty_summary() once the executor classifies the result.
        return f"get_deployments on {service}, {_window(start, end)}: 0 deployments.", []
    shown = deployments[:limit]
    rows = [
        {
            "service": d.service,
            "version": d.version,
            "deployed_at": format_iso(d.deployed_at),
            "status": d.status,
            "author": d.author,
        }
        for d in shown
    ]
    listed = "; ".join(f"{d.version} at {format_iso(d.deployed_at)} ({d.status})" for d in shown)
    truncated = f" Showing the first {len(rows)} of {len(deployments)}." if len(rows) < len(deployments) else ""
    return (
        f"get_deployments on {service}, {_window(start, end)}: {len(deployments)} deployment(s); {listed}.{truncated}",
        rows,
    )


def summarize_dependencies(deps: Dependencies) -> tuple[str, dict]:
    upstream = ", ".join(deps.upstream) or "none"
    downstream = ", ".join(deps.downstream) or "none"
    return (
        f"{deps.service} depends on: {upstream}. Services that depend on it: {downstream}.",
        deps.model_dump(),
    )


def summarize_note(note_id: str, title: str, evidence: list[str]) -> tuple[str, dict]:
    cited = ", ".join(evidence) or "no observations"
    return (
        f"Incident note {note_id} created: '{title}', citing {cited}.",
        {"note_id": note_id, "title": title, "evidence": evidence},
    )


def summarize_time_range(expression: str, start: datetime, end: datetime, assumptions: list[str]) -> tuple[str, dict]:
    note = f" Assumptions: {' '.join(assumptions)}" if assumptions else ""
    return (
        f"'{expression}' resolves to {_window(start, end)}.{note}",
        {"start_time": format_iso(start), "end_time": format_iso(end), "assumptions": assumptions},
    )
