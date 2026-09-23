"""Generates a small JSONL log dataset for the tests.

It is also the reference for the input format: one JSON object per line, with
`ts`, `service`, `level` and `message` required, and `latency_ms`, `target`,
`event_type`/`version`/`status` optional. See README.md.

The dataset contains, deliberately:
  - checkout-api: a deployment at 14:32 followed by an error spike from 14:37
  - orders-db:    latency degrading over the same period (the real culprit)
  - payment-service: steady throughout, so "nothing was wrong" is answerable
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timedelta, timezone

START = datetime(2026, 9, 22, 14, 0, tzinfo=timezone.utc)
MINUTES = 120
SPIKE_FROM, SPIKE_TO = 37, 80  # minutes after START
DEPLOY_AT = 32


def _iso(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def events(seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    out: list[dict] = []

    for minute in range(MINUTES):
        base = START + timedelta(minutes=minute)
        in_spike = SPIKE_FROM <= minute < SPIKE_TO

        if minute == DEPLOY_AT:
            out.append({"ts": _iso(base), "service": "checkout-api", "level": "INFO",
                        "message": "deployment complete: checkout-api v142 serving traffic",
                        "event_type": "deployment", "version": "v142", "status": "success"})

        # checkout-api: 24 requests a minute, each one calling orders-db.
        for index in range(24):
            at = base + timedelta(seconds=index * 2)
            failed = in_spike and rng.random() < 0.28
            out.append({
                "ts": _iso(at), "service": "checkout-api",
                "level": "ERROR" if failed else "INFO",
                "message": ("database connection timeout after 5000ms"
                            if failed else f"POST /checkout {rng.choice(['200', '201'])}"),
                "latency_ms": 5000 if failed else rng.randint(120, 260) + (900 if in_spike else 0),
                "status_code": 500 if failed else 200,
                "target": "orders-db",
            })

        # orders-db: latency climbs during the incident, but it logs no errors,
        # so error_rate alone never explains the outage.
        for index in range(12):
            out.append({
                "ts": _iso(base + timedelta(seconds=index * 5)), "service": "orders-db", "level": "INFO",
                "message": "SELECT orders WHERE customer_id = ?",
                "latency_ms": rng.randint(240, 340) if in_spike else rng.randint(8, 18),
            })

        # payment-service: untouched by any of it.
        for index in range(8):
            out.append({
                "ts": _iso(base + timedelta(seconds=index * 7)), "service": "payment-service", "level": "INFO",
                "message": "POST /authorize 200", "latency_ms": rng.randint(90, 150),
                "target": "payment-gateway",
            })

    out.append({"ts": _iso(START + timedelta(minutes=SPIKE_TO + 1)), "service": "checkout-api",
                "level": "INFO", "message": "error rate returned to baseline"})
    return out


def lines(seed: int = 7) -> list[str]:
    return [json.dumps(event) for event in events(seed)]


if __name__ == "__main__":  # write a file you can upload in the UI
    print("\n".join(lines()))
