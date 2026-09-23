"""The guardrail pipeline for a single tool call.

Every call passes through the same eight steps. The executor never raises:
a failure becomes an envelope the model can reason about, which is what lets
the agent route around a broken tool instead of stopping.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field

from pydantic import ValidationError

from ..config import Budgets, Settings
from ..session import Observation, Session, call_hash
from . import summaries
from .mock_backend import MalformedResponse, MockBackend, TransientError
from .schemas import (
    CATEGORY_BY_NAME,
    TOOLS_BY_NAME,
    Dependencies,
    Deployment,
    LogEvent,
    MetricPoint,
)
from .time_resolver import TimeResolutionError, resolve, validate_range

NON_PRODUCTIVE = ("invalid_arguments", "duplicate", "budget_exceeded")


@dataclass
class Budget:
    """Per-turn limits and the counters that enforce them."""

    limits: Budgets
    steps_used: int = 0
    tool_calls_used: int = 0
    unproductive_streak: int = 0
    repairs_used: int = 0

    def has_steps(self) -> bool:
        return self.steps_used < self.limits.max_llm_steps

    def use_step(self) -> None:
        self.steps_used += 1

    def stuck(self) -> bool:
        return self.unproductive_streak >= self.limits.stuck_threshold

    def tool_calls_left(self) -> bool:
        return self.tool_calls_used < self.limits.max_tool_calls

    def observe(self, status: str) -> None:
        if status in NON_PRODUCTIVE:
            self.unproductive_streak += 1
        else:
            self.unproductive_streak = 0


@dataclass
class ToolCall:
    """One tool call requested by the model."""

    id: str
    name: str
    arguments: dict
    arguments_error: str | None = None


@dataclass
class Batch:
    """The calls in a single model reply, so an identical repeat is caught."""

    seen: set[str] = field(default_factory=set)


class ToolExecutor:
    def __init__(self, settings: Settings, backend: MockBackend) -> None:
        self.settings = settings
        self.backend = backend
        self._pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="tool")

    # -- pipeline ----------------------------------------------------------

    def run(self, call: ToolCall, session: Session, budget: Budget, batch: Batch) -> Observation:
        observation = self._run(call, session, budget, batch)
        budget.observe(observation.status)
        return observation

    def _run(self, call: ToolCall, session: Session, budget: Budget, batch: Batch) -> Observation:
        spec = TOOLS_BY_NAME.get(call.name)
        if spec is None:
            return self._record(session, call, {}, "invalid_arguments",
                                f"Unknown tool '{call.name}'. Available: {', '.join(TOOLS_BY_NAME)}.")

        # 1. budget
        if not budget.tool_calls_left():
            return self._record(session, call, {}, "budget_exceeded",
                                "The tool-call budget for this turn is spent. Submit your answer "
                                "using the evidence you already have.")
        budget.tool_calls_used += 1

        # 2. validate arguments
        if call.arguments_error:
            return self._record(session, call, {}, "invalid_arguments", f"Invalid arguments. {call.arguments_error}")
        try:
            args_model = spec.args_model.model_validate(call.arguments)
        except ValidationError as exc:
            detail = "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:5])
            return self._record(session, call, {}, "invalid_arguments", f"Invalid arguments. {detail}")

        args = args_model.model_dump(mode="json")
        if errors := self._range_errors(args_model):
            return self._record(session, call, args, "invalid_arguments", "Invalid time range. " + " ".join(errors))

        # 3. de-duplicate. Queries only: an action is governed by its policy instead,
        #    so that a second note gets the policy's message rather than "duplicate".
        if spec.category != "action":
            key = call_hash(call.name, args)
            if key in batch.seen:
                return self._record(session, call, args, "duplicate",
                                    "You requested this exact call twice in the same step. Use the other result.")
            batch.seen.add(key)
            if earlier := session.duplicate_of(call.name, args):
                return self._record(session, call, args, "duplicate",
                                    f"This call was already made and succeeded. Use {earlier}.")

        # 4. policy
        if policy_error := self._policy_error(call.name, args_model, session):
            return self._record(session, call, args, "invalid_arguments", policy_error)

        # 5-7. execute, then validate and classify the result
        return self._execute(call, args_model, args, session)

    def _range_errors(self, args_model) -> list[str]:
        start, end = getattr(args_model, "start_time", None), getattr(args_model, "end_time", None)
        if start is None or end is None:
            return []
        return validate_range(start, end, self.settings.now, self.settings.max_window_days)

    def _policy_error(self, name: str, args_model, session: Session) -> str | None:
        if name != "create_incident_note":
            return None
        if session.notes_this_turn() >= 1:
            return "An incident note has already been created this turn. Only one note per turn is allowed."
        bad = [
            i for i in args_model.evidence
            if (observation := session.by_id(i)) is None or not observation.citable
        ]
        if bad:
            return (
                f"Evidence must cite observations that succeeded. These cannot be cited: {', '.join(bad)}."
            )
        return None

    def _execute(self, call: ToolCall, args_model, args: dict, session: Session) -> Observation:
        attempts = 0
        last_error: Exception | None = None
        while attempts <= self.settings.budgets.tool_retries:
            attempts += 1
            try:
                raw = self._invoke(call.name, args_model, session)
            except (TimeoutError, FutureTimeout) as exc:
                last_error = exc
                continue
            except TransientError as exc:
                last_error = exc
                continue
            except TimeResolutionError as exc:
                return self._record(session, call, args, "invalid_arguments", exc.message, attempts=attempts,
                                    error={"type": exc.reason, "message": exc.message})
            except MalformedResponse as exc:
                return self._record(session, call, args, "error", f"{call.name} returned an unusable response.",
                                    attempts=attempts, error={"type": "malformed_response", "message": str(exc)})

            try:
                summary, data = self._shape(call.name, args_model, raw)
            except (ValidationError, MalformedResponse, KeyError, TypeError) as exc:
                return self._record(session, call, args, "error", f"{call.name} returned an unusable response.",
                                    attempts=attempts, error={"type": "malformed_response", "message": str(exc)[:200]})

            status = "empty" if self._is_empty(call.name, raw) else "ok"
            if status == "empty":
                summary = summaries.empty_summary(
                    call.name,
                    args.get("service", "-"),
                    getattr(args_model, "start_time", None),
                    getattr(args_model, "end_time", None),
                )
                data = []
            return self._record(session, call, args, status, summary, data=data, attempts=attempts)

        kind = "timeout" if isinstance(last_error, (TimeoutError, FutureTimeout)) else "error"
        return self._record(
            session, call, args, kind,
            f"{call.name} failed after {attempts} attempt(s). Nothing can be concluded from it.",
            attempts=attempts,
            error={"type": "timeout" if kind == "timeout" else "transient", "message": str(last_error)[:200]},
        )

    def _invoke(self, name: str, args_model, session: Session):
        """Run the backend call under a timeout. Never called for control flow."""
        if name == "resolve_time_range":
            return resolve(args_model.expression, self.settings.now)
        if name == "create_incident_note":
            return f"NOTE-{sum(1 for o in session.observations if o.tool == name and o.status == 'ok') + 1:03d}"

        calls = {
            "get_metrics": lambda: self.backend.get_metrics(
                args_model.service, args_model.metric, args_model.start_time, args_model.end_time),
            "search_logs": lambda: self.backend.search_logs(
                args_model.service, args_model.start_time, args_model.end_time, args_model.query),
            "get_deployments": lambda: self.backend.get_deployments(
                args_model.service, args_model.start_time, args_model.end_time),
            "get_service_dependencies": lambda: self.backend.get_service_dependencies(args_model.service),
        }
        return self._pool.submit(calls[name]).result(timeout=self.settings.budgets.tool_timeout_s)

    def _shape(self, name: str, args_model, raw):
        """Validate the backend's payload and summarise it deterministically."""
        limit = self.settings.max_rows
        if name == "get_metrics":
            points = [MetricPoint.model_validate(row) for row in raw]
            return summaries.summarize_metrics(
                args_model.service, args_model.metric, args_model.start_time, args_model.end_time,
                points, limit, self.settings.detection)
        if name == "search_logs":
            events = [LogEvent.model_validate(row) for row in raw]
            return summaries.summarize_logs(
                args_model.service, args_model.start_time, args_model.end_time, args_model.query, events, limit)
        if name == "get_deployments":
            items = [Deployment.model_validate(row) for row in raw]
            return summaries.summarize_deployments(
                args_model.service, args_model.start_time, args_model.end_time, items, limit)
        if name == "get_service_dependencies":
            return summaries.summarize_dependencies(Dependencies.model_validate(raw))
        if name == "resolve_time_range":
            return summaries.summarize_time_range(args_model.expression, raw.start, raw.end, raw.assumptions)
        if name == "create_incident_note":
            return summaries.summarize_note(raw, args_model.title, args_model.evidence)
        raise MalformedResponse(f"no result handler for {name}")

    @staticmethod
    def _is_empty(name: str, raw) -> bool:
        if name in ("resolve_time_range", "create_incident_note"):
            return False
        if name == "get_service_dependencies":
            return not raw.get("upstream") and not raw.get("downstream")
        return len(raw) == 0

    def _record(self, session: Session, call: ToolCall, args: dict, status: str, summary: str,
                data=None, error: dict | None = None, attempts: int = 1) -> Observation:
        return session.record(Observation(
            id=session.next_observation_id(),
            turn=session.turn,
            tool=call.name,
            category=CATEGORY_BY_NAME.get(call.name, "data"),
            args=args,
            status=status,
            summary=summary,
            data=data,
            error=error,
            attempts=attempts,
        ))
