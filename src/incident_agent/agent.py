"""The agent loop.

One pass per user message. The model must call a tool at every step, so the
only way to end a turn is `submit_response`; there is no free text to parse.
Every tool call gets a reply, including rejected ones, because the API refuses
the next request if any call_id is left unanswered. The conversation is a list
of Responses API items: role messages, the model's own output items (reasoning
and function calls), and one function_call_output per call.

Hard bound per turn: `max_llm_steps` loop calls plus at most one forced final
call. Repairs happen inside the loop and cost a step, not an extra call.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import Settings
from .llm import LLMClient, tool_result_item
from .state import Recorder
from .prompts import Prompts
from .report import FinalOutcome, build_from_ledger, finalize, finalize_unverified, submit_response_schema
from .session import Session
from .tools.executor import Batch, Budget, ToolCall, ToolExecutor
from .tools.store import Store
from .tools.schemas import TOOL_SPECS


def describe_dataset(store: Store) -> str:
    info = store.info()
    if not info:
        return "none ingested yet, so no tool can return anything"
    return (f"{info['filename']}, {info['event_count']} events "
            f"from {info['first_ts']} to {info['last_ts']}")


@dataclass
class TurnResult:
    outcome: FinalOutcome
    trace: list[dict] = field(default_factory=list)
    llm_calls: int = 0
    steps: int = 0
    stopped_early: bool = False

    def to_dict(self) -> dict:
        return {
            "response": self.outcome.to_dict(),
            "trace": self.trace,
            "llm_calls": self.llm_calls,
            "steps": self.steps,
            "stopped_early": self.stopped_early,
        }


class AgentService:
    def __init__(self, settings: Settings, llm: LLMClient, store: Store) -> None:
        self.settings = settings
        self.llm = llm
        self.store = store
        self.prompts = Prompts(settings, store.services(), describe_dataset(store))
        self.executor = ToolExecutor(settings, store)
        self.tools = [spec.openai_schema() for spec in TOOL_SPECS] + [submit_response_schema()]

    def new_session(self) -> Session:
        session = Session()
        session.messages.append({"role": "system", "content": self.prompts.system})
        return session

    # -- one turn ----------------------------------------------------------

    def run_turn(self, session: Session, user_message: str) -> TurnResult:
        """One user turn, recorded as one run in the log database."""
        log = Recorder(self.settings.state_db_path, "turn", user_message)
        log.event("turn.start", user_message, session=session.id, turn=session.turn + 1)
        try:
            result = self._run_turn(session, user_message, log)
        except Exception as error:
            log.event("turn.failed", f"{type(error).__name__}: {error}", level="error")
            log.finish(f"failed: {type(error).__name__}", status="error")
            raise
        response = result.outcome.response
        log.event(
            "turn.done", f"{response.response_type}: {response.message[:120]}",
            level="warn" if result.outcome.unverified else "info",
            steps=result.steps, model_calls=result.llm_calls, tool_calls=len(result.trace),
            facts=len(response.observed_facts), hypotheses=len(response.hypotheses),
            unverified=result.outcome.unverified,
        )
        log.finish(
            f"{response.response_type} · {len(result.trace)} tool calls · {result.llm_calls} model calls"
            + (" · unverified" if result.outcome.unverified else ""))
        return result

    def _run_turn(self, session: Session, user_message: str, log: Recorder) -> TurnResult:
        session.start_turn(user_message)
        budget = Budget(self.settings.budgets)
        llm_calls = 0
        order = 0  # position of each tool call within this turn

        while budget.has_steps() and not budget.stuck():
            reply = self.llm.chat(session.messages, self.tools)
            llm_calls += 1
            budget.use_step()
            session.messages.extend(reply.items)
            wanted = [c.name for c in reply.tool_calls]
            log.event(
                "model.step",
                f"step {budget.steps_used}/{self.settings.budgets.max_llm_steps}: "
                + (f"{len(wanted)} call(s) - {', '.join(wanted)}" if wanted else "no tool call"),
                step=budget.steps_used, calls=wanted)

            if not reply.tool_calls:
                budget.unproductive_streak += 1
                continue

            batch = Batch()
            alone = len(reply.tool_calls) == 1
            for call in reply.tool_calls:
                if call.name == "submit_response":
                    outcome, payload = self._finalize(call, session, budget, alone)
                    self._reply_to(session, call, payload)
                    if outcome is not None:
                        return self._result(outcome, session, llm_calls, budget)
                    log.event("response.rejected", payload["error"][:200], level="warn")
                    budget.observe("invalid_arguments")
                else:
                    observation = self.executor.run(call, session, budget, batch)
                    order += 1
                    self._log_tool(log, observation, order, budget.steps_used)
                    self._reply_to(session, call, observation.envelope())

        reason = "step budget spent" if not budget.has_steps() else "no progress in three calls"
        log.event("turn.forced_final", f"Stopping early: {reason}.", level="warn",
                  steps=budget.steps_used, tool_calls=budget.tool_calls_used)
        return self._force_final_logged(session, budget, llm_calls, log)

    def _finalize(self, call: ToolCall, session: Session, budget: Budget, alone: bool) -> tuple[FinalOutcome | None, dict]:
        if not alone:
            return None, {"status": "rejected", "error": self.prompts.submit_not_alone}

        if call.arguments_error:
            outcome, errors = None, [call.arguments_error]
        else:
            outcome, errors = finalize(call.arguments, session)
        if outcome is not None:
            return outcome, {"status": "accepted"}

        if budget.repairs_used < self.settings.budgets.repair_attempts:
            budget.repairs_used += 1
            return None, {"status": "rejected", "error": self.prompts.submit_invalid(errors)}
        return finalize_unverified(call.arguments, session, errors), {"status": "accepted_unverified"}

    def _force_final(self, session: Session, budget: Budget, llm_calls: int) -> TurnResult:
        """One last call, offering only submit_response, so the user always gets an answer."""
        session.messages.append({"role": "system", "content": self.prompts.force_final})
        reply = self.llm.chat(session.messages, [submit_response_schema()], force="submit_response")
        llm_calls += 1
        session.messages.extend(reply.items)

        call = next((c for c in reply.tool_calls if c.name == "submit_response"), None)
        if call is None:
            outcome = build_from_ledger(session, ["the model did not return a final response"])
        else:
            outcome, errors = finalize(call.arguments, session)
            if outcome is None:
                outcome = finalize_unverified(call.arguments, session, errors)
            self._reply_to(session, call, {"status": "accepted"})

        outcome.response.gaps.append(self.prompts.stopped_early)
        return self._result(outcome, session, llm_calls, budget, stopped_early=True)

    def _force_final_logged(self, session: Session, budget: Budget, llm_calls: int, log: Recorder) -> TurnResult:
        result = self._force_final(session, budget, llm_calls)
        log.event("model.step", "forced final call: submit_response only", step=budget.steps_used + 1)
        return result

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _log_tool(log: Recorder, o, order: int, step: int) -> None:
        """One line per call, in the order it ran, with what it returned."""
        named = [str(o.args[k]) for k in ("service", "metric", "expression", "query") if o.args.get(k)]
        if o.args.get("start_time"):
            named.append(f"{o.args['start_time'][11:16]}-{o.args['end_time'][11:16]}")
        log.event(
            "tool.call",
            f"#{order} {o.tool}({', '.join(named)}) -> {o.status}"
            + (f" after {o.attempts} attempts" if o.attempts > 1 else "")
            + f" in {o.duration_ms}ms | {o.summary[:220]}",
            level="info" if o.citable else "warn",
            order=order, step=step, observation=o.id, tool=o.tool,
            status=o.status, attempts=o.attempts, ms=o.duration_ms,
        )

    @staticmethod
    def _reply_to(session: Session, call: ToolCall, payload: dict) -> None:
        session.messages.append(tool_result_item(call.id, payload))

    def _result(self, outcome: FinalOutcome, session: Session, llm_calls: int, budget: Budget,
                stopped_early: bool = False) -> TurnResult:
        trace = [
            {
                "observation_id": o.id,
                "tool": o.tool,
                "category": o.category,
                "status": o.status,
                "attempts": o.attempts,
                "args": o.args,
                "summary": o.summary,
            }
            for o in session.this_turn()
        ]
        return TurnResult(
            outcome=outcome,
            trace=trace,
            llm_calls=llm_calls,
            steps=budget.steps_used,
            stopped_early=stopped_early,
        )
