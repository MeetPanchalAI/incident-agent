"""The agent loop.

One pass per user message. The model must call a tool at every step, so the
only way to end a turn is `submit_response`; there is no free text to parse.
Every tool call gets a reply, including rejected ones, because the API
refuses the next request if any tool_call_id is left unanswered.

Hard bound per turn: `max_llm_steps` loop calls plus at most one forced final
call. Repairs happen inside the loop and cost a step, not an extra call.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .config import Settings
from .llm import LLMClient
from .prompts import FORCE_FINAL, STOPPED_EARLY_GAP, SUBMIT_INVALID, SUBMIT_NOT_ALONE, system_prompt
from .report import FinalOutcome, build_from_ledger, finalize, finalize_unverified, submit_response_schema
from .session import Session
from .tools.executor import Batch, Budget, ToolCall, ToolExecutor
from .tools.mock_backend import MockBackend
from .tools.schemas import TOOL_SPECS


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
    def __init__(self, settings: Settings, llm: LLMClient, backend: MockBackend) -> None:
        self.settings = settings
        self.llm = llm
        self.backend = backend
        self.executor = ToolExecutor(settings, backend)
        self.tools = [spec.openai_schema() for spec in TOOL_SPECS] + [submit_response_schema()]

    def new_session(self) -> Session:
        session = Session(world=self.backend.world_name)
        session.messages.append({"role": "system", "content": system_prompt(self.settings)})
        return session

    # -- one turn ----------------------------------------------------------

    def run_turn(self, session: Session, user_message: str) -> TurnResult:
        session.start_turn(user_message)
        budget = Budget(self.settings.budgets)
        llm_calls = 0

        while budget.has_steps() and not budget.stuck():
            reply = self.llm.chat(session.messages, self.tools)
            llm_calls += 1
            budget.use_step()
            session.messages.append(reply.message)

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
                    budget.observe("invalid_arguments")
                else:
                    observation = self.executor.run(call, session, budget, batch)
                    self._reply_to(session, call, observation.envelope())

        return self._force_final(session, budget, llm_calls)

    def _finalize(self, call: ToolCall, session: Session, budget: Budget, alone: bool) -> tuple[FinalOutcome | None, dict]:
        if not alone:
            return None, {"status": "rejected", "error": SUBMIT_NOT_ALONE}

        if call.arguments_error:
            outcome, errors = None, [call.arguments_error]
        else:
            outcome, errors = finalize(call.arguments, session)
        if outcome is not None:
            return outcome, {"status": "accepted"}

        if budget.repairs_used < self.settings.budgets.repair_attempts:
            budget.repairs_used += 1
            return None, {"status": "rejected", "error": SUBMIT_INVALID.format(errors="\n".join(f"- {e}" for e in errors))}
        return finalize_unverified(call.arguments, session, errors), {"status": "accepted_unverified"}

    def _force_final(self, session: Session, budget: Budget, llm_calls: int) -> TurnResult:
        """One last call, offering only submit_response, so the user always gets an answer."""
        session.messages.append({"role": "system", "content": FORCE_FINAL})
        reply = self.llm.chat(session.messages, [submit_response_schema()], force="submit_response")
        llm_calls += 1
        session.messages.append(reply.message)

        call = next((c for c in reply.tool_calls if c.name == "submit_response"), None)
        if call is None:
            outcome = build_from_ledger(session, ["the model did not return a final response"])
        else:
            outcome, errors = finalize(call.arguments, session)
            if outcome is None:
                outcome = finalize_unverified(call.arguments, session, errors)
            self._reply_to(session, call, {"status": "accepted"})

        outcome.response.gaps.append(STOPPED_EARLY_GAP)
        return self._result(outcome, session, llm_calls, budget, stopped_early=True)

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _reply_to(session: Session, call: ToolCall, payload: dict) -> None:
        session.messages.append({
            "role": "tool",
            "tool_call_id": call.id,
            "content": json.dumps(payload, default=str),
        })

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
