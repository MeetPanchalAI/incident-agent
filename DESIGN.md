# Design

## Architecture

```
Browser (index.html) ─┐                        ┌─ CLI (cli.py)
                      └─ FastAPI (api.py) ─────┤
                                               ▼
                              AgentService.run_turn(session, message)
                                               │
              ┌────────────────────────────────┼────────────────────────────────┐
              ▼                                ▼                                ▼
      Session                          LLM client                       ToolExecutor
      conversation, evidence ledger,   OpenAI Responses adapter,        budget → validate → dedupe
      de-duplication hashes            scripted fake in tests           → policy → run with timeout
                                                                        and retry → validate result
                                                                        → summarise → observation
                                                                                     │
                                                                                     ▼
                                                                        Mock backend
                                                                        JSON worlds rebased onto the
                                                                        clock, injected faults
```

The CLI, the HTTP API and the tests all call the same `AgentService`. The API and the UI hold no
logic of their own and cannot bypass any guardrail.

**Nothing tunable is hard-coded.** The model, temperature and reasoning effort, every budget, the
spike-detection thresholds, the result limits and the clock all come from `.env` through one
`Settings` object that is threaded explicitly rather than read from globals. The prompts are files
in `prompts/`. The point is that a change can be made and then measured: edit a value, run
`python -m evals.run`, compare. README.md lists every variable.

## Why this architecture

A hand-written loop, not a framework. `agent.py` is 155 lines including the docstrings.

The loop is the part of this exercise being judged. A framework would hide it, and would make the
budgets, the retry policy and the forced final answer harder to control and harder to explain. With
six tools and one control tool there is nothing for a framework to manage that the loop does not
already manage.

A planner/executor split or a multi-agent design was not used. Neither adds capability here: there
is one investigation, one evidence ledger and one answer. Both would add parts to explain and
places for state to diverge.

The principle running through the rest of this document: **the model interprets, code decides.** The
model reads the request and weighs the evidence. Anything with one correct answer — date
arithmetic, argument validation, anomaly detection, confidence limits — is computed in code, where
it can be unit tested.

## The agent loop

One pass per user message.

```python
def run_turn(session, user_message):
    session.start_turn(user_message)
    budget = Budget(...)                       # every limit comes from .env
    while budget.has_steps() and not budget.stuck():
        reply = llm.chat(session.messages, tools=ALL_TOOLS, tool_choice="required")
        budget.use_step()
        session.messages.append(reply.message)
        batch = Batch()
        for call in reply.tool_calls:
            if call.name == "submit_response":
                outcome = finalize(call, session, budget, alone=len(reply.tool_calls) == 1)
                reply_to(call, ...)            # every call gets a tool message
                if outcome:
                    return outcome
            else:
                reply_to(call, executor.run(call, session, budget, batch))
    return force_final(session)                # one last call, offering only submit_response
```

`tool_choice="required"` means the model must call a tool at every step, so the only way to end a
turn is `submit_response`. There is no free-text branch to parse and no ambiguity about whether the
model meant to stop.

Four rules make the loop safe:

- **Every tool call gets a result item**, including rejected ones. The API refuses the next request
  if any `call_id` is left unanswered, so a silently dropped call would break the conversation.
- **`submit_response` must be alone in its step.** If it arrives alongside data tools, it is answered
  with "review the other results first" and the loop continues. Answering before reading the results
  you just asked for is not a conclusion.
- **`submit_response` is never blocked by the tool-call budget.** Running out of budget must not cost
  the user their answer.
- **The turn always ends with an answer.** If the budget is spent or the loop is stuck,
  `force_final` makes one more call offering only `submit_response`, and code adds a gap saying the
  investigation was stopped early.

**Parallel tool calls are enabled.** The model may ask for several tools in one step; they run in
emission order. This is worth having because it saves model round trips — the expensive part — when
calls are genuinely independent, such as metrics and deployments for a service and window already
known. Dependent calls serialise on their own: the model cannot narrow a log search to 14:37 before
metrics has told it 14:37, and cannot query `payment-gateway` before dependencies has named it.
Running them concurrently in a thread pool was not done: the backend is in-process, so concurrency
would save nothing measurable while making observation ordering non-deterministic.

**Cost bound per turn: `AGENT_MAX_LLM_STEPS` plus one** — the loop steps plus the forced final call,
so 11 at the default. A repair of an invalid final response happens inside the loop and costs a
step, not an extra call.

### Why the Responses API

The adapter targets `/v1/responses`, not Chat Completions, because the model this agent runs on
refuses the combination it needs:

> Function tools with reasoning_effort are not supported for gpt-5.6-luna in
> /v1/chat/completions. To use function tools, use /v1/responses or set reasoning_effort to 'none'.

Omitting the parameter fails the same way; the model reasons by default. The alternative was
`reasoning_effort: none`, and it is a poor one here. `tool_choice` is `"required"`, so every reply
is tool calls with no text content — reasoning is the only place this agent can deliberate between
steps, and deciding what to check next from what the last call returned is the behaviour the whole
design is about.

The conversation is therefore a list of Responses items rather than chat messages: role messages,
the model's own output items (its reasoning and its function calls), and one `function_call_output`
per call. Nothing is stored server side (`store=False`), so the reasoning items travel in the
conversation, which is why the client asks for `reasoning.encrypted_content`. Everything above the
adapter — the executor, the ledger, the report validation, the guardrails — was unaffected by the
change.

## How the agent chooses a tool

Nothing in the code picks tools. The model does, from three inputs:

1. **The tool schemas.** Each description says what the tool is for and when it is useful, not just
   what it returns. `get_metrics` says it is usually the first step when asked why something went
   wrong, because it establishes whether anything actually changed. `get_service_dependencies` says
   to use it when a service looks affected but nothing local explains it.
2. **The system prompt**, read from `prompts/system.md`. It describes good investigative practice
   without prescribing a sequence:
   use what you learn, narrow the window around a spike, follow dependencies upward, ask rather than
   guess when the service or the window is unclear.
3. **The observations so far**, which are in the message history, including from earlier turns.

Different questions therefore produce different traces. "What does checkout-api depend on?" is one
lookup. "Why did payment-service latency rise?" needs metrics, then dependencies, then metrics on
the upstream service — and the third call's arguments only exist because of the second call's
result. Evaluation scenarios 1, 8 and 9 check exactly this, and scenario 8 asserts the chaining
directly: at least one call must use a value it could only have learnt from an earlier result.

## Preventing loops and runaway cost

| Control | Variable | Default | What happens when it is hit |
|---|---|---|---|
| Model steps per turn | `AGENT_MAX_LLM_STEPS` | 10 | `force_final` |
| Tool calls per turn, including rejected ones | `AGENT_MAX_TOOL_CALLS` | 12 | Further calls return `budget_exceeded` |
| Consecutive unproductive results | `AGENT_STUCK_THRESHOLD` | 3 | `force_final` |
| Tool timeout / retries | `AGENT_TOOL_TIMEOUT_S` / `AGENT_TOOL_RETRIES` | 2 s / 1 | Retry once on timeout or a transient error, then report `timeout` |
| Repairs of an invalid final response | `AGENT_REPAIR_ATTEMPTS` | 1 per turn | Return the response marked `unverified` |
| Forced final call | — | 1 per turn | Allowed even when the step budget is spent |

An unproductive result is `invalid_arguments`, `duplicate` or `budget_exceeded` — the three
outcomes that mean the model learnt nothing. Three in a row means it is not making progress, and
the loop stops rather than spending the rest of the budget confirming that.

**De-duplication.** Before a call runs, its tool name and validated arguments are hashed. An exact
repeat of a call that already succeeded returns `duplicate` with a pointer to the earlier
observation instead of running again. Two details matter:

- **Only successful queries are remembered.** A call that timed out must stay retryable on a later
  turn. Caching a failure would leave the agent unable either to retrieve the data or to cite it.
- **Actions are excluded.** A second `create_incident_note` is refused by the note policy, which
  gives a more useful message than "duplicate", and a note on a later turn is legitimate.

Within a single step, an identical repeat is caught regardless of status, because re-running a
failing call twice in the same batch cannot help.

De-duplication is exact match, not fuzzy. Shifting a window by a minute produces a new call. That
is deliberate: narrowing a window around a spike is the behaviour the prompt asks for, and blocking
it would be worse than the repetition it prevents. The step budget is the backstop.

## Handling poor or contradictory tool responses

The executor never raises. Every call returns the same envelope, so a failure is information the
model can reason about rather than an exception that ends the run:

```json
{
  "observation_id": "obs_004",
  "status": "ok | empty | error | timeout | invalid_arguments | duplicate | budget_exceeded",
  "summary": "error_rate on checkout-api, 2026-09-22T14:00:00Z to 2026-09-22T16:00:00Z: 120 points; baseline (median) 0.0094 (0.94%); threshold 0.0281 (2.81%); spike starts 2026-09-22T14:37:00Z; peak 0.1720 (17.20%) at 2026-09-22T14:41:00Z; 38 of 120 points above threshold. Showing 20 of 120 points.",
  "data": [],
  "error": null,
  "attempts": 1
}
```

| Situation | Behaviour |
|---|---|
| Timeout or transient error | Retried once. If it still fails: recorded, not citable, and the model must report it in `gaps` or `missing_evidence`. |
| Malformed response | The result is validated against a Pydantic model before the model sees it. A payload that does not match becomes `error: malformed_response`. Raw malformed data is never forwarded. |
| Empty result | `status: empty`. Citable, because "the query returned nothing" is a fact. |
| Invalid arguments | Not executed. The error names the valid services or metrics so the model can correct itself. |
| Contradictory evidence | Both observations are kept and can be cited, as supporting and contradicting. Confidence is capped and a next check is recommended. |
| Repeated call | Not run again; the earlier observation is returned. |

**`empty` is a fact about the query, not about the world.** `get_deployments` returning nothing does
not prove no deployment happened — the window may be wrong, the service may be wrong, or the data
may be incomplete. The summary says so in those words, every time:

> get_deployments returned no matching records for checkout-api, 2026-09-22T14:00:00Z to
> 2026-09-22T16:00:00Z. This is a result about the query, not proof that nothing happened.

This is the failure mode most likely to produce a confident wrong answer, so the wording is
generated by code rather than left to the model.

**Summaries are computed by code**, one function per tool, from the full result. Only the rows sent
to the model are capped at 20, and the summary says when that happened ("Showing 20 of 120
points"). Spike detection therefore has one correct answer, can be unit tested, and costs no extra
model call.

The metric rule:

| Step | Rule |
|---|---|
| Minimum data | At least `AGENT_MIN_METRIC_POINTS` points (5), otherwise no spike claim is made. |
| Baseline | Median of the window. |
| Usable baseline | If the median is itself more than `AGENT_SPIKE_MULTIPLIER` times the quietest tenth of the window (or exceeds it by the minimum difference), the window is mostly elevated. The summary says the median is not a usable baseline, gives no spike start, and asks for a wider window. |
| Spike | First point above `max(AGENT_SPIKE_MULTIPLIER × baseline, baseline + min_delta)`. Reports the baseline, the spike start, the peak and its time, and how many points are above the threshold. |
| `min_delta` | `error_rate` 0.01 · `latency_p95_ms` 100 · `request_rate` 10. Stops a tiny baseline turning noise into a "3× spike". |
| Scope | Increases only. |

The "usable baseline" step exists because the obvious version of this rule cannot fire. Counting how
many points exceed the threshold can never exceed half the window, because the threshold is derived
from the median of that same window: if most points are elevated, the median is elevated too, the
threshold rises with it, and the summary reports "no spike detected" — which reads as an all-clear.
Comparing the median against the quietest tenth detects it instead. This matters in practice,
because narrowing a window around a spike is exactly what the prompt asks the agent to do.

Tool output is treated as data, not instructions. The system prompt says so explicitly, because log
messages are untrusted input.

## Time

The model identifies the time expression. Code resolves it and validates it.

```
User: "Investigate checkout-api errors yesterday between 2 PM and 4 PM"
  → the model calls resolve_time_range("yesterday between 2 PM and 4 PM")
  → code, with NOW = 2026-09-23T10:00:00Z, returns
      {start: 2026-09-22T14:00:00Z, end: 2026-09-22T16:00:00Z, assumptions: []}
  → the model passes those timestamps to the data tools
  → the validator checks them again: start < end, end not after NOW, window at most 7 days
```

The grammar is small, explicit and fully unit tested, all in UTC:

- `today` / `yesterday` / `tomorrow` / an ISO date, optionally with `morning` 06–12, `afternoon`
  12–18, `evening` 18–24 or `night` 00–06
- `last N minutes / hours / days`
- a clock range such as "2 PM to 4 PM" or "14:00–16:00", with or without a date
- two ISO 8601 timestamps

A general-purpose date parser was not used, because it guesses silently on input like "during the
outage". A bounded grammar refuses instead, with a reason, and the agent asks the user.

- **A clock range or daypart with no date** resolves to the most recent occurrence that has already
  ended. The assumed date is returned as an assumption, and code copies it into the report.
- **Vague expressions** ("recently", "during the outage") return `unresolvable`; **future ranges**
  return `future_range`. Either way the agent asks rather than investigating the wrong window.
- **`today`** is clipped to the current time, and the clip is reported as an assumption.
- **Narrowing a window** from timestamps seen in a result is allowed. That is copying an observed
  value, not date arithmetic, and the validator still bounds it.

`NOW` comes from `AGENT_NOW`: a fixed timestamp, or `auto` for the system clock. It is always
injected, never read from the clock directly, so a test or an evaluation can pin it.

The mock data follows it. Each world declares an `anchor_date`, and on load its fixtures are moved
by whole days so that the incident day is the day before `NOW`. Clock times within the day do not
move: the deployment stays at 14:32 and the spike still starts at 14:37. Without this, `auto` would
be useless — "yesterday" would land on a day with no fixture data and every investigation would
come back empty.

The generated noise on a metric is keyed on the unshifted timestamp, so a rebased world produces
the same numbers as a pinned one. A run today and a run next week are comparable.

## Conversation state

One `Session` per conversation, held in memory, keyed by `session_id` for the API. It holds the full
message history including every tool call and result, the evidence ledger, the de-duplication
hashes, and the turn counter.

Follow-ups work because the model can see the previous turn's observations. "Was there a deployment
around that time?" is answered from the existing `obs_` id with no new tool call, and the citation
still validates because the ledger spans the conversation.

The service and the time window are not extracted into separate state. They are already in the
arguments of earlier tool calls, which are in the history. A second copy would be a second thing to
keep correct.

Budgets reset each turn. The ledger and the de-duplication hashes persist, which is correct here
because the mock data is static.

## The final response

```
response_type          "answer" | "clarification" | "investigation_report"
message                the direct answer, or the clarifying question
observed_facts         [{statement, evidence_ids}]         only things a tool returned
hypotheses             [{statement, supporting_evidence_ids, contradicting_evidence_ids,
                         missing_evidence, confidence}]
likely_cause           null when the evidence is inconclusive
recommended_actions    []
gaps                   failed tool calls, missing data, contradictions
assumptions            e.g. "Date not given; assumed 2026-09-22"
```

`finalize()` checks four things:

1. The payload matches the schema. Every observed fact must cite at least one observation.
2. Every cited id exists and has status `ok` or `empty`. A failed call cannot be used as evidence.
3. Every call that failed this turn is named in `gaps` or in a hypothesis's `missing_evidence`.
4. The confidence limits below.

If any check fails, the specific errors go back to the model for **one** repair attempt. If the
repair also fails, the response is returned marked `unverified` with the invalid citations removed.
If the payload cannot be parsed at all, code builds the response from the ledger — the summaries of
successful observations as facts, the failures as gaps, no hypotheses. The user always gets an
answer and always knows its status.

### Confidence

Confidence describes how well the evidence supports a hypothesis. It is not a probability and not
proof of causation. The model proposes it; code lowers it when the evidence does not support it,
and says so visibly.

| Rule | Reason |
|---|---|
| No supporting evidence → `low` | An unsupported idea is a guess. |
| Any tool call that failed this turn → at most `medium` | Part of the picture is missing, whatever the model says about it. |
| Fewer than two distinct data tools with status `ok` among the supporting evidence → at most `medium` | Independent sources must agree. "Different tools" is a deliberately simple proxy for independence. |
| Cited contradicting evidence → at most `medium` | An unresolved conflict rules out high confidence. |
| Not the single most broadly supported hypothesis → at most `medium` | Two competing explanations cannot both be well supported. |
| `likely_cause` must be null unless some hypothesis reaches `medium` | Otherwise the report says inconclusive. |

An `empty` observation may be cited — "no deployment records were found" can support a hypothesis
that no deployment was involved — but it never counts towards the two-tool rule. Absence of data
cannot raise confidence.

**The caps are read from the evidence ledger, not from the model's own fields.** This is deliberate.
An earlier version capped confidence whenever the model listed `missing_evidence`, which rewards
leaving it blank: the cheapest way to keep a high rating is to report less. Reading failures from
the ledger instead means the cap applies whether or not the model mentions them, and a separate
check requires it to mention them. `test_report.py` pins this: a silent payload and an honest one
get the same confidence.

## What enforces what

The design does not claim enforcement that code does not provide.

| Rule | Enforced by |
|---|---|
| A failed call cannot be cited | Code (`finalize`) |
| Every observed fact cites an observation | Code (schema and `finalize`) |
| A failed call must appear in `gaps` or `missing_evidence` | Code (`finalize`) |
| An empty result is worded as a query result, not proof of absence | Code (summary wording); the model's own phrasing: prompt + scenario 5 |
| An empty result cannot raise confidence | Code |
| Confidence downgrades are visible | Code |
| A forced final answer says it was stopped early | Code |
| An unverified response is marked unverified | Code |
| Time assumptions appear in the report | Code (copied from the resolver) |
| Truncation and insufficient data are disclosed | Code (summaries) |
| A note cites only observations that succeeded; at most one per turn | Code (executor policy) |
| No tool can change production | Code (no such tool exists) |
| A citation actually supports the statement it is attached to | **Not enforced** — see Limitations |
| Contradicting evidence is reported at all | Prompt; the cap applies only to what the model volunteers |
| Timing alone is correlation, not causation | Prompt; scenario 3 |
| A note is only filed for an investigation that reached a finding | Prompt; scenarios 2, 4, 9, 10 |
| `resolve_time_range` is used for worded or date-less expressions | Prompt; scenarios 1 and 11 |

## Actions and dangerous operations

`create_incident_note` runs automatically, as the brief allows. It is called during the loop, before
the final answer exists, so code cannot judge whether it is warranted. Enforcement is split
honestly: code checks the structure (citations valid, fields bounded, one per turn), the prompt
carries the judgement, and the scenarios verify it — zero notes for a healthy service, a clarifying
question, a dependency lookup, or a rollback request.

**There is no tool that changes production.** Asked to roll back, the agent does not act; it explains
that a human must carry it out or approve it, and may recommend it. This is a scope decision, not an
omission: the brief makes a rollback action optional, and the guardrail worth demonstrating is that
the agent cannot take a destructive action at all. A `rollback_deployment` tool behind an explicit
approval checkpoint — the loop pausing, returning `pending_action`, and executing only after the
user confirms that exact service and version — is the first thing to add next.

## Limitations

1. **Citation checking proves provenance, not entailment.** Code verifies that `obs_004` exists and
   succeeded. It cannot verify that `obs_004` actually says what the statement claims. A model could
   cite a real observation for a claim it does not support and pass every check. An LLM judge
   scoring entailment is the production answer.
2. **A window lying entirely inside an incident cannot be detected.** With no quiet points to compare
   against, a constant 17% error rate is indistinguishable from a service whose normal rate is 17%.
   The summary asks for a wider window whenever it can tell, but it cannot always tell.
3. **Spike detection finds increases only.** Traffic falling to zero is a real incident signature and
   is not detected.
4. **The contradiction cap depends on the model reporting the contradiction.** Code cannot detect
   that two observations conflict in meaning.
5. **Code cannot force the use of `resolve_time_range`.** The model can pass timestamps it worked out
   itself. The validator still bounds them, and the scenarios check the behaviour.
6. **Sessions are in memory.** Restarting the server loses them.

## What would change for production scale

- **Real backends** behind the same tool interface — the executor, the envelope and the summaries do
  not change. Per-tool timeouts and circuit breakers, since a real logging API is slower and less
  reliable than a dict lookup.
- **Persistent sessions** in a database, and compaction of old tool results down to their summaries
  once a conversation outgrows the context window.
- **Cost controls per tenant**, not just per turn, with the token usage already returned by the
  adapter recorded against them.
- **Tracing.** One structured record per turn — steps, tool calls, latency, tokens, cost — is the
  next thing to add, and the trace already returned to the UI is most of it.
- **Auth, RBAC and an audit log**, which become necessary the moment any tool can change something.
- **Approval workflow** for mutating actions, as described above.
- **Evaluations in CI**, with the pass rate tracked over time, so a prompt change that costs accuracy
  is visible before it ships.
- **Entailment checking** on citations, closing limitation 1.
