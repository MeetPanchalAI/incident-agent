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
      conversation,                    OpenAI Responses adapter,        budget → validate → dedupe
      evidence ledger                  scripted fake in tests           → policy → run with timeout
                                                                        and retry → validate → summarise
                                                                                     │
                                                                                     ▼
                                                                        Store (SQLite)
                                                                        events, and the metrics,
                                                                        deployments and dependencies
                                                                        derived from them
```

The CLI, the HTTP API and the tests all call the same `AgentService`. The API and the UI hold no
logic of their own and cannot bypass a guardrail.

**Nothing tunable is hard-coded.** Model, temperature, reasoning effort, every budget, the detection
thresholds, the response limits and the clock come from `.env` through one `Settings` object threaded
explicitly rather than read from globals. Prompts are files in `prompts/`. A change can therefore be
made and measured rather than argued about.

## Why this architecture

A hand-written loop, not a framework. The loop is the part being judged; a framework would hide it
and make the budgets, the retry policy and the forced final answer harder to control and to explain.
With six tools and one control tool there is nothing left for a framework to manage.

No planner/executor split, no multi-agent design. There is one investigation, one evidence ledger and
one answer; both would add parts to explain and places for state to diverge.

The principle running through everything below: **the model interprets, code decides.** The model
reads the request and weighs evidence. Anything with one correct answer — date arithmetic, argument
validation, anomaly detection, confidence limits — is computed in code, where it is unit tested.

## Data: one event stream, everything else derived

The input is a log file. The agent also needs metrics, deployments and dependencies, and there are
only two ways to get them: invent them alongside the events, or compute them from the events.

| Derived at ingest | Rule |
|---|---|
| `request_rate` | events per service per minute |
| `error_rate` | `ERROR`/`FATAL`/`CRITICAL` over total, per service per minute |
| `latency_p95_ms` | 95th percentile of `latency_ms` per service per minute, where present |
| deployments | events carrying `event_type: "deployment"` and a version |
| dependencies | distinct `service → target` pairs |

Computed at ingest rather than on read, so every metric definition lives in one readable, tested
place instead of inside a tool's SQL.

**This is what stops the agent fabricating evidence.** An earlier version generated metrics from
fixtures and quietly produced a plausible flat series for any metric a service had never emitted. The
agent reported *"orders-db showed no error-rate spike"* as an observed fact, cited it, and stopped
looking at the service that was failing. Deriving from events removes the possibility rather than
guarding against it: no rows means the tool returns nothing, which routes into the `empty` wording
the agent already handles.

SQLite: one file, no setup. Raw `sqlite3`, not an ORM. One dataset at a time — a dataset picker would
be a mode, and modes are what this design keeps removing.

## The agent loop

```python
def run_turn(session, user_message):
    budget = Budget(...)                       # every limit comes from .env
    while budget.has_steps() and not budget.stuck():
        reply = llm.chat(session.messages, tools=ALL_TOOLS, tool_choice="required")
        budget.use_step()
        for call in reply.tool_calls:
            if call.name == "submit_response":
                outcome = finalize(call, session, budget, alone=len(reply.tool_calls) == 1)
                reply_to(call, ...)            # every call gets a result item
                if outcome:
                    return outcome
            else:
                reply_to(call, executor.run(call, session, budget, batch))
    return force_final(session)                # one last call, offering only submit_response
```

`tool_choice="required"` means the model must call a tool at every step, so the only way to end a turn
is `submit_response`. No free text to parse, no ambiguity about whether it meant to stop.

Four rules keep it safe:

- **Every tool call gets a result item**, including rejected ones. The API refuses the next request if
  any `call_id` is unanswered.
- **`submit_response` must be alone in its step.** Alongside data tools it is answered with "review the
  other results first"; concluding before reading what you asked for is not a conclusion.
- **`submit_response` is never blocked by the tool budget.** Running out must not cost the answer.
- **The turn always ends with an answer.** If the budget is spent or the loop is stuck, `force_final`
  makes one more call offering only `submit_response`, and code adds a gap saying so.

**Parallel tool calls are enabled**, run in emission order. It saves model round trips — the expensive
part — when calls are independent. Dependent calls serialise on their own: the model cannot narrow a
log search to 14:37 before metrics told it 14:37. Concurrent *execution* was not added: the store is
in-process, so it would save nothing while making observation order non-deterministic.

**Cost bound per turn: `AGENT_MAX_LLM_STEPS` plus one** (11 by default). A repair of an invalid final
response happens inside the loop and costs a step, not an extra call.

**Why the Responses API.** `gpt-5.6-luna` rejects function tools combined with reasoning on Chat
Completions, and rejects the request when `reasoning_effort` is omitted because it reasons by default.
Only `reasoning_effort: none` works there — a poor trade, because `tool_choice` is `"required"`, so
every reply is tool calls with no text content, and reasoning is the only place the agent can
deliberate between steps. The conversation is therefore Responses items: role messages, the model's
own output (reasoning and function calls), and one `function_call_output` per call. With
`store=False` the reasoning items travel in the conversation, which is why the client requests
`reasoning.encrypted_content`. Nothing above the adapter changed.

## How the agent chooses a tool

Nothing in the code picks tools. The model does, from three inputs:

1. **The tool schemas.** Each description says when the tool is useful, not just what it returns.
   `get_metrics` says it is usually the first step when asked why something went wrong;
   `get_service_dependencies` says to use it when nothing local explains a problem.
2. **The system prompt** (`prompts/system.md`), which describes good investigative practice without
   prescribing a sequence.
3. **The observations so far**, in the message history, including earlier turns.

Different questions produce different traces. "What does checkout-api depend on?" is one lookup. "Why
did payment-service latency rise?" needs metrics, then dependencies, then metrics on the upstream
service — the third call's arguments exist only because of the second call's result. Evaluation
scenario E04 asserts exactly that, deterministically.

## Preventing loops and runaway cost

| Control | Variable | Default | On hit |
|---|---|---|---|
| Model steps per turn | `AGENT_MAX_LLM_STEPS` | 10 | `force_final` |
| Tool calls per turn, including rejected ones | `AGENT_MAX_TOOL_CALLS` | 12 | further calls return `budget_exceeded` |
| Consecutive unproductive results | `AGENT_STUCK_THRESHOLD` | 3 | `force_final` |
| Tool timeout / retries | `AGENT_TOOL_TIMEOUT_S` / `AGENT_TOOL_RETRIES` | 2 s / 1 | retry once, then report `timeout` |
| Repairs of an invalid final response | `AGENT_REPAIR_ATTEMPTS` | 1 per turn | return it marked `unverified` |
| Forced final call | — | 1 per turn | allowed even when the step budget is spent |

An unproductive result is `invalid_arguments`, `duplicate` or `budget_exceeded` — the outcomes that
taught the model nothing. Three in a row means no progress, and the loop stops rather than spending
the rest of the budget confirming it.

**De-duplication.** A call's tool name and validated arguments are hashed; an exact repeat of a call
that already succeeded returns `duplicate` pointing at the earlier observation. Two details:

- **Only successful queries are remembered.** Caching a failure would leave a timed-out call
  permanently unrepeatable *and* permanently uncitable.
- **Actions are excluded.** A second note is refused by the note policy, which says something more
  useful, and a note on a later turn is legitimate.

It is exact-match, not fuzzy. Shifting a window by a minute makes a new call — deliberately, since
narrowing around a spike is the behaviour the prompt asks for. The step budget is the backstop.

## Handling poor or contradictory tool responses

The executor never raises. Every call returns the same envelope, so a failure is information the model
can reason about rather than an exception that ends the run:

```json
{
  "observation_id": "obs_004",
  "status": "ok | empty | error | timeout | invalid_arguments | duplicate | budget_exceeded",
  "summary": "error_rate on checkout-api ... baseline (median) 0.0094 (0.94%); spike starts 14:37 ...",
  "data": [],
  "error": null,
  "attempts": 1
}
```

| Situation | Behaviour |
|---|---|
| Timeout or transient error | Retried once. If it still fails: recorded, not citable, and the model must report it in `gaps` or `missing_evidence`. |
| Malformed response | Validated against a Pydantic model before the model sees it; a mismatch becomes `error: malformed_response`. Raw malformed data is never forwarded. |
| Empty result | `status: empty`. Citable, because "the query returned nothing" is a fact. |
| Invalid arguments | Not executed. The error names the valid services or metrics so the model can correct itself. |
| Contradictory evidence | Both observations kept and citable, as supporting and contradicting. Confidence capped, a next check recommended. |
| Repeated call | Not run again; the earlier observation is returned. |

**`empty` is a fact about the query, not about the system.** `get_deployments` returning nothing does
not prove no deployment happened — the window or the service may be wrong, or the data incomplete.
The summary says so in those words every time:

> get_deployments returned no matching records for checkout-api, 2026-09-22T14:00:00Z to
> 2026-09-22T16:00:00Z. This is a result about the query, not proof that nothing happened.

This is the failure mode most likely to produce a confident wrong answer, so the wording is code, not
left to the model.

**Summaries are computed by code**, one function per tool, from the full result. Only the rows sent to
the model are capped at 20, and the summary says when that happened. Spike detection therefore has one
correct answer, is unit tested, and costs no extra model call.

| Metric rule | |
|---|---|
| Minimum data | at least `AGENT_MIN_METRIC_POINTS` (5), otherwise no spike claim |
| Baseline | median of the window |
| Usable baseline | if the median exceeds the quietest tenth by the spike threshold, the window is mostly elevated: no baseline, no spike start, ask for a wider window |
| Spike | first point above `max(AGENT_SPIKE_MULTIPLIER × baseline, baseline + min_delta)` |
| `min_delta` | `error_rate` 0.01 · `latency_p95_ms` 100 · `request_rate` 10, so a tiny baseline cannot turn noise into a "3× spike" |
| Scope | increases only |

The "usable baseline" step exists because the obvious version of the rule *cannot fire*. Counting
points above the threshold can never exceed half the window, because the threshold derives from the
median of that same window: if most points are elevated the median rises with them, and the summary
reports "no spike detected" — which reads as an all-clear. Comparing the median against the quietest
tenth detects it instead. This matters, because narrowing a window around a spike is exactly what the
prompt asks for.

Tool output is treated as data, not instructions — log text is untrusted input, and the system prompt
says so.

## Time

The model identifies the time expression; code resolves and validates it.

```
"yesterday between 2 PM and 4 PM"
  → resolve_time_range(...) → {start: 2026-09-22T14:00:00Z, end: ...T16:00:00Z, assumptions: []}
  → the model passes those timestamps to the data tools
  → the validator checks them again: start < end, end not after NOW, window at most 7 days
```

The grammar is small, explicit and unit tested, all UTC: `today`/`yesterday`/`tomorrow`, an ISO date
or a month-name date (`22 September`, `Sept 22`, optional year), optionally with `morning` 06–12,
`afternoon` 12–18, `evening` 18–24 or `night` 00–06; `last N minutes/hours/days`; a clock range
("2 PM to 4 PM"); or two ISO timestamps.

A general-purpose date parser was not used: it guesses silently on input like "during the outage". A
bounded grammar refuses instead, with a reason, and the agent asks.

- **A date-less expression** resolves to the most recent occurrence that has already **started**,
  clipped to now. Both the assumed date and the clip are returned as assumptions, and code copies them
  into the report. Requiring it to have *finished* would send "this afternoon" to yesterday whenever
  the data ends mid-afternoon — exactly when someone is investigating.
- **Vague expressions** return `unresolvable`; **future ranges** return `future_range`.
- **Narrowing a window** from a timestamp seen in a result is allowed: that is copying an observed
  value, not date arithmetic, and the validator still bounds it.

`AGENT_NOW` is `data` (the last event in the dataset) or a fixed timestamp. `data` makes the clock
follow the evidence, so "this afternoon" always refers to data that exists; a fixed timestamp is for
when a run has to be repeatable. It is always injected, never read from the system clock.

## Conversation state

One `Session` per conversation: the full message history including every tool call and result, the
evidence ledger, the de-duplication hashes, the turn counter. It is written to the state database
after each turn, so a conversation survives a server restart and can be replayed.

Follow-ups work because the model sees the previous turn's observations. "Was there a deployment
around that time?" is answered from the existing `obs_` id with no new tool call, and the citation
still validates because the ledger spans the conversation.

The service and time window are not extracted into separate state — they are already in the arguments
of earlier calls. A second copy would be a second thing to keep correct.

Budgets reset each turn; the ledger and hashes persist, which is right because the dataset does not
change underneath a conversation.

## The final response

```
response_type          "answer" | "clarification" | "investigation_report"
message                the answer itself, leading with the conclusion
observed_facts         [{statement, evidence_ids}]     only things a tool returned
hypotheses             [{statement, supporting_evidence_ids, contradicting_evidence_ids,
                         missing_evidence, confidence}]
likely_cause           null when the evidence is inconclusive
recommended_actions    []
gaps                   failed calls, missing data, contradictions
assumptions            e.g. "Date not given; assumed 2026-09-22"
```

`finalize()` checks four things:

1. The payload matches the schema, and every observed fact cites at least one observation.
2. Every cited id exists with status `ok` or `empty`. A failed call cannot be evidence.
3. Every call that failed this turn is named in `gaps` or in a hypothesis's `missing_evidence`.
4. The confidence limits below.

On failure the specific errors go back for **one** repair attempt. If the repair also fails, the
response is returned marked `unverified` with invalid citations removed. If it cannot be parsed at
all, code builds the response from the ledger. The user always gets an answer and always knows its
status.

### How much it says

The answer is the size of the question, enforced in two places rather than asked for in one. The
prompt says to answer what was asked and stop. The schema bounds it: 8 observed facts, 3 hypotheses,
3 recommended actions, 5 gaps, 1200 characters of message.

The failure mode is not verbosity for its own sake. An agent that made ten tool calls is under
pressure to justify all ten, and the natural way is to list every result as a fact — which buries the
two that matter. Investigating widely and reporting briefly are separate things; only the second
should be rationed.

### Confidence

Confidence describes how well evidence supports a hypothesis. It is not a probability and not proof of
causation. The model proposes it; code lowers it when the evidence does not support it, visibly.

| Rule | Reason |
|---|---|
| No supporting evidence → `low` | an unsupported idea is a guess |
| Any tool call failed this turn → at most `medium` | part of the picture is missing |
| Fewer than two distinct `ok` data tools supporting it → at most `medium` | independent sources must agree; "different tools" is a deliberately simple proxy |
| Cited contradicting evidence → at most `medium` | an unresolved conflict rules out high |
| Not the single most broadly supported hypothesis → at most `medium` | two competing explanations cannot both be well supported |
| `likely_cause` null unless some hypothesis reaches `medium` | otherwise the report says inconclusive |

An `empty` observation may be cited — "no deployment records were found" can support a hypothesis that
no deployment was involved — but never counts towards the two-tool rule. Absence of data cannot raise
confidence.

**The caps read the evidence ledger, not the model's own fields.** An earlier version capped confidence
whenever the model listed `missing_evidence`, which rewards leaving it blank: the cheapest way to keep
a high rating is to report less. Reading failures from the ledger means the cap applies whether or not
the model mentions them, and a separate check requires it to. `test_report.py` pins this: a silent
payload and an honest one get the same confidence.

## What enforces what

The design does not claim enforcement that code does not provide.

| Rule | Enforced by |
|---|---|
| A failed call cannot be cited | code (`finalize`) |
| Every observed fact cites an observation | code (schema and `finalize`) |
| A failed call appears in `gaps` or `missing_evidence` | code (`finalize`) |
| An empty result is worded as a query result, not proof of absence | code (summary wording) |
| An empty result cannot raise confidence | code |
| Confidence downgrades are visible | code |
| A forced final answer says it stopped early | code |
| An unverified response is marked unverified | code |
| Time assumptions appear in the report | code (copied from the resolver) |
| Truncation and insufficient data are disclosed | code (summaries) |
| A note cites only successful observations; one per turn | code (executor policy) |
| No tool can change production | code (no such tool exists) |
| **A citation actually supports its statement** | **not enforced** — see Limitations |
| Contradicting evidence is reported at all | prompt; the cap applies only to what the model volunteers |
| Timing alone is correlation, not causation | prompt |
| A note is filed only for an investigation with a finding | prompt; evaluation |
| `resolve_time_range` is used for worded expressions | prompt |

## Actions and dangerous operations

`create_incident_note` runs automatically, as the brief allows. It is called during the loop, before
the final answer exists, so code cannot judge whether it is warranted. Enforcement is split honestly:
code checks structure (valid citations, bounded fields, one per turn), the prompt carries the
judgement, and the evaluation verifies it — scenarios assert zero notes where none is wanted.

**There is no tool that changes production.** Asked to roll back, the agent does not act; it explains
that a human must carry it out, and may recommend it. This is a scope decision, not an omission: the
brief makes a rollback action optional, and the guardrail worth demonstrating is that the agent cannot
take a destructive action at all. A `rollback_deployment` behind an approval checkpoint — the loop
pausing, returning `pending_action`, executing only after the user confirms that exact service and
version — is the first thing to add next.

## State and run logs

Two databases. The **dataset** is replaced on every upload. The **state** database is not:
conversations, run logs and evaluation results outlive both a new upload and a restart.

| Group | Tables |
|---|---|
| Run logs | `run`, `log` |
| Conversations | `session`, `turn`, `message`, `obs` |
| Evaluations | `eval_run`, `eval_result` |

A conversation stores three things because they answer three questions: `turn` is what the UI replays,
`message` is what the agent needs to continue, `obs` is the ledger that keeps an earlier observation
citable later. It records the dataset it was asked about; one started against a previous dataset is
kept as history rather than resumed, because its evidence refers to data no longer loaded.

Every workflow — ingest, turn, evaluation — opens a run and writes ordered events, chosen so one run
reads as an account of what happened and nothing more:

| Event | Carries |
|---|---|
| `turn.start` / `turn.done` | the question; then the response type and the counts |
| `model.step` | which step of the budget, and which tools were asked for at once |
| `tool.call` | position in the turn, step, the arguments that matter, status, attempts, duration, what it found |
| `response.rejected` | a final response failed validation and went back for repair |
| `turn.forced_final` | the loop stopped early, and whether it was budget or no progress |
| `ingest.*` | what was read, what was skipped and why, what was derived |
| `eval.*` | per-scenario pass, critical flag, coverage, scores |

**Failures are recorded before they propagate.** A turn that raises still leaves a run marked `error`
with a `turn.failed` event; a crash with no trace is the one thing you cannot debug afterwards.

## Evaluation

Twelve scenarios against the ingested dataset. How they are scored is the design:

**Deterministic checks for facts about the run.** Were the required tools called — in any order, since
a fixed sequence would punish the agent for investigating well. Did an invalid range reach the
backend. Was a transient timeout retried and recovered. Was malformed output kept out of the evidence.
Did a call use a value learnt from an earlier call. Did a follow-up reuse the investigation. Was a note
created where none was wanted. Wasted calls are counted and reported, not failed: "spent two extra
calls" is information, not a defect.

**An LLM judge only for what needs reading**: factual correctness, grounding, uncertainty about
causation, completeness, actionability — 0, 1 or 2 each, and it can raise a critical error. Its prompt
is `prompts/judge.md`, a file like the others, and it is told not to reward length.

A scenario passes when nothing critical happened, every required tool was called, every deterministic
check held, and no judge score is 0.

It runs from the command line or the Evaluation tab. From the tab it runs on a worker thread and the
page polls, because twelve scenarios take minutes. Results are written after each scenario, so an
interrupted run keeps everything it scored.

Several of the critical errors the brief names are already *impossible* rather than merely checked: a
failed call cannot be cited because `finalize()` rejects it, and an invented metric cannot exist
because the store has no rows to invent from. The evaluation asserts them anyway — a guardrail never
exercised is one nobody notices has broken.

## Limitations

1. **Citation checking proves provenance, not entailment.** Code verifies `obs_004` exists and
   succeeded, not that it says what the statement claims. An LLM judge scoring entailment is the
   production answer.
2. **A window lying entirely inside an incident cannot be detected.** With no quiet points to compare
   against, a constant 17% error rate is indistinguishable from a service whose normal rate is 17%.
3. **Spike detection finds increases only.** Traffic falling to zero is a real signature and is missed.
4. **`error_rate` is only as precise as the log volume.** Errors over events in a one-minute bucket: at
   five events a minute, one error reads as 20%.
5. **The contradiction cap depends on the model reporting the contradiction.** Code cannot detect that
   two observations conflict in meaning.
6. **Code cannot force the use of `resolve_time_range`.** The model can pass timestamps it worked out
   itself; the validator still bounds them.
7. **The judge is a single model call with no second opinion**, as variable as any model. It catches
   what deterministic checks cannot read; it is not an authority.
8. **One evaluation at a time**, with progress held in memory. Restarting mid-run loses the progress,
   though every scenario already scored is in the database.

## What would change for production scale

- **Real backends** behind the same `Store` interface — the executor, envelope and summaries do not
  change. Per-tool timeouts and circuit breakers, and streaming ingest to disk rather than reading the
  upload into memory.
- **Context compaction**: old tool results collapsed to their summaries once a conversation outgrows
  the window.
- **Cost controls per tenant**, not just per turn.
- **Tracing**: the run log is most of one; it lacks token counts and cost, and a way to ship events
  somewhere other than SQLite.
- **Auth, RBAC and an audit log**, necessary the moment a tool can change something.
- **Approval workflow** for mutating actions, as above.
- **Evaluations in CI**, pass rate tracked over time, so a prompt change that costs accuracy is visible
  before it ships.
- **Entailment checking** on citations, closing limitation 1.

## Where each requirement is answered

| The brief asks for | Section |
|---|---|
| Dynamic tool selection | How the agent chooses a tool |
| Multi-step investigation | How the agent chooses a tool; scenario E04 asserts it |
| Evidence-based output, facts separate from hypotheses | The final response; What enforces what |
| Recommended next actions | The final response |
| Conversation state and follow-ups | Conversation state; State and run logs |
| Tool timeouts and temporary failures | Handling poor or contradictory tool responses |
| Empty or malformed responses | Handling poor or contradictory tool responses |
| Preventing repeated or infinite loops | Preventing loops and runaway cost |
| Validating tool arguments | The agent loop; Time |
| Invalid or nonsensical time ranges | Time |
| Not executing dangerous actions | Actions and dangerous operations |
| Why this architecture | Why this architecture |
| How poor or contradictory responses are handled | Handling poor or contradictory tool responses |
| How conversation state is maintained | Conversation state |
| What would change for production scale | What would change for production scale |
| Ten or more evaluation scenarios | `evaluations/scenarios.json` (twelve); Evaluation |
