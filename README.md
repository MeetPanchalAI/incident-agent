# Incident Agent

An AI agent that investigates production incidents in your log data. It decides which tools to
call, uses each result to choose the next step, keeps state across follow-up questions, survives
failing tools, and reports observed facts separately from its own hypotheses.

You upload a log file. The agent answers questions about it.

- `DESIGN.md` — architecture and the reasoning behind it
- `AI_USAGE.md` — how AI coding assistants were used

## Setup

```bash
pip install -e ".[dev]"
cp .env.example .env        # then set OPENAI_API_KEY
```

## Run

```bash
uvicorn incident_agent.api:app --reload           # web UI at http://localhost:8000
python -m incident_agent.cli --ingest logs.jsonl  # load a dataset from the command line
python -m incident_agent.cli                      # interactive
python -m incident_agent.cli "your question"      # one question, then exit
python -m evaluations.runner                      # the twelve evaluation scenarios
pytest                                            # 212 tests, no API key needed
```

Upload a log file with the **Upload data** button, then ask questions. Nothing works until a
dataset is loaded, and the UI says so rather than answering from nothing.

CLI commands: `/ingest <path>`, `/data`, `/trace`, `/reset`, `/help`, `/quit`.

Example questions, once data is loaded:

```
Why did checkout-api start failing this afternoon?
Was there a deployment around that time?
What does checkout-api depend on?
```

## Data format

One JSON object per line (JSONL). Four fields are required; the optional ones are what make the
richer tools possible.

```json
{"ts":"2026-09-22T14:00:03Z","service":"checkout-api","level":"INFO","message":"POST /checkout 200","latency_ms":142,"status_code":200,"target":"orders-db"}
{"ts":"2026-09-22T14:32:00Z","service":"checkout-api","level":"INFO","message":"deployment complete: v142","event_type":"deployment","version":"v142","status":"success"}
{"ts":"2026-09-22T14:37:04Z","service":"checkout-api","level":"ERROR","message":"database connection timeout after 5000ms","latency_ms":5000,"status_code":500,"target":"orders-db"}
```

| Field | Required | What it gives you |
|---|---|---|
| `ts` | yes | ISO 8601 timestamp |
| `service` | yes | Which service emitted the event. Lower-cased on ingest |
| `level` | yes | `DEBUG`, `INFO`, `WARN`/`WARNING`, `ERROR`, `FATAL`, `CRITICAL` |
| `message` | yes | The log line. `search_logs` matches against level and message |
| `latency_ms` | no | Without it there is no `latency_p95_ms` metric for that service |
| `target` | no | The service being called. This is what builds the dependency graph |
| `event_type` | no | `deployment`, with `version` and `status`, makes the event a release record |
| `status_code` | no | Stored, not currently read by a tool |

A line missing a required field is skipped, counted and reported — never silently dropped. If
nothing in the file is usable the ingest is refused, rather than leaving you with an empty dataset.

**The file only has to contain events.** The other three tool surfaces are computed from them at
ingest time:

| Derived | How |
|---|---|
| `request_rate` | events per service per minute |
| `error_rate` | `ERROR`/`FATAL`/`CRITICAL` divided by total, per service per minute |
| `latency_p95_ms` | 95th percentile of `latency_ms` per service per minute, where present |
| deployments | events with `event_type: "deployment"` |
| dependencies | distinct `service -> target` pairs |

If a service never reported something there are no rows for it and the tool returns nothing. The
agent cannot reason over data that was never collected.

### Bringing your own dataset

Four things determine what the agent will be able to find in it.

**Volume.** `error_rate` is errors divided by events in a one-minute bucket, so at five events a
minute one error reads as 20%. Twenty or more events a minute per active service keeps the rate
meaningful.

**`target` belongs on the caller.** The edge `checkout-api -> orders-db` comes from checkout-api's
own lines carrying `"target": "orders-db"`. Without it there is no dependency graph. A service that
is only ever called still appears, but has no metrics of its own.

**`latency_ms` is per service.** A service that never reports it has no `latency_p95_ms`, and the
agent will correctly say so rather than estimate one.

**Variety, if you want to test judgement rather than lookup.** Useful situations to include:

| Situation | What it tests |
|---|---|
| A service that is fine throughout | Whether the agent says "nothing is wrong" instead of inventing a cause |
| An incident with one clear cause | The basic investigation |
| Two plausible causes minutes apart | Whether it holds both open instead of picking one |
| An upstream service degrading first | Whether it follows dependencies rather than stopping at the symptom |
| A deployment long before an unrelated problem | Whether it blames the nearest deployment regardless |
| A service missing `latency_ms` | Whether it reports the gap instead of filling it |

For a small working example:

```bash
python -m tests.sample_data > sample_logs.jsonl
```

That is the generator the tests use: two hours, three services, one deployment, an error spike and
a database degrading underneath it.

## The dataset used here

`production_data.jsonl`: 6,635 events over 48 hours, 2026-09-22 to 2026-09-23. Ten services emit
events — `api-gateway`, `checkout-api`, `payment-service`, `catalog-service`, `auth-service`,
`recommendation-service`, `notification-service`, `orders-worker`, `search-service`,
`inventory-service` — and nine more appear as call targets (`orders-db`, `auth-db`, `catalog-db`,
`inventory-db`, `redis`, `payment-gateway`, `notification-provider`, `search-backend`,
`external-bank-network`), giving 19 in the dependency graph.

It is not random traffic. Every situation in the table above is present, plus independent incidents
in search, auth and inventory, and ordinary background noise: latency variation, occasional
4xx/5xx/429, several deployments.

Three lines are deliberately broken — a missing message, an invalid timestamp, an invalid log level.
Ingest skips them, counts them and gives a reason for each. The ingest report, not this paragraph,
is the source of truth for what loaded.

## Configuration

Every tunable parameter is in `.env`. `.env.example` is the same file with notes.

| Variable | Default | Meaning |
|---|---|---|
| `OPENAI_API_KEY` | — | Required for the CLI and the web UI. Not needed by the tests. |
| `AGENT_MODEL` | `gpt-5.6-luna` | Any OpenAI model with tool calling. |
| `AGENT_TEMPERATURE` | `1` | Reasoning models generally require 1. |
| `AGENT_REASONING_EFFORT` | `low` | `minimal`/`low`/`medium`/`high`. Only sent when set. |
| `AGENT_NOW` | `data` | `data` = the last event in the dataset. A fixed ISO timestamp pins it instead. |
| `AGENT_DB_PATH` | `data/incident.db` | Where the ingested dataset is stored. |
| `AGENT_STATE_DB_PATH` | `data/state.db` | Conversations, run logs and evaluation results. Separate from the dataset, so uploading never erases them. |
| `AGENT_PROMPTS_DIR` | `prompts/` | Where the prompt files live. |
| `AGENT_MAX_LLM_STEPS` | `10` | Loop steps per turn. The cost bound per turn is this plus one. |
| `AGENT_MAX_TOOL_CALLS` | `12` | Tool calls per turn, including rejected ones. |
| `AGENT_STUCK_THRESHOLD` | `3` | Consecutive results that taught the model nothing before the loop gives up. |
| `AGENT_TOOL_TIMEOUT_S` | `2.0` | Per-call timeout. |
| `AGENT_TOOL_RETRIES` | `1` | Retries on a timeout or a transient error. |
| `AGENT_REPAIR_ATTEMPTS` | `1` | Attempts to fix an invalid final response before it is marked unverified. |
| `AGENT_MAX_ROWS` | `20` | Rows of tool data sent to the model. Summaries always use the full result. |
| `AGENT_MAX_WINDOW_DAYS` | `7` | Largest window a data tool accepts. |
| `AGENT_SPIKE_MULTIPLIER` | `3` | A spike is above `max(multiplier x baseline, baseline + the metric's minimum difference)`. |
| `AGENT_MIN_METRIC_POINTS` | `5` | Below this, no spike claim is made. |

Every prompt is a file in `prompts/`, none is a literal in the code: `system.md` orchestrates the
agent, four short files are the messages the loop sends back to the model, and `judge.md` grades an
evaluation. Edit them and re-run to see the difference; a test asserts the directory and the code
agree on which files exist.

## Answer length

The answer is the size of the question. A lookup gets a sentence; an investigation gets a short
report. Both are bounded by the schema, not only asked for in the prompt: at most 8 observed facts,
3 hypotheses, 3 recommended actions, 5 gaps, and a message of 1200 characters.

```
Q: Was there a deployment of checkout-api between 14:00 and 15:00 UTC?
   1 tool call, 86 characters
   "Yes. checkout-api version v142 was deployed successfully at 14:32 UTC on 22 September."
   1 fact, no hypotheses, no recommended actions

Q: Investigate why checkout-api had increased errors that afternoon.
   8 tool calls, 504 characters
   5 facts, 2 hypotheses, 3 recommended actions, 2 gaps
```

Investigating widely and reporting briefly are separate things. The agent still makes every call it
needs; it just does not read the transcript back to you.

## Layout

```
prompts/            every prompt: system.md, the four loop messages, and judge.md
evaluations/        the twelve scenarios, the runner and the judge
src/incident_agent/
  agent.py          the loop: steps, budgets, the final response
  session.py        conversation, evidence ledger, de-duplication
  report.py         submit_response schema, citation checks, confidence limits
  prompts.py        loads the prompt files
  llm.py            OpenAI Responses adapter, and the scripted fake used by the tests
  config.py         every tunable parameter, read from the environment
  state.py          conversations, run logs and evaluation results
  cli.py            interactive command line
  api.py            five HTTP endpoints
  static/index.html the web UI, no build step
  tools/
    ingest.py       parse a log file and derive the tool surfaces from it
    store.py        the SQLite dataset the tools read
    schemas.py      argument and result models, the tool registry
    executor.py     the guardrail pipeline for one tool call
    summaries.py    deterministic summaries, including spike detection
    time_resolver.py  time expressions to UTC ranges
tests/              212 tests, no API key required
```

## Tests

`pytest` runs 212 tests against a scripted fake model. They are deterministic, free, and need no
network. `tests/sample_data.py` builds the dataset they share.

| File | Covers |
|---|---|
| `test_ingest.py` | Parsing, skipped lines, refusal of unusable files, and every derivation rule. |
| `test_config.py` | Every environment variable, the clock from the dataset, prompt loading. |
| `test_time_resolver.py` | Every supported expression, missing dates, future and unresolvable input. |
| `test_validation.py` | Unknown services and metrics, invalid time ranges, over-long arguments, malformed JSON. |
| `test_executor.py` | Timeout and retry, malformed and empty responses, de-duplication, budgets, the note policy. |
| `test_summaries.py` | Spike detection, unusable baselines, insufficient data, truncation notices. |
| `test_report.py` | Citation checks, confidence limits, the unverified and ledger fallbacks. |
| `test_agent_loop.py` | Parallel calls, budget cut-off, stuck detection, forced final answer, repair, follow-ups. |
| `test_api.py` | The endpoints, ingest, evaluation control, and conversations surviving a restart. |
| `test_state.py` | What each workflow records, and that a failure is recorded before it propagates. |
| `test_evaluations.py` | The scenario file, tool-coverage matching, every deterministic check, the runner. |

## Evaluation

Twelve scenarios in `evaluations/scenarios.json`, written against the ingested dataset. Each one
carries the question, the expected answer, the tools the investigation needs, claims the agent must
not make, and whether an incident note is appropriate.

Run it from the **Evaluation** tab, or from the command line:

```bash
python -m evaluations.runner                      # all twelve
python -m evaluations.runner --scenario E05       # one (repeatable)
python -m evaluations.runner --no-judge           # deterministic checks only, no model grading
python -m evaluations.runner --out report.json    # full results as JSON
```

Either way the run is recorded. The tab shows each scenario as it is scored, the five judge
dimensions as a row of coloured squares, and the reason for anything that failed. Past runs are
listed underneath and can be reopened, so a change to a prompt or a parameter can be compared
against the last run.

Scoring is split deliberately.

**Deterministic** for anything that is a fact about the run: were the required tools called (in any
order), did an invalid time range reach the backend, was a transient timeout retried and recovered,
was malformed output kept out of the evidence, did a follow-up reuse the existing investigation,
was an incident note created where none was wanted. Wasted calls — duplicates, rejected arguments,
calls past the budget — are counted and reported, not failed.

**An LLM judge** only for what needs reading: factual correctness, grounding in the evidence,
appropriate uncertainty about causation, completeness and actionability. Each is scored 0 (wrong),
1 (partially correct) or 2 (correct), and the judge can raise a critical error of its own.

A scenario passes when nothing critical happened, every required tool was called, every
deterministic check held, and no judge score is 0.

Two scenarios inject faults through the store: `E08` times a metrics call out on its first attempt,
`E09` returns one malformed metrics response.

## Monitoring

Every workflow — an ingest, an agent turn, an evaluation run — is recorded as a run with ordered
events. A turn reads as an account of the investigation:

```
turn.start    Was anything wrong with catalog-service on 23 September?
model.step    step 1/10: 1 call(s) - resolve_time_range
tool.call     #1 resolve_time_range(23 September) -> ok in 0ms | resolves to ...
model.step    step 2/10: 4 call(s) - get_metrics, get_metrics, get_metrics, get_deployments
tool.call     #2 get_metrics(catalog-service, error_rate, 00:00-23:55) -> ok in 6ms | baseline ...
...
turn.done     investigation_report: catalog-service had a significant incident on 23 September
```

Each tool call carries its position in the turn, the step it belonged to, the arguments that
matter, the status, the attempt count, the duration and what it found. The **Monitoring** tab lists
runs newest first; clicking one shows its events.

## Persistence

The dataset lives in `data/incident.db` and is replaced on every upload. Everything else lives in
`data/state.db` and is not: conversations with their evidence ledgers, run logs, and evaluation
results all survive both a new upload and a server restart.

A conversation is reopened by the browser when you return to the page, and can be replayed from the
database. One started against a previous dataset is kept as history rather than resumed, because
its evidence refers to data that is no longer loaded.
