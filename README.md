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
pytest                                            # 155 tests, no API key needed
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

For a dataset to try it with:

```bash
python -m tests.sample_data > sample_logs.jsonl
```

That is the same generator the tests use: two hours across three services, with a deployment, an
error spike and a database degrading underneath it.

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

Prompts are files in `prompts/`: `system.md` plus the four messages the loop sends back to the
model. Edit them and re-run to see the difference.

## Layout

```
prompts/            system.md and the four messages the loop sends the model
src/incident_agent/
  agent.py          the loop: steps, budgets, the final response
  session.py        conversation, evidence ledger, de-duplication
  report.py         submit_response schema, citation checks, confidence limits
  prompts.py        loads the prompt files
  llm.py            OpenAI Responses adapter, and the scripted fake used by the tests
  config.py         every tunable parameter, read from the environment
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
tests/              155 tests, no API key required
```

## Tests

`pytest` runs 155 tests against a scripted fake model. They are deterministic, free, and need no
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
| `test_api.py` | The endpoints, ingest, and session continuity. |

## Not built yet

**The evaluation suite.** An earlier version had twelve scenarios, but each was defined against
fixed mock fixtures that no longer exist. They were deleted rather than left pointing at data that
had been removed. The suite is being rebuilt against an ingested dataset, with its queries,
expected findings and metrics designed around that data, and will run from its own tab in the UI.
