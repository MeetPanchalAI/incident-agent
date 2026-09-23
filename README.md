# Incident Agent

An AI agent that investigates production incidents. It decides which tools to call, uses each
result to choose the next step, keeps state across follow-up questions, survives failing tools,
and reports observed facts separately from its own hypotheses.

The observability backend is mocked. The agent orchestration, the guardrails and the reasoning
are real.

- `DESIGN.md` — architecture and the reasoning behind it
- `AI_USAGE.md` — how AI coding assistants were used
- `evals/scenarios.py` — the twelve evaluation scenarios and what correct behaviour is

## Setup

```bash
pip install -e ".[dev]"
cp .env.example .env        # then set OPENAI_API_KEY
```

## Configuration

Every tunable parameter is in `.env`, so the agent can be retuned and re-measured with
`python -m evals.run` without touching the code. `.env.example` is the same file with notes.

| Variable | Default | Meaning |
|---|---|---|
| `OPENAI_API_KEY` | — | Required for the CLI, the web UI and the evals. Not needed by the tests. |
| `AGENT_MODEL` | `gpt-5.6-luna` | Any OpenAI model with tool calling. |
| `AGENT_TEMPERATURE` | `1` | Reasoning models generally require 1. |
| `AGENT_REASONING_EFFORT` | `low` | `minimal`/`low`/`medium`/`high`. Only sent when set, so non-reasoning models are unaffected. |
| `AGENT_NOW` | `2026-09-23T10:00:00Z` | A fixed timestamp, or `auto` for the system clock. See below. |
| `AGENT_WORLD` | `incident` | Which mock world the tools read from. |
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

Prompts are files in `prompts/`, not string literals: `system.md` plus the four messages the loop
sends back to the model. Edit them and re-run the evals to measure the difference.

## Run

```bash
python -m incident_agent.cli                      # interactive
python -m incident_agent.cli "your question"      # one question, then exit
uvicorn incident_agent.api:app --reload           # web UI at http://localhost:8000
pytest                                            # 136 automated tests, no API key needed
python -m evals.run                               # the twelve scenarios, against the real model
python -m evals.run --list                        # print the scenarios without running them
```

CLI commands: `/world <name>`, `/trace`, `/reset`, `/help`, `/quit`.

Try:

```
Investigate checkout-api errors yesterday between 2 PM and 4 PM.
Was there a deployment around that time?
```

## Mock data

`AGENT_NOW` is the agent's current time. Pinning it to a fixed timestamp, as the default does,
keeps evaluation runs reproducible; `auto` uses the system clock.

Either way the data follows it. Each world declares an `anchor_date`, and its fixtures are rebased
so the incident day is always the day before `AGENT_NOW` — so "yesterday afternoon" finds the
incident whichever day you run it, and the numbers come out identical to a pinned run. The clock
times within the day never move: the deployment is at 14:32 and the spike starts at 14:37.

Four worlds, selected with `AGENT_WORLD`, `/world` in the CLI, or the dropdown in the UI:

| World | What it contains |
|---|---|
| `incident` | checkout-api v142 deployed 14:32; error rate rises from under 1% at 14:37 to a 17.2% peak at 14:41 and is back to baseline by 15:20; logs show database connection timeouts; orders-db latency rises. |
| `healthy` | Flat metrics, no deployment in the window, routine logs. Nothing is wrong. |
| `ambiguous` | A deployment at 14:32 and an orders-db failover at 14:33. Both fit the timeline; the evidence does not separate them. |
| `upstream` | payment-service latency rises because payment-gateway degrades first. No deployment involved. |

Services: `checkout-api`, `payment-service`, `payment-gateway`, `orders-db`, `web-frontend`.
Metrics: `error_rate`, `latency_p95_ms`, `request_rate`.

Failures are injected per scenario rather than at random, so a timeout or a malformed response
can be reproduced exactly. The four fault types are `empty`, `timeout`, `transient` and
`malformed`; `evals/scenarios.py` shows where each is used.

## Layout

```
prompts/            system.md and the four messages the loop sends the model
src/incident_agent/
  agent.py          the loop: steps, budgets, the final response
  session.py        message history, evidence ledger, de-duplication
  report.py         submit_response schema, citation checks, confidence limits
  prompts.py        loads the prompt files
  llm.py            OpenAI Responses adapter, and the scripted fake used by the tests
  config.py         every tunable parameter, read from the environment
  cli.py            interactive command line
  api.py            four HTTP endpoints
  static/index.html the web UI, no build step
  tools/
    schemas.py      argument and result models, the tool registry
    executor.py     the guardrail pipeline for one tool call
    summaries.py    deterministic summaries, including spike detection
    time_resolver.py  time expressions to UTC ranges
    mock_backend.py   the mock observability backend
    worlds/*.json     the four worlds and the service catalog
tests/              automated tests, no API key required
evals/              the twelve scenarios and their runner
```

## Tests

Two kinds, for two different questions.

`pytest` runs 136 tests against a scripted fake model. They are deterministic, free, and need no
network. They cover the things that must always hold: argument validation, timeouts and retries,
malformed and empty responses, de-duplication, budgets, the note policy, citation checks,
confidence limits, and every path through the loop.

`python -m evals.run` runs twelve scenarios against the real model. They measure judgment, which
a fake model cannot: does the agent chain its calls, does it resist inventing a cause when
nothing is wrong, does it keep two explanations open when the evidence does not separate them.
Their checks assert behaviour — which tools ran, which citations were used, whether a note was
created — never wording.

| File | Covers |
|---|---|
| `test_config.py` | Every environment variable, `AGENT_NOW=auto`, fixture rebasing, prompt loading. |
| `test_time_resolver.py` | Every supported expression, missing dates, future and unresolvable input, range validation. |
| `test_validation.py` | Unknown services and metrics, invalid time ranges, over-long arguments, malformed JSON. |
| `test_executor.py` | Timeout and retry, malformed and empty responses, de-duplication, budgets, the note policy. |
| `test_summaries.py` | Spike detection, unusable baselines, insufficient data, truncation notices. |
| `test_report.py` | Citation checks, confidence limits, the unverified and ledger fallbacks. |
| `test_agent_loop.py` | Parallel calls, budget cut-off, stuck detection, forced final answer, repair, follow-ups. |
| `test_api.py` | The four endpoints and session continuity. |
| `test_evals.py` | The evaluation checks themselves. |
