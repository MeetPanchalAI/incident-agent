# AI usage

## Tools

Claude Code (Claude Opus 5), used throughout.

## What I used it for

**Design review before any code.** I wrote the design plan myself and then had Claude review it
adversarially, twice. That was the most useful part. Two findings changed the design:

- The confidence limits capped a hypothesis at medium whenever the model filled in
  `missing_evidence` or `contradicting_evidence_ids`. Both are fields the model writes, so the
  cheapest way to keep a high rating is to report less — the rule penalised the honesty it was
  meant to encourage. The caps now read failures from the evidence ledger, which the model cannot
  suppress, and a separate check requires it to disclose them. `test_report.py` pins this: a silent
  payload and an honest one get the same confidence.
- De-duplication would have cached failed calls. A timed-out call would then be permanently
  unrepeatable and permanently uncitable, leaving the agent with no way to get the data and no way
  to talk about it. Only `ok` and `empty` results are cached now.

**Implementation.** The time-resolver grammar, the ingest and derivation pipeline, the guardrail
pipeline, the web UI, and most of the tests. I reviewed each file and ran the suite after every
step.

**Documentation.** Drafting README.md and DESIGN.md, which I then checked line by line against the
code.

## Where I rejected or corrected what it produced

**A design recommendation I overruled.** The review argued for moving a `rollback_deployment` tool
with a human-approval checkpoint in early, and pushing the web UI back, on the grounds that a
rollback request tests nothing if the agent has no rollback tool to refuse. The reasoning is fair,
but the brief makes the rollback action optional and cautions against building a UI-heavy product,
while I wanted a working interface to drive the agent with. I kept the UI, left the approval
checkpoint designed but unbuilt, and wrote down why in DESIGN.md rather than leaving it looking
like an oversight.

**A bug the tests caught.** De-duplication was applied to every tool, including
`create_incident_note`. Queries are idempotent, so returning the earlier result for a repeat is
right. Actions are not: filing a note in a later turn about the same investigation is legitimate
work, and the executor was silently refusing it as a duplicate. The fix excludes action tools from
de-duplication and lets the note policy govern them instead, which also produces a more useful
message than "duplicate". Two tests in `test_executor.py` cover it.

**Two smaller corrections.** The summary functions crashed on an empty result, because the executor
validates and summarises before it classifies emptiness — found by the test for empty responses. And
the ISO timestamp pattern accepted a space separator, which made `2026-09-22 from 14:00 to 16:00`
parse as a single timestamp instead of a date plus a clock range; requiring the `T` separator
removed the ambiguity.

**A bug only visible in a real answer.** The first version generated metrics procedurally from
fixtures, and fell back to defaults for any metric a service had never emitted. Watching a real
investigation, I found the agent reporting *"orders-db, payment-gateway, payment-service and
web-frontend had no detected error-rate spike"* as an observed fact, with four citations — all of
it invented, and it had steered the agent away from the service that was actually failing. In a
system whose whole premise is that nothing unverified is presented as fact, the layer underneath
was manufacturing evidence. Replacing the fixtures with a database and deriving metrics from real
events removed the possibility rather than guarding against it.

**A bug the tests caught twice over.** Moving service validation out of Pydantic, I gave two
validators in a class hierarchy the same method name. Pydantic registers validators by name, so the
subclass silently replaced the parent's, and service names stopped being normalised. Nothing
errored; a de-duplication test just started failing for an unrelated-looking reason.

## One thing worth noting

The most valuable check was not a review at all. Writing a test for the rule that was supposed to
detect a mostly-elevated metric window showed the rule could never fire: the threshold is derived
from the median of the same window, so if most points are elevated the median rises with them and
nothing exceeds the threshold. The summary would have reported "no spike detected" for a window
sitting entirely inside an incident — the worst possible wrong answer, because it reads as an
all-clear. Comparing the median against the quietest tenth of the window detects it instead. Neither
I nor the review caught that by reading; the test did.
