# gaia2 — blocked_external

GAIA2 (`meta-agents-research-environments/gaia2`) is Meta's **Agents Research
Environments (ARE)** benchmark: a stateful, multi-turn, time-sensitive simulation in
which the model must act as an agent — issuing tool calls over many turns against
evolving app state (contacts, calendar, email, files, …) — and is scored by the
environment's oracle (final-state checks + timing) together with an LLM judge.

## Status: not scored (skips cleanly)
gbench issues **one request per sample** and does not host the ARE agent loop, so it
cannot produce a faithful GAIA2 score. A single-turn proxy (e.g. fuzzy
function-name matching) would not measure the benchmark, so **no approximate score
is emitted** — `gaia2` always skips with a `blocked_external` reason. The previous
heuristic scorer was removed.

## What full support requires
A Pattern-B harness (like `terminal_bench`) that:
1. installs the ARE simulator (`pip install meta-agents-research-environments`),
2. runs each GAIA2 scenario with the agent LLM pointed at the gbench endpoint,
   driving the multi-turn loop, and
3. records the environment's oracle verdict per scenario.

Until that harness is wired in, `gaia2` reports `status: "skipped"`. The gate checks
for the `are` package and `GAIA2_RUN=1`, but the run still skips (with an accurate
"harness not wired" message) because the loop is not yet implemented.
