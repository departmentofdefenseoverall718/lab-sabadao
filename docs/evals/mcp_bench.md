# mcp_bench setup

Canonical MCP-Bench ([Accenture/mcp-bench](https://github.com/Accenture/mcp-bench),
arXiv:2508.20453): a live agentic benchmark over 28 real Model Context Protocol
servers (250 tools), scored by rule-based tool metrics + an o4-mini LLM judge over
the execution trace.

## Status in gbench
The canonical run **cannot** be executed inside gbench's in-process HTTP harness:
it requires standing up the 28 live MCP servers (Node-based), several external API
keys, live network egress, and the o4-mini judge. gbench therefore loads the
**real** canonical task set (the three `tasks/*.json` files from the Accenture
repo — never any mock/toy data) and **skips** with `status="skipped"` unless a
live MCP-Bench runner is wired up.

## Enabling (advanced)
Set `GBENCH_MCP_BENCH_ENDPOINT` to a configured MCP-Bench runner and install the
`mcp` + `jsonschema` client stack. The in-process suite still delegates the
multi-round agentic execution and judge scoring to that external runner.
