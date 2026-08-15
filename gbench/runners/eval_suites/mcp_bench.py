# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# Evaluation Suite: mcp_bench
# Description: MCP-Bench (Accenture) - live multi-server Model Context Protocol agent benchmark

"""gbench native built-in runner for mcp_bench (Tool Use & Function Calling).

Canonical MCP-Bench (Accenture/mcp-bench, arXiv:2508.20453): a live agentic
benchmark over 28 real MCP servers (250 tools), scored by rule-based tool metrics
plus an o4-mini LLM judge over the execution trace. The canonical run requires
standing up the live MCP servers (Node-based), external API keys, and the judge —
none of which gbench can provision. This suite loads the REAL canonical task set
(no toy/mock data) and SKIPS cleanly unless a live MCP environment is configured;
it never substitutes a fabricated single-turn eval.
"""

import json
import logging
import os
import urllib.request
from typing import Any, Dict, List, Optional, Tuple
from .base import run_eval_suite
from .sampling import stratified_sample

logger = logging.getLogger(__name__)

PILLAR = "Tool Use & Function Calling"
DOCS_URL = "docs/evals/mcp_bench.md"

_TASK_URLS = [
    "https://raw.githubusercontent.com/Accenture/mcp-bench/main/tasks/mcpbench_tasks_single_runner_format.json",
    "https://raw.githubusercontent.com/Accenture/mcp-bench/main/tasks/mcpbench_tasks_multi_2server_runner_format.json",
    "https://raw.githubusercontent.com/Accenture/mcp-bench/main/tasks/mcpbench_tasks_multi_3server_runner_format.json",
]

_SYSTEM_PROMPT = (
    "You are an autonomous agent connected to a set of Model Context Protocol (MCP) "
    "servers. Discover the available tools, infer which are needed, plan a multi-step "
    "trajectory (sequential and parallel calls as appropriate), invoke the tools, and "
    "ground your final answer in the returned tool outputs."
)


def check_mcp_bench_prerequisites() -> Tuple[bool, str]:
    """MCP-Bench needs the live MCP client stack + running servers + a judge.

    None of this can be auto-provisioned by gbench, so this returns False unless a
    live environment is explicitly configured via GBENCH_MCP_BENCH_ENDPOINT.
    """
    if not os.environ.get("GBENCH_MCP_BENCH_ENDPOINT"):
        return False, (
            "MCP-Bench requires a live MCP environment (28 servers / 250 tools), "
            "external API keys, and the o4-mini judge; set GBENCH_MCP_BENCH_ENDPOINT "
            "to a configured MCP-Bench runner to enable it."
        )
    try:
        import mcp  # noqa: F401
        import jsonschema  # noqa: F401
    except ImportError:
        return False, "Python packages 'mcp'/'jsonschema' are not installed."
    return True, ""


def _load_mcp_bench_tasks(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], Any, Dict[str, Any]]]:
    """Load the real Accenture MCP-Bench task set (no gold; scored from the trace)."""
    samples = []
    for url in _TASK_URLS:
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            raise RuntimeError(f"mcp_bench: could not fetch canonical task file {url}: {e}") from e
        for group in data.get("server_tasks", []):
            servers = group.get("servers") or []
            for task in group.get("tasks", []):
                fuzzy = task.get("fuzzy_description")
                task_id = task.get("task_id")
                if not fuzzy or not task_id:
                    raise RuntimeError("mcp_bench: unexpected task schema; refusing to fabricate")
                messages = [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": fuzzy},
                ]
                samples.append((messages, None, {
                    "category": group.get("combination_type") or "mcp",
                    "task_id": task_id,
                    "servers": servers,
                    "distraction_servers": task.get("distraction_servers") or [],
                }))
    if not samples:
        raise RuntimeError("mcp_bench returned empty task set")
    # Stratified, not a contiguous head (audit RC-1).
    samples = stratified_sample(samples, limit, None, seed="mcp_bench")
    logger.info(f"Loaded {len(samples)} mcp_bench canonical tasks.")
    return samples


def run_mcp_bench(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run MCP-Bench (or skip: the canonical live-server + judge loop is required)."""
    ok, reason = check_mcp_bench_prerequisites()
    if not ok:
        msg = f"[SKIP] mcp_bench skipped: {reason} See '{DOCS_URL}' for setup instructions."
        logger.warning(msg)
        print(f"\n{msg}")
        return {
            "benchmark_type": "eval",
            "eval_name": "mcp_bench",
            "model_name": model_name,
            "status": "skipped",
            "total_questions": 0,
            "correct_answers": 0,
            "accuracy": 0.0,
            "skip_reason": f"{reason} (See {DOCS_URL})",
        }

    # A live MCP environment was configured: load the real tasks. The full agentic
    # executor + rule-based/LLM-judge scoring against a live MCP runner is delegated
    # to that environment and is out of scope for the in-process HTTP harness.
    raise NotImplementedError(
        "mcp_bench: live MCP execution + o4-mini judge scoring is not implemented "
        "in-process; drive the configured MCP-Bench runner at GBENCH_MCP_BENCH_ENDPOINT."
    )
