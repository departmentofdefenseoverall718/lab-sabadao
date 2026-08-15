# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# Evaluation Suite: toolbench
# Description: ToolBench (OpenBMB 16,000+ Real-world RapidAPI Multi-Turn Tool Benchmark)

"""gbench native built-in runner for toolbench (Tool Use & Function Calling)."""

import csv
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from .base import run_eval_suite
from .sampling import stratified_sample

logger = logging.getLogger(__name__)

PILLAR = "Tool Use & Function Calling"


def _load_toolbench_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Load toolbench benchmark dataset directly from HF Hub (Yhyu13/ToolBench_toolllama_G123_dfs)."""
    rows = []
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id="Yhyu13/ToolBench_toolllama_G123_dfs",
            filename="toolllama_G123_dfs_eval.json",
            repo_type="dataset",
        )
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                rows = data
            elif isinstance(data, dict):
                rows = list(data.values())
    except Exception as e:
        logger.error(f"Failed to load dataset for toolbench: {e}")
        raise RuntimeError(f"Could not load dataset for toolbench: {e}") from e

    if not rows:
        raise RuntimeError(f"Dataset for toolbench returned empty rows")

    # Stratified, not a contiguous head (audit RC-1).
    rows = stratified_sample(rows, limit, lambda r: (r or {}).get("category"), seed="toolbench")

    samples = []
    for item in rows:
        convs = item.get("conversations") or []
        sys_msg = ""
        user_msg = ""
        gold = ""
        for c in convs:
            role = c.get("from", "").lower()
            val = c.get("value", "")
            if role == "system" and not sys_msg:
                sys_msg = val
            elif role in ("human", "user") and not user_msg:
                user_msg = val
            elif role in ("gpt", "assistant") and not gold:
                gold = val

        messages = []
        if sys_msg:
            messages.append({"role": "system", "content": sys_msg})
        messages.append({"role": "user", "content": user_msg or "Execute tool action."})

        cat = str(item.get("id", "tool")).split(":")[0]
        samples.append((messages, gold, {"category": cat}))

    logger.info(f"Loaded {len(samples)} toolbench samples.")
    return samples


def _eval_toolbench(response_text: str, gold_target: str) -> bool:
    """Evaluate candidate response against gold target."""
    if not response_text:
        return False

    resp = response_text.strip()
    gold = str(gold_target).strip()

    if not gold:
        return False

    # The gold is a ToolLLaMA assistant turn: "Thought: ... Action: <fn> Action Input: {...}".
    # The old name regex matched none of that for 706/762 rows, so scoring collapsed to a
    # whole-string containment test; the ~7% that did parse matched stray capitalized words.
    from .fc_common import parse_gold_call, parse_tool_calls, call_matches
    g = parse_gold_call(gold)
    if not g:
        return False
    return call_matches(g, parse_tool_calls(resp), require_args=bool(g[1]))


def run_toolbench(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Execute toolbench native built-in evaluation suite."""
    samples = _load_toolbench_samples(limit=kwargs.get("limit"))
    return run_eval_suite(
        eval_name="toolbench",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_toolbench,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens", 4096),
    )
