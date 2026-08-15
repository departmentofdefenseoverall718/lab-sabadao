# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# Evaluation Suite: seal_tools
# Description: Seal-Tools (NLPCC Multi-Tool Single-Turn and Nested Function Calling Benchmark)

"""gbench native built-in runner for seal_tools (Tool Use & Function Calling)."""

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


def _load_seal_tools_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Load seal_tools benchmark dataset directly from HF Hub (seal-tools/seal-tools)."""
    rows = []
    try:
        from datasets import load_dataset
        # The TEST split is the evaluation set (1,354 rows). The train split (12,022) is
        # training data, and its `domain` column is the literal string "train" for every
        # row - so scoring it also produced a single meaningless category. On test,
        # `domain` is in-domain / out-domain (700 / 654), the dimension the benchmark
        # actually reports.
        ds = load_dataset('casey-martin/Seal-Tools', split='test')
        rows = list(ds)
    except Exception as e:
        logger.error(f"Failed to load dataset for seal_tools: {e}")
        raise RuntimeError(f"Could not load dataset for seal_tools: {e}") from e

    if not rows:
        raise RuntimeError(f"Dataset for seal_tools returned empty rows")

    # Stratified, not a contiguous head (audit RC-1).
    rows = stratified_sample(rows, limit, lambda r: (r or {}).get("domain"), seed="seal_tools")

    samples = []
    for item in rows:
        convs = item.get("conversations") or []
        prompt = ""
        gold = ""
        for c in convs:
            role = str(c.get("from", "")).lower()
            val = str(c.get("value", "")).strip()
            if role in ("human", "user") and not prompt:
                prompt = val
            elif role in ("gpt", "assistant") and not gold:
                gold = val

        cat = str(item.get("domain") or "tool_use")
        messages = [{"role": "user", "content": prompt or "Generate tool calls."}]
        samples.append((messages, gold, {"category": cat}))

    logger.info(f"Loaded {len(samples)} seal_tools samples.")
    return samples


def _eval_seal_tools(response_text: str, gold_target: str) -> bool:
    """Evaluate candidate response against gold target."""
    if not response_text:
        return False

    resp = response_text.strip()
    gold = str(gold_target).strip()

    if not gold:
        return False

    # Structural comparison: the gold API name AND its arguments must match a call the
    # model actually emitted. Name-substring matching credited any mention of the tool.
    from .fc_common import parse_gold_call, parse_tool_calls, call_matches
    g = parse_gold_call(gold)
    if not g:
        return False
    return call_matches(g, parse_tool_calls(resp), require_args=bool(g[1]))


def run_seal_tools(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Execute seal_tools native built-in evaluation suite."""
    samples = _load_seal_tools_samples(limit=kwargs.get("limit"))
    return run_eval_suite(
        eval_name="seal_tools",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_seal_tools,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens", 4096),
    )
