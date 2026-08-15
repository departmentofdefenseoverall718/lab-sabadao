# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# Evaluation Suite: t_eval
# Description: T-Eval (Shanghai AI Lab 6 Separable Sub-Skills for Tool-Augmented LLMs)

"""gbench native built-in runner for t_eval (Tool Use & Function Calling)."""

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


def _load_t_eval_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Load t_eval benchmark dataset directly from HF Hub (lovesnowbest/T-Eval)."""
    rows = []
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id="lovesnowbest/T-Eval",
            filename="data/instruct_v2.json",
            repo_type="dataset",
        )
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                rows = data
            elif isinstance(data, dict):
                rows = list(data.values())
    except Exception as e:
        logger.error(f"Failed to load dataset for t_eval: {e}")
        raise RuntimeError(f"Could not load dataset for t_eval: {e}") from e

    if not rows:
        raise RuntimeError(f"Dataset for t_eval returned empty rows")

    # Stratified, not a contiguous head (audit RC-1).
    rows = stratified_sample(rows, limit, lambda r: (r or {}).get("category"), seed="t_eval")

    samples = []
    for item in rows:
        origin_prompt = item.get("origin_prompt")
        if isinstance(origin_prompt, list):
            messages = origin_prompt
        elif isinstance(origin_prompt, str):
            messages = [{"role": "user", "content": origin_prompt}]
        else:
            messages = [{"role": "user", "content": "Execute tool action."}]

        gold = json.dumps(item.get("ground_truth", {}))
        cat = "instruct_tool"
        samples.append((messages, gold, {"category": cat}))

    logger.info(f"Loaded {len(samples)} t_eval samples.")
    return samples


def _eval_t_eval(response_text: str, gold_target: str) -> bool:
    """Evaluate candidate response against gold target."""
    if not response_text:
        return False

    resp = response_text.strip()
    gold = str(gold_target).strip()

    if not gold:
        return False

    # T-Eval ground_truth is {"action": <tool>, "args": {...}}. Scoring used to pass if the
    # tool NAME appeared anywhere in the response, so echoing a name from the prompt's own
    # API list counted as correct and the arguments were never checked.
    from .fc_common import parse_gold_call, parse_tool_calls, call_matches
    try:
        gold_obj = json.loads(gold)
    except Exception:
        gold_obj = gold
    g = parse_gold_call(gold_obj)
    if not g:
        return False
    return call_matches(g, parse_tool_calls(resp), require_args=bool(g[1]))


def run_t_eval(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Execute t_eval native built-in evaluation suite."""
    samples = _load_t_eval_samples(limit=kwargs.get("limit"))
    return run_eval_suite(
        eval_name="t_eval",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_t_eval,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens", 4096),
    )
