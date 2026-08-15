# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# Evaluation Suite: api_bank
# Description: API-Bank (Alibaba Multi-Level Tool Use: Call / Retrieve+Call / Plan)

"""gbench native built-in runner for api_bank (Tool Use & Function Calling)."""

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


def _load_api_bank_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Load api_bank benchmark dataset directly from HF Hub (liminghao1630/API-Bank)."""
    rows = []
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id="liminghao1630/API-Bank",
            filename="test-data/level-1-api.json",
            repo_type="dataset",
        )
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                rows = data
            elif isinstance(data, dict):
                rows = list(data.values())
    except Exception as e:
        logger.error(f"Failed to load dataset for api_bank: {e}")
        raise RuntimeError(f"Could not load dataset for api_bank: {e}") from e

    if not rows:
        raise RuntimeError(f"Dataset for api_bank returned empty rows")

    # Stratified, not a contiguous head (audit RC-1).
    rows = stratified_sample(rows, limit, lambda r: (r or {}).get("level"), seed="api_bank")

    samples = []
    for item in rows:
        instruction = item.get("instruction", "").strip()
        inp = item.get("input", "").strip()
        prompt = f"{instruction}\n\nInput:\n{inp}".strip() if inp else (instruction or "Generate API request.")
        gold = str(item.get("expected_output") or "").strip()
        cat = str(item.get("file") or "level-1")

        messages = [{"role": "user", "content": prompt}]
        samples.append((messages, gold, {"category": cat}))

    logger.info(f"Loaded {len(samples)} api_bank samples.")
    return samples


def _eval_api_bank(response_text: str, gold_target: str) -> bool:
    """Evaluate candidate response against gold target using 100% deterministic parameter matching."""
    if not response_text:
        return False

    resp = response_text.strip()
    gold = str(gold_target).strip()

    if not gold:
        return False

    # Structural (name + arguments) comparison. The previous `gold.lower() in resp.lower()`
    # could never succeed: the gold carries API-Bank's "API-Request: " prefix while the
    # prompt asks the model for a bare "[ApiName(...)]", so CORRECT answers scored wrong.
    from .fc_common import score_tool_call
    return score_tool_call(resp, gold, require_args=True)


def run_api_bank(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Execute api_bank native built-in evaluation suite."""
    samples = _load_api_bank_samples(limit=kwargs.get("limit"))
    return run_eval_suite(
        eval_name="api_bank",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_api_bank,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens", 4096),
    )
