# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# Evaluation Suite: hmmt
# Description: HMMT (Harvard-MIT Mathematics Tournament Competition Math Benchmark)

"""gbench native built-in runner for hmmt (STEM & Scientific Reasoning)."""

import csv
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from .base import run_eval_suite
from .sampling import stratified_sample

logger = logging.getLogger(__name__)

PILLAR = "STEM & Scientific Reasoning"


def _load_hmmt_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Load hmmt benchmark dataset directly from HF Hub (matharena/hmmt)."""
    rows = []
    try:
        from datasets import load_dataset
        ds = load_dataset('MathArena/hmmt_feb_2025', split='train')
        rows = list(ds)
    except Exception as e:
        logger.error(f"Failed to load dataset for hmmt: {e}")
        raise RuntimeError(f"Could not load dataset for hmmt: {e}") from e

    if not rows:
        raise RuntimeError(f"Dataset for hmmt returned empty rows")

    # Stratified, not a contiguous head (audit RC-1).
    rows = stratified_sample(rows, limit, lambda r: (r or {}).get("category"), seed="hmmt")

    samples = []
    for item in rows:
        prompt = item.get("problem")
        gold = item.get("answer")
        if not prompt or gold is None:
            raise RuntimeError(
                "hmmt: unexpected dataset schema (missing 'problem'/'answer'); "
                "refusing to fabricate sample data"
            )
        prompt = str(prompt)
        gold = str(gold).strip()
        cat = item.get("category", "number_theory")

        messages = [{"role": "user", "content": prompt}]
        samples.append((messages, gold, {"category": cat}))

    logger.info(f"Loaded {len(samples)} hmmt samples.")
    return samples


def _eval_hmmt(response_text: str, gold_target: str) -> bool:
    """Evaluate candidate response against gold target using 100% deterministic math answer matching."""
    if not response_text:
        return False

    resp = response_text.strip()
    gold = str(gold_target).strip()

    if not gold:
        return False

    # 1. Boxed match
    match_boxed = re.findall(r"\\boxed\{([^\}]+)\}", resp)
    if match_boxed:
        last_boxed = match_boxed[-1].strip()
        if last_boxed.lower() == gold.lower() or last_boxed.replace(" ", "") == gold.replace(" ", ""):
            return True

    def _norm(s: str) -> str:
        return re.sub(r"[\s$,]|\\left|\\right|\\!|\\,", "", s).strip().rstrip(".").lower()

    # 2. "Final Answer:" / "Answer:" - compare the stated answer, not containment.
    #    HMMT golds are often small integers, so `gold in text` credited any response that
    #    merely mentioned the number somewhere in its reasoning.
    match_fa = re.search(r"(?:Final Answer|Answer)\s*:\s*(.+)", resp, re.IGNORECASE)
    if match_fa:
        fa_text = match_fa.group(1).strip().split("\n")[0]
        if _norm(fa_text) == _norm(gold):
            return True
        # allow a boxed/plain answer embedded in that line, e.g. "Answer: \boxed{5}."
        inner = re.findall(r"\\boxed\{([^\}]+)\}", fa_text)
        if inner and _norm(inner[-1]) == _norm(gold):
            return True
        # numeric equality on the stated answer only
        fa_nums = re.findall(r"-?\d+(?:\.\d+)?", fa_text)
        gold_nums_fa = re.findall(r"-?\d+(?:\.\d+)?", gold)
        if len(gold_nums_fa) == 1 and len(fa_nums) == 1 and fa_nums[0] == gold_nums_fa[0]:
            return True

    # 3. Numeric fallback: the response's FINAL number must equal a purely numeric gold.
    #    (No bare `gold in resp` containment - that was the fake-pass path.)
    gold_nums = re.findall(r"-?\d+(?:\.\d+)?", gold)
    if len(gold_nums) == 1 and _norm(gold) == _norm(gold_nums[0]):
        resp_nums = re.findall(r"-?\d+(?:\.\d+)?", resp)
        if resp_nums and resp_nums[-1] == gold_nums[0]:
            return True

    return False


def run_hmmt(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Execute hmmt native built-in evaluation suite."""
    samples = _load_hmmt_samples(limit=kwargs.get("limit"))
    return run_eval_suite(
        eval_name="hmmt",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_hmmt,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens", 8192),
    )
