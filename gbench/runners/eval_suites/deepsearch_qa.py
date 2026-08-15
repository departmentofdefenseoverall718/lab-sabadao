# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# Evaluation Suite: deepsearch_qa
# Description: DeepSearchQA (xbench/DeepSearch-2510 - Autonomous Web Search & Information Retrieval Benchmark)

"""gbench native built-in runner for deepsearch_qa (Agentic & Web Research)."""

import base64
import csv
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from .search_tool import (WEB_SEARCH_TOOL, execute_tool, search_available,
                          search_backend_name, unavailable_reason)
from .swebench_common import skipped_result
from .base import run_eval_suite
from .sampling import stratified_sample

logger = logging.getLogger(__name__)

PILLAR = "Agentic & Web Research"


def _xor_decrypt(data: bytes, key: str) -> str:
    """XOR decrypt benchmark data with canary key."""
    key_bytes = key.encode("utf-8")
    k_len = len(key_bytes)
    return bytes([data[i] ^ key_bytes[i % k_len] for i in range(len(data))]).decode("utf-8")


def _load_deepsearch_qa_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Load DeepSearchQA benchmark dataset directly from HF Hub (xbench/DeepSearch-2510)."""
    rows = []
    try:
        from datasets import load_dataset
        ds = load_dataset("xbench/DeepSearch-2510", split="train")
        rows = list(ds)
    except Exception as e:
        logger.error(f"Failed to load dataset for deepsearch_qa: {e}")
        raise RuntimeError(f"Could not load dataset for deepsearch_qa: {e}") from e

    if not rows:
        raise RuntimeError("Dataset for deepsearch_qa returned empty rows")

    # Stratified, not a contiguous head (audit RC-1).
    rows = stratified_sample(rows, limit, None, seed="deepsearch_qa")

    samples = []
    for item in rows:
        canary = str(item.get("canary") or "")
        raw_prompt = item.get("prompt", "")
        raw_answer = item.get("answer", "")
        task_id = str(item.get("id", "ds_task"))

        if canary and raw_prompt:
            try:
                question = _xor_decrypt(base64.b64decode(raw_prompt), canary)
                gold = _xor_decrypt(base64.b64decode(raw_answer), canary)
            except Exception:
                question = str(raw_prompt)
                gold = str(raw_answer)
        else:
            question = str(raw_prompt)
            gold = str(raw_answer)

        prompt = (
            f"[DeepSearch Autonomous Research Question #{task_id}]\n"
            f"{question}\n\n"
            "Search, synthesize factual findings, and provide the concise factual answer.\n"
            "Conclude with: Final Answer: <answer>"
        )
        messages = [{"role": "user", "content": prompt}]
        samples.append((messages, gold.strip(), {"category": "deepsearch_web"}))

    logger.info(f"Loaded {len(samples)} deepsearch_qa samples.")
    return samples


def _normalize_answer(text: str) -> str:
    """Lowercase, strip punctuation/articles/whitespace - SQuAD-style answer normalization."""
    t = (text or "").strip().lower()
    t = re.sub(r"\b(a|an|the)\b", " ", t)
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _eval_deepsearch_qa(response_text: str, gold_target: str) -> bool:
    """Compare the model's stated final answer with the gold factual answer.

    The previous version scanned the WHOLE response for the gold string, so a research
    answer that appeared anywhere in the working - including inside a list of candidates
    the model then rejected - was scored correct. Golds here are short factual strings
    (names, dates, counts), which makes that especially easy to hit by accident.

    Now the stated answer is extracted first ("Final Answer:", else the last non-empty
    line) and compared after normalization; the gold is only searched for inside that span,
    with word boundaries, so `7` is not found inside `1978`.
    """
    if not response_text:
        return False

    gold = str(gold_target).strip()
    if not gold:
        return False

    resp = response_text.strip()
    anchored = re.findall(r"Final Answer\s*:?\s*(.+)", resp, re.IGNORECASE)
    if anchored:
        span = anchored[-1].strip()
    else:
        lines = [line.strip() for line in resp.splitlines() if line.strip()]
        span = lines[-1] if lines else ""

    norm_span, norm_gold = _normalize_answer(span), _normalize_answer(gold)
    if not norm_gold:
        return False
    if norm_span == norm_gold:
        return True
    return re.search(rf"(?<!\w){re.escape(norm_gold)}(?!\w)", norm_span) is not None


def run_deepsearch_qa(
    model_name: str,
    base_url: str,
    limit: Optional[int] = None,
    concurrency: int = 4,
    enable_thinking: bool = False,
    results_dir: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Run native DeepSearchQA evaluation benchmark."""
    samples = _load_deepsearch_qa_samples(limit=limit)
    # deepsearch_qa is a web-research benchmark: without a lookup tool the model answers "I do
    # not have access to the internet" and the suite scores a structural 0 (measured 0/20
    # on 2026-08-15). Offer the search tool and let the model drive it.
    if not search_available():
        return skipped_result("deepsearch_qa", model_name, unavailable_reason(), "docs/evals/deepsearch_qa.md")
    extra_payload = dict(kwargs.get("extra_payload") or {})
    extra_payload.setdefault("tools", [WEB_SEARCH_TOOL])

    result = run_eval_suite(
        eval_name="deepsearch_qa",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_deepsearch_qa,
        thinking=enable_thinking,
        extra_payload=extra_payload,
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens"),
        tool_executor=execute_tool,
    )
    # Search-only, not the browsing agent the public leaderboard uses: say so on the result
    # so the number is never read as leaderboard-comparable.
    result["search_backend"] = search_backend_name()
    result["leaderboard_comparable"] = False
    return result
