# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# Evaluation Suite: charxiv
# Description: CharXiv (Princeton Complex Academic Chart Reasoning & Numerical Understanding)

"""gbench native built-in runner for charxiv (Multimodal & Vision)."""

import base64
import csv
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from .base import run_eval_suite
from .sampling import stratified_sample
from .dataset_utils import extract_lossless_image_b64

logger = logging.getLogger(__name__)

PILLAR = "Multimodal & Vision"


def _load_charxiv_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Load CharXiv benchmark dataset directly from HF Hub (princeton-nlp/CharXiv)."""
    rows = []
    try:
        from datasets import load_dataset, Image
        ds = load_dataset("princeton-nlp/CharXiv", split="validation")
        try:
            ds = ds.cast_column("image", Image(decode=False))
        except Exception:
            pass
        rows = list(ds)
    except Exception as e:
        logger.error(f"Failed to load dataset for charxiv: {e}")
        raise RuntimeError(f"Could not load dataset for charxiv: {e}") from e

    if not rows:
        raise RuntimeError("Dataset for charxiv returned empty rows")

    # Stratified, not a contiguous head (audit RC-1).
    rows = stratified_sample(rows, limit, None, seed="charxiv")

    samples = []
    for item in rows:
        question = str(item.get("reasoning_q") or item.get("question") or "").strip()
        gold = str(item.get("reasoning_a") or item.get("answer") or "").strip()
        cat = str(item.get("category") or item.get("subject") or "academic_chart")
        img_val = item.get("image")

        content_payload: List[Dict[str, Any]] = [{"type": "text", "text": f"Question: {question}\n\nConclude with: Final Answer: <answer>"}]
        b64_str = extract_lossless_image_b64(img_val)
        if b64_str:
            content_payload.insert(0, {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64_str}"}
            })

        messages = [{"role": "user", "content": content_payload}]
        samples.append((messages, gold, {"category": cat}))

    logger.info(f"Loaded {len(samples)} charxiv samples with direct lossless image bytes.")
    return samples


def _eval_charxiv(response_text: str, gold_target: str):
    """Score with the canonical metric: relaxed accuracy: numeric within 5%, else exact match.

    Replaces a bidirectional substring test (gold in pred or pred in gold), which
    credited any verbose answer that merely mentioned the gold.
    """
    from .vqa_common import eval_relaxed, extract_short_answer
    if not response_text or gold_target is None:
        return False
    return eval_relaxed(extract_short_answer(response_text), gold_target)

def run_charxiv(
    model_name: str,
    base_url: str,
    limit: Optional[int] = None,
    concurrency: int = 4,
    enable_thinking: bool = False,
    results_dir: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Run native CharXiv evaluation benchmark."""
    samples = _load_charxiv_samples(limit=limit)
    return run_eval_suite(
        eval_name="charxiv",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_charxiv,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens", 2048),
    )
