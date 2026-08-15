# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# Evaluation Suite: textvqa
# Description: TextVQA (Visual Question Answering on Text in Images Benchmark)

"""Native TextVQA (Visual Question Answering on Text in Images) evaluation suite."""

import base64
import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from .base import run_eval_suite
from .sampling import limit_dataset
from .dataset_utils import extract_lossless_image_b64

logger = logging.getLogger(__name__)


def _load_textvqa_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], Any, Dict[str, Any]]]:
    """Load TextVQA samples directly from canonical HF Hub dataset ('lmms-lab/textvqa')."""
    from datasets import load_dataset, Image

    ds = load_dataset("lmms-lab/textvqa", split="validation")
    try:
        ds = ds.cast_column("image", Image(decode=False))
    except Exception:
        pass

    # Stratified, not a contiguous head (audit RC-1).
    ds = limit_dataset(ds, limit, None, seed="textvqa")
    samples = []
    for item in ds:
        q_text = item.get("question", "")
        answers = item.get("answers", [""])
        img_obj = item.get("image")

        b64_str = extract_lossless_image_b64(img_obj)
        if not b64_str:
            continue

        prompt = f"Question: {q_text}\nAnswer with only the short text string found in the image."
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_str}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        samples.append((messages, answers, {"category": "scene_ocr"}))

    logger.info(f"Loaded {len(samples)} TextVQA samples with direct lossless image bytes.")
    return samples


def _eval_textvqa(response_text: str, gold_answers: Any):
    """Score with the canonical metric: VQA accuracy min(#annotators/3, 1) >= 0.5.

    Replaces a bidirectional substring test (gold in pred or pred in gold), which
    credited any verbose answer that merely mentioned the gold.
    """
    from .vqa_common import eval_vqa, extract_short_answer
    if not response_text or gold_answers is None:
        return False
    return eval_vqa(extract_short_answer(response_text), gold_answers)

def run_textvqa(
    model_name: str,
    base_url: str,
    concurrency: int = 4,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run native TextVQA evaluation suite."""
    samples = _load_textvqa_samples(limit=kwargs.get("limit"))
    return run_eval_suite(
        eval_name="textvqa",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_textvqa,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens", 2048),
    )
