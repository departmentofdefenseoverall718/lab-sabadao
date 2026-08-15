# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# Evaluation Suite: docvqa
# Description: DocVQA (Document Visual Question Answering Benchmark)

"""Native DocVQA (Document Visual Question Answering) evaluation suite."""

import base64
import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from .base import run_eval_suite
from .sampling import limit_dataset
from .dataset_utils import extract_lossless_image_b64

logger = logging.getLogger(__name__)


def _load_docvqa_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], Any, Dict[str, Any]]]:
    """Load DocVQA document samples with direct lossless raw image byte decoding."""
    from datasets import load_dataset, Image

    try:
        ds = load_dataset("lmms-lab-encoder/DocVQA", "DocVQA", split="validation")
    except Exception:
        ds = load_dataset("hf-internal-testing/fixtures_docvqa", split="test")

    try:
        ds = ds.cast_column("image", Image(decode=False))
    except Exception:
        pass

    # Stratified, not a contiguous head (audit RC-1).
    ds = limit_dataset(ds, limit, None, seed="docvqa")
    samples = []
    for item in ds:
        q_text = item.get("question", "")
        answers = item.get("answers", [""])
        doc_type = item.get("data_type", "document")
        img_obj = item.get("image")

        b64_str = extract_lossless_image_b64(img_obj)
        if not b64_str:
            continue

        prompt = (
            f"Question: {q_text}\n"
            "Extract the exact answer directly from the document image. Output only the short answer text."
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_str}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        samples.append((messages, answers, {"category": doc_type}))

    logger.info(f"Loaded {len(samples)} DocVQA samples with direct lossless PNG bytes.")
    return samples


def _eval_docvqa(response_text: str, gold_answers: Any):
    """Score with the canonical metric: ANLS >= 0.5 (the DocVQA metric).

    Replaces a bidirectional substring test (gold in pred or pred in gold), which
    credited any verbose answer that merely mentioned the gold.
    """
    from .vqa_common import eval_anls, extract_short_answer
    if not response_text or gold_answers is None:
        return False
    return eval_anls(extract_short_answer(response_text), gold_answers)

def run_docvqa(
    model_name: str,
    base_url: str,
    concurrency: int = 4,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run native DocVQA document understanding evaluation suite."""
    samples = _load_docvqa_samples(limit=kwargs.get("limit"))
    return run_eval_suite(
        eval_name="docvqa",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_docvqa,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens", 2048),
    )
