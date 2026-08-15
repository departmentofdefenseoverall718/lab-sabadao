# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Native LMSYS / WildBench Hard Non-Coding Reasoning evaluation suite."""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from .base import run_eval_suite
from .sampling import limit_dataset

logger = logging.getLogger(__name__)


def _load_lmsys_noncoding_hard_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], List[str], Dict[str, Any]]]:
    """Load hard non-coding prompts from canonical HF dataset ('WildEval/WildBench', config 'v2-hard')."""
    from datasets import load_dataset

    ds = load_dataset("WildEval/WildBench", "v2-hard", split="test")
    # Stratified, not a contiguous head (audit RC-1).
    ds = limit_dataset(ds, limit, 'primary_tag', seed="lmsys_noncoding_hard")
    raw_samples = list(ds)
    logger.info(f"Loaded {len(raw_samples)} Hard Non-Coding samples from HF Hub ('WildEval/WildBench').")

    samples = []
    for item in raw_samples:
        conv = item.get("conversation_input", [])
        if not conv:
            continue
        messages = [{"role": msg.get("role", "user"), "content": msg.get("content", "")} for msg in conv]
        checklist = item.get("checklist", [])
        tag = item.get("primary_tag", "general")
        samples.append((messages, checklist, {"category": tag}))

    return samples


def _eval_lmsys_noncoding_hard(response_text: str, checklist: List[str]) -> bool:
    """Evaluate response against checklist constraints and substantive length."""
    if not response_text or len(response_text.strip()) < 30:
        return False

    text = response_text.lower()
    if not checklist:
        # No checklist -> nothing to verify. The old `len(resp) >= 50` auto-passed here.
        return False

    # Check key constraint keywords from checklist
    hits = 0
    for criterion in checklist:
        words = [w for w in re.findall(r"\w+", criterion.lower()) if len(w) > 4]
        if not words:
            hits += 1
            continue
        matched_words = sum(1 for w in words if w in text)
        if matched_words / len(words) >= 0.4:
            hits += 1

    return (hits / len(checklist)) >= 0.5


def run_lmsys_noncoding_hard(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run native Hard Non-Coding evaluation suite."""
    samples = _load_lmsys_noncoding_hard_samples(limit=kwargs.get("limit"))
    return run_eval_suite(
        eval_name="lmsys_noncoding_hard",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_lmsys_noncoding_hard,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens"),
    )
