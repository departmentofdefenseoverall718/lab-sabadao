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

"""Native MMLU zero-shot / CoT evaluation suite (57 subjects, 4 choices A-D)."""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from .base import run_eval_suite
from .sampling import limit_dataset

logger = logging.getLogger(__name__)

OPTION_LETTERS = ["A", "B", "C", "D"]


def _load_mmlu_samples(
    enable_thinking: bool = False,
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Load canonical MMLU samples directly from HF Hub ('cais/mmlu', 'all', split='test')."""
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test")
    # Stratified, not a contiguous head: the rows are stored grouped by
    # `subject`, so `[:limit]` returned a single category (audit RC-1).
    ds = limit_dataset(ds, limit, "subject", seed="mmlu")
    raw_samples = list(ds)
    logger.info(f"Loaded {len(raw_samples)} MMLU samples from HF Hub ('cais/mmlu').")

    samples = []
    for item in raw_samples:
        q_text = str(item.get("question", "")).strip()
        choices = item.get("choices", [])
        subject = str(item.get("subject", "general")).strip()
        ans = item.get("answer", 0)

        if isinstance(ans, str) and ans.strip().upper() in OPTION_LETTERS:
            gold_letter = ans.strip().upper()
        else:
            gold_idx = int(ans)
            gold_letter = OPTION_LETTERS[gold_idx] if 0 <= gold_idx < len(OPTION_LETTERS) else "A"

        options_str = "\n".join(
            f"({let}) {choice}" for let, choice in zip(OPTION_LETTERS[:len(choices)], choices)
        )
        if enable_thinking:
            prompt = (
                f"Subject: {subject.replace('_', ' ').title()}\n"
                f"Question: {q_text}\n\n{options_str}\n\n"
                "Let's think step by step and then output the correct option letter in the format: 'Final Answer: (X)'.\n"
                "Answer:"
            )
        else:
            prompt = (
                f"Subject: {subject.replace('_', ' ').title()}\n"
                f"Question: {q_text}\n\n{options_str}\n\n"
                "Answer with only the correct option letter in the format: 'Final Answer: (X)'.\n"
                "Answer:"
            )
        messages = [{"role": "user", "content": prompt}]
        samples.append((messages, gold_letter, {"category": subject}))
    return samples


def _eval_mmlu(response_text: str, gold_letter: str) -> bool:
    """Check if predicted answer matches gold MMLU letter (A-D)."""
    text = response_text.strip().upper()
    # Anchored forms first.
    for pat in (r"FINAL\s+ANSWER:\s*\(?([ABCD])\)?", r"ANSWER:\s*\(?([ABCD])\)?",
                r"\\BOXED\{\(?([ABCD])\)?\}"):
        m = re.findall(pat, text)
        if m:
            return m[-1] == gold_letter
    # Whole response is just the letter.
    if text.strip().strip("().") in ("A", "B", "C", "D"):
        return text.strip().strip("().") == gold_letter
    # CC7: fall back to the LAST standalone option letter, not the FIRST. `^\(?([ABCD])`
    # matched the leading "A" of a sentence like "A careful analysis shows C", and
    # `\b([ABCD])\b` took the first such letter anywhere - both credited wrong answers.
    standalone = re.findall(r"\b([ABCD])\b", text)
    if standalone:
        return standalone[-1] == gold_letter
    return False


def run_mmlu(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run native MMLU evaluation suite."""
    samples = _load_mmlu_samples(enable_thinking=enable_thinking, limit=kwargs.get("limit"))
    return run_eval_suite(
        eval_name="mmlu",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_mmlu,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens"),
    )
