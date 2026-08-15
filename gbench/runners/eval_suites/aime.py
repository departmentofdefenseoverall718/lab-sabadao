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

"""Native AIME Olympiad Competition Math evaluation suite.

Dataset: `AI-MO/aimo-validation-aime`, which is AIME **2022-2024**, not 2025. The suite
was labelled "AIME 2025"; the label is corrected here rather than the data, because no
2025 source is wired up. Treat the score as potentially contaminated: these problems
pre-date the training cutoff of most current models.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from .base import run_eval_suite
from .sampling import limit_dataset
from .extraction_common import final_number

logger = logging.getLogger(__name__)

def _load_aime_samples(enable_thinking: bool = False, limit: Optional[int] = None) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Load AIME samples directly from canonical HF Hub dataset ('AI-MO/aimo-validation-aime')."""
    from datasets import load_dataset
    ds = load_dataset("AI-MO/aimo-validation-aime", split="train")
    # Stratified, not a contiguous head (audit RC-1).
    ds = limit_dataset(ds, limit, 'category', seed="aime")
    raw_samples = list(ds)
    logger.info(f"Loaded {len(raw_samples)} AIME samples from HF Hub ('AI-MO/aimo-validation-aime').")

    samples = []
    for item in raw_samples:
        q_text = item.get("problem") or item.get("question", "")
        gold_ans = str(item["answer"]).strip()

        if enable_thinking:
            prompt = (
                f"Problem: {q_text}\n\n"
                "Solve the math problem step by step. Write your final answer as an integer between 000 and 999 "
                "at the end in the format: 'The final answer is X'."
            )
        else:
            prompt = (
                f"Problem: {q_text}\n\n"
                "Solve the math problem. Write your final answer as an integer between 000 and 999 "
                "at the end in the format: 'The final answer is X'."
            )
        messages = [{"role": "user", "content": prompt}]
        samples.append((messages, gold_ans, {"category": str(item.get("category", "math"))}))
    return samples


def _eval_aime(response_text: str, gold_answer: str) -> bool:
    """Check the model's stated integer against the gold AIME answer (0-999).

    Shared extraction (audit CC7): \\boxed{} first, then any answer anchor, then the last
    number. The previous version matched one literal phrasing and otherwise took the last
    1-4 digit token, so `\\boxed{042}` was missed and a trailing year or step number in the
    working could be read as the answer.
    """
    pred = final_number(response_text)
    if pred is None:
        return False
    try:
        return int(float(pred)) == int(float(gold_answer))
    except ValueError:
        return False


def run_aime(model_name: str, base_url: str, concurrency: int, enable_thinking: bool = False, **kwargs) -> Dict[str, Any]:
    """Run native AIME competition math evaluation suite (AIME 2022-2024 set)."""
    samples = _load_aime_samples(enable_thinking, limit=kwargs.get("limit"))
    return run_eval_suite(
        eval_name="aime",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_aime,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens"),
    )
