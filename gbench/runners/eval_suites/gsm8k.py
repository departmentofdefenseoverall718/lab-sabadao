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

"""Native GSM8K math reasoning evaluation suite."""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from .base import run_eval_suite
from .sampling import limit_dataset
from .extraction_common import final_number

logger = logging.getLogger(__name__)


def _extract_gold_num(gold_str: str) -> str:
    """Extract gold number after '#### ' in GSM8K answer."""
    if "#### " in gold_str:
        return gold_str.split("#### ")[-1].strip().replace(",", "")
    return re.sub(r"[^\d.-]", "", gold_str)


def _load_gsm8k_samples(limit: Optional[int] = None) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Load GSM8K test set directly from canonical HF Hub dataset ('openai/gsm8k')."""
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    # Stratified, not a contiguous head (audit RC-1).
    ds = limit_dataset(ds, limit, None, seed="gsm8k")
    raw_samples = list(ds)
    logger.info(f"Loaded {len(raw_samples)} GSM8K samples from HF Hub ('openai/gsm8k').")

    samples = []
    for item in raw_samples:
        question = item["question"]
        gold_num = _extract_gold_num(item["answer"])

        prompt = (
            f"Question: {question}\n\n"
            "Let's think step by step and finish your answer with 'Final Answer: X' where X is the number."
        )
        messages = [{"role": "user", "content": prompt}]
        samples.append((messages, gold_num, {}))
    return samples


def _eval_gsm8k(response_text: str, gold_num: str) -> bool:
    """Check the model's stated final number against the gold.

    Extraction is shared (audit CC7): \\boxed{} and every "Final Answer:"/"the answer is"
    phrasing are honoured before the last-number fallback, and the LAST anchor wins. The
    previous version recognised one exact phrasing, so a boxed answer - or a second,
    corrected "Final Answer:" line - fell through to the bare last number in the response.
    """
    pred = final_number(response_text)
    if pred is None:
        return False
    try:
        return abs(float(pred) - float(gold_num)) < 1e-4
    except ValueError:
        return pred == str(gold_num).strip()


def run_gsm8k(model_name: str, base_url: str, concurrency: int, enable_thinking: bool = False, **kwargs) -> Dict[str, Any]:
    """Run native GSM8K math reasoning evaluation suite."""
    samples = _load_gsm8k_samples(limit=kwargs.get("limit"))
    return run_eval_suite(
        eval_name="gsm8k",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_gsm8k,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens"),
    )
