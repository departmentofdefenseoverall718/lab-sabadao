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

"""Native AMC and AIME high-school math competition evaluation suite."""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from .base import run_eval_suite
from .extraction_common import final_number

logger = logging.getLogger(__name__)


def _load_new_amc_aime_samples(
    enable_thinking: bool = False,
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Load AMC and AIME competition math problems from canonical HF Hub datasets."""
    from datasets import load_dataset

    samples = []

    # 1. Load AMC problems
    try:
        ds_amc = load_dataset("AI-MO/aimo-validation-amc", split="train")
        amc_limit = (limit // 2) if limit is not None else None
        if amc_limit is not None:
            ds_amc = ds_amc.select(range(min(amc_limit, len(ds_amc))))
        for item in list(ds_amc):
            q_text = item.get("problem", "")
            raw_ans = str(item.get("answer", "")).strip()
            try:
                gold_ans = str(int(float(raw_ans)))
            except ValueError:
                gold_ans = raw_ans

            if enable_thinking:
                prompt = (
                    f"Problem: {q_text}\n\n"
                    "Solve the math competition problem step by step. Write your final answer as an integer "
                    "at the end in the format: 'The final answer is X'."
                )
            else:
                prompt = (
                    f"Problem: {q_text}\n\n"
                    "Solve the math competition problem. Write your final answer as an integer "
                    "at the end in the format: 'The final answer is X'."
                )
            messages = [{"role": "user", "content": prompt}]
            samples.append((messages, gold_ans, {"category": "AMC"}))
    except Exception as e:
        logger.warning(f"Could not load AMC dataset ({e}).")

    # 2. Load AIME problems
    try:
        ds_aime = load_dataset("AI-MO/aimo-validation-aime", split="train")
        aime_limit = (limit - len(samples)) if limit is not None else None
        if aime_limit is not None and aime_limit > 0:
            ds_aime = ds_aime.select(range(min(aime_limit, len(ds_aime))))
        elif aime_limit is not None and aime_limit <= 0:
            ds_aime = []
        for item in list(ds_aime):
            q_text = item.get("problem", "")
            raw_ans = str(item.get("answer", "")).strip()
            try:
                gold_ans = str(int(float(raw_ans)))
            except ValueError:
                gold_ans = raw_ans

            if enable_thinking:
                prompt = (
                    f"Problem: {q_text}\n\n"
                    "Solve the math competition problem step by step. Write your final answer as an integer "
                    "at the end in the format: 'The final answer is X'."
                )
            else:
                prompt = (
                    f"Problem: {q_text}\n\n"
                    "Solve the math competition problem. Write your final answer as an integer "
                    "at the end in the format: 'The final answer is X'."
                )
            messages = [{"role": "user", "content": prompt}]
            samples.append((messages, gold_ans, {"category": "AIME"}))
    except Exception as e:
        logger.warning(f"Could not load AIME dataset ({e}).")

    # CC6: both loads are wrapped in warn-and-continue try/excepts. If AMC fails but AIME
    # succeeds the suite silently reports a partial benchmark under the full name; if both
    # fail it would report success on 0 samples. Neither may pass silently.
    cats = {m.get("category") for _, _, m in samples}
    if not samples:
        raise RuntimeError(
            "new_amc_aime: neither the AMC nor the AIME dataset could be loaded; refusing "
            "to report a result over zero samples (see warnings above).")
    if len(cats) < 2:
        logger.warning("new_amc_aime: only %s loaded - the reported score covers that "
                       "subset ONLY, not the full AMC+AIME set.", cats)

    logger.info(f"Loaded {len(samples)} AMC/AIME math competition samples from HF Hub.")
    return samples


def _eval_new_amc_aime(response_text: str, gold_answer: str) -> bool:
    """Check the model's stated number against the gold AMC/AIME answer.

    Shared extraction (audit CC7): \\boxed{} and every answer anchor are honoured before
    the last-number fallback, so a boxed answer is no longer skipped in favour of whatever
    number happened to appear last in the working.
    """
    if not response_text:
        return False
    pred = final_number(response_text)
    if pred is None:
        return False
    try:
        return int(float(pred)) == int(float(gold_answer))
    except ValueError:
        return pred.strip().lower() == str(gold_answer).strip().lower()


def run_new_amc_aime(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run native AMC & AIME competition math evaluation suite."""
    samples = _load_new_amc_aime_samples(enable_thinking, limit=kwargs.get("limit"))
    return run_eval_suite(
        eval_name="new_amc_aime",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_new_amc_aime,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens"),
    )
