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

"""Native RULER high-entropy long-context multi-needle evaluation suite."""

import logging
import random
import uuid
from typing import Any, Dict, List, Optional, Tuple
from .base import run_eval_suite
from .sampling import limit_dataset

logger = logging.getLogger(__name__)

def _length_band(length: Any) -> str:
    """Bucket a context length into a RULER band.

    `length` is the sample's own token count, so using it directly as the category
    produced a near-unique bucket per sample - the per-category table had one row each and
    said nothing about how accuracy degrades with context.
    """
    try:
        n = int(length)
    except (TypeError, ValueError):
        return "unknown"
    for bound, name in ((4096, "4k"), (8192, "8k"), (16384, "16k"), (32768, "32k"),
                        (65536, "64k"), (131072, "128k")):
        if n <= bound:
            return name
    return "128k+"


def ruler_recall(response_text: str, gold_answers: Any) -> float:
    """Fraction of the required outputs the response contains (canonical RULER scoring).

    RULER's multi-output tasks (multi-key/multi-value/multi-query NIAH, variable tracking)
    score partial recall. All-or-nothing collapsed "found 7 of 8 needles" to the same
    score as "found none", which understates long-context performance and hides where it
    starts to degrade.
    """
    if not response_text:
        return 0.0
    text = str(response_text).strip().lower()
    if isinstance(gold_answers, (list, tuple, set)):
        needles = [str(x).strip().lower() for x in gold_answers if str(x).strip()]
    else:
        needles = [str(gold_answers).strip().lower()] if str(gold_answers).strip() else []
    if not needles:
        return 0.0
    return sum(1 for n in needles if n in text) / len(needles)


def _load_ruler_samples(limit: Optional[int] = None) -> List[Tuple[List[Dict[str, Any]], Any, Dict[str, Any]]]:
    """Load canonical RULER benchmark dataset directly from HF Hub ('rayonlabs/ruler-all')."""
    from datasets import load_dataset
    ds = load_dataset("rayonlabs/ruler-all", split="train")
    if limit is not None:
        if hasattr(ds, "select"):
            ds = limit_dataset(ds, limit, "task", seed="ruler")
        else:
            ds = ds[:limit]
    raw_samples = list(ds)
    logger.info(f"Loaded {len(raw_samples)} RULER samples from HF Hub ('rayonlabs/ruler-all').")

    samples = []
    for item in raw_samples:
        prompt = item["input"]
        gold_outputs = item["outputs"]
        length = item.get("length", 0)
        task = item.get("task", "ruler")
        messages = [{"role": "user", "content": prompt}]
        samples.append((messages, gold_outputs,
                        {"category": f"{task}@{_length_band(length)}",
                         "length": length, "task": task}))
    return samples


def _eval_ruler(response_text: str, gold_answers: Any) -> bool:
    """Harness pass/fail: every required output present. Reported metric is mean recall."""
    return ruler_recall(response_text, gold_answers) >= 1.0


def run_ruler(model_name: str, base_url: str, concurrency: int, enable_thinking: bool = False, **kwargs) -> Dict[str, Any]:
    """Run native RULER long-context multi-needle evaluation suite.

    Headline is the canonical **mean per-item recall**; the all-needles-found rate is kept
    alongside. Categories are `<task>@<length band>`, so the per-category table shows how
    each task degrades with context instead of one row per sample.

    Scope: `rayonlabs/ruler-all` tops out well below the 128k RULER advertises - check
    `length_bands` on the result before comparing with a published RULER number.
    """
    samples = _load_ruler_samples(limit=kwargs.get("limit"))
    result = run_eval_suite(
        eval_name="ruler",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_ruler,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens"),
    )
    recalls = []
    for trace in result.get("sample_traces", []):
        if trace.get("response_text") is None:
            continue
        r = ruler_recall(trace["response_text"], trace.get("gold_answer"))
        trace["recall"] = round(r, 4)
        recalls.append(r)
    result["all_needles_rate"] = result.get("accuracy")
    result["metric"] = "mean per-item needle recall (canonical RULER partial credit)"
    bands = sorted({_length_band(m.get("length")) for _, _, m in samples})
    result["length_bands"] = bands
    if recalls:
        result["accuracy"] = round(sum(recalls) / len(recalls) * 100.0, 2)
    return result
