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

"""Native Berkeley Function Calling Leaderboard (BFCL) evaluation suite."""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple
from .base import run_eval_suite

logger = logging.getLogger(__name__)

SUPPORTED_BFCL_CATEGORIES = [
    "simple",
    "multiple",
    "parallel",
    "parallel_multiple",
    "java",
    "javascript",
    "sql",
    "live_simple",
    "live_multiple",
    "live_parallel",
    "live_parallel_multiple",
]


def _load_bfcl_samples(categories: Optional[str]) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Load BFCL train/test samples directly from canonical Gorilla HF Hub repository."""
    import json
    from huggingface_hub import hf_hub_download

    cats_to_load = (
        SUPPORTED_BFCL_CATEGORIES
        if not categories or categories == "all"
        else [c.strip() for c in categories.split(",") if c.strip()]
    )

    raw_samples = []
    for cat in cats_to_load:
        try:
            fpath_q = hf_hub_download("gorilla-llm/Berkeley-Function-Calling-Leaderboard", f"BFCL_v3_{cat}.json", repo_type="dataset")
            fpath_a = hf_hub_download("gorilla-llm/Berkeley-Function-Calling-Leaderboard", f"possible_answer/BFCL_v3_{cat}.json", repo_type="dataset")

            ans_map = {}
            with open(fpath_a, "r", encoding="utf-8") as fa:
                for line in fa:
                    if line.strip():
                        obj = json.loads(line)
                        ans_map[obj.get("id")] = obj.get("ground_truth")

            with open(fpath_q, "r", encoding="utf-8") as fq:
                for line in fq:
                    if line.strip():
                        item = json.loads(line)
                        item["ground_truth"] = ans_map.get(item.get("id"))
                        item["test_category"] = cat
                        raw_samples.append(item)
        except Exception as cat_err:
            logger.warning(f"Could not load BFCL category '{cat}': {cat_err}")

    logger.info(f"Loaded {len(raw_samples)} BFCL samples across {len(cats_to_load)} categories from HF Hub.")

    allowed_cats = set(categories.split(",")) if categories and categories != "all" else None

    samples = []
    for item in raw_samples:
        cat = item.get("test_category", "simple_python")
        if allowed_cats and cat not in allowed_cats:
            continue

        q_obj = item["question"]
        if isinstance(q_obj, list) and len(q_obj) > 0:
            if isinstance(q_obj[0], list) and len(q_obj[0]) > 0 and isinstance(q_obj[0][0], dict):
                user_content = q_obj[0][0].get("content", "")
            elif isinstance(q_obj[0], dict):
                user_content = q_obj[0].get("content", "")
            else:
                user_content = str(q_obj[0])
        else:
            user_content = str(q_obj)

        tools = []
        for fn in item.get("function", []):
            tools.append({"type": "function", "function": fn})

        messages = [{"role": "user", "content": user_content}]
        gold_call = item.get("ground_truth")

        samples.append((messages, gold_call, {"tools": tools} if tools else {}))
    return samples


def _eval_bfcl(response_text: str, gold_call: Any) -> bool:
    """Structurally match the emitted call against BFCL's `possible_answer` gold.

    Gold is a list of {func_name: {param: [accepted values]}}. Previously this only asked
    whether the function name and one accepted value appeared ANYWHERE in the response
    text, so a model could pass by echoing the prompt's own API list without emitting a
    call, and wrong arguments elsewhere in the prose still counted.
    """
    if not gold_call or not response_text:
        return False
    from .fc_common import score_possible_answer, score_tool_call
    if isinstance(gold_call, list):
        return score_possible_answer(response_text, gold_call)
    if isinstance(gold_call, dict):
        return score_possible_answer(response_text, [gold_call])
    return score_tool_call(response_text, gold_call, require_args=True)

def run_bfcl(
    model_name: str,
    base_url: str,
    concurrency: int,
    eval_categories: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Run native BFCL evaluation suite."""
    samples = _load_bfcl_samples(eval_categories)
    return run_eval_suite(
        eval_name="bfcl",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_bfcl,
        thinking=kwargs.get("enable_thinking", False),
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens"),
    )
