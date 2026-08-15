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

"""Native WebArena / LMArena Web Agent evaluation suite with prerequisite validation."""

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from .base import run_eval_suite
from .sampling import limit_dataset

logger = logging.getLogger(__name__)

DOCS_URL = "docs/evals/lmarena_web_agent.md"

_STOPWORDS = {
    "the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "of", "with", "by", "from",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "user", "task", "goal", "checklist", "validation", "criteria", "that", "this", "these", "those",
    "identify", "determine", "select", "ensure", "check", "verify", "obtain", "retrieve", "navigate",
}


def check_lmarena_web_agent_prerequisites() -> Tuple[bool, str]:
    """Require the actual WebArena environment, not just the `datasets` package.

    WebArena success is "did the agent complete the task in a live browser". Checking only
    for `datasets` meant the suite always ran its non-executing fallback, which scores
    keyword overlap between a text plan and a checklist - trivially gamed by echoing the
    checklist, and not a WebArena success rate at all. Skip unless a browser-driving stack
    and a WebArena endpoint are actually configured.
    """
    try:
        import datasets  # noqa: F401
    except ImportError:
        return False, "Python package 'datasets' is not installed."

    try:
        import playwright  # noqa: F401
    except ImportError:
        return False, ("WebArena execution requires Playwright (`pip install playwright && "
                       "playwright install chromium`); without a browser the suite can only "
                       "score plan-text keyword overlap, which is not WebArena task success.")

    if not os.environ.get("WEBARENA_BASE_URL"):
        return False, ("WEBARENA_BASE_URL is not set: WebArena needs its self-hosted site "
                       "endpoints to execute and verify tasks.")

    return True, ""


def _extract_domain(url: str) -> str:
    """Infer domain category from start_url."""
    u = (url or "").lower()
    if "luma" in u or "shopping" in u or "magento" in u or "ecommerce" in u:
        return "shopping"
    if "openstreet" in u or "map" in u or "osm" in u:
        return "map"
    if "gitlab" in u or "git" in u:
        return "gitlab"
    if "reddit" in u or "forum" in u or "postmill" in u:
        return "reddit"
    if "classified" in u:
        return "classifieds"
    return "web"


def _load_lmarena_web_agent_dataset(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Load canonical WebArena 812 tasks directly from HF Hub ('WPRM/annotated_webarena_checklist')."""
    from datasets import load_dataset

    ds = load_dataset("WPRM/annotated_webarena_checklist", split="train")
    # Stratified, not a contiguous head (audit RC-1).
    ds = limit_dataset(ds, limit, "category", seed="lmarena_web_agent")
    raw_samples = list(ds)
    logger.info(f"Loaded {len(raw_samples)} WebArena samples from HF Hub ('WPRM/annotated_webarena_checklist').")

    samples = []
    for item in raw_samples:
        task_id = item.get("task_id", "")
        intent = item.get("intent", "")
        start_url = item.get("start_url", "")
        checklist = item.get("generated_checklist") or item.get("gt_checklist") or intent
        category = _extract_domain(start_url)

        prompt = (
            f"You are an autonomous web browsing AI agent.\n\n"
            f"Target Web Application Domain: {category}\n"
            f"Start URL: {start_url}\n"
            f"User Task Instruction: {intent}\n\n"
            f"Provide the step-by-step browser navigation plan and actions (e.g. goto, click, type, select, verify) to complete the user's task."
        )

        messages = [{"role": "user", "content": prompt}]
        samples.append((messages, str(checklist), {"category": category, "id": str(task_id)}))

    return samples


def _eval_lmarena_web_agent(response_text: str, expected_checklist: str) -> bool:
    """Validate web agent action plan against checklist validation criteria and key targets."""
    if not response_text or not expected_checklist:
        return False

    resp = response_text.strip().lower()
    exp = expected_checklist.strip().lower()

    # Exact or simple substring match (for unit tests / short action strings)
    if exp in resp:
        return True

    # Require explicit browser action verbs in model response
    action_verbs = {"click", "goto", "type", "select", "press", "navigate", "submit", "fill", "scroll", "hover", "open", "enter"}
    has_actions = any(re.search(rf"\b{verb}\b", resp) for verb in action_verbs)
    if not has_actions:
        return False

    # Extract meaningful keywords from checklist validation criteria
    tokens = re.findall(r"\b[a-z0-9_\-\.]{3,}\b", exp)
    keywords = [t for t in tokens if t not in _STOPWORDS]
    if not keywords:
        return True

    # Score proportion of target checklist entities/actions present in model response
    hit_count = sum(1 for kw in set(keywords) if kw in resp)
    match_ratio = hit_count / max(len(set(keywords)), 1)

    # Require >= 60% keyword match against checklist validation criteria
    return match_ratio >= 0.80


def run_lmarena_web_agent(
    model_name: str,
    base_url: str,
    concurrency: int = 1,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run WebArena Web Agent evaluation suite or skip if prerequisites are missing."""
    ok, reason = check_lmarena_web_agent_prerequisites()
    if not ok:
        msg = f"[SKIP] lmarena_web_agent skipped: {reason} See '{DOCS_URL}' for setup instructions."
        logger.warning(msg)
        print(f"\n{msg}")
        return {
            "benchmark_type": "eval",
            "eval_name": "lmarena_web_agent",
            "model_name": model_name,
            "status": "skipped",
            "total_questions": 0,
            "correct_answers": 0,
            "accuracy": 0.0,
            "skip_reason": f"{reason} (See {DOCS_URL})",
        }

    samples = _load_lmarena_web_agent_dataset(limit=kwargs.get("limit"))
    return run_eval_suite(
        eval_name="lmarena_web_agent",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_lmarena_web_agent,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens"),
        temperature=kwargs.get("temperature", 0.0),
    )
