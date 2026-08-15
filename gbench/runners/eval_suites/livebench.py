# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# Evaluation Suite: livebench
# Description: LiveBench (Abacus/NYU Monthly Refreshed Contamination-Limited Benchmark across 6 categories)

"""gbench native built-in runner for livebench (General Knowledge & Reasoning)."""

import csv
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from .base import run_eval_suite, gemini_required_skip, DEFAULT_JUDGE_MODEL, parse_grade_verdict

logger = logging.getLogger(__name__)

PILLAR = "General Knowledge & Reasoning"


def _load_livebench_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Load LiveBench from HF Hub across all answer-gradeable categories (no fabricated fallback)."""
    # LiveBench is SIX categories; loading only livebench/math and reporting it as
    # "livebench" misrepresents the benchmark. Load every category this scorer can grade
    # and tag each row so category_accuracy shows the real breakdown.
    #
    # Verified schema: math/reasoning/language/data_analysis expose `ground_truth` and are
    # answer-graded here. `coding` (public/private test cases -> needs execution) and
    # `instruction_following` (instruction_id_list/kwargs -> needs IFEval rule checking)
    # carry NO ground_truth, so they are excluded rather than mis-scored; run them via the
    # dedicated code-exec / ifeval suites. Override with LIVEBENCH_CATEGORIES.
    _GRADEABLE = "math,reasoning,language,data_analysis"
    _NEEDS_OTHER_SCORER = {"coding": "test-case execution",
                           "instruction_following": "IFEval rule checking"}
    categories = [c.strip() for c in os.environ.get("LIVEBENCH_CATEGORIES", _GRADEABLE).split(",")
                  if c.strip()]
    for c in categories:
        if c in _NEEDS_OTHER_SCORER:
            logger.warning("livebench: category '%s' needs %s and has no ground_truth column; "
                           "its rows will be skipped by the answer scorer.",
                           c, _NEEDS_OTHER_SCORER[c])
    rows = []
    loaded, failed = [], []
    from datasets import load_dataset
    for cat in categories:
        try:
            ds = load_dataset(f"livebench/{cat}", split="test")
            for r in ds:
                r = dict(r)
                r["_livebench_category"] = cat
                rows.append(r)
            loaded.append(cat)
        except Exception as e:
            failed.append(f"{cat} ({type(e).__name__})")
    if not rows:
        raise RuntimeError(
            f"Could not load any LiveBench category (tried {categories}; failures: {failed})")
    if failed:
        logger.warning("livebench: loaded %s; could NOT load %s - the reported score covers "
                       "only the loaded categories.", loaded, failed)
    else:
        logger.info("livebench: loaded all %d categories %s", len(loaded), loaded)

    if not rows:
        raise RuntimeError(f"Dataset for livebench returned empty rows")

    if limit is not None and limit > 0:
        rows = rows[:limit]

    samples = []
    ungradeable: Dict[str, int] = {}
    for item in rows:
        # livebench/* columns: 'turns' (list; the prompt), 'ground_truth', 'task', 'category'.
        turns = item.get("turns")
        prompt = turns[0] if isinstance(turns, list) and turns else None
        gold = item.get("ground_truth")
        cat = str(item.get("_livebench_category") or item.get("category")
                  or item.get("task") or "unknown")
        if not prompt or gold is None or str(gold).strip() == "":
            # Not gradeable by an answer scorer (e.g. coding / instruction_following rows).
            # Skip it and account for it rather than aborting the whole load or, worse,
            # scoring it against a missing key.
            ungradeable[cat] = ungradeable.get(cat, 0) + 1
            continue
        messages = [{"role": "user", "content": str(prompt).strip()}]
        samples.append((messages, str(gold).strip(), {"category": cat}))

    if ungradeable:
        logger.warning("livebench: skipped %d rows with no ground_truth (needs a "
                       "different scorer): %s", sum(ungradeable.values()), ungradeable)
    logger.info(f"Loaded {len(samples)} livebench samples.")
    return samples


def _eval_livebench(response_text: str, gold_target: str) -> bool:
    """Fast deterministic evaluation for code, math, and data analysis in LiveBench."""
    if not response_text:
        return False
    resp = response_text.strip()
    gold = str(gold_target).strip()
    if not gold:
        return False
    def _norm(s):
        return re.sub(r"[\s$,]|\\left|\\right|\\!|\\,", "", str(s)).strip().rstrip(".").lower()

    # CC4: no bare containment. A gold that merely appears somewhere in the reasoning is
    # not an answer - require the boxed / "Final Answer:" span to EQUAL the gold. Anything
    # else is deferred to the LLM judge rather than silently credited here.
    match_boxed = re.findall(r"\\boxed\{([^\}]+)\}", resp)
    if match_boxed and _norm(match_boxed[-1]) == _norm(gold):
        return True
    match_fa = re.search(r"(?:Final Answer|Answer)\s*:\s*(.+)", resp, re.IGNORECASE)
    if match_fa and _norm(match_fa.group(1).split("\n")[0]) == _norm(gold):
        return True
    if _norm(resp) == _norm(gold):
        return True
    return False


async def _async_judge_livebench(
    sample_traces: List[Dict[str, Any]],
    judge_model: str = DEFAULT_JUDGE_MODEL,
    judge_concurrency: int = 64,
) -> None:
    """Canonical LiveBench Reasoning/Language 100% LLM Judge executed in parallel post-generation."""
    from tqdm import tqdm
    import asyncio

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.warning("GEMINI_API_KEY not set; using local fallback for LiveBench.")
        for trace in sample_traces:
            cleaned = str(trace.get("response_text") or "").strip()
            gold = str(trace.get("gold_answer") or "").strip()
            trace["is_correct"] = _eval_livebench(cleaned, gold)
            trace["status"] = "OK"
        return

    try:
        from google import genai
        client = genai.Client(api_key=api_key)
    except Exception as e:
        logger.error(f"Failed to initialize google-genai client: {e}")
        return

    semaphore = asyncio.Semaphore(judge_concurrency)

    async def _judge_single(trace: Dict[str, Any], pbar: tqdm) -> None:
        resp_text = trace.get("response_text")
        gold = str(trace.get("gold_answer") or "").strip()

        # Fast deterministic pass
        if _eval_livebench(str(resp_text or ""), gold):
            trace["is_correct"] = True
            trace["judge_grade"] = "CORRECT"
            trace["status"] = "OK"
            pbar.update(1)
            return

        if not resp_text:
            trace["is_correct"] = False
            trace["judge_grade"] = "FAILED"
            pbar.update(1)
            return

        messages = trace.get("messages", [])
        question = messages[0].get("content", "") if messages else ""

        prompt = (
            "You are an expert evaluator for the LiveBench benchmark.\n"
            "Evaluate whether the candidate response correctly answers the problem according to the gold target.\n\n"
            f"Question:\n{question}\n\n"
            f"Gold Target Answer:\n{gold}\n\n"
            f"Candidate Model Response:\n{resp_text}\n\n"
            "Output exactly:\n"
            "Grade: CORRECT / INCORRECT"
        )

        grade_str = "INCORRECT"
        async with semaphore:
            for attempt in range(3):
                try:
                    res = await client.aio.models.generate_content(
                        model=judge_model,
                        contents=prompt,
                    )
                    grade_str = (res.text or "").strip().upper()
                    break
                except Exception as e:
                    if attempt == 2:
                        logger.warning(f"Judge request failed after 3 attempts: {e}")
                    await asyncio.sleep(1.0 * (attempt + 1))

        is_corr = parse_grade_verdict(grade_str)
        trace["is_correct"] = is_corr
        trace["judge_grade"] = grade_str
        trace["status"] = "OK"
        pbar.update(1)

    with tqdm(total=len(sample_traces), desc="Judging [LIVEBENCH]") as pbar:
        tasks = [_judge_single(trace, pbar) for trace in sample_traces]
        await asyncio.gather(*tasks)


def run_livebench(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Execute livebench native built-in evaluation suite."""
    skip = gemini_required_skip("livebench", model_name)
    if skip is not None:
        return skip
    samples = _load_livebench_samples(limit=kwargs.get("limit"))
    return run_eval_suite(
        eval_name="livebench",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_livebench,
        async_eval_fn=_async_judge_livebench,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens", 2048),
    )
