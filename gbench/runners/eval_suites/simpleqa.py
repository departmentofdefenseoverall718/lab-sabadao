# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# Evaluation Suite: simpleqa
# Description: OpenAI SimpleQA (Factuality & Hallucination Benchmark - 4,326 short questions)

"""gbench native built-in runner for simpleqa (Factuality & Knowledge)."""

import ast
import csv
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from .base import run_eval_suite, gemini_required_skip, DEFAULT_JUDGE_MODEL, parse_grade_verdict
from .sampling import stratified_sample

logger = logging.getLogger(__name__)

PILLAR = "Factuality & Knowledge"


def _simpleqa_topic(metadata: Any) -> Optional[str]:
    """Topic out of SimpleQA's `metadata` column, which is a stringified Python dict."""
    if isinstance(metadata, dict):
        topic = metadata.get("topic")
        return str(topic).strip() if topic else None
    text = str(metadata or "").strip()
    if not text:
        return None
    for loader in (json.loads, ast.literal_eval):
        try:
            value = loader(text)
        except Exception:
            continue
        if isinstance(value, dict) and value.get("topic"):
            return str(value["topic"]).strip()
    m = re.search(r"['\"]topic['\"]\s*:\s*['\"]([^'\"]+)['\"]", text)
    return m.group(1).strip() if m else None


def _load_simpleqa_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Load simpleqa benchmark dataset directly from HF Hub (openai/simple-evals)."""
    rows = []
    try:
        from datasets import load_dataset
        ds = load_dataset('basicv8vc/SimpleQA', split='test')
        rows = list(ds)
    except Exception as e:
        logger.error(f"Failed to load dataset for simpleqa: {e}")
        raise RuntimeError(f"Could not load dataset for simpleqa: {e}") from e

    if not rows:
        raise RuntimeError(f"Dataset for simpleqa returned empty rows")

    # Stratified, not a contiguous head (audit RC-1).
    rows = stratified_sample(rows, limit, lambda r: (r or {}).get("metadata"), seed="simpleqa")

    samples = []
    for item in rows:
        prompt = item.get("problem") or item.get("question")
        gold = item.get("answer") or item.get("target")
        if not prompt or gold is None:
            raise RuntimeError(
                "simpleqa: unexpected dataset schema (missing 'problem'/'answer'); "
                "refusing to fabricate sample data"
            )
        gold = str(gold).strip()
        # SimpleQA has no top-level `category`; the topic lives inside `metadata`, which
        # the row stores as a stringified dict. The default therefore applied to every
        # row and the per-category table reported one bucket, "geography", for the whole
        # benchmark.
        cat = item.get("category") or _simpleqa_topic(item.get("metadata")) or "uncategorized"

        messages = [{"role": "user", "content": prompt}]
        samples.append((messages, gold, {"category": cat}))

    logger.info(f"Loaded {len(samples)} simpleqa samples.")
    return samples


def _eval_simpleqa(response_text: str, gold_target: str) -> bool:
    """Fallback local evaluation if GEMINI_API_KEY is not available (unit tests/offline)."""
    if not response_text:
        return False
    resp = response_text.strip()
    gold = str(gold_target).strip()
    if not gold:
        return False
    if gold.lower() in resp.lower():
        return True
    return False


async def _async_judge_simpleqa(
    sample_traces: List[Dict[str, Any]],
    judge_model: str = DEFAULT_JUDGE_MODEL,
    judge_concurrency: int = 64,
) -> None:
    """Canonical OpenAI SimpleQA 100% LLM Judge executed in parallel post-generation."""
    from tqdm import tqdm
    import asyncio

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.warning("GEMINI_API_KEY not set; using local string fallback for SimpleQA.")
        for trace in sample_traces:
            cleaned = str(trace.get("response_text") or "").strip()
            gold = str(trace.get("gold_answer") or "").strip()
            trace["is_correct"] = _eval_simpleqa(cleaned, gold)
            trace["scoring_mode"] = "judge_fallback"
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
        if not resp_text:
            trace["is_correct"] = False
            trace["judge_grade"] = "FAILED"
            pbar.update(1)
            return

        messages = trace.get("messages", [])
        question = messages[0].get("content", "") if messages else ""
        gold = str(trace.get("gold_answer") or "")

        prompt = (
            "You are an expert evaluator. Your task is to evaluate if a candidate response correctly answers the question, given a ground truth answer.\n\n"
            f"Question: {question}\n"
            f"Ground Truth Answer: {gold}\n"
            f"Candidate Model Response:\n{resp_text}\n\n"
            "Please judge the candidate response strictly as one of the following:\n"
            "- 'CORRECT': The candidate response correctly answers the question matching the ground truth.\n"
            "- 'INCORRECT': The candidate response gives an incorrect factual answer.\n"
            "- 'NOT_ATTEMPTED': The candidate model refused to answer, stated it does not know the answer, or abstained.\n\n"
            "Respond in this format:\n"
            "Grade: CORRECT / INCORRECT / NOT_ATTEMPTED"
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

    with tqdm(total=len(sample_traces), desc="Judging [SIMPLEQA]") as pbar:
        tasks = [_judge_single(trace, pbar) for trace in sample_traces]
        await asyncio.gather(*tasks)


def run_simpleqa(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Execute simpleqa native built-in evaluation suite."""
    skip = gemini_required_skip("simpleqa", model_name)
    if skip is not None:
        return skip
    samples = _load_simpleqa_samples(limit=kwargs.get("limit"))
    return run_eval_suite(
        eval_name="simpleqa",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_simpleqa,
        async_eval_fn=_async_judge_simpleqa,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens", 512),
    )
