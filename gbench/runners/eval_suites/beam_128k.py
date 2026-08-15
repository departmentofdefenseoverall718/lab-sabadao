# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# Evaluation Suite: beam_128k
# Description: BEAM-128K (Beyond a Million Tokens: Long-Term Memory in LLMs - 400 conversations spanning 128k context)

"""gbench native built-in runner for beam_128k (Long Context & Retrieval)."""

import ast
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from .base import run_eval_suite, gemini_required_skip, DEFAULT_JUDGE_MODEL, parse_grade_verdict
from .sampling import stratified_sample

logger = logging.getLogger(__name__)

PILLAR = "Long Context & Retrieval"


def _load_beam_128k_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Load BEAM-128K benchmark dataset directly from HF Hub ('Mohammadta/BEAM')."""
    samples = []
    try:
        from huggingface_hub import hf_hub_download
        import pandas as pd
        parquet_file = hf_hub_download(repo_id="Mohammadta/BEAM", filename="data/100K-00000-of-00001.parquet", repo_type="dataset")
        df = pd.read_parquet(parquet_file)

        for _, row in df.iterrows():
            conv_id = str(row.get("conversation_id", ""))
            chat = row.get("chat", [])
            chat_list = list(chat) if hasattr(chat, "__iter__") else []

            # Format conversation transcript
            conv_turns = []
            for turn in chat_list:
                if isinstance(turn, dict):
                    role = turn.get("role", "user").capitalize()
                    content = turn.get("content", "").strip()
                    conv_turns.append(f"[{role}]: {content}")
                else:
                    conv_turns.append(str(turn))

            history_text = "\n\n".join(conv_turns)

            raw_pq = row.get("probing_questions", {})
            if isinstance(raw_pq, str):
                try:
                    pq_dict = ast.literal_eval(raw_pq)
                except Exception:
                    pq_dict = json.loads(raw_pq)
            else:
                pq_dict = raw_pq

            if isinstance(pq_dict, dict):
                for memory_ability, q_items in pq_dict.items():
                    if isinstance(q_items, list):
                        for q_item in q_items:
                            if isinstance(q_item, dict):
                                question = q_item.get("question", "").strip()
                                ideal_resp = q_item.get("ideal_response", "") or q_item.get("answer", "")
                                if question:
                                    prompt = (
                                        f"[Conversation History]\n{history_text}\n\n"
                                        f"[Memory Probing Question]\n{question}\n\n"
                                        "Answer the probing question based strictly on the conversation history above:"
                                    )
                                    messages = [{"role": "user", "content": prompt}]
                                    samples.append(
                                        (
                                            messages,
                                            ideal_resp,
                                            {
                                                "category": memory_ability,
                                                "conversation_id": conv_id,
                                                "difficulty": q_item.get("difficulty", "medium"),
                                            },
                                        )
                                    )
    except Exception as e:
        logger.error(f"Failed to load dataset for beam_128k: {e}")
        raise RuntimeError(f"Could not load dataset for beam_128k: {e}") from e

    if not samples:
        raise RuntimeError("Dataset for beam_128k returned empty rows")

    # Stratified, not a contiguous head (audit RC-1). `samples` holds built
    # (messages, gold, meta) tuples, not raw dict rows, so the key must read meta - `r.get`
    # raised "'tuple' object has no attribute 'get'" and killed the whole suite.
    samples = stratified_sample(
        samples, limit,
        lambda s: (s[2] or {}).get("memory_ability") if len(s) > 2 and isinstance(s[2], dict) else None,
        seed="beam_128k")

    logger.info(f"Loaded {len(samples)} beam_128k samples.")
    return samples


def _eval_beam_128k(response_text: str, gold_target: str) -> bool:
    """Fallback local evaluation if GEMINI_API_KEY is not available (unit tests/offline)."""
    if not response_text:
        return False
    resp = response_text.strip()
    gold = str(gold_target).strip()
    if not gold:
        return False
    if gold.lower() in resp.lower():
        return True
    if "no information" in gold.lower() or "not mentioned" in gold.lower():
        if any(w in resp.lower() for w in ["not mentioned", "no information", "not found", "cannot determine"]):
            return True
    return False


async def _async_judge_beam_128k(
    sample_traces: List[Dict[str, Any]],
    judge_model: str = DEFAULT_JUDGE_MODEL,
    judge_concurrency: int = 64,
) -> None:
    """Canonical Meta BEAM 100% LLM Judge executed in parallel post-generation."""
    from tqdm import tqdm
    import asyncio

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.warning("GEMINI_API_KEY not set; using local fallback for BEAM.")
        for trace in sample_traces:
            cleaned = str(trace.get("response_text") or "").strip()
            gold = str(trace.get("gold_answer") or "").strip()
            trace["is_correct"] = _eval_beam_128k(cleaned, gold)
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
        question = messages[-1].get("content", "") if messages else ""
        gold = str(trace.get("gold_answer") or "")

        prompt = (
            "You are an expert evaluator for the Meta BEAM long-context memory benchmark.\n"
            "Evaluate whether the candidate model response accurately answers the probing question given the ideal target response.\n\n"
            f"Question:\n{question}\n\n"
            f"Ideal Target Response:\n{gold}\n\n"
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

    with tqdm(total=len(sample_traces), desc="Judging [BEAM_128K]") as pbar:
        tasks = [_judge_single(trace, pbar) for trace in sample_traces]
        await asyncio.gather(*tasks)


def run_beam_128k(
    model_name: str,
    base_url: str,
    limit: Optional[int] = None,
    concurrency: int = 4,
    enable_thinking: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run native beam_128k evaluation suite."""
    skip = gemini_required_skip("beam_128k", model_name)
    if skip is not None:
        return skip
    samples = _load_beam_128k_samples(limit=limit)
    return run_eval_suite(
        eval_name="beam_128k",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_beam_128k,
        async_eval_fn=_async_judge_beam_128k,
        limit=limit,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        max_output_tokens=kwargs.get("max_output_tokens", 4096),
    )
