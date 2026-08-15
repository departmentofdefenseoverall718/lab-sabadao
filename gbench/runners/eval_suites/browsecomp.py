# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# Evaluation Suite: browsecomp
# Description: OpenAI BrowseComp (Agentic Web Search & Deep Research Fact Retrieval)

"""gbench native built-in runner for browsecomp (Agentic & Web Research).

The smolagents/browse_comp dataset stores the 'problem' and 'answer' fields
XOR-encrypted with a per-row 'canary' password (the canonical OpenAI
simple-evals scheme). We decrypt them at load time and grade with the canonical
BrowseComp LLM grader. NOTE: BrowseComp is canonically a *browsing* benchmark;
gbench has no browser tool wired, so this runs the model closed-book.
"""

import base64
import hashlib
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from .base import run_eval_suite, gemini_required_skip, DEFAULT_JUDGE_MODEL
from .sampling import stratified_sample

logger = logging.getLogger(__name__)

PILLAR = "Agentic & Web Research"

# Canonical OpenAI simple-evals BrowseComp templates.
QUERY_TEMPLATE = """{question}

Your response should be in the following format:
Explanation: {{your explanation for your final answer}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100% for your answer}}"""

GRADER_TEMPLATE = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0% and 100% from [response]. Put 100 if there is no confidence score available."""


def _derive_key(password: str, length: int) -> bytes:
    """Derive a fixed-length key from the canary password (SHA-256 keystream)."""
    hasher = hashlib.sha256()
    hasher.update(password.encode())
    key = hasher.digest()
    return key * (length // len(key)) + key[: length % len(key)]


def _decrypt(ciphertext_b64: str, password: str) -> str:
    """Decrypt base64 XOR-ciphertext using the canary-derived keystream."""
    encrypted = base64.b64decode(ciphertext_b64)
    key = _derive_key(password, len(encrypted))
    decrypted = bytes(a ^ b for a, b in zip(encrypted, key))
    return decrypted.decode()


def _load_browsecomp_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Load BrowseComp from HF Hub (smolagents/browse_comp) and canary-decrypt each row.

    Raises on load/schema/decrypt failure (no fabricated fallback).
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("smolagents/browse_comp", split="test")
        rows = list(ds)
    except Exception as e:
        logger.error(f"Failed to load dataset for browsecomp: {e}")
        raise RuntimeError(f"Could not load dataset for browsecomp: {e}") from e

    if not rows:
        raise RuntimeError("Dataset for browsecomp returned empty rows")

    # Stratified, not a contiguous head (audit RC-1).
    rows = stratified_sample(rows, limit, lambda r: (r or {}).get("problem_topic"), seed="browsecomp")

    samples = []
    for item in rows:
        enc_problem = item.get("problem")
        enc_answer = item.get("answer")
        canary = item.get("canary")
        if not enc_problem or not enc_answer or not canary:
            raise RuntimeError(
                "browsecomp: unexpected dataset schema "
                "(missing 'problem'/'answer'/'canary'); refusing to fabricate sample data"
            )
        try:
            question = _decrypt(enc_problem, canary)
            gold = _decrypt(enc_answer, canary)
        except Exception as e:
            raise RuntimeError(
                f"browsecomp: failed to canary-decrypt a row: {e}"
            ) from e
        topic = str(item.get("problem_topic") or "General")
        messages = [{"role": "user", "content": QUERY_TEMPLATE.format(question=question)}]
        samples.append((messages, gold, {"category": topic}))

    logger.info(f"Loaded {len(samples)} browsecomp samples (canary-decrypted).")
    return samples


async def _async_judge_browsecomp(
    sample_traces: List[Dict[str, Any]],
    judge_model: str = DEFAULT_JUDGE_MODEL,
    judge_concurrency: int = 64,
) -> None:
    """Canonical BrowseComp LLM grader (correct: yes/no). Requires GEMINI_API_KEY."""
    from tqdm import tqdm
    import asyncio

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "browsecomp requires GEMINI_API_KEY for canonical LLM grading; "
            "refusing to emit a heuristic score"
        )

    try:
        from google import genai
        client = genai.Client(api_key=api_key)
    except Exception as e:
        raise RuntimeError(
            f"browsecomp: failed to initialize google-genai client: {e}"
        ) from e

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
        prompt = GRADER_TEMPLATE.format(
            question=question, response=resp_text, correct_answer=gold
        )

        grade_str = ""
        async with semaphore:
            for attempt in range(3):
                try:
                    res = await client.aio.models.generate_content(
                        model=judge_model,
                        contents=prompt,
                    )
                    grade_str = res.text or ""
                    break
                except Exception as e:
                    if attempt == 2:
                        logger.warning(f"Judge request failed after 3 attempts: {e}")
                    await asyncio.sleep(1.0 * (attempt + 1))

        match = re.search(r"correct:\s*(yes|no)", grade_str, re.IGNORECASE)
        trace["is_correct"] = bool(match and match.group(1).lower() == "yes")
        # A judge that never answered is not a judge that said "no". Recording it as
        # `correct: no` made three failed API calls indistinguishable from a wrong answer,
        # so an outage read as a low score instead of a broken run.
        if match:
            trace["judge_grade"] = match.group(0)
            trace["status"] = "OK"
        else:
            trace["judge_grade"] = "JUDGE_FAILED"
            trace["status"] = "FAILED"
        pbar.update(1)

    with tqdm(total=len(sample_traces), desc="Judging [BROWSECOMP]") as pbar:
        tasks = [_judge_single(trace, pbar) for trace in sample_traces]
        await asyncio.gather(*tasks)


def run_browsecomp(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Execute browsecomp native built-in evaluation suite."""
    skip = gemini_required_skip("browsecomp", model_name)
    if skip:
        return skip
    samples = _load_browsecomp_samples(limit=kwargs.get("limit"))
    return run_eval_suite(
        eval_name="browsecomp",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        async_eval_fn=_async_judge_browsecomp,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens", 4096),
    )
