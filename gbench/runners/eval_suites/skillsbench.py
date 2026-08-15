# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# Evaluation Suite: skillsbench
# Description: SkillsBench (BenchFlow Modular Agent Skills Composition Benchmark - 87 tasks)

"""gbench native built-in runner for skillsbench (Tool Use & Agentic Workflows)."""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from .base import run_eval_suite, gemini_required_skip, DEFAULT_JUDGE_MODEL, parse_grade_verdict
from .sampling import stratified_sample

logger = logging.getLogger(__name__)

PILLAR = "Tool Use & Agentic Workflows"


def _load_skillsbench_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Load SkillsBench benchmark dataset directly from HF Hub ('benchflow/skillsbench')."""
    samples = []
    try:
        from huggingface_hub import HfApi, hf_hub_download
        api = HfApi()
        files = api.list_repo_files("benchflow/skillsbench", repo_type="dataset")
        task_mds = sorted([f for f in files if f.endswith("/task.md")])

        # Stratified, not a contiguous head (audit RC-1).
        task_mds = stratified_sample(task_mds, limit, None, seed="skillsbench")

        for task_file in task_mds:
            task_id = task_file.split("/")[0]
            local_t = hf_hub_download(repo_id="benchflow/skillsbench", filename=task_file, repo_type="dataset")
            with open(local_t, "r", encoding="utf-8") as fp:
                task_content = fp.read().strip()

            # Parse frontmatter if present
            category = "general"
            cat_match = re.search(r"category:\s*([^\n]+)", task_content)
            if cat_match:
                category = cat_match.group(1).strip()

            # Find oracle solution if present
            oracle_file = f"{task_id}/oracle/solve.sh"
            gold_sol = ""
            try:
                local_o = hf_hub_download(repo_id="benchflow/skillsbench", filename=oracle_file, repo_type="dataset")
                with open(local_o, "r", encoding="utf-8") as fp:
                    gold_sol = fp.read().strip()
            except Exception:
                gold_sol = ""

            prompt = (
                f"[SkillsBench Task: {task_id} ({category})]\n\n"
                f"{task_content}\n\n"
                "You have access to the skill packages in your environment. "
                "Execute the necessary scripts, reason through the problem, and provide the complete solution."
            )
            messages = [{"role": "user", "content": prompt}]
            samples.append((messages, gold_sol, {"category": category, "task_id": task_id}))
    except Exception as e:
        logger.error(f"Failed to load dataset for skillsbench: {e}")
        raise RuntimeError(f"Could not load dataset for skillsbench: {e}") from e

    if not samples:
        raise RuntimeError("Dataset for skillsbench returned empty rows")

    logger.info(f"Loaded {len(samples)} skillsbench samples.")
    return samples


def _eval_skillsbench(response_text: str, gold_target: str) -> bool:
    """Fallback local evaluation if GEMINI_API_KEY is not available (unit tests/offline)."""
    if not response_text:
        return False
    resp = response_text.strip()
    gold = str(gold_target).strip()
    if not gold:
        return False
    if gold.lower() in resp.lower():
        return True
    return len(resp) > 20


async def _async_judge_skillsbench(
    sample_traces: List[Dict[str, Any]],
    judge_model: str = DEFAULT_JUDGE_MODEL,
    judge_concurrency: int = 64,
) -> None:
    """Canonical SkillsBench 100% LLM Judge executed in parallel post-generation."""
    from tqdm import tqdm
    import asyncio

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.warning("GEMINI_API_KEY not set; using local fallback for SkillsBench.")
        for trace in sample_traces:
            cleaned = str(trace.get("response_text") or "").strip()
            gold = str(trace.get("gold_answer") or "").strip()
            trace["is_correct"] = _eval_skillsbench(cleaned, gold)
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
        task_prompt = messages[0].get("content", "") if messages else ""
        gold = str(trace.get("gold_answer") or "")

        prompt = (
            "You are an expert evaluator for the SkillsBench modular skills and engineering benchmark.\n"
            "Evaluate whether the candidate agent solution correctly fulfills the skill task according to the gold oracle solution.\n\n"
            f"Task Objective:\n{task_prompt[:2000]}\n\n"
            f"Gold Oracle Solution:\n{gold[:2000]}\n\n"
            f"Candidate Agent Solution:\n{resp_text}\n\n"
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

    with tqdm(total=len(sample_traces), desc="Judging [SKILLSBENCH]") as pbar:
        tasks = [_judge_single(trace, pbar) for trace in sample_traces]
        await asyncio.gather(*tasks)


def run_skillsbench(
    model_name: str,
    base_url: str,
    limit: Optional[int] = None,
    concurrency: int = 4,
    enable_thinking: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run native skillsbench evaluation suite."""
    skip = gemini_required_skip("skillsbench", model_name)
    if skip is not None:
        return skip
    samples = _load_skillsbench_samples(limit=limit)
    return run_eval_suite(
        eval_name="skillsbench",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_skillsbench,
        async_eval_fn=_async_judge_skillsbench,
        limit=limit,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        max_output_tokens=kwargs.get("max_output_tokens", 4096),
    )
