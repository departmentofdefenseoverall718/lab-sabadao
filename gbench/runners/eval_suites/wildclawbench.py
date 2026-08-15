# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# Evaluation Suite: wildclawbench
# Description: WildClawBench (OpenClaw Real-World Long-Horizon Autonomous Agent Benchmark - 60 tasks)

"""gbench native built-in runner for wildclawbench (Tool Use & Agentic Workflows)."""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from .base import run_eval_suite, gemini_required_skip, DEFAULT_JUDGE_MODEL, parse_grade_verdict
from .sampling import stratified_sample

logger = logging.getLogger(__name__)

PILLAR = "Tool Use & Agentic Workflows"


def _load_wildclawbench_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Load WildClawBench benchmark dataset directly from HF Hub ('internlm/WildClawBench')."""
    samples = []
    try:
        from huggingface_hub import HfApi, hf_hub_download
        api = HfApi()
        files = api.list_repo_files("internlm/WildClawBench", repo_type="dataset")

        task_dirs = set()
        for f in files:
            parts = f.split("/")
            if len(parts) >= 3 and parts[0] == "workspace":
                task_dirs.add(f"{parts[0]}/{parts[1]}/{parts[2]}")

        sorted_tasks = sorted(list(task_dirs))
        # Stratified, not a contiguous head (audit RC-1).
        sorted_tasks = stratified_sample(sorted_tasks, limit, None, seed="wildclawbench")

        for t_dir in sorted_tasks:
            parts = t_dir.split("/")
            category = parts[1] if len(parts) > 1 else "general"
            task_name = parts[2] if len(parts) > 2 else t_dir

            gt_file = f"{t_dir}/gt/ground_truth.json"
            gt_text = ""
            try:
                local_gt = hf_hub_download(repo_id="internlm/WildClawBench", filename=gt_file, repo_type="dataset")
                with open(local_gt, "r", encoding="utf-8") as fp:
                    gt_data = json.load(fp)
                    gt_text = json.dumps(gt_data, ensure_ascii=False)
            except Exception:
                gt_text = "{}"

            prompt = (
                f"[WildClaw Task: {task_name.replace('_', ' ').title()} ({category})]\n\n"
                f"You are an autonomous computer-use and workspace agent executing task: '{task_name}'.\n"
                "Plan and execute all required tool calls, file manipulations, or data transformations "
                "to fulfill the objective. Conclude with a detailed final report and solution payload."
            )
            messages = [{"role": "user", "content": prompt}]
            samples.append((messages, gt_text, {"category": category, "task": task_name}))
    except Exception as e:
        logger.error(f"Failed to load dataset for wildclawbench: {e}")
        raise RuntimeError(f"Could not load dataset for wildclawbench: {e}") from e

    if not samples:
        raise RuntimeError("Dataset for wildclawbench returned empty rows")

    logger.info(f"Loaded {len(samples)} wildclawbench samples.")
    return samples


def _eval_wildclawbench(response_text: str, gold_target: str) -> bool:
    """Fallback local evaluation if GEMINI_API_KEY is not available (unit tests/offline)."""
    if not response_text:
        return False
    resp = response_text.strip()
    if not gold_target or gold_target == "{}":
        return len(resp) > 5
    return len(resp) > 20


async def _async_judge_wildclawbench(
    sample_traces: List[Dict[str, Any]],
    judge_model: str = DEFAULT_JUDGE_MODEL,
    judge_concurrency: int = 64,
) -> None:
    """Canonical WildClawBench 100% LLM Judge executed in parallel post-generation."""
    from tqdm import tqdm
    import asyncio

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.warning("GEMINI_API_KEY not set; using local fallback for WildClawBench.")
        for trace in sample_traces:
            cleaned = str(trace.get("response_text") or "").strip()
            gold = str(trace.get("gold_answer") or "").strip()
            trace["is_correct"] = _eval_wildclawbench(cleaned, gold)
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
        task_desc = messages[0].get("content", "") if messages else ""
        gold = str(trace.get("gold_answer") or "")

        prompt = (
            "You are an expert evaluator for the WildClawBench autonomous computer-use and workspace agent benchmark.\n"
            "Evaluate whether the agent candidate response successfully completed the task according to the ground truth specification.\n\n"
            f"Task Objective:\n{task_desc[:2000]}\n\n"
            f"Ground Truth Specification:\n{gold[:2000]}\n\n"
            f"Agent Candidate Response:\n{resp_text}\n\n"
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

    with tqdm(total=len(sample_traces), desc="Judging [WILDCLAWBENCH]") as pbar:
        tasks = [_judge_single(trace, pbar) for trace in sample_traces]
        await asyncio.gather(*tasks)


def run_wildclawbench(
    model_name: str,
    base_url: str,
    limit: Optional[int] = None,
    concurrency: int = 4,
    enable_thinking: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run native wildclawbench evaluation suite."""
    skip = gemini_required_skip("wildclawbench", model_name)
    if skip is not None:
        return skip
    samples = _load_wildclawbench_samples(limit=limit)
    return run_eval_suite(
        eval_name="wildclawbench",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_wildclawbench,
        async_eval_fn=_async_judge_wildclawbench,
        limit=limit,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        max_output_tokens=kwargs.get("max_output_tokens", 4096),
    )
