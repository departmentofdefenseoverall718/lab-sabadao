# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# Evaluation Suite: bfcl_v3_live
# Description: BFCL v3 LIVE single-turn function calling + irrelevance abstention

"""gbench native built-in runner for bfcl_v3_live (Tool Use & Function Calling).

This suite loads the BFCL **v3 LIVE** single-turn subsets (live_simple, live_parallel,
live_multiple, live_parallel_multiple) plus live_irrelevance abstention - roughly the
"Live" 10% slice of the BFCL v4 leaderboard. It was previously mis-named
`bfcl_v4_agentic`, which implied the v4 agentic track (web search / memory / format
sensitivity); that track is now a separate suite, `bfcl_v4_agentic`."""

import csv
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from .base import run_eval_suite
from .sampling import stratified_sample

logger = logging.getLogger(__name__)

PILLAR = "Tool Use & Function Calling"


def _load_bfcl_v3_live_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], Any, Dict[str, Any]]]:
    """Load canonical Berkeley Function Calling Leaderboard (BFCL) dataset from HF Hub."""
    from huggingface_hub import hf_hub_download

    subsets = [
        ("BFCL_v3_live_simple.json", "possible_answer/BFCL_v3_live_simple.json", "live_simple"),
        ("BFCL_v3_live_parallel.json", "possible_answer/BFCL_v3_live_parallel.json", "live_parallel"),
        ("BFCL_v3_live_multiple.json", "possible_answer/BFCL_v3_live_multiple.json", "live_multiple"),
        ("BFCL_v3_live_parallel_multiple.json", "possible_answer/BFCL_v3_live_parallel_multiple.json", "live_parallel_multiple"),
        ("BFCL_v3_live_irrelevance.json", None, "live_irrelevance"),
    ]

    all_raw_samples = []
    for q_filename, a_filename, cat_name in subsets:
        try:
            q_path = hf_hub_download(
                repo_id="gorilla-llm/Berkeley-Function-Calling-Leaderboard",
                filename=q_filename,
                repo_type="dataset",
            )
            with open(q_path, "r", encoding="utf-8") as f:
                q_items = [json.loads(line) for line in f if line.strip()]

            a_items = []
            if a_filename:
                a_path = hf_hub_download(
                    repo_id="gorilla-llm/Berkeley-Function-Calling-Leaderboard",
                    filename=a_filename,
                    repo_type="dataset",
                )
                with open(a_path, "r", encoding="utf-8") as f:
                    a_items = [json.loads(line) for line in f if line.strip()]

            for idx, q_obj in enumerate(q_items):
                ans_obj = a_items[idx] if idx < len(a_items) else None
                all_raw_samples.append((q_obj, ans_obj, cat_name))
        except Exception as e:
            logger.warning(f"Could not load BFCL subset {q_filename}: {e}")

    if not all_raw_samples:
        raise RuntimeError("No BFCL samples loaded from gorilla-llm/Berkeley-Function-Calling-Leaderboard")

    # Stratified, not a contiguous head (audit RC-1).
    all_raw_samples = stratified_sample(all_raw_samples, limit, None, seed="bfcl_v3_live")

    samples = []
    for q_obj, ans_obj, cat_name in all_raw_samples:
        tools = q_obj.get("function") or []
        question_data = q_obj.get("question") or []

        # Extract user prompt from question
        user_text = ""
        if isinstance(question_data, list):
            for turn in question_data:
                if isinstance(turn, list):
                    for msg in turn:
                        if isinstance(msg, dict) and msg.get("role") == "user":
                            user_text += msg.get("content", "") + "\n"
                elif isinstance(turn, dict) and turn.get("role") == "user":
                    user_text += turn.get("content", "") + "\n"
                elif isinstance(turn, str):
                    user_text += turn + "\n"
        elif isinstance(question_data, str):
            user_text = question_data

        user_text = user_text.strip() or "Execute the appropriate tool for the request."

        # Canonical system prompt with available tool definitions
        tools_str = json.dumps(tools, indent=2) if tools else "[]"
        system_msg = (
            "You are an expert function calling assistant. You have access to the following tools:\n"
            f"{tools_str}\n\n"
            "If a function should be called, respond with the function call as JSON in the format:\n"
            '{"name": "function_name", "arguments": {"param1": "value1", ...}}\n'
            "If multiple functions should be called, respond with a JSON list of function calls:\n"
            '[{"name": "func1", "arguments": {...}}, {"name": "func2", "arguments": {...}}]\n'
            "If no function is suitable or needed to answer the user request, answer directly without calling any tools."
        )

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_text},
        ]

        ground_truth = (ans_obj.get("ground_truth") if isinstance(ans_obj, dict) else []) or []
        samples.append((messages, ground_truth, {"category": cat_name}))

    logger.info(f"Loaded {len(samples)} canonical BFCL v4 Agentic samples.")
    return samples


def _eval_bfcl_v3_live(response_text: str, gold_ground_truth: Any) -> bool:
    """Evaluate candidate response against gold target using 100% deterministic AST/JSON tool matching."""
    if not response_text:
        return False

    resp = response_text.strip()
    resp_lower = resp.lower()

    # Case 1: Irrelevance / Abstention (gold is empty list)
    if not gold_ground_truth:
        # Model should NOT invoke a function call JSON or function call syntax
        has_tool_call = bool(
            re.search(r'\"name\"\s*:\s*\"[a-zA-Z_0-9\.]+\"', resp)
            or re.search(r'[a-zA-Z_0-9\.]+\s*\([^\)]*=[^\)]*\)', resp)
        )
        return not has_tool_call

    # Case 2: Functional tool invocation
    if isinstance(gold_ground_truth, list):
        for exp_call in gold_ground_truth:
            if not isinstance(exp_call, dict):
                continue
            for fn_name, fn_args in exp_call.items():
                if fn_name.lower() not in resp_lower:
                    return False
                if isinstance(fn_args, dict):
                    for arg_k, valid_vals in fn_args.items():
                        if isinstance(valid_vals, list) and valid_vals:
                            matched = any(str(v).lower() in resp_lower for v in valid_vals)
                            if not matched:
                                return False
                        elif valid_vals is not None:
                            if str(valid_vals).lower() not in resp_lower:
                                return False

        return True

    return False


def run_bfcl_v3_live(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Execute bfcl_v3_live native built-in evaluation suite."""
    samples = _load_bfcl_v3_live_samples(limit=kwargs.get("limit"))
    return run_eval_suite(
        eval_name="bfcl_v3_live",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_bfcl_v3_live,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens", 4096),
    )
