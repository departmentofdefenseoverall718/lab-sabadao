# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# Evaluation Suite: complexfuncbench
# Description: ComplexFuncBench (THUDM Complex Multi-Axis Tool & Function Calling Benchmark)

"""gbench native built-in runner for complexfuncbench (Tool Use & Function Calling)."""

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


def _load_complexfuncbench_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Load ComplexFuncBench benchmark dataset directly from HF Hub (THUDM/ComplexFuncBench)."""
    rows = []
    try:
        from datasets import load_dataset
        ds = load_dataset("THUDM/ComplexFuncBench", split="train")
        rows = list(ds)
    except Exception as e:
        logger.error(f"Failed to load dataset for complexfuncbench: {e}")
        raise RuntimeError(f"Could not load dataset for complexfuncbench: {e}") from e

    if not rows:
        raise RuntimeError("Dataset for complexfuncbench returned empty rows")

    # Stratified, not a contiguous head (audit RC-1).
    rows = stratified_sample(rows, limit, lambda r: (r or {}).get("id"), seed="complexfuncbench")

    samples = []
    for item in rows:
        convs = item.get("conversations") or []
        funcs = item.get("functions") or []
        user_content = ""
        gold = ""
        gold_fn = ""
        gold_args = {}

        for c in convs:
            if c.get("role") == "user" and not user_content:
                user_content = c.get("content", "")
            elif c.get("role") == "assistant" and not gold:
                fc = c.get("function_call")
                # ComplexFuncBench stores function_call as a LIST of calls; the dict-only
                # branch skipped them, leaving the NL summary as "gold" (meaningless noise).
                if isinstance(fc, list) and fc:
                    first = next((x for x in fc if isinstance(x, dict) and x.get("name")), None)
                    if first:
                        gold_fn = first.get("name", "")
                        gold_args = first.get("arguments", {})
                        gold = json.dumps(fc)
                elif fc and isinstance(fc, dict):
                    gold_fn = fc.get("name", "")
                    gold_args = fc.get("arguments", {})
                    gold = json.dumps(fc)
                elif c.get("content"):
                    gold = str(c.get("content", ""))

        tools_payload = []
        tools_desc_lines = []
        for f in funcs:
            if isinstance(f, dict) and f.get("name"):
                tools_payload.append({"type": "function", "function": f})
                fname = f.get("name")
                fdesc = f.get("description", "")
                params = f.get("parameters", {}).get("properties", {})
                param_strs = [f"{p}: {info.get('type', 'any')} ({info.get('description', '')})" for p, info in params.items()]
                tools_desc_lines.append(f"- {fname}({', '.join(params.keys())}): {fdesc}\n  Parameters: {'; '.join(param_strs)}")

        tools_prompt_str = "\n".join(tools_desc_lines)
        prompt = (
            f"You have access to the following functions:\n{tools_prompt_str}\n\n"
            f"User Request: {user_content}\n\n"
            "Select and call the appropriate function with the correct arguments to fulfill the request.\n"
            "Respond with: Tool: <function_name>(<arguments>)"
        )
        cat = str(item.get("id", "func")).split("-")[0]
        messages = [{"role": "user", "content": prompt}]
        sample_meta = {"category": cat, "gold_fn": gold_fn, "gold_args": gold_args}
        if tools_payload:
            sample_meta["tools"] = tools_payload

        samples.append((messages, gold, sample_meta))

    logger.info(f"Loaded {len(samples)} complexfuncbench samples with full function schemas.")
    return samples


def _eval_complexfuncbench(response_text: str, gold_target: str) -> bool:
    """Evaluate candidate response against expected function call."""
    if not response_text:
        return False

    resp = response_text.strip()
    gold = str(gold_target).strip()

    if not gold:
        return False

    # Compare the emitted tool call structurally (name + arguments). Gold may be a JSON
    # object OR a JSON list of calls - the loader previously only handled the dict case,
    # so for list-valued function_call rows the gold degraded to the NL summary text.
    from .fc_common import score_tool_call
    return score_tool_call(resp, gold, require_args=True)


def run_complexfuncbench(
    model_name: str,
    base_url: str,
    concurrency: int = 4,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Execute complexfuncbench native built-in evaluation suite."""
    samples = _load_complexfuncbench_samples(limit=kwargs.get("limit"))
    return run_eval_suite(
        eval_name="complexfuncbench",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_complexfuncbench,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens", 2048),
    )
