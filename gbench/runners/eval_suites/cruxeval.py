# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# Evaluation Suite: cruxeval
# Description: CRUXEval (MIT/Meta Code Reasoning & I/O Execution Simulation Benchmark)

"""gbench native built-in runner for cruxeval (Code Reasoning & Execution)."""

import ast
import csv
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from .base import run_eval_suite
from .sampling import stratified_sample

logger = logging.getLogger(__name__)

PILLAR = "Code Reasoning & Execution"


def _load_cruxeval_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Load CRUXEval from HF Hub (cruxeval-org/cruxeval); raises on load/schema failure (no fabricated fallback)."""
    rows = []
    try:
        from datasets import load_dataset
        ds = load_dataset('cruxeval-org/cruxeval', split='test')
        rows = list(ds)
    except Exception as e:
        logger.error(f"Failed to load dataset for cruxeval: {e}")
        raise RuntimeError(f"Could not load dataset for cruxeval: {e}") from e

    if not rows:
        raise RuntimeError(f"Dataset for cruxeval returned empty rows")

    # Stratified, not a contiguous head (audit RC-1).
    rows = stratified_sample(rows, limit, None, seed="cruxeval")

    samples = []
    for item in rows:
        # cruxeval-org/cruxeval columns: 'code' (a function named f), 'input', 'output'.
        # This is the canonical CRUXEval-O (output prediction) task: given code + input,
        # predict the exact output of f(input).
        code = item.get("code")
        inp = item.get("input")
        out = item.get("output")
        if code is None or inp is None or out is None:
            raise RuntimeError(
                "cruxeval: unexpected dataset schema (missing 'code'/'input'/'output'); "
                "refusing to fabricate sample data"
            )
        prompt = (
            "You are given a Python function and an input. Determine the exact output "
            "returned when the function is executed on that input.\n\n"
            f"```python\n{str(code).strip()}\n```\n\n"
            f"What does `f({str(inp).strip()})` return? "
            "Respond with only the literal Python value in the format:\n"
            "Final Answer: <output>"
        )
        gold = str(out).strip()
        messages = [{"role": "user", "content": prompt}]
        samples.append((messages, gold, {"category": "output_prediction"}))

    logger.info(f"Loaded {len(samples)} cruxeval samples.")
    return samples


def _extract_cruxeval_answer(response_text: str) -> Optional[str]:
    """The literal the model states as its answer, in the prompt's requested format.

    Order: \\boxed{} -> an explicit "Final Answer:"/"Output:"/"==>" anchor -> the last
    non-empty line. Markdown fencing/backticks are stripped, quotes are NOT: in CRUXEval
    the gold is a Python literal, so `'0'` (a string) and `0` (an int) are different
    answers and must not be conflated.
    """
    if not response_text:
        return None
    resp = response_text.strip()

    boxed = re.findall(r"\\boxed\{([^{}]+)\}", resp)
    if boxed:
        return boxed[-1].strip().strip("`").strip()

    anchored = re.findall(r"(?:Final Answer|Output|==>)\s*:?\s*(.+)", resp, re.IGNORECASE)
    if anchored:
        return anchored[-1].strip().strip("`").strip()

    lines = [line.strip() for line in resp.splitlines() if line.strip()]
    if not lines:
        return None
    last = lines[-1]
    if last.startswith("```"):                      # a bare fenced block: use its contents
        body = [line for line in lines if not line.startswith("```")]
        last = body[-1] if body else last
    return last.strip("`").strip()


def _eval_cruxeval(response_text: str, gold_target: str) -> bool:
    """Compare the model's stated output literal with the gold literal.

    Two leniencies are deliberately gone (audit 3A):
      * bare containment (`gold in resp`) credited any response that happened to include a
        short literal - golds such as `0`, `True` or `[]` occur in almost every response;
      * case-insensitive matching, which equates the Python literals `true` and `True`.
    Equality is either textual (exact, after fencing is stripped) or, when both sides parse
    as Python literals, structural - so `[1, 2]` and `[1,2]` agree but `'0'` and `0` do not.
    """
    gold = str(gold_target).strip()
    if not gold or not response_text:
        return False

    pred = _extract_cruxeval_answer(response_text)
    if pred is None:
        return False
    if pred == gold:
        return True

    try:
        gold_value = ast.literal_eval(gold)
        pred_value = ast.literal_eval(pred)
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return False
    # bool is an int subclass in Python: True must not equal 1.
    if isinstance(gold_value, bool) != isinstance(pred_value, bool):
        return False
    return type(gold_value) is type(pred_value) and gold_value == pred_value


def run_cruxeval(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Execute cruxeval native built-in evaluation suite."""
    samples = _load_cruxeval_samples(limit=kwargs.get("limit"))
    return run_eval_suite(
        eval_name="cruxeval",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_cruxeval,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens", 2048),
    )
