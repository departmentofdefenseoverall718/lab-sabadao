# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# Evaluation Suite: nestful
# Description: NESTFUL (IBM Nested Output-to-Input Function Calling Benchmark)

"""gbench native built-in runner for nestful (Tool Use & Function Calling)."""

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


#: NESTFUL declares parameters as a bare {name: {description, type}} map with a non-JSON
#: type vocabulary ("int or float"). Convert to a JSON-Schema `function` object so the
#: endpoint accepts it; an unparseable schema is rejected and the whole request fails.
_TYPE_MAP = {"int": "integer", "float": "number", "int or float": "number",
             "str": "string", "string": "string", "bool": "boolean",
             "list": "array", "dict": "object"}


def _openai_tool(fn: Dict[str, Any]) -> Dict[str, Any]:
    props, required = {}, []
    for pname, spec in (fn.get("parameters") or {}).items():
        spec = spec if isinstance(spec, dict) else {}
        raw = str(spec.get("type", "string")).strip().lower()
        props[pname] = {"type": _TYPE_MAP.get(raw, "string"),
                        "description": str(spec.get("description", ""))}
        required.append(pname)
    return {
        "name": fn.get("name", ""),
        "description": str(fn.get("description", "")),
        "parameters": {"type": "object", "properties": props, "required": required},
    }


def _load_nestful_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Load nestful benchmark dataset directly from HF Hub (ibm/nestful)."""
    rows = []
    try:
        from datasets import load_dataset
        ds = load_dataset('ibm-research/nestful', split='train')
        rows = list(ds)
    except Exception as e:
        logger.error(f"Failed to load dataset for nestful: {e}")
        raise RuntimeError(f"Could not load dataset for nestful: {e}") from e

    if not rows:
        raise RuntimeError(f"Dataset for nestful returned empty rows")

    # Stratified, not a contiguous head (audit RC-1).
    rows = stratified_sample(rows, limit, None, seed="nestful")

    samples = []
    for item in rows:
        prompt = str(item.get("input") or item.get("prompt") or "").strip()
        # NESTFUL scores the SEQUENCE OF NESTED API CALLS (`output`), not the final value.
        # `gold_answer` is the arithmetic result ("40.0"); scoring the call-matching
        # evaluator against that could only ever return 0, which is what the 2026-08-15
        # sweep measured (0/20 on every row).
        gold = item.get("output")
        if isinstance(gold, str):
            gold = gold.strip()
        if not gold:
            continue
        cat = "math_planning"

        # The tool schemas MUST be offered: NESTFUL asks the model to compose calls to
        # these specific functions. Without them the model has no way to know `divide`,
        # `multiply`, ... exist and just does the arithmetic in prose.
        raw_tools = item.get("tools")
        if isinstance(raw_tools, str):
            try:
                raw_tools = json.loads(raw_tools)
            except Exception:
                raw_tools = []
        tools = [{"type": "function", "function": _openai_tool(fn)}
                 for fn in (raw_tools or []) if isinstance(fn, dict)]

        messages = [{"role": "user", "content": prompt}]
        meta = {"category": cat, "final_answer": item.get("gold_answer")}
        if tools:
            meta["tools"] = tools
        samples.append((messages, gold if isinstance(gold, str) else json.dumps(gold), meta))

    logger.info(f"Loaded {len(samples)} nestful samples.")
    return samples


def _eval_nestful(response_text: str, gold_target: str) -> bool:
    """Evaluate candidate response against gold target."""
    if not response_text:
        return False

    resp = response_text.strip()
    gold = str(gold_target).strip()

    if not gold:
        return False

    # NESTFUL golds are a SEQUENCE of (possibly nested) API calls. Compare structurally:
    # every gold call must appear with a matching name and arguments. The old path credited
    # bare containment of the gold string, a boxed answer, or any nearby number - none of
    # which show the model actually produced the call sequence.
    from .fc_common import parse_gold_call, parse_tool_calls, call_matches
    import json as _json

    try:
        gold_obj = _json.loads(gold)
    except Exception:
        gold_obj = gold

    gold_calls = gold_obj if isinstance(gold_obj, list) else [gold_obj]
    parsed_gold = [g for g in (parse_gold_call(x) for x in gold_calls) if g]
    if not parsed_gold:
        return False
    candidates = parse_tool_calls(resp)
    if not candidates:
        return False
    return all(call_matches(g, candidates, require_args=bool(g[1])) for g in parsed_gold)


def run_nestful(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Execute nestful native built-in evaluation suite."""
    samples = _load_nestful_samples(limit=kwargs.get("limit"))
    return run_eval_suite(
        eval_name="nestful",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_nestful,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens", 4096),
    )
