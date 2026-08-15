# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# Evaluation Suite: gdpval
# Description: GDPval (OpenAI Real Economic Knowledge-Work Deliverables & GDPval-AA Benchmark)

"""gbench native built-in runner for gdpval (Economic & Knowledge Work)."""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from .swebench_common import skipped_result

logger = logging.getLogger(__name__)

PILLAR = "Economic & Knowledge Work"
DOCS_URL = "docs/evals/gdpval.md"


def _load_gdpval_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Load gdpval benchmark dataset directly from HF Hub (openai/gdpval)."""
    rows = []
    try:
        from datasets import load_dataset
        ds = load_dataset('openai/gdpval', split='train')
        rows = list(ds)
    except Exception as e:
        logger.error(f"Failed to load dataset for gdpval: {e}")
        raise RuntimeError(f"Could not load dataset for gdpval: {e}") from e

    if not rows:
        raise RuntimeError("Dataset for gdpval returned empty rows")

    # Stratified, not a contiguous head (audit RC-1).
    rows = stratified_sample(rows, limit, lambda r: (r or {}).get("occupation"), seed="gdpval")

    samples = []
    for item in rows:
        prompt = str(item.get("prompt") or "").strip()
        gold = str(item.get("rubric_pretty") or json.dumps(item.get("rubric_json") or [])).strip()
        cat = str(item.get("occupation") or item.get("sector") or "knowledge_work")

        messages = [{"role": "user", "content": prompt}]
        samples.append((messages, gold, {"category": cat}))

    logger.info(f"Loaded {len(samples)} gdpval samples.")
    return samples


def _eval_gdpval(response_text: str, gold_target: str) -> bool:
    """Evaluate candidate response against canonical GDPval deliverable rubric criteria."""
    if not response_text or not gold_target:
        return False

    resp = response_text.strip()
    resp_lower = resp.lower()
    gold = str(gold_target).strip()

    if not gold:
        return False

    # Extract distinct criteria / sentences from the rubric
    rubric_points = [p.strip() for p in re.split(r"[\n;•\-]+", gold) if len(p.strip()) > 10]
    if not rubric_points:
        return False

    # Every criterion must have its core semantic terms satisfied
    satisfied_count = 0
    for pt in rubric_points:
        key_terms = [
            w.lower()
            for w in re.findall(r"\b[a-zA-Z0-9_\-]{4,}\b", pt)
            if w.lower() not in ("should", "must", "include", "provide", "ensure", "deliverable", "criteria", "analysis", "table", "section", "report")
        ]
        if not key_terms:
            continue
        term_matches = sum(1 for t in key_terms if t in resp_lower)
        if (term_matches / len(key_terms)) >= 0.70:
            satisfied_count += 1

    # Must satisfy at least 75% of distinct rubric criteria
    return (satisfied_count / len(rubric_points)) >= 0.75


def run_gdpval(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Execute gdpval native built-in evaluation suite.

    GDPval grades produced *file deliverables* (.xlsx / .pptx / .docx, etc.) against
    file-property rubrics. A plain text /v1 endpoint cannot produce deliverables, and the
    deterministic rubric check cannot meaningfully grade prose, so scoring against a text
    endpoint yields a misleading ~0%. gbench therefore reports a clean skip until a
    file-producing agent harness + rubric LLM judge is wired in. The canonical loader and
    rubric parser remain (`_load_gdpval_samples`, `_eval_gdpval`) for that future harness.
    """
    return skipped_result(
        "gdpval", model_name,
        "GDPval grades produced file deliverables (.xlsx/.pptx/.docx) against file-property "
        "rubrics; a plain text /v1 endpoint cannot produce them and the deterministic rubric "
        "check cannot grade prose. Requires a file-producing agent harness + rubric LLM judge",
        DOCS_URL)
