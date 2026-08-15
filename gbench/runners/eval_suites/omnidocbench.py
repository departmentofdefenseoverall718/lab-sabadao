# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# Evaluation Suite: omnidocbench
# Description: OmniDocBench v1.5 (Multimodal Document Parsing, Layout & Formula Recognition)

"""gbench native built-in runner for omnidocbench (Multimodal & Vision)."""

import base64
import csv
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from huggingface_hub import hf_hub_download
from .base import run_eval_suite
from .dataset_utils import extract_lossless_image_b64

logger = logging.getLogger(__name__)

PILLAR = "Multimodal & Vision"


def _load_omnidocbench_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Load OmniDocBench benchmark dataset directly from HF Hub (opendatalab/OmniDocBench)."""
    rows = []
    try:
        from datasets import load_dataset, Image
        ds = load_dataset("opendatalab/OmniDocBench", split="train")
        json_file = hf_hub_download(repo_id="opendatalab/OmniDocBench", filename="OmniDocBench.json", repo_type="dataset")
        with open(json_file) as f:
            annotations = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load dataset for omnidocbench: {e}")
        raise RuntimeError(f"Could not load dataset for omnidocbench: {e}") from e

    if not ds:
        raise RuntimeError("Dataset for omnidocbench returned empty rows")

    # Pair image <-> annotation by page filename. Indexing both by position assumes the
    # HF split and OmniDocBench.json enumerate the pages in the same order; when they do
    # not, every page is scored against a different page's ground truth.
    anno_by_name = {}
    for anno in annotations:
        name = str((anno.get("page_info") or {}).get("image_path") or "").strip()
        if name:
            anno_by_name[os.path.basename(name)] = anno

    total_len = min(len(ds), len(annotations))
    if limit is not None and limit > 0:
        total_len = min(total_len, limit)

    samples = []
    unpaired = 0
    for i in range(total_len):
        item = ds[i]
        page_name = os.path.basename(str(item.get("image_path") or item.get("file_name")
                                         or item.get("page_name") or "").strip())
        anno = anno_by_name.get(page_name)
        if anno is None:
            unpaired += 1
            anno = annotations[i] if i < len(annotations) else {}
        image_val = item.get("image")

        layout_dets = anno.get("layout_dets", [])
        page_info = anno.get("page_info", {})
        page_attr = page_info.get("page_attribute", {})
        if isinstance(page_attr, dict):
            doc_type = str(page_attr.get("subset") or page_attr.get("data_source") or "unknown")
        else:
            doc_type = str(page_attr or "unknown")

        # A page's ground truth is not text alone: tables are annotated as `html` and
        # formulas as `latex`. Keeping only `text` asked the model to transcribe the whole
        # page and then graded it against a reference with the tables and equations
        # removed, so a correct table cost the page its score.
        parts: List[str] = []
        for det in layout_dets:
            value = det.get("text") or det.get("html") or det.get("latex")
            if value:
                parts.append(str(value))
        gt_text = "\n".join(parts)

        prompt = (
            "[Document Parsing Task]\n"
            "Carefully transcribe all text, tables, and mathematical formulas from this document image in clean markdown format."
        )
        content_payload: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        b64_str = extract_lossless_image_b64(image_val)
        if b64_str:
            content_payload.insert(0, {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64_str}"}
            })

        messages = [{"role": "user", "content": content_payload}]
        samples.append((messages, gt_text, {"category": doc_type}))

    if unpaired:
        logger.warning("[omnidocbench] %d/%d pages could not be paired with an annotation "
                       "by filename and fell back to positional pairing; their ground "
                       "truth may belong to a different page.", unpaired, len(samples))
    logger.info(f"Loaded {len(samples)} omnidocbench samples with lossless direct images.")
    return samples


def _normalize_doc_text(text: str) -> str:
    """OmniDocBench text normalization before edit distance.

    Whitespace, markdown emphasis and heading markers are transcription style, not
    transcription accuracy, so they are removed on both sides.
    """
    t = str(text or "")
    t = re.sub(r"!?\[[^\]]*\]\([^)]*\)", " ", t)          # images/links -> their absence
    t = re.sub(r"[*_`#>|]+", " ", t)                       # markdown emphasis/heading/table pipes
    t = re.sub(r"\s+", " ", t)
    return t.strip().lower()


def normalized_edit_distance(pred: str, gold: str) -> float:
    """Character-level normalized edit distance in [0, 1]; 0 is a perfect transcription.

    This is OmniDocBench's canonical text metric. `rapidfuzz` is used when present (C++,
    linear memory); `difflib` is the fallback so the suite still runs without it, and the
    substitution is recorded on the result rather than hidden.
    """
    p, g = _normalize_doc_text(pred), _normalize_doc_text(gold)
    if not g:
        return 0.0 if not p else 1.0
    try:
        from rapidfuzz.distance import Levenshtein
        return float(Levenshtein.normalized_distance(p, g))
    except ImportError:
        import difflib
        return 1.0 - difflib.SequenceMatcher(None, p, g, autojunk=False).ratio()


def _eval_omnidocbench(response_text: str, gold_target: str) -> bool:
    """Strict per-page pass: a transcription within 10% edit distance of the annotation.

    The headline metric of the suite is the *continuous* mean edit distance (see
    `run_omnidocbench`); this boolean only feeds the harness' pass count. The previous
    scorer passed a page when 40% of the ground truth's 3+ character word types appeared
    anywhere in the response, in any order - a page of the right document with more than
    half its content missing scored as a correct transcription.
    """
    if not response_text or not str(gold_target).strip():
        return False
    return normalized_edit_distance(response_text, str(gold_target)) <= 0.10


def run_omnidocbench(
    model_name: str,
    base_url: str,
    limit: Optional[int] = None,
    concurrency: int = 4,
    enable_thinking: bool = False,
    results_dir: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Run native OmniDocBench evaluation benchmark.

    Headline metric is the canonical **mean normalized edit distance** (lower is better),
    not a pass rate: document parsing is a continuous transcription task, and a threshold
    hides the difference between a near-perfect page and an empty one. `accuracy` carries
    the derived similarity (100 - mean edit distance x 100) so the field stays
    interpretable, and the strict pass rate is kept alongside it.
    """
    samples = _load_omnidocbench_samples(limit=limit)
    result = run_eval_suite(
        eval_name="omnidocbench",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_omnidocbench,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens", 2048),
    )

    distances = []
    for trace in result.get("sample_traces", []):
        gold = str(trace.get("gold_answer") or "")
        response = trace.get("response_text")
        if not gold or response is None:
            continue
        d = normalized_edit_distance(response, gold)
        trace["edit_distance"] = round(d, 4)
        distances.append(d)

    result["strict_pass_rate"] = result.get("accuracy")
    result["metric"] = "mean normalized edit distance (canonical; lower is better)"
    try:
        import rapidfuzz  # noqa: F401
        result["edit_distance_backend"] = "rapidfuzz"
    except ImportError:
        result["edit_distance_backend"] = "difflib (rapidfuzz not installed - approximate)"
    if distances:
        mean_distance = sum(distances) / len(distances)
        result["mean_edit_distance"] = round(mean_distance, 4)
        result["accuracy"] = round((1.0 - mean_distance) * 100.0, 2)
    return result
