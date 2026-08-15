# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# Evaluation Suite: gaia
# Description: GAIA (General AI Assistants) - canonical question_scorer exact-match

"""gbench native built-in runner for gaia (Agentic & Assistant Tasks).

Canonical GAIA (gaia-benchmark/GAIA, 2023 validation) with the official
`question_scorer` (number/list/string normalized exact-match on the FINAL ANSWER).
The gated dataset requires HF auth (skips cleanly without it). NOTE: gbench runs
GAIA closed-book (no tool/web loop, file attachments not passed), so scores are
expected to be low - this measures scorer fidelity, not agentic capability.
"""

import logging
import os
import re
import string
from typing import Any, Dict, List, Optional, Tuple
from .search_tool import (WEB_SEARCH_TOOL, execute_tool, search_available,
                          search_backend_name, unavailable_reason)
from .swebench_common import skipped_result
from .base import run_eval_suite
from .sampling import stratified_sample

logger = logging.getLogger(__name__)

PILLAR = "Agentic & Assistant Tasks"
DOCS_URL = "docs/evals/gaia.md"

# Verbatim canonical GAIA system prompt (paper / leaderboard).
_GAIA_SYSTEM = (
    "You are a general AI assistant. I will ask you a question. Report your thoughts, "
    "and finish your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER]. "
    "YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated "
    "list of numbers and/or strings. If you are asked for a number, don't use comma to write "
    "your number neither use units such as $ or percent sign unless specified otherwise. If "
    "you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), "
    "and write the digits in plain text unless specified otherwise. If you are asked for a "
    "comma separated list, apply the above rules depending of whether the element to be put in "
    "the list is a number or a string."
)


# ---- Official GAIA question_scorer (huggingface.co/spaces/gaia-benchmark/leaderboard) ----
def _is_float(x: Any) -> bool:
    try:
        float(x)
        return True
    except (ValueError, TypeError):
        return False


def _normalize_number_str(s: str) -> float:
    for ch in ["$", "%", ","]:
        s = s.replace(ch, "")
    try:
        return float(s)
    except ValueError:
        return float("inf")


def _split_string(s: str, char_list: Optional[List[str]] = None) -> List[str]:
    char_list = char_list or [",", ";"]
    return re.split("[" + "".join(char_list) + "]", s)


def _normalize_str(s: str, remove_punct: bool = True) -> str:
    no_spaces = re.sub(r"\s", "", s)
    if remove_punct:
        return no_spaces.lower().translate(str.maketrans("", "", string.punctuation))
    return no_spaces.lower()


def _question_scorer(model_answer: str, ground_truth: str) -> bool:
    gt, ma = str(ground_truth), str(model_answer)
    if _is_float(gt):
        return _normalize_number_str(ma) == float(gt)
    if any(c in gt for c in [",", ";"]):
        gt_elems, ma_elems = _split_string(gt), _split_string(ma)
        if len(gt_elems) != len(ma_elems):
            return False
        for g, m in zip(gt_elems, ma_elems):
            if _is_float(g):
                if _normalize_number_str(m) != float(g):
                    return False
            elif _normalize_str(m, remove_punct=False) != _normalize_str(g, remove_punct=False):
                return False
        return True
    return _normalize_str(ma) == _normalize_str(gt)


_FINAL_ANSWER_RE = re.compile(r"FINAL\s+ANSWER\s*:?\s*(.*)", re.IGNORECASE)


def _extract_final_answer(text: str) -> str:
    """Text after the last `FINAL ANSWER` marker.

    The old version sliced a fixed `len("FINAL ANSWER: ")` == 14 characters past the
    marker, so any spacing other than exactly `FINAL ANSWER: ` mis-cut the answer:
    `FINAL ANSWER:42` lost its first two characters and `FINAL ANSWER:  42` kept a
    stray space. It also took the FIRST marker, not the model's last word on it.
    """
    matches = _FINAL_ANSWER_RE.findall(text or "")
    if not matches:
        return ""
    return matches[-1].strip().splitlines()[0].strip() if matches[-1].strip() else ""


def _eval_gaia(response_text: str, gold_target: Any) -> bool:
    ans = _extract_final_answer(response_text or "")
    if not ans:
        return False
    return _question_scorer(ans, gold_target)


def _hf_token() -> Optional[str]:
    tok = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not tok:
        try:
            from huggingface_hub import get_token
            tok = get_token()
        except Exception:
            pass
    return tok


def check_gaia_prerequisites() -> Tuple[bool, str]:
    """datasets/huggingface_hub importable + HF token granting access to the gated repo."""
    try:
        import datasets  # noqa: F401
        import huggingface_hub  # noqa: F401
    except ImportError:
        return False, "Python packages 'datasets'/'huggingface_hub' are not installed."
    tok = _hf_token()
    gate_msg = ("gaia-benchmark/GAIA is gated: set HF_TOKEN and accept the license at "
                "https://huggingface.co/datasets/gaia-benchmark/GAIA")
    if not tok:
        return False, gate_msg
    try:
        from huggingface_hub import HfApi
        HfApi().dataset_info("gaia-benchmark/GAIA", token=tok)
    except Exception as e:
        return False, f"{gate_msg} ({type(e).__name__})"
    return True, ""


def _load_gaia_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], Any, Dict[str, Any]]]:
    """Load GAIA 2023 validation from HF Hub (gated); raises on load/schema failure."""
    tok = _hf_token()
    try:
        from datasets import load_dataset
        try:
            from huggingface_hub import snapshot_download
            data_dir = snapshot_download(
                repo_id="gaia-benchmark/GAIA", repo_type="dataset", token=tok)
            ds = load_dataset(data_dir, "2023_all", split="validation")
        except Exception:
            ds = load_dataset("gaia-benchmark/GAIA", "2023_all", split="validation", token=tok)
        rows = list(ds)
    except Exception as e:
        logger.error(f"Failed to load dataset for gaia: {e}")
        raise RuntimeError(f"Could not load dataset for gaia: {e}") from e

    if not rows:
        raise RuntimeError("Dataset for gaia returned empty rows")
    # Stratified, not a contiguous head (audit RC-1).
    rows = stratified_sample(rows, limit, lambda r: (r or {}).get("Level"), seed="gaia")

    samples = []
    for item in rows:
        question = item.get("Question")
        gold = item.get("Final answer")  # exact key: capital F, space
        if not question or gold is None:
            raise RuntimeError(
                "gaia: unexpected dataset schema (missing 'Question'/'Final answer'); "
                "refusing to fabricate sample data"
            )
        messages = [
            {"role": "system", "content": _GAIA_SYSTEM},
            {"role": "user", "content": str(question)},
        ]
        samples.append((messages, str(gold), {
            "category": f"Level_{item.get('Level', '1')}",
            "has_file": bool(item.get("file_name")),
        }))

    logger.info(
        f"Loaded {len(samples)} gaia samples (closed-book; ~half need file attachments "
        "not passed here, so scores are expected to be low - this checks scorer fidelity)."
    )
    return samples


def run_gaia(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run GAIA with the canonical question_scorer (skips without HF gated access)."""
    ok, reason = check_gaia_prerequisites()
    if not ok:
        msg = f"[SKIP] gaia skipped: {reason} See '{DOCS_URL}' for setup instructions."
        logger.warning(msg)
        print(f"\n{msg}")
        return {
            "benchmark_type": "eval",
            "eval_name": "gaia",
            "model_name": model_name,
            "status": "skipped",
            "total_questions": 0,
            "correct_answers": 0,
            "accuracy": 0.0,
            "skip_reason": f"{reason} (See {DOCS_URL})",
        }
    samples = _load_gaia_samples(limit=kwargs.get("limit"))
    # gaia is a web-research benchmark: without a lookup tool the model answers "I do
    # not have access to the internet" and the suite scores a structural 0 (measured 0/20
    # on 2026-08-15). Offer the search tool and let the model drive it.
    if not search_available():
        return skipped_result("gaia", model_name, unavailable_reason(), "docs/evals/gaia.md")
    extra_payload = dict(kwargs.get("extra_payload") or {})
    extra_payload.setdefault("tools", [WEB_SEARCH_TOOL])

    result = run_eval_suite(
        eval_name="gaia",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_gaia,
        thinking=enable_thinking,
        extra_payload=extra_payload,
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens"),
        tool_executor=execute_tool,
    )
    # Search-only, not the browsing agent the public leaderboard uses: say so on the result
    # so the number is never read as leaderboard-comparable.
    result["search_backend"] = search_backend_name()
    result["leaderboard_comparable"] = False
    return result
