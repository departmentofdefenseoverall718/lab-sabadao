# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Native Putnam Mathematical Competition evaluation suite with LLM-as-a-Judge for pure proofs."""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from .base import run_eval_suite, DEFAULT_JUDGE_MODEL, parse_grade_verdict
from .sampling import limit_dataset

logger = logging.getLogger(__name__)


def check_putnam_prerequisites(api_key: Optional[str] = None) -> Tuple[bool, str]:
    """Validate prerequisites for Putnam evaluation with LLM proof judge."""
    try:
        import google.genai  # noqa: F401
    except ImportError:
        return False, "Package 'google-genai' is not installed. Install via: pip install google-genai"

    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        return False, "GEMINI_API_KEY is not set. Provide via --gemini-api-key or export GEMINI_API_KEY='...'"

    return True, ""





def _load_putnam_samples(
    enable_thinking: bool = False,
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Load Putnam samples directly from canonical HF Hub dataset ('amitayusht/PutnamBench')."""
    from datasets import load_dataset

    ds = load_dataset("amitayusht/PutnamBench", split="train")
    # Stratified, not a contiguous head (audit RC-1).
    ds = limit_dataset(ds, limit, 'tags', seed="putnam")
    raw_samples = list(ds)
    logger.info(f"Loaded {len(raw_samples)} Putnam samples from HF Hub ('amitayusht/PutnamBench').")

    samples = []
    for item in raw_samples:
        q_text = item.get("informal_statement") or item.get("question", "")
        gold_ans = str(item.get("informal_solution") or item.get("answer", "") or "").strip()
        tags = item.get("tags")
        if isinstance(tags, str) and tags.startswith("["):
            import ast
            try:
                tags = ast.literal_eval(tags)
            except Exception:
                pass
        if isinstance(tags, list) and len(tags) > 0:
            category = str(tags[0]).strip("'\"")
        elif isinstance(tags, str):
            category = tags.strip("'\"[] ")
        else:
            category = str(item.get("category", "proof"))

        if enable_thinking:
            prompt = (
                f"Putnam Competition Problem:\n{q_text}\n\n"
                "Think step by step to construct a complete mathematical proof, and state your final answer on the last line "
                "in the format: 'Final Answer: <answer>'."
            )
        else:
            prompt = (
                f"Putnam Competition Problem:\n{q_text}\n\n"
                "Solve the mathematical problem and state your final answer on the last line "
                "in the format: 'Final Answer: <answer>'."
            )
        messages = [{"role": "user", "content": prompt}]
        samples.append((messages, gold_ans, {"category": category}))

    return samples


def _normalize_putnam_math(s: str) -> str:
    """Normalize LaTeX fractions, sqrt, and delimiters for robust comparison."""
    text = s.replace("$", "").strip().lower()
    text = re.sub(r"\\?frac\{([^}]+)\}\{([^}]+)\}", r"\1/\2", text)
    text = re.sub(r"\\?sqrt\{([^}]+)\}", r"sqrt(\1)", text)
    text = text.replace("\\", "").strip()
    # Collapse (but keep) whitespace: word boundaries are needed to tell the `2` of an
    # answer from the `2` inside `12`. A trailing sentence period is not part of the value.
    text = re.sub(r"\s+", " ", text).strip().rstrip(".,;").strip()
    return text


def _spaceless(s: str) -> str:
    """Layout must not decide correctness: `12 - 8sqrt(2)` == `12-8sqrt(2)`."""
    return re.sub(r"\s+", "", s or "")


# Alphabetic tokens allowed inside a bare math answer; anything else means prose.
_MATH_WORDS = {"sqrt", "frac", "pi", "log", "ln", "exp", "sin", "cos", "tan", "inf",
               "infty", "mod", "max", "min", "sum", "prod", "lim", "cdot", "times", "e"}


def _gold_closed_form(gold_answer: str) -> Optional[str]:
    """The single math expression a PutnamBench `informal_solution` asserts, if it has one.

    The golds are prose that embeds the answer ("The minimum is $12 - 8\\sqrt{2}$."), not a
    bare value. Of the 271 non-empty golds in the set, 130 contain exactly one `$...$`
    span, which is unambiguously the answer. The other 141 either contain none ("The limit
    does not exist.") or several (a case split, a parameterised family) and cannot be
    reduced to one value - those are for the judge, not for string comparison.

    A gold that is already a bare value (`100`, `1/2`) is returned as-is.
    """
    text = str(gold_answer or "").strip()
    if not text:
        return None
    spans = re.findall(r"\$([^$]+)\$", text)
    if len(spans) == 1:
        return spans[0].strip()
    if spans:
        return None                      # several values: a case split, not one answer
    words = [w.lower() for w in re.findall(r"[A-Za-z]{2,}", text)]
    if any(w not in _MATH_WORDS for w in words):
        return None                      # prose ("The limit does not exist.")
    return text


def _pred_closed_form(response_text: str) -> Tuple[str, bool]:
    """The model's stated answer and whether it was explicitly anchored.

    `\\boxed{}` or "Final Answer:" means the model committed to a value, so it is held to
    it exactly. With no anchor there is only a concluding sentence to search.
    """
    text = str(response_text or "").strip()
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        return boxed[-1].strip(), True
    anchored = re.findall(r"Final Answer\s*:?\s*(.+)$", text, re.IGNORECASE | re.MULTILINE)
    if anchored:
        return anchored[-1].strip(), True
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return (lines[-1] if lines else ""), False


def _states_answer(pred: str, target: str) -> bool:
    """Is `target` the answer `pred` states? Equality, or a boundary-delimited occurrence.

    The occurrence check exists only for unanchored responses. It is boundary-delimited so
    a gold of `2` is not found inside `12`; the reverse direction (the prediction being a
    substring of the reference) is never allowed, because that is what credited a bare `2`
    against "The minimum is $12 - 8\\sqrt{2}$.".
    """
    if not target:
        return False
    if _spaceless(pred) == _spaceless(target):
        return True
    for m in re.finditer(re.escape(target), pred):
        before = pred[m.start() - 1] if m.start() else " "
        after = pred[m.end()] if m.end() < len(pred) else " "
        if not (before.isalnum() or before in "._") and not (after.isalnum() or after in "._"):
            return True
    return False


def _eval_putnam(response_text: str, gold_answer: str) -> bool:
    """Deterministic scoring for the golds that state a single closed-form answer.

    The previous comparison was bidirectional containment of the whole gold *sentence*
    (`norm_gold in norm_pred or norm_pred in norm_gold`), which passes any short answer
    that happens to be a substring of the reference prose: a prediction of `2` was scored
    correct against "The minimum is $12 - 8\\sqrt{2}$." Both sides are now reduced to the
    answer expression and compared for equality (audit 3A).
    """
    if not response_text:
        return False

    gold = str(gold_answer).strip()
    if not gold:
        # A pure-proof problem: nothing to match against, so this is not a scoring
        # authority. The judge decides; reaching here means no judge was available.
        return False

    target = _gold_closed_form(gold)
    if target is None:
        return False        # not reducible to one value -> judge territory, fail closed

    pred, anchored = _pred_closed_form(response_text)
    # The model may wrap its value in prose ("The limit equals $1/8$."). Reduce the
    # prediction the same way the gold was reduced, so both sides are comparable.
    pred = _gold_closed_form(pred) or pred
    norm_pred = _normalize_putnam_math(pred)
    norm_gold = _normalize_putnam_math(target)
    if not norm_gold:
        return False
    if anchored:
        return _spaceless(norm_pred) == _spaceless(norm_gold)
    return _states_answer(norm_pred, norm_gold)


async def _async_judge_putnam(
    sample_traces: List[Dict[str, Any]],
    judge_model: str = DEFAULT_JUDGE_MODEL,
    judge_concurrency: int = 64,
) -> None:
    """Canonical Putnam proof judging executed in parallel post-generation."""
    from tqdm import tqdm
    import asyncio

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        for trace in sample_traces:
            resp = str(trace.get("response_text") or "").strip()
            gold = str(trace.get("gold_answer") or "").strip()
            trace["is_correct"] = _eval_putnam(resp, gold)
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
        gold = str(trace.get("gold_answer") or "").strip()

        # Only golds that state a single closed-form value are decided by string equality.
        # A gold such as "The limit does not exist." or a case-split answer cannot be
        # compared that way; those go to the judge WITH the reference, which is what the
        # suite documents ("LLM judge vs reference") and what it previously never did -
        # every non-empty gold took the deterministic path and every judged sample had an
        # empty reference, so the judge always graded blind.
        if gold and _gold_closed_form(gold) is not None:
            trace["is_correct"] = _eval_putnam(str(resp_text or ""), gold)
            trace["status"] = "OK"
            pbar.update(1)
            return

        if not resp_text:
            trace["is_correct"] = False
            trace["status"] = "FAILED"
            pbar.update(1)
            return

        messages = trace.get("messages", [])
        problem_statement = messages[0].get("content", "") if messages else ""

        reference = (f"Reference Solution (the official answer):\n{gold}\n\n"
                     "The candidate need not match the reference's wording, but its "
                     "conclusion must agree with it.\n\n") if gold else ""
        prompt = (
            "You are an expert mathematician and Putnam Competition grader.\n\n"
            f"Problem Statement:\n{problem_statement}\n\n"
            f"{reference}"
            f"Candidate Mathematical Proof:\n{resp_text}\n\n"
            "Evaluate if the proof is mathematically rigorous, sound, and fully completes the proof.\n"
            "Respond strictly in the format:\n"
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

    with tqdm(total=len(sample_traces), desc="Judging [PUTNAM]") as pbar:
        tasks = [_judge_single(trace, pbar) for trace in sample_traces]
        await asyncio.gather(*tasks)


def run_putnam(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run native Putnam Mathematical Competition evaluation suite with requirements check."""
    gemini_key = kwargs.get("gemini_api_key") or os.environ.get("GEMINI_API_KEY")
    ok, reason = check_putnam_prerequisites(gemini_key)
    if not ok:
        msg = f"[SKIP] putnam skipped: {reason}"
        logger.warning(msg)
        return {
            "benchmark_type": "eval",
            "eval_name": "putnam",
            "model_name": model_name,
            "thinking": enable_thinking,
            "status": "skipped",
            "skip_reason": reason,
        }

    samples = _load_putnam_samples(enable_thinking=enable_thinking, limit=kwargs.get("limit"))
    return run_eval_suite(
        eval_name="putnam",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_putnam,
        async_eval_fn=_async_judge_putnam,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens", 8192),
    )
