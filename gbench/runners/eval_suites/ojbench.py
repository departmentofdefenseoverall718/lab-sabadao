# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# Evaluation Suite: ojbench
# Description: OJBench (NOI/ICPC) - online-judge Pass@1 via the official ojbench + DMOJ sandbox

"""gbench native built-in runner for ojbench (Code & Competitive Programming).

Canonical OJBench (He-Ren/OJBench_testdata): the model emits a full stdin/stdout
solution; correctness is online-judge Pass@1 (Accepted iff ALL testcases pass
within per-problem time/memory limits), computed by the official `ojbench` library
over the DMOJ sandbox. SANDBOX_EVAL. Skips cleanly unless ojbench + DMOJ
judge-server + PyPy3 + g++ + the testdata (OJBENCH_TESTDATA) are all present.
"""

import importlib.util
import json
import logging
import os
import shutil
from typing import Any, Dict, List, Optional, Tuple
from .base import run_eval_suite, strip_thinking_tags
from .sampling import stratified_sample

logger = logging.getLogger(__name__)

PILLAR = "Coding & Software Engineering"
DOCS_URL = "docs/evals/ojbench.md"
_INITIALIZED = False


def _testdata_dir() -> Optional[str]:
    return os.getenv("OJBENCH_TESTDATA")


def check_ojbench_prerequisites() -> Tuple[bool, str]:
    """ojbench + DMOJ judge-server + PyPy3 + g++ + the NOI/ICPC testdata."""
    if importlib.util.find_spec("ojbench") is None:
        return False, "Python package 'ojbench' is not installed (git clone He-Ren/OJBench; pip install -e)."
    if importlib.util.find_spec("dmoj") is None:
        return False, "DMOJ judge-server is not installed (git clone DMOJ/judge-server @f098cd3; pip install)."
    if not shutil.which("pypy3"):
        return False, "PyPy3 is not on PATH (required to run Python-language solutions)."
    if not shutil.which("g++"):
        return False, "g++ is not on PATH (required to compile C++ solutions)."
    td = _testdata_dir()
    if not td or not os.path.isdir(os.path.join(td, "NOI")) or not os.path.isdir(os.path.join(td, "ICPC")):
        return False, ("OJBench testdata not found: set OJBENCH_TESTDATA to a snapshot of "
                       "He-Ren/OJBench_testdata containing NOI/ and ICPC/ (7.85 GB).")
    return True, ""


def _load_ojbench_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], Any, Dict[str, Any]]]:
    """Load OJBench prompts (full.jsonl); prompt sent verbatim; raises on load/schema failure."""
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id="He-Ren/OJBench_testdata",
                               filename="prompts/full.jsonl", repo_type="dataset")
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    except Exception as e:
        logger.error(f"Failed to load dataset for ojbench: {e}")
        raise RuntimeError(f"Could not load dataset for ojbench: {e}") from e

    if not rows:
        raise RuntimeError("Dataset for ojbench returned empty rows")
    # Stratified, not a contiguous head (audit RC-1).
    rows = stratified_sample(rows, limit, None, seed="ojbench")

    samples = []
    for item in rows:
        pid = item.get("id")
        prompt = item.get("prompt")
        lang = item.get("language")
        dataset = item.get("dataset")
        difficulty = item.get("difficulty")
        if pid is None or not prompt or not lang:
            raise RuntimeError(
                "ojbench: unexpected schema (id/prompt/language); refusing to fabricate sample data")
        # Canonical: the prompt already embeds the response-format constraint; send verbatim.
        messages = [{"role": "user", "content": str(prompt)}]
        samples.append((messages, pid, {
            "category": f"{dataset}_{difficulty}",
            "row": {"id": pid, "dataset": dataset, "language": lang, "difficulty": difficulty},
        }))

    logger.info(f"Loaded {len(samples)} ojbench samples.")
    return samples


def _make_scorer(num_workers: int):
    async def _score(sample_traces: List[Dict[str, Any]]) -> None:
        import asyncio
        global _INITIALIZED
        import ojbench  # lazy (gated by prereq)
        from pathlib import Path

        td = _testdata_dir()
        if not _INITIALIZED:
            ojbench.init(problem_dirs=[Path(td) / "NOI", Path(td) / "ICPC"])
            _INITIALIZED = True

        records = []
        for tr in sample_traces:
            resp = tr.get("response_text")
            row = (tr.get("extra_payload") or {}).get("row") or {}
            if resp and row.get("id") is not None:
                records.append({**row, "content": strip_thinking_tags(resp)})

        def _judge():
            return ojbench.judge_jsonl_data(records, num_workers=max(1, num_workers))
        results = await asyncio.to_thread(_judge)

        by_key = {(r.get("id"), r.get("language")): bool(r.get("is_passed"))
                  for r in (results or [])}
        for tr in sample_traces:
            row = (tr.get("extra_payload") or {}).get("row") or {}
            tr["is_correct"] = by_key.get((row.get("id"), row.get("language")), False)
            tr["status"] = "OK"
    return _score


def run_ojbench(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run OJBench online-judge Pass@1 (or skip if the judge/testdata are unavailable)."""
    ok, reason = check_ojbench_prerequisites()
    if not ok:
        msg = f"[SKIP] ojbench skipped: {reason} See '{DOCS_URL}' for setup instructions."
        logger.warning(msg)
        print(f"\n{msg}")
        return {
            "benchmark_type": "eval",
            "eval_name": "ojbench",
            "model_name": model_name,
            "status": "skipped",
            "total_questions": 0,
            "correct_answers": 0,
            "accuracy": 0.0,
            "skip_reason": f"{reason} (See {DOCS_URL})",
        }
    samples = _load_ojbench_samples(limit=kwargs.get("limit"))
    return run_eval_suite(
        eval_name="ojbench",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        async_eval_fn=_make_scorer(concurrency),
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens", 8192),
        temperature=kwargs.get("temperature", 0.0),
    )
