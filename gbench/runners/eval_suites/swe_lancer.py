# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# Evaluation Suite: swe_lancer
# Description: SWE-Lancer (OpenAI) - real-world freelance SWE tasks, resolved via the official Docker harness

"""gbench native built-in runner for swe_lancer (Coding & Software Engineering).

Canonical SWE-Lancer (openai/SWELancer-Benchmark) scored by execution: the model's
patch is applied inside the task's Docker image and the hidden end-to-end
(Playwright/pytest) test suite is run - an IC-SWE task resolves iff those tests pass
(SWE-Manager tasks are scored by the correct proposal selection). Only the official
harness can score this; a substring/filename check cannot verify program behaviour,
so the previous heuristic scorer was removed. SANDBOX_EVAL. A real run pulls very
large per-task images, so it is gated behind an explicit opt-in (SWELANCER_RUN=1)
and skips cleanly otherwise.
"""

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Tuple
from .base import run_eval_suite, strip_thinking_tags
from .sampling import stratified_sample
from .swebench_common import skipped_result

logger = logging.getLogger(__name__)

PILLAR = "Coding & Software Engineering"
DOCS_URL = "docs/evals/swe_lancer.md"
_DATASET = "DCAgent2/swe-lancer"
# Default harness invocation; override with SWELANCER_EVAL_CMD (placeholders below).
_DEFAULT_EVAL_CMD = ("python {harness}/run_swelancer_eval.py "
                     "--predictions={predictions} --output_dir={output_dir} --num_workers={num_workers}")


def _harness_dir() -> Optional[str]:
    return os.getenv("SWELANCER_HARNESS_DIR")


def _eval_cmd_template() -> str:
    return os.getenv("SWELANCER_EVAL_CMD") or _DEFAULT_EVAL_CMD


def check_swe_lancer_prerequisites() -> Tuple[bool, str]:
    """Docker + docker SDK + the SWELancer harness checkout + explicit opt-in (expensive)."""
    if not shutil.which("docker"):
        return False, "Docker CLI is not found on PATH."
    try:
        if subprocess.run(["docker", "info"], capture_output=True, timeout=20).returncode != 0:
            return False, "Docker daemon is not reachable."
    except Exception as e:
        return False, f"Cannot connect to Docker daemon: {e}"
    try:
        import docker  # noqa: F401
    except ImportError:
        return False, "Python 'docker' SDK is not installed (pip install gbench[evals])."
    hd = _harness_dir()
    if not hd or not os.path.isdir(hd):
        return False, ("SWE-Lancer harness not found: set SWELANCER_HARNESS_DIR to a clone of "
                       "openai/SWELancer-Benchmark.")
    # The default runner script must exist unless the user supplies their own command.
    if not os.getenv("SWELANCER_EVAL_CMD") and not os.path.isfile(os.path.join(hd, "run_swelancer_eval.py")):
        return False, ("SWELancer runner not found: add run_swelancer_eval.py to the harness dir, or set "
                       "SWELANCER_EVAL_CMD to the command that scores predictions (see docs).")
    if os.getenv("SWELANCER_RUN") != "1":
        return False, ("SWE-Lancer is gated: it pulls very large per-task Docker images. "
                       "Set SWELANCER_RUN=1 to enable.")
    return True, ""


def _extract_patch(text: str) -> str:
    t = strip_thinking_tags(text or "")
    m = re.search(r"```(?:diff|patch)?\s*([\s\S]*?)```", t)
    if m and "diff --git" in m.group(1):
        return m.group(1).strip() + "\n"
    idx = t.find("diff --git")
    return (t[idx:].strip() + "\n") if idx != -1 else ""


def _load_swe_lancer_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Load SWE-Lancer task instructions (real instruction.md per task); raises on failure.

    The harness owns the tests + Docker images; gbench uses these only to build the
    prompt and to key each prediction by task_id.
    """
    samples: List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]] = []
    try:
        from huggingface_hub import HfApi, hf_hub_download
        files = HfApi().list_repo_files(_DATASET, repo_type="dataset")
        instructions = sorted(f for f in files if f.endswith("/instruction.md"))
        # Stratified, not a contiguous head (audit RC-1).
        instructions = stratified_sample(instructions, limit, None, seed="swe_lancer")
        for inst_file in instructions:
            task_id = inst_file.split("/")[0]
            local_inst = hf_hub_download(repo_id=_DATASET, filename=inst_file, repo_type="dataset")
            with open(local_inst, encoding="utf-8") as fp:
                instruction_text = fp.read().strip()
            if not task_id or not instruction_text:
                raise RuntimeError("swe_lancer: empty task_id/instruction; refusing to fabricate sample data")
            prompt = (
                f"[SWE-Lancer freelance task {task_id}]\n\n"
                f"{instruction_text}\n\n"
                "Resolve this task. Output a single unified git diff (`diff --git a/... b/...`) "
                "inside a ```diff code block."
            )
            samples.append(([{"role": "user", "content": prompt}], task_id,
                            {"category": "swe_lancer", "task_id": task_id}))
    except Exception as e:
        logger.error(f"Failed to load dataset for swe_lancer: {e}")
        raise RuntimeError(f"Could not load dataset for swe_lancer: {e}") from e

    if not samples:
        raise RuntimeError("Dataset for swe_lancer returned empty rows")
    logger.info(f"Loaded {len(samples)} swe_lancer samples.")
    return samples


def _parse_results(out_dir: str) -> Dict[str, bool]:
    """Tolerant parse of the harness report: {task_id: resolved} or {resolved_ids: [...]}."""
    for root, _, fs in os.walk(out_dir):
        for fn in fs:
            if fn.endswith(".json"):
                try:
                    with open(os.path.join(root, fn), encoding="utf-8") as f:
                        data = json.load(f)
                except Exception:
                    continue
                if isinstance(data, dict) and "resolved_ids" in data:
                    return {str(t): True for t in data.get("resolved_ids", [])}
                if isinstance(data, dict) and all(isinstance(v, bool) for v in data.values()) and data:
                    return {str(k): bool(v) for k, v in data.items()}
    return {}


def _make_scorer(num_workers: int, metrics: Dict[str, Any]):
    async def _score(sample_traces: List[Dict[str, Any]]) -> None:
        import asyncio
        hd = _harness_dir()
        workdir = tempfile.mkdtemp(prefix="gbench_swelancer_")
        out_dir = os.path.join(workdir, "out")
        os.makedirs(out_dir, exist_ok=True)
        preds_path = os.path.join(workdir, "predictions.jsonl")

        with open(preds_path, "w", encoding="utf-8") as f:
            for tr in sample_traces:
                tid = (tr.get("extra_payload") or {}).get("task_id")
                if tid:
                    f.write(json.dumps({"task_id": tid,
                                        "patch": _extract_patch(tr.get("response_text") or "")}) + "\n")

        cmd = _eval_cmd_template().format(
            harness=hd, predictions=preds_path, output_dir=out_dir, num_workers=max(1, num_workers))

        def _run():
            return subprocess.run(cmd, shell=True, cwd=hd, capture_output=True, text=True)
        proc = await asyncio.to_thread(_run)

        results = _parse_results(out_dir)
        if results:
            # Prompts come from the `DCAgent2/swe-lancer` mirror while the harness is a
            # clone of `openai/SWELancer-Benchmark`. If the two number their tasks
            # differently, every lookup below misses and the suite reports a clean 0%
            # that looks like a model failure. Fail loudly on zero overlap instead.
            sample_ids = {str((tr.get("extra_payload") or {}).get("task_id"))
                          for tr in sample_traces}
            overlap = sample_ids & {str(k) for k in results}
            if not overlap:
                logger.error(
                    "swe_lancer: the harness report shares NO task ids with the loaded "
                    "prompts (%d predictions vs %d scored ids). The %s mirror's ids do not "
                    "match this harness checkout; the run cannot be scored.",
                    len(sample_ids), len(results), _DATASET)
                metrics["swe_lancer_report"] = {
                    "error": "task_id mismatch between the prompt mirror and the harness",
                    "prompt_ids": len(sample_ids), "report_ids": len(results)}
                for tr in sample_traces:
                    tr["is_correct"] = False
                    tr["status"] = "FAILED"
                return
            metrics["swe_lancer_report"] = {"total_instances": len(sample_traces),
                                            "resolved_instances": sum(1 for v in results.values() if v),
                                            "matched_instances": len(overlap)}
        else:
            logger.error("swe_lancer: no harness report parsed. stderr tail: %s", (proc.stderr or "")[-800:])
            metrics["swe_lancer_report"] = {"error": "harness report not produced"}

        for tr in sample_traces:
            tid = (tr.get("extra_payload") or {}).get("task_id")
            tr["is_correct"] = bool(results.get(str(tid), False))
            tr["status"] = "OK"
    return _score


def run_swe_lancer(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run SWE-Lancer resolved-rate (or skip if harness/Docker/opt-in absent)."""
    ok, reason = check_swe_lancer_prerequisites()
    if not ok:
        return skipped_result("swe_lancer", model_name, reason, DOCS_URL)
    samples = _load_swe_lancer_samples(limit=kwargs.get("limit"))
    metrics: Dict[str, Any] = {}
    result = run_eval_suite(
        eval_name="swe_lancer",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        async_eval_fn=_make_scorer(concurrency, metrics),
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens", 8192),
        temperature=kwargs.get("temperature", 0.0),
    )
    result.update(metrics)
    return result
