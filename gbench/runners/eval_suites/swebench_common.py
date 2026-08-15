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

"""Shared SWE-bench execution-harness helpers.

Every SWE-bench-style suite (swe_bench_live, swe_bench_multilingual,
copilot_bench_swe, ...) resolves a GitHub issue by emitting one unified-diff
patch, then scores execution-based resolved-rate via the swebench Docker harness
(apply patch + run FAIL_TO_PASS/PASS_TO_PASS). This module holds the common
prereq check, dataset loader, patch extractor, and harness scorer so each suite
is a thin, consistent wrapper.
"""

import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple
from .base import run_eval_suite
from .sampling import stratified_sample

logger = logging.getLogger(__name__)


def check_swebench_prereq(
    dataset: str, split: str, namespace: Optional[str], requires_fork: bool = False
) -> Tuple[bool, str]:
    """datasets + swebench + reachable Docker, and the harness can build a TestSpec."""
    try:
        import datasets  # noqa: F401
    except ImportError:
        return False, "Python package 'datasets' is not installed."
    try:
        from swebench.harness.run_evaluation import load_swebench_dataset  # noqa: F401
        from swebench.harness.test_spec.test_spec import make_test_spec  # noqa: F401
    except ImportError:
        return False, "Python package 'swebench' is not installed (pip install gbench[evals])."
    try:
        r = subprocess.run(["docker", "info"], capture_output=True, timeout=20)
        if r.returncode != 0:
            return False, "Docker daemon is not reachable."
    except Exception:
        return False, "Docker CLI/daemon is not available."
    try:
        from swebench.harness.run_evaluation import load_swebench_dataset
        from swebench.harness.test_spec.test_spec import make_test_spec
        ds = load_swebench_dataset(dataset, split)
        try:
            make_test_spec(ds[0], namespace=namespace)
        except TypeError:
            make_test_spec(ds[0])
    except Exception as e:
        hint = ""
        if requires_fork:
            # The fork and upstream swebench are the SAME package name at different
            # versions, so only one can be installed at a time - and they support
            # different suites. Measured 2026-08-15 on this box:
            #     swebench 4.1.0 (upstream)   swe_bench_multilingual OK, swe_bench_live FAIL
            #     swebench 4.0.3 (Live fork)  swe_bench_live OK, swe_bench_multilingual FAIL
            #                                 (KeyError: 'parse_log_maven')
            # copilot_bench_swe works on both. Installing the fork into the main
            # environment therefore trades one suite for another, which is why this says
            # "separate virtualenv" rather than "pip install".
            hint = (" This suite needs the SWE-bench-Live harness fork "
                    "(github.com/SWE-bench-Live/SWE-bench-Live), which is the same "
                    "`swebench` package at an older version than the one upstream suites "
                    "need - installing it here would break swe_bench_multilingual. Install "
                    "it in a SEPARATE virtualenv and run this suite from there; see "
                    "docs/evals/swe_bench_live.md.")
        return False, (
            f"installed 'swebench' cannot build {dataset} test specs.{hint} "
            f"({type(e).__name__}: {str(e)[:100]})"
        )
    return True, ""


def extract_patch(text: str) -> str:
    """Extract a unified-diff patch from a model response."""
    if not text:
        return ""
    m = re.search(r"```(?:diff|patch)?\s*([\s\S]*?)```", text)
    if m and "diff --git" in m.group(1):
        return m.group(1).strip() + "\n"
    idx = text.find("diff --git")
    if idx != -1:
        return text[idx:].strip() + "\n"
    return ""


def load_swe_samples(
    dataset: str, split: str, limit: Optional[int], eval_name: str
) -> List[Tuple[List[Dict[str, Any]], Any, Dict[str, Any]]]:
    """Load a SWE-bench-schema dataset into (messages, gold='', extra) samples; raises on failure."""
    try:
        from datasets import load_dataset
        ds = load_dataset(dataset, split=split)
        rows = list(ds)
    except Exception as e:
        logger.error(f"Failed to load dataset for {eval_name}: {e}")
        raise RuntimeError(f"Could not load dataset for {eval_name}: {e}") from e

    if not rows:
        raise RuntimeError(f"{eval_name} returned empty rows")
    # Stratified, not a contiguous head (audit RC-1).
    rows = stratified_sample(rows, limit, lambda r: (r or {}).get("repo"), seed="swebench_common")

    samples = []
    for item in rows:
        repo = item.get("repo")
        instance_id = item.get("instance_id")
        problem = item.get("problem_statement")
        if not instance_id or not problem:
            raise RuntimeError(
                f"{eval_name}: unexpected dataset schema (instance_id/problem_statement); "
                "refusing to fabricate sample data"
            )
        prompt = (
            f"You are an expert software engineer resolving a GitHub issue in {repo}.\n\n"
            f"Issue ({instance_id}):\n{problem}\n\n"
            "Produce a single unified-diff patch in git format (`diff --git a/... b/...`) "
            "that resolves the issue. Output only the patch."
        )
        messages = [{"role": "user", "content": prompt}]
        samples.append((messages, "", {"category": str(repo), "instance_id": instance_id, "split": split}))

    logger.info(f"Loaded {len(samples)} {eval_name} samples from HF Hub ('{dataset}' {split}).")
    return samples


def make_swebench_scorer(
    eval_name: str, model_name: str, dataset: str, split: str,
    namespace: Optional[str], max_workers: int, metrics: Dict[str, Any],
):
    """Async scorer: run the swebench harness over the model patches, mark resolved."""
    async def _score(sample_traces: List[Dict[str, Any]]) -> None:
        import asyncio
        model_tag = "gbench__" + re.sub(r"[^A-Za-z0-9_.-]", "_", model_name)[:48]
        run_id = "gbench_" + re.sub(r"[^A-Za-z0-9_.-]", "_", f"{eval_name}_{model_name}")[:40]

        preds: Dict[str, Dict[str, str]] = {}
        for tr in sample_traces:
            iid = (tr.get("extra_payload") or {}).get("instance_id")
            if not iid:
                continue
            preds[iid] = {
                "instance_id": iid,
                "model_name_or_path": model_tag,
                "model_patch": extract_patch(tr.get("response_text") or ""),
            }

        workdir = tempfile.mkdtemp(prefix="gbench_swe_")
        preds_path = os.path.join(workdir, "preds.jsonl")
        with open(preds_path, "w", encoding="utf-8") as f:
            for p in preds.values():
                f.write(json.dumps(p) + "\n")

        cmd = [
            sys.executable, "-m", "swebench.harness.run_evaluation",
            "--dataset_name", dataset, "--split", split,
            "--predictions_path", preds_path, "--run_id", run_id,
            "--max_workers", str(max(1, max_workers)), "--cache_level", "env",
            "--instance_ids", *list(preds.keys()),
        ]
        if namespace:
            cmd += ["--namespace", namespace]

        def _run():
            return subprocess.run(cmd, cwd=workdir, capture_output=True, text=True)
        proc = await asyncio.to_thread(_run)

        report_path = os.path.join(workdir, f"{model_tag}.{run_id}.json")
        resolved: set = set()
        if os.path.exists(report_path):
            with open(report_path, encoding="utf-8") as f:
                rep = json.load(f)
            resolved = set(rep.get("resolved_ids", []))
            metrics["swebench_report"] = {
                k: rep.get(k) for k in (
                    "total_instances", "submitted_instances", "completed_instances",
                    "resolved_instances", "unresolved_instances", "error_instances",
                    "empty_patch_instances",
                )
            }
        else:
            logger.error(
                "%s: harness report not found (%s). stderr tail: %s",
                eval_name, report_path, (proc.stderr or "")[-800:],
            )
            metrics["swebench_report"] = {"error": "harness report not produced"}

        for tr in sample_traces:
            iid = (tr.get("extra_payload") or {}).get("instance_id")
            tr["is_correct"] = iid in resolved
            tr["status"] = "OK"
    return _score


def execute_swebench(
    eval_name: str, model_name: str, base_url: str, concurrency: int,
    dataset: str, split: str, namespace: Optional[str], **kwargs,
) -> Dict[str, Any]:
    """Load samples, generate patches, and score via the harness. Assumes prereqs pass."""
    samples = load_swe_samples(dataset, split, kwargs.get("limit"), eval_name)
    metrics: Dict[str, Any] = {}
    result = run_eval_suite(
        eval_name=eval_name,
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        async_eval_fn=make_swebench_scorer(
            eval_name, model_name, dataset, split, namespace, concurrency, metrics),
        thinking=kwargs.get("enable_thinking", False),
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens", 8192),
        temperature=kwargs.get("temperature", 0.0),
    )
    result.update(metrics)
    # CC6: if the harness never produced its report, every instance was marked
    # unresolved - that is an error, not a 0% score.
    if isinstance(metrics.get('swebench_report'), dict) and metrics['swebench_report'].get('error'):
        result['status'] = 'error'
        result['error'] = metrics['swebench_report']['error']
    return result


def skipped_result(eval_name: str, model_name: str, reason: str, docs_url: str) -> Dict[str, Any]:
    """Standard skip dict."""
    msg = f"[SKIP] {eval_name} skipped: {reason} See '{docs_url}' for setup instructions."
    logger.warning(msg)
    print(f"\n{msg}")
    return {
        "benchmark_type": "eval",
        "eval_name": eval_name,
        "model_name": model_name,
        "status": "skipped",
        "total_questions": 0,
        "correct_answers": 0,
        "accuracy": 0.0,
        "skip_reason": f"{reason} (See {docs_url})",
    }
