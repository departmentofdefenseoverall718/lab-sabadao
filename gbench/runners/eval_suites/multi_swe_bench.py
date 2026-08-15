# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# Evaluation Suite: multi_swe_bench
# Description: Multi-SWE-bench (ByteDance, 7 languages) - resolved rate via its own harness

"""gbench native built-in runner for multi_swe_bench (Coding & Software Engineering).

Canonical Multi-SWE-bench (ByteDance-Seed/Multi-SWE-bench) scored by execution-based
resolved-rate via the project's OWN harness (`multi_swe_bench.harness.run_evaluation`;
vanilla swebench cannot score it - different prediction schema and test fields).
SANDBOX_EVAL. Skips cleanly if the harness/Docker are absent.
"""

import importlib.util
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple
from .base import run_eval_suite, strip_thinking_tags

logger = logging.getLogger(__name__)

PILLAR = "Coding & Software Engineering"
DOCS_URL = "docs/evals/multi_swe_bench.md"
_REPO = "ByteDance-Seed/Multi-SWE-bench"
_SRC_FILES: Dict[str, str] = {}  # instance_id -> local dataset jsonl path (for the scorer)


def check_multi_swe_bench_prerequisites() -> Tuple[bool, str]:
    if not shutil.which("docker"):
        return False, "Docker CLI is not found on PATH."
    try:
        if subprocess.run(["docker", "info"], capture_output=True, timeout=20).returncode != 0:
            return False, "Docker daemon is not reachable."
    except Exception as e:
        return False, f"Cannot connect to Docker daemon: {e}"
    try:
        # find_spec raises (not returns None) when the PARENT package is absent.
        if importlib.util.find_spec("multi_swe_bench.harness.run_evaluation") is None:
            raise ModuleNotFoundError
    except ModuleNotFoundError:
        return False, "Python package 'multi_swe_bench' is not installed (pip install multi-swe-bench)."
    return True, ""


def _extract_patch(text: str) -> str:
    t = strip_thinking_tags(text or "")
    m = re.search(r"```(?:diff|patch)?\s*([\s\S]*?)```", t)
    if m and "diff --git" in m.group(1):
        return m.group(1).strip() + "\n"
    idx = t.find("diff --git")
    return (t[idx:].strip() + "\n") if idx != -1 else ""


def _load_multi_swe_bench_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], Any, Dict[str, Any]]]:
    """Load Multi-SWE-bench instances via list_repo_files (load_dataset is broken for this repo)."""
    global _SRC_FILES
    _SRC_FILES = {}
    try:
        from huggingface_hub import HfApi, hf_hub_download
        files = HfApi().list_repo_files(_REPO, repo_type="dataset")
        jsonl_files = sorted(f for f in files if f.endswith("_dataset.jsonl"))
    except Exception as e:
        logger.error(f"Failed to list dataset for multi_swe_bench: {e}")
        raise RuntimeError(f"Could not load dataset for multi_swe_bench: {e}") from e

    samples: List[Tuple[List[Dict[str, Any]], Any, Dict[str, Any]]] = []
    for jf in jsonl_files:
        lang = jf.split("/")[0] if "/" in jf else "unknown"
        local_p = hf_hub_download(repo_id=_REPO, filename=jf, repo_type="dataset")
        with open(local_p, encoding="utf-8") as fp:
            for line in fp:
                if not line.strip():
                    continue
                item = json.loads(line)
                org, repo, number = item.get("org"), item.get("repo"), item.get("number")
                if not org or not repo or number is None:
                    raise RuntimeError(
                        "multi_swe_bench: unexpected schema (org/repo/number); "
                        "refusing to fabricate sample data"
                    )
                instance_id = item.get("instance_id") or f"{org}__{repo}-{number}"
                _SRC_FILES[instance_id] = local_p
                title = str(item.get("title") or "").strip()
                body = str(item.get("body") or "").strip()
                prompt = (
                    f"Repository: {org}/{repo} ({lang})\n"
                    f"Base commit: {(item.get('base') or {}).get('sha', '')}\n\n"
                    f"Issue: {title}\n\n{body}\n\n"
                    "Output ONLY a unified git diff (`diff --git a/... b/...`), rooted at the "
                    "repository top, that resolves the issue."
                )
                samples.append(([{"role": "user", "content": prompt}],
                                item.get("fix_patch") or "",
                                {"category": lang, "instance_id": instance_id,
                                 "org": org, "repo": repo, "number": number}))
                if limit and len(samples) >= limit:
                    break
        if limit and len(samples) >= limit:
            break

    if not samples:
        raise RuntimeError("multi_swe_bench returned empty rows")
    logger.info(f"Loaded {len(samples)} multi_swe_bench samples.")
    return samples


def _make_scorer(model_name: str, max_workers: int, metrics: Dict[str, Any]):
    async def _score(sample_traces: List[Dict[str, Any]]) -> None:
        import asyncio
        workdir = tempfile.mkdtemp(prefix="gbench_mswe_")
        preds_path = os.path.join(workdir, "preds.jsonl")
        subset_path = os.path.join(workdir, "dataset.jsonl")
        out_dir = os.path.join(workdir, "out")
        os.makedirs(out_dir, exist_ok=True)

        # predictions: {org, repo, number, fix_patch}
        wanted = set()
        with open(preds_path, "w", encoding="utf-8") as pf:
            for tr in sample_traces:
                e = tr.get("extra_payload") or {}
                iid = e.get("instance_id")
                if not iid:
                    continue
                wanted.add(iid)
                pf.write(json.dumps({"org": e["org"], "repo": e["repo"],
                                     "number": e["number"],
                                     "fix_patch": _extract_patch(tr.get("response_text") or "")}) + "\n")

        # dataset subset: pull the exact rows we ran from the cached source files
        seen_files = set(_SRC_FILES.get(i) for i in wanted if _SRC_FILES.get(i))
        with open(subset_path, "w", encoding="utf-8") as sf:
            for src in seen_files:
                for line in open(src, encoding="utf-8"):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    iid = row.get("instance_id") or f"{row.get('org')}__{row.get('repo')}-{row.get('number')}"
                    if iid in wanted:
                        sf.write(line)

        config = {
            "mode": "evaluation", "workdir": workdir, "output_dir": out_dir,
            "log_dir": os.path.join(workdir, "logs"),
            "patch_files": [preds_path], "dataset_files": [subset_path],
            "force_build": False, "need_clone": True, "clear_env": True,
            "max_workers": max(1, max_workers),
        }
        cfg_path = os.path.join(workdir, "config.json")
        with open(cfg_path, "w", encoding="utf-8") as cf:
            json.dump(config, cf)

        def _run():
            return subprocess.run(
                [sys.executable, "-m", "multi_swe_bench.harness.run_evaluation", "--config", cfg_path],
                cwd=workdir, capture_output=True, text=True)
        proc = await asyncio.to_thread(_run)

        resolved_ids: List[str] = []
        report = None
        for root, _, fs in os.walk(workdir):
            if "final_report.json" in fs:
                with open(os.path.join(root, "final_report.json"), encoding="utf-8") as f:
                    report = json.load(f)
                resolved_ids = report.get("resolved_ids", [])
                break
        if report is None:
            logger.error("multi_swe_bench: final_report.json not found. stderr tail: %s",
                         (proc.stderr or "")[-800:])
            metrics["multi_swe_bench_report"] = {"error": "harness report not produced"}
        else:
            metrics["multi_swe_bench_report"] = {
                k: report.get(k) for k in ("total_instances", "resolved_instances",
                                           "unresolved_instances", "error_instances",
                                           "empty_patch_instances")
            }

        def _resolved(e: Dict[str, Any]) -> bool:
            if e.get("instance_id") in resolved_ids:
                return True
            # spelling of ids varies; fall back to matching repo + number
            rep, num = str(e.get("repo")), str(e.get("number"))
            return any(rep in str(r) and num in str(r) for r in resolved_ids)

        for tr in sample_traces:
            tr["is_correct"] = _resolved(tr.get("extra_payload") or {})
            tr["status"] = "OK"
    return _score


def run_multi_swe_bench(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run Multi-SWE-bench resolved-rate (or skip if the harness/Docker are unavailable)."""
    ok, reason = check_multi_swe_bench_prerequisites()
    if not ok:
        msg = f"[SKIP] multi_swe_bench skipped: {reason} See '{DOCS_URL}' for setup instructions."
        logger.warning(msg)
        print(f"\n{msg}")
        return {
            "benchmark_type": "eval",
            "eval_name": "multi_swe_bench",
            "model_name": model_name,
            "status": "skipped",
            "total_questions": 0,
            "correct_answers": 0,
            "accuracy": 0.0,
            "skip_reason": f"{reason} (See {DOCS_URL})",
        }
    samples = _load_multi_swe_bench_samples(limit=kwargs.get("limit"))
    metrics: Dict[str, Any] = {}
    result = run_eval_suite(
        eval_name="multi_swe_bench",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        async_eval_fn=_make_scorer(model_name, concurrency, metrics),
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens", 8192),
        temperature=kwargs.get("temperature", 0.0),
    )
    result.update(metrics)
    return result
