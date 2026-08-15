# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# Evaluation Suite: swe_bench_pro
# Description: SWE-bench Pro (ScaleAI) - resolved rate via the Pro-specific Docker harness

"""gbench native built-in runner for swe_bench_pro (Coding & Software Engineering).

Canonical SWE-bench Pro (ScaleAI/SWE-bench_Pro) scored by resolved-rate via the
Pro-specific Docker harness (scaleapi/SWE-bench_Pro-os `swe_bench_pro_eval.py` +
`jefzda/sweap-images`; vanilla swebench cannot score it). SANDBOX_EVAL. A real
run pulls tens-hundreds of GB of images, so it is gated behind an explicit opt-in
(SWE_BENCH_PRO_RUN=1) and skips cleanly otherwise.
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

logger = logging.getLogger(__name__)

PILLAR = "Coding & Software Engineering"
DOCS_URL = "docs/evals/swe_bench_pro.md"
_DATASET = "ScaleAI/SWE-bench_Pro"


def _harness_dir() -> Optional[str]:
    return os.getenv("SWE_BENCH_PRO_HARNESS_DIR")


#: Columns `swe_bench_pro_eval.py` reads off the raw-sample frame (lines 96-99, 556-557).
_RAW_SAMPLE_COLUMNS = ("instance_id", "base_commit", "before_repo_set_cmd",
                       "selected_test_files_to_run", "fail_to_pass", "pass_to_pass")


def raw_sample_path() -> Optional[str]:
    """Path to the harness' `--raw_sample_path` table, generating it if the clone lacks it.

    `scaleapi/SWE-bench_Pro-os` documents `swe_bench_pro_full.csv` in its README but does
    not ship it, so a fresh clone always failed the prerequisite check and the suite
    skipped. The file is not privileged data: every column the harness reads is a column
    of the canonical HF dataset (`ScaleAI/SWE-bench_Pro`, test split), already stored in
    the string-of-python-literal form its `eval()` calls expect. So gbench writes it once
    into the harness directory (or `SWE_BENCH_PRO_RAW_SAMPLE` if set, or a temp file when
    the clone is read-only) instead of asking the operator to find it.

    Returns None when neither the file exists nor the dataset can be read.
    """
    explicit = os.getenv("SWE_BENCH_PRO_RAW_SAMPLE")
    if explicit:
        return explicit if os.path.isfile(explicit) else None
    hd = _harness_dir()
    if not hd:
        return None
    shipped = os.path.join(hd, "swe_bench_pro_full.csv")
    if os.path.isfile(shipped):
        return shipped

    try:
        from datasets import load_dataset
        df = load_dataset(_DATASET, split="test").to_pandas()
    except Exception as e:
        logger.warning("swe_bench_pro: cannot build the raw-sample table from %s (%s)",
                       _DATASET, e)
        return None
    missing = [c for c in _RAW_SAMPLE_COLUMNS if c not in df.columns]
    if missing:
        logger.warning("swe_bench_pro: %s is missing the harness columns %s; the raw-sample "
                       "table cannot be generated.", _DATASET, missing)
        return None

    for target in (shipped, os.path.join(tempfile.gettempdir(), "swe_bench_pro_full.csv")):
        try:
            df.to_csv(target, index=False)
            logger.info("swe_bench_pro: wrote the raw-sample table (%d instances) to %s",
                        len(df), target)
            return target
        except OSError as e:
            logger.info("swe_bench_pro: could not write %s (%s)", target, e)
    return None


def check_swe_bench_pro_prerequisites() -> Tuple[bool, str]:
    """Docker + docker SDK + the Pro harness checkout + explicit opt-in (expensive)."""
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
    if not hd or not os.path.isfile(os.path.join(hd, "swe_bench_pro_eval.py")) \
            or not os.path.isdir(os.path.join(hd, "run_scripts")):
        return False, ("SWE-bench Pro harness not found: set SWE_BENCH_PRO_HARNESS_DIR to a clone of "
                       "scaleapi/SWE-bench_Pro-os (needs swe_bench_pro_eval.py and run_scripts/).")
    # The clone does not ship swe_bench_pro_full.csv; gbench builds it from the canonical
    # dataset. Only a genuinely unobtainable table is a prerequisite failure.
    if raw_sample_path() is None:
        return False, ("the harness raw-sample table is unavailable: "
                       f"{os.path.join(hd, 'swe_bench_pro_full.csv')} does not exist and it "
                       f"could not be generated from {_DATASET} (needs the dataset cached or "
                       "network access). Set SWE_BENCH_PRO_RAW_SAMPLE to point at your own copy.")
    if os.getenv("SWE_BENCH_PRO_RUN") != "1":
        return False, ("SWE-bench Pro is gated: it pulls tens-hundreds of GB of images per run. "
                       "Set SWE_BENCH_PRO_RUN=1 to enable.")
    return True, ""


def _extract_patch(text: str) -> str:
    t = strip_thinking_tags(text or "")
    m = re.search(r"```(?:diff|patch)?\s*([\s\S]*?)```", t)
    if m and "diff --git" in m.group(1):
        return m.group(1).strip() + "\n"
    idx = t.find("diff --git")
    return (t[idx:].strip() + "\n") if idx != -1 else ""


def _load_swe_bench_pro_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], Any, Dict[str, Any]]]:
    """Load SWE-bench Pro tasks; raises on load/schema failure. Tests are looked up by the harness."""
    try:
        from datasets import load_dataset
        ds = load_dataset(_DATASET, split="test")
        rows = list(ds)
    except Exception as e:
        logger.error(f"Failed to load dataset for swe_bench_pro: {e}")
        raise RuntimeError(f"Could not load dataset for swe_bench_pro: {e}") from e

    if not rows:
        raise RuntimeError("Dataset for swe_bench_pro returned empty rows")
    # Stratified, not a contiguous head (audit RC-1).
    rows = stratified_sample(rows, limit, lambda r: (r or {}).get("repo"), seed="swe_bench_pro")

    samples = []
    for item in rows:
        instance_id = item.get("instance_id")
        repo = item.get("repo")
        problem = item.get("problem_statement")
        if not instance_id or not problem:
            raise RuntimeError(
                "swe_bench_pro: unexpected schema (instance_id/problem_statement); "
                "refusing to fabricate sample data")
        parts = [f"Repository: {repo}", "", f"Issue:\n{problem}"]
        if item.get("requirements"):
            parts += ["", f"Requirements:\n{item['requirements']}"]
        if item.get("interface"):
            parts += ["", f"Interface:\n{item['interface']}"]
        parts += ["", "Output ONLY a single unified git diff (`diff --git a/... b/...`) that resolves "
                      "the issue, inside a ```diff code block."]
        messages = [{"role": "user", "content": "\n".join(parts)}]
        samples.append((messages, item.get("patch") or "",
                        {"category": str(item.get("repo_language") or repo), "instance_id": instance_id}))

    logger.info(f"Loaded {len(samples)} swe_bench_pro samples.")
    return samples


def _make_scorer(model_name: str, num_workers: int, metrics: Dict[str, Any]):
    async def _score(sample_traces: List[Dict[str, Any]]) -> None:
        import asyncio
        hd = _harness_dir()
        workdir = tempfile.mkdtemp(prefix="gbench_swepro_")
        out_dir = os.path.join(workdir, "out")
        os.makedirs(out_dir, exist_ok=True)
        preds_path = os.path.join(workdir, "preds.json")
        prefix = "gbench__" + re.sub(r"[^A-Za-z0-9_.-]", "_", model_name)[:40]

        preds = []
        for tr in sample_traces:
            iid = (tr.get("extra_payload") or {}).get("instance_id")
            if iid:
                preds.append({"instance_id": iid, "patch": _extract_patch(tr.get("response_text") or ""),
                              "prefix": prefix})
        with open(preds_path, "w", encoding="utf-8") as f:
            json.dump(preds, f)

        cmd = [
            "python", os.path.join(hd, "swe_bench_pro_eval.py"),
            f"--raw_sample_path={raw_sample_path()}",
            f"--patch_path={preds_path}", f"--output_dir={out_dir}",
            f"--scripts_dir={os.path.join(hd, 'run_scripts')}",
            f"--num_workers={max(1, num_workers)}",
            "--dockerhub_username=jefzda", "--use_local_docker",
        ]

        def _run():
            return subprocess.run(cmd, cwd=hd, capture_output=True, text=True)
        proc = await asyncio.to_thread(_run)

        results: Dict[str, bool] = {}
        report_path = os.path.join(out_dir, "eval_results.json")
        if os.path.isfile(report_path):
            with open(report_path, encoding="utf-8") as f:
                results = {k: bool(v) for k, v in json.load(f).items()}
            metrics["swe_bench_pro_report"] = {
                "total_instances": len(results),
                "resolved_instances": sum(results.values()),
            }
        else:
            logger.error("swe_bench_pro: eval_results.json not found. stderr tail: %s",
                         (proc.stderr or "")[-800:])
            metrics["swe_bench_pro_report"] = {"error": "harness report not produced"}

        for tr in sample_traces:
            iid = (tr.get("extra_payload") or {}).get("instance_id")
            tr["is_correct"] = bool(results.get(iid, False))
            tr["status"] = "OK"
    return _score


def run_swe_bench_pro(
    model_name: str,
    base_url: str,
    concurrency: int = 4,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run SWE-bench Pro resolved-rate (or skip if harness/Docker/opt-in absent)."""
    ok, reason = check_swe_bench_pro_prerequisites()
    if not ok:
        msg = f"[SKIP] swe_bench_pro skipped: {reason} See '{DOCS_URL}' for setup instructions."
        logger.warning(msg)
        print(f"\n{msg}")
        return {
            "benchmark_type": "eval",
            "eval_name": "swe_bench_pro",
            "model_name": model_name,
            "status": "skipped",
            "total_questions": 0,
            "correct_answers": 0,
            "accuracy": 0.0,
            "skip_reason": f"{reason} (See {DOCS_URL})",
        }
    samples = _load_swe_bench_pro_samples(limit=kwargs.get("limit"))
    metrics: Dict[str, Any] = {}
    result = run_eval_suite(
        eval_name="swe_bench_pro",
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
