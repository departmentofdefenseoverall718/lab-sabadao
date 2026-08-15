# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# Evaluation Suite: scicode
# Description: SciCode (Scientific Python Problem Solving & Numerical Code Execution Benchmark)

"""gbench native built-in runner for scicode (Coding & Algorithmic)."""

import csv
import json
import logging
import os
import re
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Tuple
from .base import run_eval_suite
from .sampling import stratified_sample
from .sandbox import run_sandboxed
from .swebench_common import skipped_result

logger = logging.getLogger(__name__)

DOCS_URL = "docs/evals/scicode.md"

PILLAR = "Coding & Algorithmic"


def _load_scicode_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Load SciCode benchmark dataset directly from HF Hub (SciCode1/SciCode)."""
    rows = []
    try:
        from datasets import load_dataset
        ds = load_dataset("SciCode1/SciCode", split="test")
        rows = list(ds)
    except Exception as e:
        logger.error(f"Failed to load dataset for scicode: {e}")
        raise RuntimeError(f"Could not load dataset for scicode: {e}") from e

    if not rows:
        raise RuntimeError("Dataset for scicode returned empty rows")

    # Stratified, not a contiguous head (audit RC-1).
    rows = stratified_sample(rows, limit, None, seed="scicode")

    samples = []
    for item in rows:
        problem_name = str(item.get("problem_name") or "Scientific Problem")
        prob_desc = str(item.get("problem_description_main") or "").strip()
        bg_desc = str(item.get("problem_background_main") or "").strip()
        sub_steps = item.get("sub_steps") or []
        general_tests = item.get("general_tests") or []
        gold_sol = str(item.get("general_solution") or "").strip()

        step_desc = ""
        if sub_steps and isinstance(sub_steps, list):
            step_desc = "\n".join([f"- Step {i+1}: {s.get('step_description_prompt', '')}" for i, s in enumerate(sub_steps) if isinstance(s, dict)])

        test_code_str = "\n".join(general_tests) if isinstance(general_tests, list) else str(general_tests)

        prompt = (
            f"[Scientific Python Coding Problem: {problem_name}]\n"
            f"Background:\n{bg_desc}\n\n"
            f"Problem Description:\n{prob_desc}\n\n"
            f"{step_desc}\n\n"
            "Write complete, working Python functions with all necessary imports (numpy, scipy, etc.) to solve this scientific problem.\n"
            "Enclose the complete Python code within a ```python ... ``` code block."
        )
        messages = [{"role": "user", "content": prompt}]
        samples.append((messages, test_code_str, {"category": "scientific_python", "name": problem_name, "gold_sol": gold_sol}))

    logger.info(f"Loaded {len(samples)} scicode samples.")
    return samples


#: Third-party mirror of SciCode's reference-output HDF5. The canonical source is a Google
#: Drive folder linked from the SciCode README, which is not scriptable without gdown/auth,
#: and the HF dataset (`SciCode1/SciCode`) ships only the problem JSONLs. Override with
#: SCICODE_TEST_DATA (a local path) or SCICODE_TEST_DATA_REPO (a different mirror).
_TEST_DATA_REPO = os.environ.get("SCICODE_TEST_DATA_REPO", "Srimadh/Scicode-test-data-h5")
_TEST_DATA_FILE = "test_data.h5"


def resolve_test_data() -> Tuple[Optional[str], Optional[str]]:
    """Path to SciCode's `test_data.h5`, fetching it if needed. Returns (path, provenance).

    Every SciCode test binds its expected values as `target` from this file
    (`process_hdf5_to_tuple`); without it every test raises NameError and even a perfect
    solution scores 0 - a structural zero, not a model result. It is ~1 GB and is fetched
    at eval time like any other dataset, then cached by huggingface_hub.
    """
    local = os.environ.get("SCICODE_TEST_DATA")
    if local:
        return (local, "SCICODE_TEST_DATA") if os.path.exists(local) else (None, None)
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=_TEST_DATA_REPO, filename=_TEST_DATA_FILE,
                               repo_type="dataset")
    except Exception as e:
        logger.warning("scicode: could not fetch %s/%s (%s)", _TEST_DATA_REPO,
                       _TEST_DATA_FILE, e)
        return None, None
    # The canonical file lives on Google Drive, so this mirror is unverified against it.
    # Say so, and record the digest, rather than let an unchecked artefact decide a score.
    logger.warning(
        "scicode: using the third-party mirror %s for %s. The canonical copy is the Google "
        "Drive folder linked from the SciCode README; verify the digest before quoting a "
        "headline number. Set SCICODE_TEST_DATA to use your own copy.",
        _TEST_DATA_REPO, _TEST_DATA_FILE)
    return path, f"hf:{_TEST_DATA_REPO}/{_TEST_DATA_FILE}"


def _digest(path: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _eval_scicode(response_text: str, gold_target: str) -> bool:
    """Evaluate Python code response by executing unit tests in Python subprocess."""
    if not response_text:
        return False

    resp = response_text.strip()
    test_code = str(gold_target).strip()

    # Extract code from python block
    code_match = re.search(r"```python\s*(.*?)\s*```", resp, re.DOTALL)
    if code_match:
        code_to_run = code_match.group(1)
    elif "def " in resp:
        code_to_run = resp
    else:
        return False

    if not test_code:
        # No tests -> nothing was verified. Crediting "def "/"return" would be a fake pass.
        return False

    full_script = f"{code_to_run}\n\n{test_code}\n"

    try:
        res = run_sandboxed(
            [sys.executable, "-c", full_script],
            capture_output=True,
            text=True,
            timeout=8,
        )
        if res.returncode == 0:
            return True
    except Exception:
        pass

    # Fallback to structural code similarity
    return False


def run_scicode(
    model_name: str,
    base_url: str,
    limit: Optional[int] = None,
    concurrency: int = 4,
    enable_thinking: bool = False,
    results_dir: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Run native SciCode evaluation benchmark.

    SciCode's `general_tests` compare against a bare `target` variable that the canonical
    harness binds from the per-problem HDF5 reference outputs (process_hdf5_to_tuple over
    SciCode's test_data.h5). Without that file every test raises NameError, so a perfect
    solution still scores 0 - a structural zero, not a model result. Require the reference
    data before reporting a number.
    """
    h5, provenance = resolve_test_data()
    if not h5:
        return skipped_result(
            "scicode", model_name,
            "SciCode tests evaluate against per-problem reference outputs bound as `target` "
            f"from the benchmark's test_data HDF5, and it could not be fetched from "
            f"{_TEST_DATA_REPO}. Set SCICODE_TEST_DATA to a local copy (the canonical file "
            "is the Google Drive folder linked from the SciCode README). Without it every "
            "test errors and the suite would report a structural 0%",
            DOCS_URL)
    os.environ["SCICODE_TEST_DATA"] = h5      # the scorer subprocess reads it from the env

    samples = _load_scicode_samples(limit=limit)
    result = run_eval_suite(
        eval_name="scicode",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_scicode,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens"),
    )
    # Provenance of the reference outputs decides every score in this suite, so it is part
    # of the result rather than a log line.
    result["test_data"] = {"path": h5, "source": provenance, "sha256": _digest(h5)}
    return result
