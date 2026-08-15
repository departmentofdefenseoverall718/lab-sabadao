# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# Evaluation Suite: swe_bench_live
# Description: SWE-bench-Live - continuously-updated real GitHub issue resolution (resolved rate)

"""gbench native built-in runner for swe_bench_live (Coding & Software Engineering).

Canonical SWE-bench-Live scored by the SWE-bench-Live fork of the swebench Docker
harness (per-instance DockerHub images under namespace 'starryzhang'). Thin wrapper
over swebench_common; SANDBOX_EVAL. Skips cleanly if the fork/Docker are absent
(vanilla swebench cannot build SWE-bench-Live test specs).
"""

from typing import Any, Dict, Optional
from .swebench_common import (
    check_swebench_prereq, load_swe_samples, execute_swebench, skipped_result,
)

PILLAR = "Coding & Software Engineering"
DOCS_URL = "docs/evals/swe_bench_live.md"
_DATASET = "SWE-bench-Live/SWE-bench-Live"
_NAMESPACE = "starryzhang"
_DEFAULT_SPLIT = "lite"


def check_swe_bench_live_prerequisites():
    return check_swebench_prereq(_DATASET, _DEFAULT_SPLIT, _NAMESPACE, requires_fork=True)


def _load_swe_bench_live_samples(limit: Optional[int] = None, split: str = _DEFAULT_SPLIT):
    return load_swe_samples(_DATASET, split, limit, "swe_bench_live")


def run_swe_bench_live(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run SWE-bench-Live (or skip if the fork/Docker are unavailable)."""
    ok, reason = check_swe_bench_live_prerequisites()
    if not ok:
        return skipped_result("swe_bench_live", model_name, reason, DOCS_URL)
    kwargs["enable_thinking"] = enable_thinking
    return execute_swebench(
        "swe_bench_live", model_name, base_url, concurrency,
        _DATASET, kwargs.get("split", _DEFAULT_SPLIT), _NAMESPACE, **kwargs,
    )
