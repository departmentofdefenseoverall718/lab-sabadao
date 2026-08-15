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

"""Native Codeforces competitive programming evaluation suite."""

import io
import logging
import re
import sys
from typing import Any, Dict, List, Optional, Tuple
from .base import run_eval_suite
from .sandbox import run_sandboxed

logger = logging.getLogger(__name__)

DEFAULT_CODEFORCES_BENCHMARK_LIMIT = 500


def _load_codeforces_samples(enable_thinking: bool = False, limit: Optional[int] = None) -> List[Tuple[List[Dict[str, Any]], Any, Dict[str, Any]]]:
    """Load Codeforces competitive programming samples covering CF, ICPC, and IOI contests."""
    from datasets import load_dataset
    ds = load_dataset("open-r1/codeforces-cots", split="train")
    effective_limit = limit if limit is not None else DEFAULT_CODEFORCES_BENCHMARK_LIMIT

    target_cf = max(1, int(effective_limit * 0.40))
    target_icpc = max(1, int(effective_limit * 0.30))
    target_ioi = max(1, effective_limit - target_cf - target_icpc)

    cf_samples = []
    icpc_samples = []
    ioi_samples = []

    for item in ds:
        ctype = str(item.get("contest_type", "CF")).upper()
        if ctype == "CF" and len(cf_samples) < target_cf:
            cf_samples.append(item)
        elif ctype == "ICPC" and len(icpc_samples) < target_icpc:
            icpc_samples.append(item)
        elif ctype == "IOI" and len(ioi_samples) < target_ioi:
            ioi_samples.append(item)

        if len(cf_samples) >= target_cf and len(icpc_samples) >= target_icpc and len(ioi_samples) >= target_ioi:
            break

    raw_samples = (cf_samples + icpc_samples + ioi_samples)[:effective_limit]
    logger.info(f"Loaded {len(raw_samples)} canonical Codeforces competition samples (CF={len(cf_samples)}, ICPC={len(icpc_samples)}, IOI={len(ioi_samples)}).")

    samples = []
    for item in raw_samples:
        title = item.get("title", "")
        desc = item.get("description") or item.get("question", "")
        in_fmt = item.get("input_format", "")
        out_fmt = item.get("output_format", "")
        examples = item.get("examples") or item.get("tests", [])
        category = str(item.get("contest_type") or item.get("category", "competitive"))

        prompt = f"Problem: {title}\n\n{desc}\n\n"
        if in_fmt:
            prompt += f"Input Format:\n{in_fmt}\n\n"
        if out_fmt:
            prompt += f"Output Format:\n{out_fmt}\n\n"
        prompt += (
            "Write a complete, optimized Python 3 solution that reads inputs from standard input (stdin) "
            "and prints the solution to standard output (stdout). "
            "Provide your solution code in a single markdown ```python ... ``` code block."
        )
        messages = [{"role": "user", "content": prompt}]
        samples.append((messages, examples, {"category": category}))
    return samples


def _eval_codeforces(response_text: str, gold_tests: Any) -> bool:
    """Check if predicted Python code passes stdin/stdout test cases with strict subprocess timeout."""
    import subprocess

    code_match = re.search(r"```(?:python)?\s*\n(.*?)\n```", response_text, re.DOTALL)
    if code_match:
        code = code_match.group(1)
    else:
        code = response_text

    if not isinstance(gold_tests, list) or len(gold_tests) == 0:
        # No tests means the submission was never validated. Returning True made every
        # untestable problem a free pass and inflated the suite's accuracy with rows that
        # measured nothing.
        return False

    for test_case in gold_tests:
        if not isinstance(test_case, dict):
            continue
        in_str = str(test_case.get("input") or test_case.get("stdin", ""))
        expected_out = str(test_case.get("output") or test_case.get("stdout", "")).strip()

        try:
            res = run_sandboxed(
                [sys.executable, "-c", code],
                input=in_str,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if res.returncode != 0:
                return False
            actual_out = res.stdout.strip()
            if actual_out != expected_out:
                return False
        except (subprocess.TimeoutExpired, Exception):
            return False

    return True


def run_codeforces(model_name: str, base_url: str, concurrency: int, enable_thinking: bool = False, **kwargs) -> Dict[str, Any]:
    """Run native Codeforces competitive programming evaluation suite."""
    samples = _load_codeforces_samples(enable_thinking=enable_thinking, limit=kwargs.get("limit"))
    return run_eval_suite(
        eval_name="codeforces",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_codeforces,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens"),
    )
