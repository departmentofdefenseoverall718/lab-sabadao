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

"""Native LiveCodeBench algorithmic coding evaluation suite."""

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm
from .base import run_eval_suite
from .sampling import limit_dataset
from .sandbox import run_sandboxed

logger = logging.getLogger(__name__)

def _load_lcb_samples(enable_thinking: bool = False, limit: Optional[int] = None) -> List[Tuple[List[Dict[str, Any]], Any, Dict[str, Any]]]:
    """Load LiveCodeBench samples covering Code Generation (public + private tests), Code Execution, and Test Generation."""
    import json
    from datasets import load_dataset

    samples = []

    # 1. Code Generation (with both public and private test cases)
    ds_gen = load_dataset("livecodebench/code_generation", split="test")
    # Stratified by difficulty, not a contiguous head (audit RC-1): the split is
    # ordered, so a head skewed the easy/medium/hard mix.
    ds_gen = limit_dataset(ds_gen, limit, "difficulty", seed="lcb")
    raw_gen = list(ds_gen)
    for item in raw_gen:
        title = item.get("question_title", "")
        content = item.get("question_content") or item.get("question", "")
        starter = item.get("starter_code", "")
        difficulty = item.get("difficulty", "coding")

        # Collect both public and private test cases
        all_tests = []
        for test_field in ["public_test_cases", "private_test_cases"]:
            tc_data = item.get(test_field) or []
            if isinstance(tc_data, str):
                try:
                    tc_data = json.loads(tc_data)
                except Exception:
                    tc_data = []
            if isinstance(tc_data, list):
                all_tests.extend(tc_data)

        prompt = f"Problem: {title}\n\n{content}\n\n"
        if starter and starter.strip():
            prompt += f"Starter code:\n```python\n{starter}\n```\n\n"
        prompt += (
            "Write a complete Python 3 solution to solve this problem. "
            "Your solution should read from standard input if required, or implement the starter code function. "
            "Provide your solution code in a single markdown ```python ... ``` code block."
        )
        messages = [{"role": "user", "content": prompt}]
        samples.append((messages, {"task": "generation", "tests": all_tests}, {"category": f"code_gen_{difficulty}"}))
    logger.info(f"Loaded {len(raw_gen)} LCB Code Generation samples (public + private tests).")

    # If limit reached for overall run, stop early
    if limit is not None and len(samples) >= limit:
        return samples[:limit]

    # 2. Code Execution
    try:
        ds_exec = load_dataset("livecodebench/execution", split="test")
        if limit is not None:
            exec_limit = limit - len(samples)
            if exec_limit > 0:
                ds_exec = ds_exec.select(range(min(exec_limit, len(ds_exec))))
            else:
                ds_exec = []
        for item in list(ds_exec):
            code = item.get("code", "")
            call_input = item.get("input", "")
            expected_output = str(item.get("output", "")).strip()

            prompt = (
                "Predict the exact return value of executing the following Python code:\n\n"
                f"```python\n{code}\n```\n\n"
                f"Execution call: `{call_input}`\n\n"
                "State your final predicted return value on the last line in the format: 'Final Output: <output>'."
            )
            messages = [{"role": "user", "content": prompt}]
            samples.append((messages, {"task": "execution", "expected": expected_output}, {"category": "code_execution"}))
        logger.info("Loaded LCB Code Execution samples.")
    except Exception as e:
        logger.warning(f"Could not load LCB 'execution' ({e}).")

    if limit is not None and len(samples) >= limit:
        return samples[:limit]

    # 3. Test Generation — EXCLUDED by default: faithful scoring needs the model's generated
    # tests to be run against a reference solution (not available in this harness), and the
    # cheap length proxy fakes ~100%. Set LCB_INCLUDE_TEST_GEN=1 to load it anyway (it will
    # not be credited by the scorer; see _verify_single_lcb_sample).
    if os.getenv("LCB_INCLUDE_TEST_GEN") != "1":
        logger.info("LCB: skipping test_generation (needs reference-solution execution to "
                    "score faithfully; set LCB_INCLUDE_TEST_GEN=1 to include).")
        return samples[:limit] if (limit is not None) else samples
    try:
        ds_test = load_dataset("livecodebench/test_generation", split="test")
        if limit is not None:
            test_limit = limit - len(samples)
            if test_limit > 0:
                ds_test = ds_test.select(range(min(test_limit, len(ds_test))))
            else:
                ds_test = []
        for item in list(ds_test):
            title = item.get("question_title", "")
            content = item.get("question_content", "")
            fn_name = item.get("function_name", "")
            starter = item.get("starter_code", "")
            difficulty = item.get("difficulty", "medium")
            gold_test = item.get("test", "")

            prompt = (
                f"Problem: {title}\n\n{content}\n\n"
                f"Function Name: `{fn_name}`\n"
                f"Starter code:\n```python\n{starter}\n```\n\n"
                "Generate valid inputs and expected outputs that test this function correctly. "
                "Provide your test case in the format: 'Test: input=<input>, output=<output>'."
            )
            messages = [{"role": "user", "content": prompt}]
            samples.append((messages, {"task": "test_gen", "expected": gold_test}, {"category": f"test_gen_{difficulty}"}))
        logger.info("Loaded LCB Test Generation samples.")
    except Exception as e:
        logger.warning(f"Could not load LCB 'test_generation' ({e}).")

    return samples


def _verify_single_lcb_sample(resp_text: str, gold_payload: Any) -> bool:
    """Verify single LCB sample in an isolated process with strict timeout."""
    import subprocess
    import sys
    import json

    if not resp_text:
        return False

    if isinstance(gold_payload, list):
        gold_payload = {"task": "generation", "tests": gold_payload}
    elif not isinstance(gold_payload, dict):
        # An unparseable payload means nothing was executed against the submission.
        # Returning True auto-passed those rows.
        return False

    task = gold_payload.get("task", "generation")

    # Task 1: Code Generation
    if task == "generation":
        gold_tests = gold_payload.get("tests", [])
        code_match = re.search(r"```(?:python)?\s*\n(.*?)\n```", resp_text, re.DOTALL)
        code = code_match.group(1) if code_match else resp_text

        if not gold_tests:
            # A problem whose tests could not be decoded is unverified, not solved.
            return False

        # Test string assertions
        if isinstance(gold_tests[0], str):
            test_script = f"{code}\n" + "\n".join(gold_tests)
            try:
                r = run_sandboxed(
                    [sys.executable, "-c", test_script],
                    capture_output=True,
                    timeout=5,
                )
                return r.returncode == 0
            except (subprocess.TimeoutExpired, Exception):
                return False

        for tc in gold_tests:
            if not isinstance(tc, dict):
                continue
            test_type = tc.get("testtype", "stdin")
            inp = str(tc.get("input") or "")
            expected = str(tc.get("output") or "").strip()

            if test_type == "stdin":
                try:
                    r = run_sandboxed(
                        [sys.executable, "-c", code],
                        input=inp,
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    if r.returncode != 0 or r.stdout.strip() != expected:
                        return False
                except (subprocess.TimeoutExpired, Exception):
                    return False
            elif test_type == "functional":
                fn_name = tc.get("fn_name")
                args = tc.get("args", [])
                test_script = (
                    f"{code}\n\n"
                    f"import sys\n"
                    f"args = {repr(args)}\n"
                    f"expected = {repr(expected)}\n"
                    f"res = {fn_name}(*args) if isinstance(args, list) else {fn_name}(args)\n"
                    f"if str(res).strip() == str(expected).strip():\n"
                    f"    sys.exit(0)\n"
                    f"sys.exit(1)\n"
                )
                try:
                    r = run_sandboxed(
                        [sys.executable, "-c", test_script],
                        capture_output=True,
                        timeout=5,
                    )
                    if r.returncode != 0:
                        return False
                except (subprocess.TimeoutExpired, Exception):
                    return False

        return True

    # Task 2: Code Execution — the prompt asks for 'Final Output: <output>' on the last
    # line. Extract the model's *stated* answer and compare to the gold, instead of a bare
    # substring test (`expected in resp`), which passed whenever the value appeared anywhere
    # in the model's reasoning and heavily inflated this category.
    elif task == "execution":
        expected = str(gold_payload.get("expected", "")).strip()
        if not expected:
            return False
        resp = resp_text.strip()

        def _norm(s: str) -> str:
            s = str(s).strip()
            if s.startswith("```") and s.endswith("```"):
                s = s.strip("`").strip()
            s = s.strip().strip("`").strip()
            if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
                s = s[1:-1]
            return s.strip()

        m = re.search(r"(?i)final\s*output\s*[:=]\s*(.+)", resp)
        if m and _norm(m.group(1)) == _norm(expected):
            return True
        # fallbacks: whole response is just the value, or the last non-empty line is
        if _norm(resp) == _norm(expected):
            return True
        lines = [ln for ln in resp.splitlines() if ln.strip()]
        if lines and _norm(lines[-1]) == _norm(expected):
            return True
        return False

    # Task 3: Test Generation — the canonical LCB metric runs the model's generated tests
    # against a reference solution to check they discriminate correct vs. buggy code, which
    # this lightweight harness cannot do. The previous `len(resp) > 10` check passed almost
    # everything (fake ~100%), so test_gen is EXCLUDED from the loader by default
    # (LCB_INCLUDE_TEST_GEN=1 to force-load it). If it is loaded, we do not credit it here
    # rather than report an unvalidated pass.
    elif task == "test_gen":
        return False

    return True


async def _async_judge_lcb(sample_traces: List[Dict[str, Any]]) -> Tuple[int, int, Dict[str, Dict[str, Any]]]:
    """Execute all LCB test suites in parallel using ProcessPoolExecutor post-generation."""
    import asyncio
    import concurrent.futures

    loop = asyncio.get_running_loop()
    max_workers = min(32, (os.cpu_count() or 4))
    
    with tqdm(total=len(sample_traces), desc="Judging [LCB]") as pbar:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
            tasks = []
            for sample in sample_traces:
                resp = sample.get("response_text") or sample.get("response") or ""
                gold = sample.get("gold_answer")
                fut = loop.run_in_executor(pool, _verify_single_lcb_sample, resp, gold)
                tasks.append((sample, fut))

            correct_count = 0
            category_stats = {}
            for sample, fut in tasks:
                verdict = await fut
                sample["is_correct"] = verdict
                sample["correct"] = verdict
                sample["status"] = "OK" if verdict else "FAILED"
                cat = sample.get("category")
                if cat:
                    if cat not in category_stats:
                        category_stats[cat] = {"correct": 0, "total": 0}
                    category_stats[cat]["total"] += 1
                    if verdict:
                        category_stats[cat]["correct"] += 1
                if verdict:
                    correct_count += 1
                pbar.update(1)

    return correct_count, len(sample_traces), category_stats


def run_lcb(model_name: str, base_url: str, concurrency: int, enable_thinking: bool = False, **kwargs) -> Dict[str, Any]:
    """Run native LiveCodeBench algorithmic coding evaluation suite."""
    samples = _load_lcb_samples(enable_thinking, limit=kwargs.get("limit"))
    return run_eval_suite(
        eval_name="lcb",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        async_eval_fn=_async_judge_lcb,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens"),
    )
