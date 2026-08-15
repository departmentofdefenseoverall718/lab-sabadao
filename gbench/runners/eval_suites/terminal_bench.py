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

"""Official Terminal-Bench 2.1 evaluation suite via Harbor framework & Docker sandboxes."""

import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        def __init__(self, total=0, desc="", **kwargs):
            self.total = total
            self.desc = desc
            self.n = 0
        def update(self, n=1):
            self.n += n
        def set_postfix(self, **kwargs):
            pass
        def refresh(self):
            pass
        def close(self):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

logger = logging.getLogger(__name__)

DOCS_URL = "docs/evals/terminal_bench.md"
DEFAULT_DATASET = "terminal-bench/terminal-bench-2-1"


def check_terminal_bench_prerequisites() -> Tuple[bool, str]:
    """Check if Harbor CLI and Docker daemon are available and accessible."""
    if not shutil.which("docker"):
        return False, "Docker CLI is not found on PATH."

    try:
        res = subprocess.run(["docker", "info"], capture_output=True, text=True, timeout=5)
        if res.returncode != 0:
            return False, f"Docker daemon is not running or accessible: {res.stderr.strip()}"
    except Exception as e:
        return False, f"Cannot connect to Docker daemon: {e}"

    if not shutil.which("harbor"):
        return False, "Harbor CLI is not installed. Install via: pip install harbor (or uv tool install harbor)"

    return True, ""


def run_terminal_bench(
    model_name: str,
    base_url: str,
    concurrency: int = 1,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run official Terminal-Bench 2.1 evaluation suite via Harbor framework."""
    ok, reason = check_terminal_bench_prerequisites()
    if not ok:
        msg = f"[SKIP] terminal_bench skipped: {reason} See '{DOCS_URL}' for setup instructions."
        logger.warning(msg)
        print(f"\n{msg}")
        return {
            "benchmark_type": "eval",
            "eval_name": "terminal_bench",
            "model_name": model_name,
            "status": "skipped",
            "total_questions": 0,
            "correct_answers": 0,
            "accuracy": 0.0,
            "skip_reason": f"{reason} (See {DOCS_URL})",
        }

    start_time = time.time()
    limit = kwargs.get("limit")
    total = 0
    correct = 0
    category_acc: Dict[str, Dict[str, Any]] = {}
    sample_traces: List[Dict[str, Any]] = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        env = os.environ.copy()
        # Direct Harbor to target OpenAI-compatible endpoint
        endpoint = base_url.rstrip("/")
        if not endpoint.endswith("/v1"):
            endpoint = f"{endpoint}/v1"
        env["OPENAI_BASE_URL"] = endpoint
        env["OPENAI_API_KEY"] = env.get("OPENAI_API_KEY", "EMPTY")

        temperature = kwargs.get("temperature", 0.0)
        agent_name = kwargs.get("agent", "terminus-2")
        harbor_model = model_name if model_name.startswith("openai/") else f"openai/{model_name}"
        cmd = [
            "harbor",
            "run",
            "-d",
            DEFAULT_DATASET,
            "--agent",
            agent_name,
            "--model",
            harbor_model,
            "--n-concurrent",
            str(concurrency),
            "--jobs-dir",
            tmp_dir,
            "--agent-kwarg",
            f"api_base={endpoint}",
            "--agent-kwarg",
            f"temperature={temperature}",
            "--agent-kwarg",
            f'llm_call_kwargs={{"extra_body": {{"chat_template_kwargs": {{"enable_thinking": {str(enable_thinking).lower()}}}}}}}',
        ]

        if limit:
            cmd.extend(["--n-tasks", str(limit)])

        import threading

        full_output_lines = []
        total_expected = limit if limit else 89
        seen_trials = set()
        correct_trials = 0

        try:
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            # Consume any direct stdout from Harbor asynchronously
            def _read_stdout():
                if proc.stdout:
                    line_iter = iter(proc.stdout.readline, "") if hasattr(proc.stdout, "readline") else iter(proc.stdout or [])
                    for line in line_iter:
                        full_output_lines.append(line)

            stdout_thread = threading.Thread(target=_read_stdout, daemon=True)
            stdout_thread.start()

            with tqdm(total=total_expected, desc="Eval [TERMINAL_BENCH]") as pbar:
                # Monitor trial progress via result.json files on disk
                while proc.poll() is None:
                    time.sleep(2)
                    for root, _, files in os.walk(tmp_dir):
                        if "result.json" in files:
                            rf_path = os.path.join(root, "result.json")
                            try:
                                with open(rf_path, "r", encoding="utf-8") as rf:
                                    rdata = json.load(rf)
                                if isinstance(rdata, dict):
                                    if "n_total_trials" in rdata and rdata["n_total_trials"] and pbar.total != rdata["n_total_trials"]:
                                        pbar.total = rdata["n_total_trials"]
                                        pbar.refresh()
                                    if "verifier_result" in rdata or "trial_name" in rdata:
                                        trial_id = rdata.get("trial_name", os.path.basename(root))
                                        if trial_id not in seen_trials:
                                            seen_trials.add(trial_id)
                                            vr = rdata.get("verifier_result") or {}
                                            rewards = vr.get("rewards") if isinstance(vr, dict) else {}
                                            r_val = rewards.get("reward", 0.0) if isinstance(rewards, dict) else (rdata.get("reward") or 0.0)
                                            if float(r_val) >= 1.0:
                                                correct_trials += 1
                                            pbar.update(1)
                                            acc = (correct_trials / len(seen_trials) * 100.0) if seen_trials else 0.0
                                            pbar.set_postfix(correct=f"{correct_trials}/{len(seen_trials)} ({acc:.1f}%)")
                            except Exception:
                                pass

                proc.wait()
                stdout_thread.join(timeout=1.0)

                # Final sweep to catch any remaining results on completion
                for root, _, files in os.walk(tmp_dir):
                    if "result.json" in files:
                        rf_path = os.path.join(root, "result.json")
                        try:
                            with open(rf_path, "r", encoding="utf-8") as rf:
                                rdata = json.load(rf)
                            if isinstance(rdata, dict) and ("verifier_result" in rdata or "trial_name" in rdata):
                                trial_id = rdata.get("trial_name", os.path.basename(root))
                                if trial_id not in seen_trials:
                                    seen_trials.add(trial_id)
                                    vr = rdata.get("verifier_result") or {}
                                    rewards = vr.get("rewards") if isinstance(vr, dict) else {}
                                    r_val = rewards.get("reward", 0.0) if isinstance(rewards, dict) else (rdata.get("reward") or 0.0)
                                    if float(r_val) >= 1.0:
                                        correct_trials += 1
                                    pbar.update(1)
                                    acc = (correct_trials / len(seen_trials) * 100.0) if seen_trials else 0.0
                                    pbar.set_postfix(correct=f"{correct_trials}/{len(seen_trials)} ({acc:.1f}%)")
                        except Exception:
                            pass

            full_output = "".join(full_output_lines)
            if proc.returncode != 0:
                logger.error(f"Harbor CLI exited with code {proc.returncode}")
        except Exception as e:
            logger.error(f"Failed to execute Harbor CLI: {e}")
            return {
                "benchmark_type": "eval",
                "eval_name": "terminal_bench",
                "model_name": model_name,
                "status": "failed",
                "total_questions": 0,
                "correct_answers": 0,
                "accuracy": 0.0,
                "error": str(e),
            }

        # Specifically scan for Harbor result.json files (ignoring config.json, lock.json, trajectory.json)
        trial_results = []
        job_stats = {}

        for root, _, files in os.walk(tmp_dir):
            for file in files:
                if file == "result.json":
                    fpath = os.path.join(root, file)
                    try:
                        with open(fpath, "r", encoding="utf-8") as rf:
                            data = json.load(rf)
                        if isinstance(data, dict):
                            if "stats" in data and isinstance(data["stats"], dict):
                                job_stats = data["stats"]
                            elif "verifier_result" in data or "trial_name" in data:
                                vr = data.get("verifier_result") or {}
                                rewards = vr.get("rewards") if isinstance(vr, dict) else {}
                                reward_val = rewards.get("reward", 0.0) if isinstance(rewards, dict) else (data.get("reward") or 0.0)
                                is_passed = float(reward_val) >= 1.0
                                trial_name = data.get("trial_name", os.path.basename(root))
                                exc = data.get("exception_info", {}).get("exception_type") if data.get("exception_info") else None
                                trial_results.append({
                                    "trial_name": trial_name,
                                    "reward": float(reward_val),
                                    "passed": is_passed,
                                    "exception": exc,
                                    "agent_turns": data.get("agent_result", {}).get("n_turns") if isinstance(data.get("agent_result"), dict) else None,
                                })
                    except Exception as e:
                        logger.debug(f"Could not parse {fpath}: {e}")

        if trial_results:
            sample_traces = sorted(trial_results, key=lambda x: x.get("trial_name", ""))
            total = len(trial_results)
            correct = sum(1 for t in trial_results if t.get("passed"))
        elif job_stats:
            total = job_stats.get("n_trials", 0)
            correct = job_stats.get("n_passed", job_stats.get("n_success", 0))
            if correct == 0 and "mean_reward" in job_stats and total > 0:
                correct = int(round(job_stats["mean_reward"] * total))

        accuracy = (correct / total * 100.0) if total > 0 else 0.0
        duration = time.time() - start_time

        # CC6: zero trials means nothing was measured. Previously this only tripped when
        # the CLI also exited non-zero, so a clean exit with no trials became "success" 0%.
        if total == 0:
            return {
                "benchmark_type": "eval",
                "eval_name": "terminal_bench",
                "model_name": model_name,
                "status": "failed",
                "total_questions": 0,
                "correct_answers": 0,
                "accuracy": 0.0,
                "error": (full_output.strip()
                          or ("Harbor CLI produced no trials" if proc.returncode == 0
                              else "Harbor CLI execution failed")),
                "duration_s": round(duration, 2),
            }

        return {
            "benchmark_type": "eval",
            "eval_name": "terminal_bench",
            "model_name": model_name,
            "thinking": enable_thinking,
            "total_questions": total,
            "correct_answers": correct,
            "failed_requests": 0,
            "accuracy": round(accuracy, 2),
            "category_accuracy": category_acc,
            "sample_traces": sample_traces,
            "status": "success" if proc.returncode == 0 or total > 0 else "failed",
            "duration_s": round(duration, 2),
        }
