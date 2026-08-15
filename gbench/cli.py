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

"""Command-line interface for Gemma benchmark suite.

This module provides a rich CLI with comprehensive options for running
benchmarks on Gemma models.
"""

import argparse
import logging
import os
import re
import sys
import time
from types import ModuleType
dummy_check = ModuleType("transformers.dependency_versions_check")
def dep_version_check(*args, **kwargs):
    pass
dummy_check.dep_version_check = dep_version_check
sys.modules["transformers.dependency_versions_check"] = dummy_check
from pathlib import Path
from typing import Optional

from . import __version__
from .core import (
    BenchmarkConfig,
    DEFAULT_CONFIG,
    DEFAULT_GEMMACLAW_COMMIT,
    QUICK_CONFIG,
    ModelCategory,
    ModelFormat,
    Priority,
    registry,
    check_gpu_ready,
    validate_gpu_config,
    get_available_gpus,
    get_batch_sizes,
)

logger = logging.getLogger(__name__)

# Exit codes. A caller pointing gbench at its own endpoint needs to tell
# "the model got something wrong" apart from "the run never happened",
# because only the first is evidence about the model.
EXIT_OK = 0
EXIT_MODEL_FAILURE = 1
EXIT_HARNESS_ERROR = 2

GOLDEN_VERDICT = {"passed": "PASS", "failed": "FAIL", "error": "ERROR"}


def golden_exit_code(results: list[dict]) -> int:
    """Reduce Golden Set results to a process exit code.

    Harness errors outrank model failures: a run that could not reach the
    endpoint has no verdict on the model at all, so it must not be
    reported with the same code as a real regression.
    """
    golden = [r for r in results if r.get("benchmark_type") == "golden"]
    if any(r.get("status") == "error" for r in golden):
        return EXIT_HARNESS_ERROR
    if any(r.get("status") == "failed" for r in golden):
        return EXIT_MODEL_FAILURE
    return EXIT_OK


def golden_category_breakdown(task_results: list[dict]) -> list[dict]:
    """Aggregate per-case Golden Set results into one row per category.

    A bare "9/12" says something regressed but not where, which is the
    first thing you need in order to judge whether it matters. Losing
    both tool_use cases is a different morning from losing one
    translation.

    Args:
        task_results: Per-case result dicts from the Golden runner.

    Returns:
        One row per category, worst first, so problem areas sort to the
        top instead of landing wherever the alphabet puts them. The
        offending task ids are deliberately not repeated here: the
        FAIL and ERROR lines printed below the table already name each
        one, along with the detail you need to reproduce it.
    """
    buckets: dict[str, dict] = {}
    for task in task_results:
        category = task.get("category") or "uncategorized"
        bucket = buckets.setdefault(category, {
            "category": category,
            "total": 0,
            "passed": 0,
            "statuses": set(),
        })
        bucket["total"] += 1
        status = task.get("status")
        bucket["statuses"].add(status)
        if status == "passed":
            bucket["passed"] += 1

    rows = []
    for bucket in buckets.values():
        # Same precedence as golden_exit_code: a category that could not
        # be measured is not a category that passed.
        if "error" in bucket["statuses"]:
            bucket["status"] = "error"
        elif "failed" in bucket["statuses"]:
            bucket["status"] = "failed"
        else:
            bucket["status"] = "passed"
        del bucket["statuses"]
        rows.append(bucket)

    rank = {"error": 0, "failed": 1, "passed": 2}
    return sorted(rows, key=lambda b: (rank.get(b["status"], 3),
                                       b["category"]))


def _save_eval_summary_csv(
    results_dir: Path,
    eval_results: list,
    eval_failures: list,
    eval_pillars: list,
    custom_pillars: dict,
) -> Optional[Path]:
    """Save clean, structured CSV report for all evaluation results to eval_summary.csv."""
    import csv

    csv_path = results_dir / "eval_summary.csv"
    suite_to_pillar = {}
    for p_title, keys in eval_pillars:
        clean_title = re.sub(r"^\d+\.\s*", "", p_title).strip()
        for k in keys:
            suite_to_pillar[k.lower()] = clean_title

    fieldnames = [
        "Pillar",
        "Model",
        "Format",
        "Eval Suite",
        "Subcategory",
        "Thinking",
        "Questions",
        "Effective N",
        "Correct",
        "Accuracy (%)",
        "Status",
        "Duration (s)",
    ]

    rows = []
    all_runs = [(r, True) for r in eval_results] + [(r, False) for r in eval_failures]

    for r, is_success in all_runs:
        eval_name = r.get("eval_name", "").lower()
        model = r.get("model_short", r.get("model_name", "N/A"))
        fmt = r.get("format", "N/A")
        thinking = "Yes" if r.get("thinking", False) else "No"
        total_q = r.get("total_questions", 0) if is_success else 0
        effective_n = r.get("effective_n", total_q) if is_success else "-"
        correct = r.get("correct_answers", 0) if is_success else 0
        acc = f"{r.get('accuracy', 0.0):.2f}" if is_success else "-"
        status = r.get("status", "success" if is_success else "failed")
        duration = r.get("duration_s", "")

        if eval_name in suite_to_pillar:
            pillar = suite_to_pillar[eval_name]
        elif eval_name in custom_pillars:
            clean_cp = re.sub(r"^\d+\.\s*", "", custom_pillars[eval_name]).strip()
            pillar = f"[CUSTOM] {clean_cp}"
        else:
            pillar = "[CUSTOM] OTHER"

        # Extract subcategory information
        cat_acc = r.get("category_accuracy", {})
        traces = r.get("sample_traces", [])
        trace_cat = traces[0].get("category") if traces and isinstance(traces[0], dict) else None

        if len(cat_acc) == 1:
            single_cat = list(cat_acc.keys())[0]
            rows.append({
                "Pillar": pillar,
                "Model": model,
                "Format": fmt,
                "Eval Suite": eval_name.upper(),
                "Subcategory": single_cat,
                "Thinking": thinking,
                "Questions": total_q,
                "Effective N": effective_n,
                "Correct": correct,
                "Accuracy (%)": acc,
                "Status": status,
                "Duration (s)": duration,
            })
        elif len(cat_acc) > 1:
            # Main overall suite row
            rows.append({
                "Pillar": pillar,
                "Model": model,
                "Format": fmt,
                "Eval Suite": eval_name.upper(),
                "Subcategory": "OVERALL",
                "Thinking": thinking,
                "Questions": total_q,
                "Effective N": effective_n,
                "Correct": correct,
                "Accuracy (%)": acc,
                "Status": status,
                "Duration (s)": duration,
            })
            # Individual subcategory rows
            for cat, cstats in sorted(cat_acc.items()):
                ctot = cstats.get("total", 0)
                ccorr = cstats.get("correct", 0)
                cacc = f"{float(cstats.get('accuracy', 0.0)):.2f}"
                rows.append({
                    "Pillar": pillar,
                    "Model": model,
                    "Format": fmt,
                    "Eval Suite": eval_name.upper(),
                    "Subcategory": cat,
                    "Thinking": thinking,
                    "Questions": ctot,
                    "Effective N": ctot,
                    "Correct": ccorr,
                    "Accuracy (%)": cacc,
                    "Status": status,
                    "Duration (s)": "",
                })
        else:
            # Fallback to trace category if available
            rows.append({
                "Pillar": pillar,
                "Model": model,
                "Format": fmt,
                "Eval Suite": eval_name.upper(),
                "Subcategory": trace_cat or "-",
                "Thinking": thinking,
                "Questions": total_q,
                "Effective N": effective_n,
                "Correct": correct,
                "Accuracy (%)": acc,
                "Status": status,
                "Duration (s)": duration,
            })

    if not rows:
        return None

    try:
        results_dir.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"Saved evaluation CSV report to: {csv_path}")
        return csv_path
    except Exception as e:
        logger.error(f"Failed to write CSV summary to {csv_path}: {e}")
        return None


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    for noisy_logger in ["httpx", "httpcore", "datasets", "huggingface_hub", "urllib3", "fsspec", "filelock", "google_genai", "google", "grpc"]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def wait_for_gpu_memory(min_free_gb: float = 70.0, max_wait_seconds: int = 120):
    """Wait for GPU memory to be reclaimed by CUDA driver.
    
    Uses nvidia-smi to actively poll until sufficient memory is free.
    
    Args:
        min_free_gb: Minimum free memory required in GB
        max_wait_seconds: Maximum time to wait before giving up
    """
    import subprocess
    
    logger.info(f"Waiting for GPU memory (need {min_free_gb:.0f}GB free)...")
    
    for elapsed in range(0, max_wait_seconds, 5):
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                free_mb = float(result.stdout.strip().split('\n')[0])
                free_gb = free_mb / 1024
                if free_gb >= min_free_gb:
                    logger.info(f"GPU memory ready: {free_gb:.1f}GB free")
                    return
                logger.info(f"  GPU memory: {free_gb:.1f}GB free, waiting... ({elapsed}s)")
        except Exception:
            pass
        time.sleep(5)
    
    logger.warning(f"GPU memory wait timed out after {max_wait_seconds}s")


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        description="Open Model Performance Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick smoke test on single model
  gbench --preset quick --models gemma-4-E4B-it

  # Run full baseline (HF only)
  gbench --format hf --models gemma-4-E4B-it gemma-4-31B-it

  # Benchmark any HuggingFace model (auto-registers)
  gbench --models google/gemma-4-31B-it

  # Multimodal models with vision benchmarks
  gbench --category multimodal --multimodal-only

  # Dry run to see what would be executed
  gbench --dry-run --models gemma-4-E4B-it
        """,
    )

    parser.add_argument(
        "--version",
        "-v",
        action="version",
        version=f"gbench {__version__}",
    )

    # Model selection
    model_group = parser.add_argument_group("Model Selection")
    model_group.add_argument(
        "--models",
        nargs="+",
        help="Models to benchmark — use registry names (e.g., gemma-4-E4B-it) or HuggingFace IDs (e.g., google/gemma-4-31B-it)",
    )
    model_group.add_argument(
        "--category",
        choices=["text", "embedding", "multimodal"],
        help="Filter by model category",
    )
    model_group.add_argument(
        "--priority",
        choices=["P0", "P1", "P2"],
        help="Filter by priority level",
    )
    model_group.add_argument(
        "--format",
        choices=["hf", "gguf", "both"],
        default="both",
        help="Model format to test (default: both)",
    )

    # Benchmark type selection
    bench_group = parser.add_argument_group("Benchmark Types")
    bench_group.add_argument(
        "--serving-only",
        action="store_true",
        help="Run only serving benchmarks",
    )
    bench_group.add_argument(
        "--throughput-only",
        action="store_true",
        help="Run only throughput benchmarks",
    )
    bench_group.add_argument(
        "--text-only",
        action="store_true",
        help="Run only text benchmarks, skip multimodal even for capable models",
    )
    bench_group.add_argument(
        "--multimodal-only",
        action="store_true",
        help="Run only multimodal benchmarks (for models that support it)",
    )
    bench_group.add_argument(
        "--stress-test",
        action="store_true",
        help="Run only stress test (ramp-up to find max sustainable throughput)",
    )
    bench_group.add_argument(
        "--no-stress-test",
        action="store_true",
        help="Skip stress test (included by default in all presets)",
    )
    bench_group.add_argument(
        "--stress-threshold",
        type=int,
        default=5000,
        help="P99 TTFT threshold in ms for stress test (default: 5000ms)",
    )
    bench_group.add_argument(
        "--quality",
        action="store_true",
        help="Run quality (gemmaclaw agentic) benchmarks",
    )
    bench_group.add_argument(
        "--quality-only",
        action="store_true",
        help="Run only quality (gemmaclaw agentic) benchmarks",
    )
    bench_group.add_argument(
        "--scenarios",
        nargs="+",
        help="List of specific scenario file paths to run (relative to qa/scenarios/)",
    )
    bench_group.add_argument(
        "--golden",
        action="store_true",
        help="Run Golden Set deterministic smoke-test tasks",
    )
    bench_group.add_argument(
        "--golden-only",
        action="store_true",
        help="Run only Golden Set deterministic smoke-test tasks",
    )
    bench_group.add_argument(
        "--golden-tasks",
        nargs="+",
        help="List of specific Golden task IDs or JSON files to run",
    )
    bench_group.add_argument(
        "--golden-model-id",
        help=(
            "Model name to send in the Golden Set request payload. "
            "Defaults to whatever the endpoint's /models listing reports, "
            "falling back to the model's HF id. Required when an endpoint "
            "serves several models"
        ),
    )

    # Evaluation benchmarks
    eval_group = parser.add_argument_group("Evaluation Benchmarks")
    eval_group.add_argument(
        "--evals",
        nargs="+",
        help="Select evaluation benchmark suites to run (space or comma-separated, or 'all', 'plugins')",
    )
    eval_group.add_argument(
        "--eval-thinking",
        "--enable-thinking",
        "--thinking",
        dest="eval_thinking",
        action="store_true",
        help="Enable reasoning/thinking mode for evals that support it (aime, bfcl, bundled_detection, causalbench, gpqa_diamond, infographicvqa, loft_x_arxiv, mmlu_pro, mmmu_pro, new_amc_aime, putnam, semantic_keypoint)",
    )
    eval_group.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        help="Maximum generation tokens for evals (default: 8192 for non-thinking, 16384 for thinking mode)",
    )
    eval_group.add_argument(
        "--eval-max-soft-tokens",
        type=int,
        choices=[70, 140, 280, 560, 1120],
        default=None,
        help="Image soft token budget for vision evals (70, 140, 280, 560, 1120). Default: 1120",
    )
    eval_group.add_argument(
        "--suite-timeout",
        type=int,
        default=None,
        help="Per-suite wall-clock budget in seconds (default: OFF). When set, a suite "
             "that exceeds it is reported as 'timeout' and the sweep continues, so one "
             "wedged harness cannot stall the run. Destructive: the suite's partial "
             "results are lost and its child processes/containers are not reaped, so "
             "leave it unset unless running unattended.",
    )
    eval_group.add_argument(
        "--eval-n-shot",
        type=int,
        default=None,
        help="Few-shot CoT exemplar count for MMLU-Pro, taken from the dataset's official validation split (max 5 per category; default: 0)",
    )
    eval_group.add_argument(
        "--eval-categories",
        type=str,
        default=None,
        help="Comma-separated category override for BFCL (e.g. 'simple_python,multi_turn')",
    )
    eval_group.add_argument(
        "--eval-limit",
        "--limit",
        dest="eval_limit",
        type=int,
        default=None,
        help="Limit number of evaluation samples per benchmark suite (e.g. --eval-limit 10)",
    )
    eval_group.add_argument(
        "--sandboxes",
        type=int,
        default=None,
        help="Concurrency level for containerized/sandboxed evaluations (e.g. terminal_bench, copilot_bench_swe, ui_control_osworld, lmarena_web_agent). Defaults to --batch-sizes.",
    )
    eval_group.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for model generations across all benchmarks (default: 0.0)",
    )
    eval_group.add_argument(
        "--eval-plugins-dir",
        "--eval-plugins-path",
        dest="eval_plugins_dir",
        nargs="+",
        default=None,
        help="Directory or Python file path(s) containing custom/private eval suite plugins (*.py)",
    )
    eval_group.add_argument(
        "--eval-custom-jsonl",
        type=str,
        default=None,
        help="Path to custom JSONL evaluation dataset file for zero-code adhoc evaluation",
    )
    eval_group.add_argument(
        "--evals-only",
        action="store_true",
        help="Run only specific evaluation benchmarks (skip serving, throughput, stress, quality, golden)",
    )
    eval_group.add_argument(
        "--list-plugins",
        nargs="?",
        const=".",
        default=None,
        metavar="PATH",
        help="List all evaluation plugins discovered at PATH (defaults to current directory) and exit",
    )
    eval_group.add_argument(
        "--list",
        dest="list_what",
        nargs="?",
        const="evals",
        default=None,
        choices=["evals", "pillars", "presets", "campaigns", "golden", "quality", "scenarios", "models", "plugins", "all"],
        help="List what gbench can run (evals, pillars, presets, campaigns, golden, quality, scenarios, models, plugins, all) and exit",
    )

    # Configuration options
    config_group = parser.add_argument_group("Configuration")
    config_group.add_argument(
        "--preset",
        choices=["quick", "default"],
        default="default",
        help="Configuration preset (default: default)",
    )
    config_group.add_argument(
        "--campaign",
        nargs="+",
        choices=["chat-like", "agentic", "prefill-heavy", "decode-heavy", "mixed", "long-decode"],
        help="Specific performance campaign scenario(s) to run",
    )
    config_group.add_argument(
        "--dataset",
        choices=["sharegpt", "random", "custom", "hf"],
        help="Override the dataset type (sharegpt, random, custom, or hf)",
    )
    config_group.add_argument(
        "--dataset-path",
        type=str,
        help="Path or HF ID for custom dataset",
    )
    config_group.add_argument(
        "--num-iterations",
        type=int,
        help="Number of iterations per config (default: 3)",
    )
    config_group.add_argument(
        "--warmup-iterations",
        type=int,
        help="Number of warmup iterations (default: 1)",
    )
    config_group.add_argument(
        "--max-cv-percent",
        type=float,
        help="Max acceptable CV%% for validation (default: 5.0)",
    )
    config_group.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        help="Custom batch sizes to test",
    )
    config_group.add_argument(
        "--input-lengths",
        nargs="+",
        type=int,
        help="Custom input lengths for throughput tests",
    )
    config_group.add_argument(
        "--output-lengths",
        nargs="+",
        type=int,
        help="Custom output lengths for throughput tests",
    )
    config_group.add_argument(
        "--num-prompts",
        type=int,
        help="Number of prompts for serving benchmarks",
    )
    config_group.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs to use (validates against available GPUs)",
    )
    config_group.add_argument(
        "--tensor-parallel",
        type=int,
        help="Tensor parallel size (default: same as num-gpus)",
    )
    config_group.add_argument(
        "--gemmaclaw-commit",
        type=str,
        default=DEFAULT_GEMMACLAW_COMMIT,
        help=(
            "Target gemmaclaw git commit, branch or tag (default: the pinned "
            f"release {DEFAULT_GEMMACLAW_COMMIT[:7]}). A branch or tag is "
            "resolved to a commit sha before the run, so the quality "
            "scaffold_id moves when the scorer does. Pass 'main' to score "
            "against the current development tip instead"
        ),
    )
    config_group.add_argument(
        "--gemmaclaw-path",
        type=str,
        help="Path to local gemmaclaw repository checkout (optional)",
    )
    config_group.add_argument(
        "--remote-endpoint",
        type=str,
        help="Remote API endpoint URL to benchmark instead of starting local vLLM (e.g. https://.../v1)",
    )
    config_group.add_argument(
        "--tokenizer",
        type=str,
        help="HuggingFace model ID or local directory path for tokenization when benchmarking custom remote endpoints or Ollama tags",
    )

    # Output options
    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--results-dir",
        type=Path,
        default=Path("./results"),
        help="Directory for benchmark results (default: ./results)",
    )
    output_group.add_argument(
        "--tags",
        nargs="+",
        help="List of arbitrary key:value tags to assign to this run (e.g., family:gemma-1b stage:prod)",
    )
    output_group.add_argument(
        "--skip-existing",
        action="store_true",
        default=False,
        help="Reuse a previous result for a suite instead of running it, if one is found "
             "anywhere under --results-dir (newest match wins). OFF by default: reusing a "
             "result silently mixes it with whatever the code measures now, so opt in only "
             "when resuming an interrupted run against unchanged code.",
    )
    output_group.add_argument(
        "--no-skip-existing",
        action="store_false",
        dest="skip_existing",
        help="Explicitly re-run every benchmark (this is the default).",
    )

    # Execution options
    exec_group = parser.add_argument_group("Execution")
    exec_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be executed without running",
    )
    exec_group.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    exec_group.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization (default: 0.9)",
    )

    # Staging utilities
    staging_group = parser.add_argument_group("Staging Utilities")
    staging_group.add_argument(
        "--stage-to-gcs",
        type=str,
        help="Stage models locally (downloading if TFHub) and upload to the specified GCS destination (e.g. gs://my-bucket/path/). Benchmarks will not be run.",
    )

    return parser


def apply_campaign_to_config(config, campaign: str, args=None) -> None:
    """Apply specific campaign scenario parameters to config."""
    if campaign == "chat-like":
        config.dataset = "sharegpt"
        if not (args and getattr(args, "num_prompts", None)):
            config.num_prompts = 1000
    elif campaign == "agentic":
        config.dataset = "random"
        config.input_lengths = [8000]
        config.output_lengths = [400]
        if not (args and getattr(args, "num_prompts", None)):
            config.num_prompts = 500
    elif campaign == "prefill-heavy":
        config.dataset = "random"
        config.input_lengths = [8192]
        config.output_lengths = [128]
    elif campaign == "decode-heavy":
        config.dataset = "random"
        config.input_lengths = [128]
        config.output_lengths = [2048]
    elif campaign == "mixed":
        config.dataset = "random"
        config.input_lengths = [4096]
        config.output_lengths = [1024]
    elif campaign == "long-decode":
        config.dataset = "random"
        config.input_lengths = [8192]
        config.output_lengths = [8192]
    if args and getattr(args, "dataset", None):
        config.dataset = args.dataset
    if args and getattr(args, "dataset_path", None):
        config.dataset_path = args.dataset_path


def _split_golden_tasks(raw: Optional[list[str]]) -> Optional[list[str]]:
    """Normalise --golden-tasks into a flat list of task ids.

    argparse nargs="+" only splits on spaces, so a comma separated list
    arrives as a single token that matches nothing. Accept either form,
    and both mixed together.

    Args:
        raw: Values as argparse produced them, or None if the flag was
            not passed.

    Returns:
        The flattened task ids, or None when the flag was not passed. An
        explicitly empty selection stays empty rather than becoming None,
        so it is reported as "matched nothing" instead of silently
        running the whole dataset.
    """
    if raw is None:
        return None
    return [part.strip() for item in raw for part in item.split(",")
            if part.strip()]


def get_config_from_args(args: argparse.Namespace) -> BenchmarkConfig:
    """Create BenchmarkConfig from CLI arguments."""
    # Start with preset
    if args.preset == "quick":
        config = QUICK_CONFIG
    else:
        config = DEFAULT_CONFIG

    # Override with custom iteration values
    if args.num_iterations:
        config.num_iterations = args.num_iterations
    if args.warmup_iterations is not None:
        config.warmup_iterations = args.warmup_iterations
    if args.max_cv_percent:
        config.min_acceptable_cv_percent = args.max_cv_percent

    # Override with custom configuration values
    if hasattr(args, "campaign") and args.campaign:
        campaigns = args.campaign if isinstance(args.campaign, list) else [args.campaign]
        apply_campaign_to_config(config, campaigns[0], args)

    if args.batch_sizes:
        config.batch_sizes = args.batch_sizes
    if args.input_lengths:
        config.input_lengths = args.input_lengths
    if args.output_lengths:
        config.output_lengths = args.output_lengths
    if args.num_prompts:
        config.num_prompts = args.num_prompts
    if getattr(args, "dataset", None):
        config.dataset = args.dataset
    if getattr(args, "dataset_path", None):
        config.dataset_path = args.dataset_path
    # Set GPU configuration
    if args.num_gpus:
        config.num_gpus = args.num_gpus
    if args.tensor_parallel:
        config.tensor_parallel_size = args.tensor_parallel
    else:
        config.tensor_parallel_size = config.num_gpus  # Default: TP = num GPUs

    # Set output options
    config.results_dir = args.results_dir
    config.skip_existing = args.skip_existing
    config.dry_run = args.dry_run
    config.gpu_memory_utilization = args.gpu_memory_utilization

    # Quality benchmark parameters
    config.gemmaclaw_commit = args.gemmaclaw_commit
    config.gemmaclaw_path = args.gemmaclaw_path
    config.remote_endpoint = args.remote_endpoint
    config.tokenizer = getattr(args, "tokenizer", None)
    config.selected_scenarios = args.scenarios
    config.tags = args.tags

    # Evaluation parameters
    config.evals = getattr(args, "evals", None)
    if getattr(args, "evals_only", False) and not config.evals:
        if getattr(args, "eval_plugins_dir", None):
            config.evals = ["plugins"]
        else:
            config.evals = ["all"]
    config.eval_thinking = getattr(args, "eval_thinking", False)
    config.eval_max_output_tokens = getattr(args, "max_output_tokens", None)
    config.eval_max_soft_tokens = getattr(args, "eval_max_soft_tokens", 1120)
    config.eval_n_shot = getattr(args, "eval_n_shot", 0)
    config.suite_timeout = getattr(args, "suite_timeout", None)
    config.eval_categories = getattr(args, "eval_categories", None)
    config.eval_limit = getattr(args, "eval_limit", None)
    config.sandboxes = getattr(args, "sandboxes", None)
    config.temperature = getattr(args, "temperature", 0.0)
    config.eval_plugins_dir = getattr(args, "eval_plugins_dir", None)
    config.eval_custom_jsonl = getattr(args, "eval_custom_jsonl", None)

    # Golden Set parameters
    config.golden = args.golden or args.golden_only
    config.golden_only = args.golden_only
    # nargs="+" splits on spaces, but a comma separated list is the more
    # natural thing to type and silently matches no tasks. Accept both.
    config.selected_golden_tasks = _split_golden_tasks(args.golden_tasks)
    config.golden_model_id = args.golden_model_id

    # Initialize LogManager NOW — after all config fields are set
    # (must be after results_dir is assigned from CLI args)
    config.initialize()

    return config


def get_models_from_args(args: argparse.Namespace) -> list:
    """Get list of models to benchmark based on CLI arguments."""
    models = []

    # If specific models requested (takes precedence over auto-detection)
    if args.models:
        for model_name in args.models:
            model = registry.get(model_name)
            if not model:
                model = registry.register_hf_model(model_name)
            models.append(model)
        return models

    # If remote endpoint specified without --models, auto-detect served model ID from /v1/models
    if getattr(args, "remote_endpoint", None):
        base_url = args.remote_endpoint
        if not base_url.endswith("/v1") and not base_url.endswith("/v1/"):
            base_url = f"{base_url.rstrip('/')}/v1"
        import urllib.request
        import json

        served_id = None
        try:
            url = f"{base_url.rstrip('/')}/models"
            with urllib.request.urlopen(url, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models_data = data.get("data", [])
                if models_data:
                    served_id = models_data[0]["id"]
        except Exception as e:
            logger.warning(
                f"Could not query /v1/models from {base_url} to auto-detect model: {e}"
            )
        if served_id:
            model = registry.get(served_id)
            if not model:
                model = registry.register_hf_model(served_id)
            logger.info(
                f"Auto-detected model '{served_id}' served at {args.remote_endpoint}"
            )
            return [model]

    # If specific models requested
    if args.models:
        for model_name in args.models:
            model = registry.get(model_name)
            if not model:
                logger.warning(
                    f"Unknown model: {model_name}. "
                    f"Use a HuggingFace model ID (e.g. org/model-name) "
                    f"to auto-register any vLLM-compatible model."
                )
                continue
            models.append(model)
    else:
        # Filter by category and priority
        category = (
            ModelCategory[args.category.upper()]
            if args.category
            else None
        )
        priority = Priority[args.priority] if args.priority else None

        models = registry.filter(
            category=category,
            priority=priority,
            supports_gguf=(args.format == "gguf"),
        )

    return models


def get_formats_from_args(args: argparse.Namespace) -> list:
    """Get list of formats to test."""
    if getattr(args, "remote_endpoint", None):
        return [ModelFormat.REMOTE]
    elif args.format == "hf":
        return [ModelFormat.HF]
    elif args.format == "gguf":
        return [ModelFormat.GGUF]
    else:  # both
        return [ModelFormat.HF, ModelFormat.GGUF]



def _handle_list(args) -> int:
    """`--list <what>`: show what gbench can run, then exit.

    The suites, pillars and presets were previously only discoverable by reading the
    source or scraping --help, which made it hard to know what a sweep would actually
    execute (or which suites would skip).
    """
    what = args.list_what
    show = (lambda k: what in (k, "all"))

    if show("pillars") or show("evals"):
        from .runners.evals import BUILTIN_PILLARS
        from .runners.eval_suites import SUITES
        in_pillars = {e for _, evs in BUILTIN_PILLARS for e in evs}
        if show("pillars"):
            print(f"\nBuilt-in pillars ({len(BUILTIN_PILLARS)}):\n")
            for pillar, evs in BUILTIN_PILLARS:
                print(f"  {pillar}  ({len(evs)})")
                print("      " + "  ".join(sorted(evs)))
        if show("evals"):
            print(f"\nBuilt-in eval suites ({len(SUITES)}):\n")
            pillar_of = {e: p for p, evs in BUILTIN_PILLARS for e in evs}
            width = max((len(n) for n in SUITES), default=10) + 2
            for name in sorted(SUITES):
                tag = pillar_of.get(name, "(not in --evals all)")
                print(f"  {name:<{width}} {tag}")
            orphan = sorted(set(SUITES) - in_pillars)
            if orphan:
                print(f"\n  NB {len(orphan)} suite(s) are registered but NOT part of "
                      f"`--evals all`: {', '.join(orphan)}")

    if show("presets"):
        print("\nPresets (--preset):\n  quick    reduced scenario set\n  default  full scenario set")

    if show("campaigns"):
        campaign_defs = [
            ("chat-like", "Multi-turn conversational traffic (ShareGPT dataset)"),
            ("agentic", "Long prompt (8000 tokens), short tool response (400 tokens)"),
            ("decode-heavy", "Short prompt (128 tokens), long generation (2048 tokens)"),
            ("prefill-heavy", "Long prompt (8192 tokens), short generation (128 tokens)"),
            ("mixed", "Balanced prompt (4096 tokens), output (1024 tokens)"),
            ("long-decode", "High-context prompt (8192 tokens), output (8192 tokens)"),
        ]
        print(f"\nWorkload Campaigns ({len(campaign_defs)}) - use with --campaign:\n")
        width = max(len(c[0]) for c in campaign_defs) + 2
        for name, desc in campaign_defs:
            print(f"  {name:<{width}} {desc}")

    if show("golden"):
        # Golden tasks are JSON files under gbench/golden_dataset/, each with an "id";
        # there is no module-level constant to import.
        import json as _json
        from pathlib import Path as _Path
        dataset_dir = _Path(__file__).parent / "golden_dataset"
        if not dataset_dir.is_dir():
            print(f"\nGolden Set tasks: dataset directory not found at {dataset_dir}")
        else:
            rows = []
            for fp in sorted(dataset_dir.glob("*.json")):
                try:
                    data = _json.loads(fp.read_text(encoding="utf-8"))
                except Exception as e:
                    rows.append((fp.stem, f"<unreadable: {type(e).__name__}>", ""))
                    continue
                rows.append((str(data.get("id", fp.stem)),
                             str(data.get("category", "uncategorized")),
                             str(data.get("description", data.get("name", "")))[:52]))
            print(f"\nGolden Set tasks ({len(rows)}) - use with --golden / --golden-tasks:\n")
            width = max((len(r[0]) for r in rows), default=10) + 2
            cat_w = max((len(r[1]) for r in rows), default=8) + 2
            for tid, cat, desc in rows:
                print(f"  {tid:<{width}} {cat:<{cat_w}} {desc}")

    if show("quality") or show("scenarios"):
        quality_scenarios = [
            ("memory/session_recall.json", "Long-term multi-turn session recall & context retention"),
            ("memory/profile_update.json", "Dynamic user profile & entity memory extraction"),
            ("plugins/mcp_routing.json", "Model Context Protocol (MCP) server discovery & routing"),
            ("plugins/schema_validation.json", "Strict JSON schema validation for tool arguments"),
            ("tool_calling/parallel_dispatch.json", "Concurrent multi-tool invocation with argument isolation"),
            ("tool_calling/error_recovery.json", "Autonomous error recovery on failed tool executions"),
            ("reasoning/planning_decomposition.json", "Step-by-step agent task planning & execution"),
            ("safety/instruction_injection.json", "Prompt injection & malicious payload resistance"),
        ]
        print(f"\nAgent Quality Scenarios ({len(quality_scenarios)}) - use with --scenarios:\n")
        width = max(len(s[0]) for s in quality_scenarios) + 2
        for sc_name, sc_desc in quality_scenarios:
            print(f"  {sc_name:<{width}} {sc_desc}")

    if show("models"):
        try:
            from .core.models import MODELS
            entries = MODELS.values() if isinstance(MODELS, dict) else MODELS
            entries = list(entries)
            print(f"\nModels ({len(entries)}) - use with --models:\n")
            width = max((len(str(getattr(m, "name", m))) for m in entries), default=10) + 2
            for m in sorted(entries, key=lambda x: str(getattr(x, "name", x))):
                name = str(getattr(m, "name", m))
                cat = getattr(getattr(m, "category", None), "value", "") or ""
                prio = getattr(getattr(m, "priority", None), "value", "") or ""
                hf = getattr(m, "hf_model_id", "") or ""
                print(f"  {name:<{width}} {str(cat):<12} {str(prio):<4} {hf}")
        except Exception as e:
            print(f"\nModels: unavailable ({type(e).__name__}: {e}); see --models")

    if show("plugins"):
        from .runners.eval_suites.loader import discover_and_register_plugins, CUSTOM_PILLARS
        paths = getattr(args, "eval_plugins_dir", None)
        if not paths:
            env_paths = os.environ.get("GBENCH_EVAL_PLUGINS_PATH", "").strip()
            if env_paths:
                paths = [p.strip() for p in env_paths.split(os.pathsep) if p.strip()]
            elif Path("plugins").is_dir():
                paths = ["plugins"]
            elif Path("gbench/plugins").is_dir():
                paths = ["gbench/plugins"]
        if not paths:
            print("\nPlugins (--list plugins): no plugin directory specified.\n  Pass --eval-plugins-dir <path> or set GBENCH_EVAL_PLUGINS_PATH to discover custom suites.")
        else:
            discovered = discover_and_register_plugins(paths if isinstance(paths, list) else [paths])
            print(f"\nPlugins discovered in {paths} ({len(discovered)}):")
            for name in sorted(discovered):
                print(f"  {name:<52} [{CUSTOM_PILLARS.get(name, 'Custom Plugin')}]")

    print()
    return 0


def main(argv: Optional[list[str]] = None):
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args(argv)

    # Show help if no arguments provided
    import sys
    if (argv is None and len(sys.argv) == 1) or (argv is not None and len(argv) == 0):
        parser.print_help()
        return 0

    # List plugins mode (independent of any benchmark execution)
    if getattr(args, "list_what", None):
        return _handle_list(args)

    if getattr(args, "list_plugins", None) is not None:
        target_path = args.list_plugins
        from .runners.eval_suites.loader import discover_and_register_plugins, CUSTOM_PILLARS
        discovered = discover_and_register_plugins([target_path])
        if not discovered:
            print(f"No plugins found in '{target_path}'.")
            return 0
        print(f"\nDiscovered {len(discovered)} evaluation plugin(s) in '{target_path}':\n")
        max_name_len = max(len(name) for name in discovered.keys())
        for name, fn in sorted(discovered.items()):
            pillar = CUSTOM_PILLARS.get(name, "Custom Plugin")
            print(f"  • {name:<{max_name_len + 2}} [{pillar}]")
        print()
        return 0

    # Validation constraints
    if getattr(args, "remote_endpoint", None) and not getattr(args, "tokenizer", None):
        parser.error(
            "--tokenizer <hf_repo_or_local_path> is required when using --remote-endpoint (e.g. --tokenizer google/gemma-4-E4B-it)."
        )
    if getattr(args, "remote_endpoint", None) and getattr(args, "format", None) and args.format != "both":
        parser.error("--remote-endpoint and --format are mutually exclusive.")
    if getattr(args, "campaign", None) and (getattr(args, "input_lengths", None) or getattr(args, "output_lengths", None)):
        parser.error("--campaign and --input-lengths/--output-lengths are mutually exclusive.")
    if not getattr(args, "campaign", None) and not (getattr(args, "input_lengths", None) or getattr(args, "output_lengths", None)) and not getattr(args, "preset", None):
        # --stage-to-gcs only copies model artifacts to a bucket. It never
        # runs a benchmark, so it has no workload geometry to describe and
        # belongs with the other modes exempted here.
        if not getattr(args, "evals", None) and not getattr(args, "evals_only", False) and not getattr(args, "golden_only", False) and not getattr(args, "quality_only", False) and not getattr(args, "stress_test", False) and not getattr(args, "stage_to_gcs", None):
            parser.error("Either --preset, --campaign, or --input-lengths/--output-lengths must be provided.")

    # Strict Eval Argument Compatibility Validation
    from .runners.eval_suites import discover_and_register_plugins, SUITES
    discover_and_register_plugins(getattr(args, "eval_plugins_dir", None))

    raw_evals = getattr(args, "evals", None) or []
    flat_evals = []
    for item in raw_evals:
        for sub in str(item).split(","):
            sub_clean = sub.strip()
            if sub_clean:
                flat_evals.append(sub_clean)
    args.evals = flat_evals
    evals_selected = set(flat_evals)
    if evals_selected:
        for e in evals_selected:
            if e not in {"all", "plugins"} and e.lower() not in SUITES:
                parser.error(f"Unknown eval suite '{e}'. Available: {', '.join(sorted(SUITES.keys()))}")

    if getattr(args, "eval_categories", None):
        if not evals_selected or any(e not in {"bfcl", "all"} for e in evals_selected):
            parser.error("--eval-categories is only supported when running 'bfcl'. Cannot combine with unsupported evals.")
    if getattr(args, "eval_max_soft_tokens", None) is not None:
        if getattr(args, "remote_endpoint", None):
            parser.error("--eval-max-soft-tokens cannot be used with --remote-endpoint (soft tokens are configured server-side).")
        if not evals_selected or any(e not in {"mmmu_pro", "screenspot", "semantic_keypoint", "textvqa", "infographicvqa", "bundled_detection", "all"} for e in evals_selected):
            parser.error("--eval-max-soft-tokens is only supported by vision evals ('mmmu_pro', 'screenspot', 'semantic_keypoint', 'textvqa', 'infographicvqa', 'bundled_detection').")
    if getattr(args, "eval_n_shot", None) is not None:
        if not evals_selected or any(e not in {"mmlu_pro", "all"} for e in evals_selected):
            parser.error("--eval-n-shot is only supported by 'mmlu_pro'.")




    # Setup logging
    setup_logging(args.verbose)

    # Get configuration
    config = get_config_from_args(args)
    evals_only_mode = getattr(args, "evals_only", False) or (
        bool(getattr(args, "evals", None))
        and not getattr(args, "campaign", None)
        and not (getattr(args, "input_lengths", None) or getattr(args, "output_lengths", None))
    )
    logger.info(f"Using configuration: {args.preset}")
    if evals_only_mode:
        logger.info(f"  Mode: Evaluation Benchmarks Only")
        logger.info(f"  Evals: {config.evals}")
        logger.info(f"  Concurrency: {config.batch_sizes[0] if config.batch_sizes else 8} (from --batch-sizes)")
        logger.info(f"  Thinking enabled: {config.eval_thinking}")
        eff_max = config.eval_max_output_tokens or (16384 if config.eval_thinking else 8192)
        logger.info(f"  Max output tokens: {eff_max}")
    else:
        logger.info(f"  Iterations: {config.num_iterations} (+{config.warmup_iterations} warmup)")
        logger.info(f"  Batch sizes: {config.batch_sizes}")
        logger.info(f"  Input/Output lengths: {config.input_lengths}/{config.output_lengths}")
        logger.info(f"  Num prompts: serving={config.num_prompts}, throughput={config.num_prompts_throughput}")
        stress_disabled = args.no_stress_test or args.serving_only or args.throughput_only
        logger.info(f"  Stress test: {'disabled' if stress_disabled else 'enabled (ramp-up)'}")
    logger.info(f"Results directory: {config.results_dir}")
    
    # Pre-flight GPU check: ensure vLLM engine and GPU are ready for local benchmarking
    if not config.dry_run and not args.stage_to_gcs and not config.remote_endpoint:
        from gbench.utils import require_vllm_engine
        require_vllm_engine("Local GPU benchmarking")
        gpu_ready, gpu_msg = check_gpu_ready()
        if not gpu_ready:
            logger.error(f"GPU pre-flight check failed:\n{gpu_msg}")
            return 1
        else:
            logger.info(gpu_msg)

    # Get models to benchmark
    models = get_models_from_args(args)
    if not models:
        logger.error("No models selected")
        return 1

    if args.stage_to_gcs:
        stage_models_to_gcs(models, args.stage_to_gcs)
        return 0

    logger.info(f"Selected {len(models)} model(s) for benchmarking")
    
    # Save run metadata
    if config.log_manager and not args.stage_to_gcs and not config.dry_run:
        metadata = {
            "timestamp": config.log_manager.timestamp,
            "models": [m.name for m in models],
            "tags": config.tags or [],
            "preset": args.preset,
            "benchmark_mode": args.quality_only and "quality" or "performance",
        }
        config.log_manager.save_metadata(metadata)
    for model in models:
        moe_tag = f", MoE {model.num_experts}x{model.num_active_experts}" if model.is_moe else ""
        logger.info(f"  - {model.name} ({model.total_params_b:.0f}B{moe_tag})")

    # Validate GPU configuration against first model (skip if remote endpoint is used)
    first_model = models[0]
    if config.remote_endpoint:
        logger.info(f"Remote endpoint specified ({config.remote_endpoint}); checking if endpoint is functional...")
        from .utils import verify_endpoint_functional
        is_up, err_msg, max_model_len = verify_endpoint_functional(config.remote_endpoint)
        if not is_up:
            logger.error(f"❌ Remote endpoint '{config.remote_endpoint}' is unreachable or not functional: {err_msg}")
            logger.error("Cancelling benchmark run.")
            return 1
        logger.info(f"✅ Remote endpoint '{config.remote_endpoint}' is functional and answering requests.")
        if max_model_len:
            logger.info(f"   Server reported max_model_len: {max_model_len} tokens")
        logger.info(f"Remote endpoint specified ({config.remote_endpoint}); skipping local GPU count check.")
    else:
        is_valid, gpu_msg = validate_gpu_config(config.num_gpus, first_model.total_params_b)
        if not is_valid:
            if config.dry_run or args.golden_only or args.quality_only:
                logger.warning(f"GPU configuration error (ignored because golden/quality/dry-run mode is active): {gpu_msg}")
            else:
                logger.error(f"GPU configuration error: {gpu_msg}")
                return 1
        elif "Warning" in gpu_msg:
            logger.warning(gpu_msg)
        else:
            logger.info(gpu_msg)

    # Get formats
    formats = get_formats_from_args(args)
    if config.remote_endpoint and len(formats) > 1:
        logger.info("Remote endpoint specified; testing single active server format (hf).")
        formats = [ModelFormat.HF]
    logger.info(f"Testing formats: {[f.value for f in formats]}")

    # Determine benchmark modes
    # Default: run text + multimodal for multimodal-capable models
    # --text-only: skip multimodal benchmarks entirely
    # --multimodal-only: skip text benchmarks, only run multimodal
    # --serving-only / --throughput-only: control benchmark type, not text vs multimodal
    evals_only = getattr(args, "evals_only", False) or (
        bool(getattr(args, "evals", None))
        and not getattr(args, "campaign", None)
        and not (getattr(args, "input_lengths", None) or getattr(args, "output_lengths", None))
    )
    run_golden = args.golden or args.golden_only
    run_quality = args.quality or args.quality_only
    run_text = not args.multimodal_only and not args.quality_only and not args.golden_only and not evals_only
    run_multimodal = not args.text_only and not args.quality_only and not args.golden_only and not evals_only
    
    if args.multimodal_only:
        run_multimodal = True
        run_text = False
    
    # Display benchmark mode
    if evals_only:
        benchmark_mode = "evals only"
    elif args.golden_only:
        benchmark_mode = "golden set only"
    elif args.quality_only:
        benchmark_mode = "quality only"
    elif args.multimodal_only:
        benchmark_mode = "multimodal only"
    elif run_text and run_multimodal:
        benchmark_mode = "text and multimodal (for capable models)"
    elif run_text:
        benchmark_mode = "text only"
    else:
        benchmark_mode = "multimodal only"
    logger.info(f"Benchmark mode: {benchmark_mode}")

    # Create runners
    from .runners import ServingBenchmarkRunner, ThroughputBenchmarkRunner, StressTestRunner, QualityBenchmarkRunner, GoldenBenchmarkRunner, EvalsBenchmarkRunner
    serving_runner = ServingBenchmarkRunner(config)
    throughput_runner = ThroughputBenchmarkRunner(config)
    stress_runner = StressTestRunner(config, ttft_threshold_ms=args.stress_threshold)
    quality_runner = QualityBenchmarkRunner(config)
    golden_runner = GoldenBenchmarkRunner(config)
    evals_runner = EvalsBenchmarkRunner(config)

    # Determine if stress test should run
    # --stress-test means run stress test only (stress_test_only)
    # --no-stress-test means skip stress test entirely
    # --serving-only or --throughput-only also skip stress test
    stress_test_only = args.stress_test  # --stress-test means ONLY run stress test
    run_stress_test = (
        (args.stress_test or not args.no_stress_test)
        and not args.serving_only
        and not args.throughput_only
        and not args.quality_only
        and not args.golden_only
        and not evals_only
    )

    # Scaffold contract: record every (pillar, model, format) condition this
    # run plans to produce. See docs/scaffold-versioning.md.
    #
    # `conditions` is the planned matrix, not the outcome. Multimodal pillars
    # are listed from the model's declared capability, and a remote endpoint
    # probe can still skip one at run time. The results say what completed.
    from .core.scaffold import (
        Pillar,
        build_condition,
        gbench_commit,
        render_conditions,
        resolve_gemmaclaw_sha,
    )

    # A branch name is not a pin. The default is already a sha, so this is
    # a no-op that touches nothing for an unflagged run, but anyone passing
    # `--gemmaclaw-commit main` or a tag needs it resolved. Resolve once,
    # here, so the scaffold built below and the checkout the quality runner
    # performs later are the same commit by construction. Resolving in both
    # places instead would let `main` advance in between and pin a scorer
    # the run never used.
    #
    # Gated on the pillar actually being planned, because a non-sha ref
    # reaches the network and a serving-only run has no business doing that.
    if run_quality:
        resolved = resolve_gemmaclaw_sha(config.gemmaclaw_commit, config.gemmaclaw_path)
        if resolved:
            if resolved != config.gemmaclaw_commit:
                logger.info(
                    f"Resolved gemmaclaw ref '{config.gemmaclaw_commit}' to {resolved}"
                )
            config.gemmaclaw_commit = resolved
        else:
            logger.warning(
                f"Could not resolve gemmaclaw ref '{config.gemmaclaw_commit}' to a "
                "commit. The quality scaffold_id will pin the ref itself, so it "
                "will not move when that ref does."
            )

    conditions = []
    is_remote = bool(getattr(args, "remote_endpoint", None))
    for model in models:
        mm = run_multimodal and model.supports_multimodal
        pillars = [
            (Pillar.SERVING, run_text and not args.throughput_only and not stress_test_only),
            (Pillar.THROUGHPUT, run_text and not args.serving_only and not stress_test_only and not is_remote),
            (Pillar.SERVING_MULTIMODAL,
             mm and not args.throughput_only and not stress_test_only),
            (Pillar.THROUGHPUT_MULTIMODAL,
             mm and not args.serving_only and not stress_test_only and not is_remote),
            (Pillar.STRESS_TEST, run_stress_test),
            (Pillar.QUALITY, run_quality),
            (Pillar.GOLDEN, run_golden),
            (Pillar.EVALS, bool(config.evals)),
        ]
        for pillar, planned in pillars:
            if not planned:
                continue
            for fmt in formats:
                is_golden = pillar is Pillar.GOLDEN
                conditions.append(build_condition(
                    pillar,
                    model,
                    fmt,
                    dataset_dir=golden_runner.dataset_dir if is_golden else None,
                    selected_tasks=getattr(args, "golden_tasks", None) if is_golden else None,
                    config=config,
                ))

    # Printed before the persist, and outside its condition, because the
    # console and metadata.json are two views of the same objects rather
    # than one depending on the other. A run with no log manager, a dry
    # run, and a GCS staging run are all still entitled to know what
    # scaffold they are about to measure against.
    if conditions:
        print("\n" + render_conditions(conditions))

    # A SECOND save_metadata rather than a move of the first one. The early
    # write happens before the plan exists, and removing it would leave an
    # aborted run as a directory with no metadata.json at all, which
    # service/utils/storage.py renders as a run with unknown models.
    if config.log_manager and not args.stage_to_gcs and not config.dry_run:
        metadata["conditions"] = [c.to_dict() for c in conditions]
        # Beside the conditions rather than inside any id. Results are
        # never uploaded anywhere central by a CLI run, so a run
        # directory has to be self-describing: there is no index to
        # reconstruct later which harness produced it.
        metadata["gbench"] = gbench_commit()
        config.log_manager.save_metadata(metadata)

    if config.dry_run:
        logger.info("Dry-run complete. Exiting without executing benchmarks.")
        return 0

    # Track all results for final summary
    all_results = []

    # Execute benchmarks
    total_runs = 0
    failed_models = []
    for model in models:
      try:
        # Check multimodal compatibility (dynamic HTTP probe for remote endpoints)
        if getattr(args, "remote_endpoint", None) and run_multimodal:
            from gbench.utils import probe_multimodal_support
            logger.info(f"Probing remote endpoint multimodal capability for {model.name}...")
            model_supports_multimodal = probe_multimodal_support(args.remote_endpoint, model.hf_model_id or model.name)
            if model_supports_multimodal:
                logger.info(f"✅ Remote model {model.name} supports multimodal requests.")
            else:
                logger.info(f"ℹ️ Remote model {model.name} does not support multimodal requests.")
        else:
            model_supports_multimodal = model.supports_multimodal or model.category == ModelCategory.MULTIMODAL

        if run_multimodal and not model_supports_multimodal:
            logger.warning(
                f"{model.name} does not support multimodal, skipping multimodal benchmarks"
            )
        
        for format in formats:
            # Skip GGUF if not available
            if format == ModelFormat.GGUF and not model.gguf_model_id:
                logger.warning(
                    f"GGUF not available for {model.name}, skipping"
                )
                continue

            logger.info(
                f"\n{'='*60}\n"
                f"Benchmarking: {model.name} ({format.value})\n"
                f"{'='*60}"
            )

            # Apply param-based batch sizes for performance benchmarks
            if not args.batch_sizes and not args.golden_only and not args.quality_only:
                model_batch_sizes = get_batch_sizes(model.total_params_b, args.preset)
                config.batch_sizes = model_batch_sizes
                logger.info(f"  Batch sizes for {model.total_params_b:.0f}B: {model_batch_sizes}")

            # Run text-only serving benchmarks
            if run_text and not args.throughput_only and not stress_test_only:
                campaigns_to_run = (args.campaign if isinstance(args.campaign, list) else [args.campaign]) if getattr(args, "campaign", None) else [None]
                for camp in campaigns_to_run:
                    if camp:
                        logger.info(f"Running serving benchmarks (text) - campaign: {camp}...")
                        apply_campaign_to_config(config, camp, args)
                    else:
                        logger.info("Running serving benchmarks (text)...")
                    results = serving_runner.run_all(model, format)
                    for result in results:
                        result['benchmark_type'] = 'serving'
                        result['model'] = model.short_name
                        result['model_name'] = model.name
                        result['model_short'] = model.short_name
                        result['format'] = format.value
                        if camp:
                            result['campaign'] = camp
                        all_results.append(result)
                    total_runs += 1

            # Run text-only throughput benchmarks
            if run_text and not args.serving_only and not stress_test_only and not getattr(args, "remote_endpoint", None):
                campaigns_to_run = (args.campaign if isinstance(args.campaign, list) else [args.campaign]) if getattr(args, "campaign", None) else [None]
                for camp in campaigns_to_run:
                    if camp:
                        logger.info(f"Running throughput benchmarks (text) - campaign: {camp}...")
                        apply_campaign_to_config(config, camp, args)
                    else:
                        logger.info("Running throughput benchmarks (text)...")
                    results = throughput_runner.run_all(model, format)
                    for result in results:
                        result['benchmark_type'] = 'throughput'
                        result['model'] = model.short_name
                        result['model_name'] = model.name
                        result['model_short'] = model.short_name
                        result['format'] = format.value
                        if camp:
                            result['campaign'] = camp
                        all_results.append(result)
                    total_runs += 1

            # Run multimodal serving benchmarks — sends real image+text requests
            if run_multimodal and model_supports_multimodal and not args.throughput_only and not stress_test_only:
                campaigns_to_run = (args.campaign if isinstance(args.campaign, list) else [args.campaign]) if getattr(args, "campaign", None) else [None]
                for camp in campaigns_to_run:
                    if camp:
                        logger.info(f"Running serving benchmarks (multimodal) - campaign: {camp}...")
                        apply_campaign_to_config(config, camp, args)
                    else:
                        logger.info("Running serving benchmarks (multimodal)...")
                    results = serving_runner.run_all(model, format, multimodal=True)
                    for result in results:
                        result['benchmark_type'] = 'serving_multimodal'
                        result['model'] = model.short_name
                        result['model_name'] = model.name
                        result['model_short'] = model.short_name
                        result['format'] = format.value
                        if camp:
                            result['campaign'] = camp
                        all_results.append(result)
                    total_runs += 1

            # Run multimodal benchmarks (throughput) - uses custom offline inference
            if run_multimodal and model_supports_multimodal and not args.serving_only and not stress_test_only and not getattr(args, "remote_endpoint", None):
                campaigns_to_run = (args.campaign if isinstance(args.campaign, list) else [args.campaign]) if getattr(args, "campaign", None) else [None]
                for camp in campaigns_to_run:
                    if camp:
                        logger.info(f"Running throughput benchmarks (multimodal) - campaign: {camp}...")
                        apply_campaign_to_config(config, camp, args)
                    else:
                        logger.info("Running throughput benchmarks (multimodal)...")
                    results = throughput_runner.run_all(model, format, multimodal=True)
                    for result in results:
                        result['benchmark_type'] = 'throughput_multimodal'
                        result['model'] = model.short_name
                        result['model_name'] = model.name
                        result['model_short'] = model.short_name
                        result['format'] = format.value
                        if camp:
                            result['campaign'] = camp
                        all_results.append(result)
                    total_runs += 1

            # Run stress test (ramp-up to find max sustainable throughput)
            if run_stress_test:
                logger.info("Running stress test (finding max sustainable throughput)...")
                stress_results = stress_runner.run_all(model, format)
                for result in stress_results:
                    result['benchmark_type'] = 'stress_test'
                    result['model'] = model.short_name
                    result['model_name'] = model.name
                    result['model_short'] = model.short_name
                    result['format'] = format.value
                    all_results.append(result)
                total_runs += 1

            # Run quality benchmarks (gemmaclaw agentic)
            if run_quality:
                logger.info("Running quality benchmarks (gemmaclaw)...")
                result = quality_runner.run(model, format)
                result['benchmark_type'] = 'quality'
                result['model'] = model.short_name
                result['model_name'] = model.name
                result['model_short'] = model.short_name
                result['format'] = format.value
                all_results.append(result)
                total_runs += 1

            # Run Golden Set exact-match benchmarks
            if run_golden:
                logger.info("Running Golden Set benchmarks...")
                result = golden_runner.run(model)
                result['benchmark_type'] = 'golden'
                result['model'] = model.short_name
                result['model_name'] = model.name
                result['model_short'] = model.short_name
                result['format'] = format.value
                all_results.append(result)
                total_runs += 1

            # Run Evaluation benchmarks (bfcl, gpqa, gsm8k, mmlu, mmmu_pro, mrcr, screenspot)
            if config.evals:
                logger.info("Running Evaluation benchmarks...")
                eval_results = evals_runner.run(model, format)
                for eval_res in eval_results:
                    all_results.append(eval_res)
                    total_runs += 1


      except Exception as e:
        logger.error(f"❌ Model {model.name} failed: {e}")
        logger.error("Continuing with next model...")
        failed_models.append(model.name)

    # Print comprehensive results summary
    print_results_summary(all_results, config)

    if failed_models:
        logger.error(f"❌ {len(failed_models)} model(s) failed: {', '.join(failed_models)}")
        return EXIT_MODEL_FAILURE

    golden_code = golden_exit_code(all_results)
    if golden_code == EXIT_HARNESS_ERROR:
        logger.error(
            "❌ Golden Set could not complete. Exiting %d (harness error). "
            "This is not a verdict on the model.", EXIT_HARNESS_ERROR
        )
    elif golden_code == EXIT_MODEL_FAILURE:
        logger.error(
            "❌ Golden Set has failing cases. Exiting %d (model failure).",
            EXIT_MODEL_FAILURE,
        )
    return golden_code


def print_results_summary(results: list[dict], config: BenchmarkConfig):
    """Print comprehensive summary of all benchmark results."""
    if not results:
        logger.warning("No benchmark results to summarize")
        return

    print("\n" + "="*80)
    print("BENCHMARK RESULTS SUMMARY".center(80))
    print("="*80 + "\n")

    # Separate successful and failed results
    successful_results = [r for r in results if not r.get('failed', False)]
    failed_results = [r for r in results if r.get('failed', False)]

    # Group results by benchmark type (text and multimodal separately)
    serving_results = [r for r in successful_results if r.get('benchmark_type') == 'serving']
    serving_mm_results = [r for r in successful_results if r.get('benchmark_type') == 'serving_multimodal']
    throughput_results = [r for r in successful_results if r.get('benchmark_type') == 'throughput']
    throughput_mm_results = [r for r in successful_results if r.get('benchmark_type') == 'throughput_multimodal']
    stress_results = [r for r in successful_results if r.get('benchmark_type') == 'stress_test']
    quality_results = [r for r in successful_results if r.get('benchmark_type') == 'quality']
    golden_results = [r for r in successful_results
                      if r.get('benchmark_type') == 'golden']
    serving_failures = [r for r in failed_results if r.get('benchmark_type') == 'serving']
    serving_mm_failures = [r for r in failed_results if r.get('benchmark_type') == 'serving_multimodal']
    throughput_failures = [r for r in failed_results if r.get('benchmark_type') in ('throughput', 'throughput_multimodal')]
    stress_failures = [r for r in failed_results if r.get('benchmark_type') == 'stress_test']
    quality_failures = [r for r in failed_results if r.get('benchmark_type') == 'quality']

    # Print serving results table (text)
    if serving_results or serving_failures:
        print("SERVING BENCHMARKS (TEXT)")
        has_campaigns = any(r.get('campaign') for r in serving_results + serving_failures)
        if has_campaigns:
            print("-" * 115)
            print(f"{'Model':<20} {'Format':<8} {'Campaign':<14} {'Batch':<6} {'TTFT(ms)':<10} {'TPOT(ms)':<10} {'ITL(ms)':<10} {'TP(req/s)':<10}")
            print("-" * 115)
        else:
            print("-" * 100)
            print(f"{'Model':<20} {'Format':<8} {'Batch':<6} {'TTFT(ms)':<10} {'TPOT(ms)':<10} {'ITL(ms)':<10} {'TP(req/s)':<10}")
            print("-" * 100)
        
        for r in serving_results:
            model = r.get('model_short', 'N/A')[:20]
            fmt = r.get('format', 'N/A')
            batch = r.get('batch_size', 'N/A')
            camp = str(r.get('campaign', '-'))[:14]
            
            # Get metrics from vllm bench serve output
            # Handle both raw results (mean_ttft_ms) and aggregated results (mean_ttft_ms_mean)
            ttft = r.get('mean_ttft_ms', r.get('mean_ttft_ms_mean', r.get('median_ttft_ms', 'N/A')))
            tpot = r.get('mean_tpot_ms', r.get('mean_tpot_ms_mean', r.get('median_tpot_ms', 'N/A')))
            itl = r.get('mean_itl_ms', r.get('mean_itl_ms_mean', r.get('median_itl_ms', 'N/A')))
            throughput = r.get('request_throughput', r.get('request_throughput_mean', 'N/A'))
            
            # Format values
            ttft_str = f"{ttft:.2f}" if isinstance(ttft, (int, float)) else str(ttft)
            tpot_str = f"{tpot:.2f}" if isinstance(tpot, (int, float)) else str(tpot)
            itl_str = f"{itl:.2f}" if isinstance(itl, (int, float)) else str(itl)
            tp_str = f"{throughput:.2f}" if isinstance(throughput, (int, float)) else str(throughput)
            
            # Validation indicator
            valid = r.get('request_throughput_repeatability_valid', True)
            indicator = "✓" if valid else "✗"
            
            if has_campaigns:
                print(f"{model:<20} {fmt:<8} {camp:<14} {batch:<6} {ttft_str:<10} {tpot_str:<10} {itl_str:<10} {tp_str:<10} {indicator}")
            else:
                print(f"{model:<20} {fmt:<8} {batch:<6} {ttft_str:<10} {tpot_str:<10} {itl_str:<10} {tp_str:<10} {indicator}")
        
        # Print failures
        for r in serving_failures:
            model = r.get('model', 'N/A')[:20]
            fmt = r.get('format', 'N/A')
            batch = r.get('batch_size', 'N/A')
            camp = str(r.get('campaign', '-'))[:14]
            if has_campaigns:
                print(f"{model:<20} {fmt:<8} {camp:<14} {batch:<6} {'FAILED':<10} {'-':<10} {'-':<10} {'-':<10} ✗")
            else:
                print(f"{model:<20} {fmt:<8} {batch:<6} {'FAILED':<10} {'-':<10} {'-':<10} {'-':<10} ✗")
        
        print()

    # Print throughput results table
    if throughput_results or throughput_failures:
        print("THROUGHPUT BENCHMARKS")
        has_campaigns = any(r.get('campaign') for r in throughput_results + throughput_failures)
        if has_campaigns:
            print("-" * 95)
            print(f"{'Model':<20} {'Format':<8} {'Campaign':<14} {'Batch':<7} {'In/Out':<12} {'TP(tok/s)':<12} {'CV%':<8}")
            print("-" * 95)
        else:
            print("-" * 80)
            print(f"{'Model':<20} {'Format':<8} {'Batch':<7} {'In/Out':<12} {'TP(tok/s)':<12} {'CV%':<8}")
            print("-" * 80)
        
        for r in throughput_results:
            model = r.get('model_short', 'N/A')[:20]
            fmt = r.get('format', 'N/A')
            batch = r.get('batch_size', 'N/A')
            camp = str(r.get('campaign', '-'))[:14]
            in_len = r.get('input_length', 'N/A')
            out_len = r.get('output_length', 'N/A')
            in_out = f"{in_len}/{out_len}"
            
            # Get metrics from parsed throughput output
            throughput = r.get('output_tokens_per_second', r.get('total_tokens_per_second', 'N/A'))
            cv = r.get('output_tokens_per_second_cv_percent', 'N/A')
            
            # Format values
            tp_str = f"{throughput:.2f}" if isinstance(throughput, (int, float)) else str(throughput)
            cv_str = f"{cv:.1f}%" if isinstance(cv, (int, float)) else str(cv)
            
            # Validation indicator
            valid = r.get('output_tokens_per_second_repeatability_valid', True)
            indicator = "✓" if valid else "✗"
            
            if has_campaigns:
                print(f"{model:<20} {fmt:<8} {camp:<14} {batch:<7} {in_out:<12} {tp_str:<12} {cv_str:<8} {indicator}")
            else:
                print(f"{model:<20} {fmt:<8} {batch:<7} {in_out:<12} {tp_str:<12} {cv_str:<8} {indicator}")
        
        # Print failures
        for r in throughput_failures:
            model = r.get('model', 'N/A')[:20]
            fmt = r.get('format', 'N/A')
            batch = r.get('batch_size', 'N/A')
            camp = str(r.get('campaign', '-'))[:14]
            in_len = r.get('input_length', 'N/A')
            out_len = r.get('output_length', 'N/A')
            in_out = f"{in_len}/{out_len}"
            if has_campaigns:
                print(f"{model:<20} {fmt:<8} {camp:<14} {batch:<7} {in_out:<12} {'FAILED':<12} {'-':<8} ✗")
            else:
                print(f"{model:<20} {fmt:<8} {batch:<7} {in_out:<12} {'FAILED':<12} {'-':<8} ✗")
        
        print()

    # Print multimodal serving results table
    if serving_mm_results or serving_mm_failures:
        print("SERVING BENCHMARKS (MULTIMODAL)")
        has_campaigns = any(r.get('campaign') for r in serving_mm_results + serving_mm_failures)
        if has_campaigns:
            print("-" * 115)
            print(f"{'Model':<20} {'Format':<8} {'Campaign':<14} {'Batch':<6} {'TTFT(ms)':<10} {'TPOT(ms)':<10} {'ITL(ms)':<10} {'TP(req/s)':<10}")
            print("-" * 115)
        else:
            print("-" * 100)
            print(f"{'Model':<20} {'Format':<8} {'Batch':<6} {'TTFT(ms)':<10} {'TPOT(ms)':<10} {'ITL(ms)':<10} {'TP(req/s)':<10}")
            print("-" * 100)
        
        for r in serving_mm_results:
            model = r.get('model_short', 'N/A')[:20]
            fmt = r.get('format', 'N/A')
            batch = r.get('batch_size', 'N/A')
            camp = str(r.get('campaign', '-'))[:14]
            # Handle both raw results and aggregated results (with _mean suffix)
            ttft = r.get('mean_ttft_ms', r.get('mean_ttft_ms_mean', r.get('median_ttft_ms', 'N/A')))
            tpot = r.get('mean_tpot_ms', r.get('mean_tpot_ms_mean', r.get('median_tpot_ms', 'N/A')))
            itl = r.get('mean_itl_ms', r.get('mean_itl_ms_mean', r.get('median_itl_ms', 'N/A')))
            throughput = r.get('request_throughput', r.get('request_throughput_mean', 'N/A'))
            
            ttft_str = f"{ttft:.2f}" if isinstance(ttft, (int, float)) else str(ttft)
            tpot_str = f"{tpot:.2f}" if isinstance(tpot, (int, float)) else str(tpot)
            itl_str = f"{itl:.2f}" if isinstance(itl, (int, float)) else str(itl)
            tp_str = f"{throughput:.2f}" if isinstance(throughput, (int, float)) else str(throughput)
            
            if has_campaigns:
                print(f"{model:<20} {fmt:<8} {camp:<14} {batch:<6} {ttft_str:<10} {tpot_str:<10} {itl_str:<10} {tp_str:<10} ✓")
            else:
                print(f"{model:<20} {fmt:<8} {batch:<6} {ttft_str:<10} {tpot_str:<10} {itl_str:<10} {tp_str:<10} ✓")

        # Print MM failures
        for r in serving_mm_failures:
            model = r.get('model', 'N/A')[:20]
            fmt = r.get('format', 'N/A')
            batch = r.get('batch_size', 'N/A')
            camp = str(r.get('campaign', '-'))[:14]
            if has_campaigns:
                print(f"{model:<20} {fmt:<8} {camp:<14} {batch:<6} {'FAILED':<10} {'-':<10} {'-':<10} {'-':<10} ✗")
            else:
                print(f"{model:<20} {fmt:<8} {batch:<6} {'FAILED':<10} {'-':<10} {'-':<10} {'-':<10} ✗")
        
        print()

    # Print multimodal throughput results table
    if throughput_mm_results:
        print("THROUGHPUT BENCHMARKS (MULTIMODAL)")
        print("-" * 80)
        print(f"{'Model':<20} {'Format':<8} {'Batch':<7} {'In/Out':<12} {'TP(tok/s)':<12}")
        print("-" * 80)
        
        for r in throughput_mm_results:
            model = r.get('model_short', 'N/A')[:20]
            fmt = r.get('format', 'N/A')
            batch = r.get('batch_size', 'N/A')
            in_len = r.get('input_length', 'N/A')
            out_len = r.get('output_length', 'N/A')
            in_out = f"{in_len}/{out_len}"
            throughput = r.get('output_tokens_per_second', r.get('total_tokens_per_second', 'N/A'))
            tp_str = f"{throughput:.2f}" if isinstance(throughput, (int, float)) else str(throughput)
            
            print(f"{model:<20} {fmt:<8} {batch:<7} {in_out:<12} {tp_str:<12} ✓")
        
        print()

    # Print stress test results - ramp-up format
    if stress_results or stress_failures:
        print("STRESS TEST BENCHMARKS (Ramp-Up)")
        print("=" * 80)
        
        for r in stress_results:
            model = r.get('model', r.get('model_short', 'N/A'))
            fmt = r.get('format', 'N/A')
            mode = 'Multimodal' if r.get('multimodal', False) else 'Text'
            threshold = r.get('ttft_threshold_ms', 5000)
            
            print(f"Model: {model} ({fmt}) - {mode} Mode")
            print(f"Threshold: {threshold}ms P99 TTFT")
            print("-" * 80)
            
            # Print search results table if available
            search_results = r.get('search_results', r.get('ramp_up_results', []))
            if search_results:
                # Sort by concurrency for display
                sorted_results = sorted(search_results, key=lambda x: x.get('concurrency', 0))
                print(f"{'Concurrency':<12} {'Median P99':<10} {'Range':<14} {'Throughput':<12} {'Status'}")
                print("-" * 70)
                for level in sorted_results:
                    conc = level.get('concurrency', 0)
                    p99 = level.get('p99_ttft_ms', 0)
                    tp = level.get('request_throughput', 0)
                    passed = level.get('passed', False)
                    status = "✓ PASSED" if passed else "⚠ DEGRADED"
                    # Show sample range if available (from 3x iterations)
                    samples = level.get('p99_ttft_samples', [])
                    if samples:
                        range_str = f"[{min(samples):.0f}-{max(samples):.0f}]"
                    else:
                        range_str = ""
                    print(f"{conc:<12} {p99:>8.1f}ms {range_str:<14} {tp:>8.1f} req/s   {status}")
                print("-" * 70)
            
            # Print max sustainable load
            max_conc = r.get('max_sustainable_concurrency', 0)
            max_tp = r.get('max_sustainable_throughput', 0)
            
            if max_conc > 0:
                print(f"✅ Maximum Sustainable Load: {max_conc} concurrent requests (~{max_tp:.1f} req/s)")
                print(f"   At this concurrency, P99 TTFT stays within {threshold}ms threshold.")
            else:
                print(f"⚠ No sustainable load found - even lowest concurrency exceeded {threshold}ms threshold")
                print(f"   Consider increasing --stress-threshold or investigating model performance.")
            
            print("=" * 80)
            print()
        
        # Print failures
        for r in stress_failures:
            model = r.get('model', 'N/A')
            fmt = r.get('format', 'N/A')
            mode = 'Multimodal' if r.get('multimodal', False) else 'Text'
            error = r.get('error', 'Unknown error')
            print(f"Model: {model} ({fmt}) - {mode} Mode")
            print("-" * 80)
            print(f"Status: ✗ FAILED")
            print(f"Error:  {error}")
            print("=" * 80)
            print()

    # Print quality benchmarks results table
    if quality_results or quality_failures:
        print("QUALITY BENCHMARKS (AGENTIC)")
        print("-" * 80)
        print(f"{'Model':<20} {'Format':<8} {'Commit':<12} {'Scenarios':<12} {'Pass Rate':<10}")
        print("-" * 80)
        
        for r in quality_results:
            model = r.get('model_short', 'N/A')[:20]
            fmt = r.get('format', 'N/A')
            commit = r.get('gemmaclaw_commit', 'N/A')[:7]
            passed = r.get('passed_scenarios', 0)
            total = r.get('total_scenarios', 0)
            scenarios_str = f"{passed}/{total}"
            pass_rate = r.get('pass_rate', 0.0)
            
            print(f"{model:<20} {fmt:<8} {commit:<12} {scenarios_str:<12} {pass_rate:.1f}%")
        
        # Print failures
        for r in quality_failures:
            model = r.get('model', 'N/A')[:20]
            fmt = r.get('format', 'N/A')
            print(f"{model:<20} {fmt:<8} {'-':<12} {'FAILED':<12} -")
        
        print()

    # Print evaluation benchmarks results table
    eval_results = [r for r in successful_results if r.get('benchmark_type') == 'eval']
    eval_failures = [r for r in failed_results if r.get('benchmark_type') == 'eval']
    if eval_results or eval_failures:
        from .runners.evals import BUILTIN_PILLARS
        eval_pillars = BUILTIN_PILLARS

        builtin_suite_keys = set()
        for _, keys in eval_pillars:
            builtin_suite_keys.update(keys)

        builtin_results = [r for r in eval_results if r.get('eval_name', '').lower() in builtin_suite_keys]
        builtin_fails = [r for r in eval_failures if r.get('eval_name', '').lower() in builtin_suite_keys]

        custom_results = [r for r in eval_results if r.get('eval_name', '').lower() not in builtin_suite_keys]
        custom_fails = [r for r in eval_failures if r.get('eval_name', '').lower() not in builtin_suite_keys]

        from .runners.eval_suites import CUSTOM_PILLARS

        table_width = 113
        label_width = 83
        grand_total_q = 0
        grand_correct_q = 0

        # -------------------------------------------------------------------------
        # SECTION 1: BUILT-IN EVALUATION BENCHMARKS (GBENCH STANDARD)
        # -------------------------------------------------------------------------
        if builtin_results or builtin_fails:
            print("BUILT-IN EVALUATION BENCHMARKS (GBENCH STANDARD)")
            print("=" * table_width)
            print(f"{'Model':<18} {'Format':<16} {'Eval Suite':<36} {'Thinking':<10} {'Questions':>9} {'Correct':>9} {'Accuracy':>10}")
            print("=" * table_width)

            builtin_tot = 0
            builtin_corr = 0

            for pillar_title, suite_keys in eval_pillars:
                p_res = [r for r in builtin_results if r.get('eval_name', '').lower() in suite_keys]
                p_fails = [r for r in builtin_fails if r.get('eval_name', '').lower() in suite_keys]
                if not p_res and not p_fails:
                    continue

                print(f"\n▶ {pillar_title}")
                print("-" * table_width)
                p_tot = 0
                p_corr = 0

                for r in p_res:
                    model = r.get('model_short', r.get('model_name', 'N/A'))[:18]
                    fmt = r.get('format', 'N/A')[:16]
                    suite = r.get('eval_name', 'N/A').upper()[:36]
                    thinking = "Yes" if r.get('thinking', False) else "No"
                    total_q = r.get('total_questions', 0)
                    correct = r.get('correct_answers', 0)
                    acc = r.get('accuracy', 0.0)
                    p_tot += total_q
                    p_corr += correct
                    print(f"{model:<18} {fmt:<16} {suite:<36} {thinking:<10} {total_q:>9} {correct:>9} {acc:>9.2f}%")

                    if r.get("low_diversity"):
                        eff_n = r.get("effective_n", total_q)
                        print(f"  └ WARNING: low diversity (effective n={eff_n} / {total_q} questions)")

                    cat_acc = r.get("category_accuracy", {})
                    if 1 < len(cat_acc) <= 25:
                        for cat, cstats in sorted(cat_acc.items()):
                            cat_label = f"  └ {cat}"[:label_width]
                            ctot = cstats.get("total", 0)
                            ccorr = cstats.get("correct", 0)
                            cacc = cstats.get("accuracy", 0.0)
                            print(f"{cat_label:<{label_width}} {ctot:>9} {ccorr:>9} {cacc:>9.2f}%")

                for r in p_fails:
                    model = r.get('model_short', r.get('model_name', 'N/A'))[:18]
                    fmt = r.get('format', 'N/A')[:16]
                    suite = r.get('eval_name', 'N/A').upper()[:36]
                    thinking = "Yes" if r.get('thinking', False) else "No"
                    print(f"{model:<18} {fmt:<16} {suite:<36} {thinking:<10} {'FAILED':>9} {'-':>9} {'-':>10}")

                if p_tot > 0:
                    p_acc = (p_corr / p_tot * 100.0)
                    print(f"{'  └ Pillar Subtotal':<{label_width}} {p_tot:>9} {p_corr:>9} {p_acc:>9.2f}%")
                    builtin_tot += p_tot
                    builtin_corr += p_corr

            if builtin_tot > 0:
                b_acc = (builtin_corr / builtin_tot * 100.0)
                print("-" * table_width)
                print(f"{'BUILT-IN EVALS SUBTOTAL':<{label_width}} {builtin_tot:>9} {builtin_corr:>9} {b_acc:>9.2f}%")
                print("=" * table_width)
                grand_total_q += builtin_tot
                grand_correct_q += builtin_corr
            print()

        # -------------------------------------------------------------------------
        # SECTION 2: CUSTOM / PLUGIN EVALUATION BENCHMARKS
        # -------------------------------------------------------------------------
        if custom_results or custom_fails:
            print("CUSTOM / PLUGIN EVALUATION BENCHMARKS")
            print("=" * table_width)
            print(f"{'Model':<18} {'Format':<16} {'Eval Suite':<36} {'Thinking':<10} {'Questions':>9} {'Correct':>9} {'Accuracy':>10}")
            print("=" * table_width)

            custom_pillar_map: Dict[str, Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]] = {}
            truly_uncat_res = []
            truly_uncat_fails = []

            for r in custom_results:
                ename = r.get('eval_name', '').lower()
                if ename in CUSTOM_PILLARS:
                    p_name = CUSTOM_PILLARS[ename]
                    custom_pillar_map.setdefault(p_name, ([], []))[0].append(r)
                else:
                    truly_uncat_res.append(r)

            for r in custom_fails:
                ename = r.get('eval_name', '').lower()
                if ename in CUSTOM_PILLARS:
                    p_name = CUSTOM_PILLARS[ename]
                    custom_pillar_map.setdefault(p_name, ([], []))[1].append(r)
                else:
                    truly_uncat_fails.append(r)

            custom_tot = 0
            custom_corr = 0

            for cp_title, (cp_res, cp_fails) in sorted(custom_pillar_map.items()):
                clean_cp_title = re.sub(r"^\d+\.\s*", "", cp_title).strip()
                print(f"\n▶ [CUSTOM] {clean_cp_title.upper()}")
                print("-" * table_width)
                cp_tot = 0
                cp_corr = 0
                for r in cp_res:
                    model = r.get('model_short', r.get('model_name', 'N/A'))[:18]
                    fmt = r.get('format', 'N/A')[:16]
                    suite = r.get('eval_name', 'N/A').upper()[:36]
                    thinking = "Yes" if r.get('thinking', False) else "No"
                    total_q = r.get('total_questions', 0)
                    correct = r.get('correct_answers', 0)
                    acc = r.get('accuracy', 0.0)
                    cp_tot += total_q
                    cp_corr += correct
                    print(f"{model:<18} {fmt:<16} {suite:<36} {thinking:<10} {total_q:>9} {correct:>9} {acc:>9.2f}%")

                    if r.get("low_diversity"):
                        eff_n = r.get("effective_n", total_q)
                        print(f"  └ WARNING: low diversity (effective n={eff_n} / {total_q} questions)")

                    cat_acc = r.get("category_accuracy", {})
                    if 1 < len(cat_acc) <= 25:
                        for cat, cstats in sorted(cat_acc.items()):
                            cat_label = f"  └ {cat}"[:label_width]
                            ctot = cstats.get("total", 0)
                            ccorr = cstats.get("correct", 0)
                            cacc = cstats.get("accuracy", 0.0)
                            print(f"{cat_label:<{label_width}} {ctot:>9} {ccorr:>9} {cacc:>9.2f}%")

                for r in cp_fails:
                    model = r.get('model_short', r.get('model_name', 'N/A'))[:18]
                    fmt = r.get('format', 'N/A')[:16]
                    suite = r.get('eval_name', 'N/A').upper()[:36]
                    thinking = "Yes" if r.get('thinking', False) else "No"
                    print(f"{model:<18} {fmt:<16} {suite:<36} {thinking:<10} {'FAILED':>9} {'-':>9} {'-':>10}")

                if cp_tot > 0:
                    cp_acc = (cp_corr / cp_tot * 100.0)
                    print(f"{'  └ Pillar Subtotal':<{label_width}} {cp_tot:>9} {cp_corr:>9} {cp_acc:>9.2f}%")
                    custom_tot += cp_tot
                    custom_corr += cp_corr

            if truly_uncat_res or truly_uncat_fails:
                print(f"\n▶ [CUSTOM] OTHER EVALUATIONS")
                print("-" * table_width)
                uncat_tot = 0
                uncat_corr = 0
                for r in truly_uncat_res:
                    model = r.get('model_short', r.get('model_name', 'N/A'))[:18]
                    fmt = r.get('format', 'N/A')[:16]
                    suite = r.get('eval_name', 'N/A').upper()[:36]
                    thinking = "Yes" if r.get('thinking', False) else "No"
                    total_q = r.get('total_questions', 0)
                    correct = r.get('correct_answers', 0)
                    acc = r.get('accuracy', 0.0)
                    uncat_tot += total_q
                    uncat_corr += correct
                    print(f"{model:<18} {fmt:<16} {suite:<36} {thinking:<10} {total_q:>9} {correct:>9} {acc:>9.2f}%")

                    if r.get("low_diversity"):
                        eff_n = r.get("effective_n", total_q)
                        print(f"  └ WARNING: low diversity (effective n={eff_n} / {total_q} questions)")

                for r in truly_uncat_fails:
                    model = r.get('model_short', r.get('model_name', 'N/A'))[:18]
                    fmt = r.get('format', 'N/A')[:16]
                    suite = r.get('eval_name', 'N/A').upper()[:36]
                    thinking = "Yes" if r.get('thinking', False) else "No"
                    print(f"{model:<18} {fmt:<16} {suite:<36} {thinking:<10} {'FAILED':>9} {'-':>9} {'-':>10}")

                if uncat_tot > 0:
                    u_acc = (uncat_corr / uncat_tot * 100.0)
                    print(f"{'  └ Pillar Subtotal':<{label_width}} {uncat_tot:>9} {uncat_corr:>9} {u_acc:>9.2f}%")
                    custom_tot += uncat_tot
                    custom_corr += uncat_corr

            if custom_tot > 0:
                c_acc = (custom_corr / custom_tot * 100.0)
                print("-" * table_width)
                print(f"{'CUSTOM PLUGINS SUBTOTAL':<{label_width}} {custom_tot:>9} {custom_corr:>9} {c_acc:>9.2f}%")
                print("=" * table_width)
                grand_total_q += custom_tot
                grand_correct_q += custom_corr
            print()

        # Print overall total if both built-in and custom were executed
        if (builtin_results or builtin_fails) and (custom_results or custom_fails) and grand_total_q > 0:
            grand_acc = (grand_correct_q / grand_total_q * 100.0)
            print("=" * table_width)
            print(f"{'OVERALL EVALS TOTAL':<{label_width}} {grand_total_q:>9} {grand_correct_q:>9} {grand_acc:>9.2f}%")
            print("=" * table_width)
            print()

        print()

        # Generate CSV summary report
        target_results_dir = config.log_manager.results_dir if getattr(config, "log_manager", None) else Path(getattr(config, "results_dir", "results"))
        csv_file = _save_eval_summary_csv(
            target_results_dir,
            eval_results,
            eval_failures,
            eval_pillars,
            CUSTOM_PILLARS,
        )

    # Print Golden Set results table
    if golden_results:
        print("GOLDEN SET (DETERMINISTIC SMOKE TEST)")
        print("-" * 80)
        print(f"{'Model':<20} {'Requested':<24} {'Cases':<10} {'Verdict':<10}")
        print("-" * 80)

        for r in golden_results:
            model = r.get('model_short', r.get('model', 'N/A'))[:20]
            requested = (r.get('requested_model') or '(endpoint default)')[:24]
            passed = r.get('passed_cases', 0)
            total = r.get('total_tasks', 0)
            cases = f"{passed}/{total}"
            verdict = GOLDEN_VERDICT.get(r.get('status'), 'UNKNOWN')
            print(f"{model:<20} {requested:<24} {cases:<10} {verdict:<10}")

            rows = golden_category_breakdown(r.get('task_results', []))
            if len(rows) > 1:
                for row in rows:
                    cat_label = f"  └ {row['category']}"[:45]
                    cat_cases = f"{row['passed']}/{row['total']}"
                    cat_verdict = GOLDEN_VERDICT.get(row['status'], 'UNKNOWN')
                    print(f"{cat_label:<45} {cat_cases:<10} "
                          f"{cat_verdict:<10}".rstrip())

        print()

        for r in golden_results:
            for err in r.get('harness_errors', []):
                print(f"  ERROR  {err}")
            for t in r.get('task_results', []):
                if t.get('status') == 'failed':
                    print(f"  FAIL   {t.get('task_id')}: {t.get('details')}")
        print()

    # Print file locations using LogManager
    print("="*80)
    print("RESULTS & LOGS")
    print("="*80)
    
    lm = getattr(config, "log_manager", None)
    if lm:
        # Save top-level aggregated summary.json for programmatic consumption
        summary_payload = {
            "timestamp": lm.timestamp,
            "results_dir": str(lm.results_dir.absolute()),
            "total_configurations": len(results),
            "successful_configurations": len(successful_results),
            "failed_configurations": len(failed_results),
            "models": successful_results,
            "failures": failed_results,
        }
        lm.save_summary(summary_payload)

        summary = lm.get_summary()
        print(f"\nResults Directory: {summary['results_dir']}")
        
        # CSV summary file
        csv_path = lm.results_dir / "eval_summary.csv"
        if csv_path.exists():
            print(f"\nSummary Report (CSV):")
            print(f"  - {csv_path.name} (ready for Google Sheets / Excel)")

        # List result files
        result_files = summary['result_files']
        if result_files:
            print(f"\nResult Files ({len(result_files)} total):")
            for f in result_files[:10]:
                print(f"  - {f}")
            if len(result_files) > 10:
                print(f"  ... and {len(result_files) - 10} more files")
        
        # List log files
        log_files = summary['log_files']
        if log_files:
            print(f"\nLog Files ({len(log_files)} total):")
            for f in log_files[:5]:
                print(f"  - {f}")
            if len(log_files) > 5:
                print(f"  ... and {len(log_files) - 5} more files")
    else:
        results_dir = getattr(config, "results_dir", "results")
        print(f"\nResults Directory: {results_dir}")

    # Statistical validation summary
    print("\n" + "="*80)
    print("STATISTICAL VALIDATION")
    print("="*80)
    
    total_configs = len(results)
    num_failed = len(failed_results)
    num_passed = len(successful_results)
    
    print(f"\nTotal Configurations: {total_configs}")
    print(f"Successful: {num_passed}")
    if num_failed > 0:
        print(f"Failed: {num_failed}")
    
    print("\n" + "="*80)
    if num_failed > 0:
        print(f"Benchmark suite complete! {num_passed} passed, {num_failed} failed.")
    else:
        print(f"Benchmark suite complete! All {total_configs} configurations passed.")

    # A configuration "passed" here only means its runner returned. A
    # Golden Set run that returned a FAIL or ERROR verdict is one of
    # those, so without this line the footer reads "All 1 configurations
    # passed" directly above a non-zero exit code.
    golden_bad = [r for r in golden_results
                  if r.get('status') in ('failed', 'error')]
    if golden_bad:
        verdicts = ", ".join(
            f"{r.get('model_short', r.get('model', 'N/A'))} "
            f"{GOLDEN_VERDICT.get(r.get('status'), 'UNKNOWN')}"
            for r in golden_bad
        )
        print(f"Golden Set did not pass: {verdicts}. See the table above.")
    print("="*80 + "\n")


def stage_models_to_gcs(models: list, gcs_destination: str) -> None:
    """Stage model weights from TFHub/local to GCS for remote deployment."""
    import subprocess
    import os
    from gbench.core.models import ModelFormat

    logger.info(f"Starting GCS staging to: {gcs_destination}")

    for model in models:
        logger.info(f"Processing model: {model.name}")
        try:
            local_path = model.get_model_path(ModelFormat.HF)
        except Exception as e:
            logger.error(f"Failed to resolve local path for {model.name}: {e}")
            continue

        dest_url = gcs_destination.rstrip("/") + "/" + model.short_name + "/"
        logger.info(f"Uploading {local_path} to {dest_url} ...")

        src_items = [os.path.join(local_path, item) for item in os.listdir(local_path)]
        if not src_items:
            logger.warning(f"No files found in {local_path} to upload.")
            continue

        cmd = ["gcloud", "storage", "cp", "-r"] + src_items + [dest_url]
        logger.info(f"Running command: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
            logger.info(f"Successfully uploaded {model.name} to GCS.")
        except subprocess.SubprocessError as e:
            logger.error(f"Failed to upload {model.name} to GCS: {e}")


if __name__ == "__main__":
    sys.exit(main())

