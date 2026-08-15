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

"""Benchmark configuration management.

This module defines all benchmark configurations and derives GPU allocation,
batch sizes, and timeouts directly from model parameters (total_params_b
and is_moe), rather than using hand-tuned lookup tables.

GPU allocation uses discrete tiers with fair boundaries so that competing
models of similar size always receive the same resources:
    ≤20B total params  → 1 GPU
    ≤80B total params  → 2 GPUs
    >80B total params  → 8 GPUs
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ── Param-based benchmark functions ─────────────────────────


def get_num_gpus(total_params_b: float) -> int:
    """Derive GPU count from total model parameters.

    Uses discrete tiers for fair benchmarking — all models in the same
    size range get identical GPU allocation regardless of vendor.

    Boundaries chosen so competing models always share the same tier:
      ≤20B:  1 GPU  — small/mid dense and edge MoE
      ≤80B:  2 GPUs — large dense (23-36B), mid MoE
      >80B:  8 GPUs — all large MoE (100B+) on equal footing
    """
    if total_params_b <= 20:
        return 1
    elif total_params_b <= 80:
        return 2
    else:
        return 8


def get_tensor_parallel(total_params_b: float) -> int:
    """Tensor parallel size = GPU count (1:1 mapping)."""
    return get_num_gpus(total_params_b)


def get_batch_sizes(total_params_b: float, preset: str = "default") -> list[int]:
    """Derive batch sizes from total model parameters.

    Smaller models have more KV cache headroom per GPU, so they get
    larger batch sizes to saturate the GPU. Larger models are memory-
    constrained and need smaller batches.

    Args:
        total_params_b: Total model parameters in billions.
        preset: "quick" for smoke tests, "default" for production.
    """
    if total_params_b <= 15:
        return [1, 64] if preset == "quick" else [1, 32, 128, 256]
    elif total_params_b <= 80:
        return [1, 32] if preset == "quick" else [1, 16, 50, 100]
    else:
        return [1, 32] if preset == "quick" else [1, 16, 32, 64]


def get_server_timeout(total_params_b: float, is_moe: bool) -> int:
    """Derive vLLM server startup timeout from model parameters.

    Generous timeouts for parallel execution — under parallel GPU load,
    weight loading competes for PCIe/NVLink bandwidth and CUDA graph
    compilation is slower. Uses 3x safety margin over solo estimates.

    Args:
        total_params_b: Total model parameters in billions.
        is_moe: Whether the model uses Mixture of Experts.

    Returns:
        Timeout in seconds. Floor 600s, cap 3600s.
    """
    solo = 300 + total_params_b * 10
    parallel = solo * (1.3 if is_moe else 1.0) * 3
    return int(max(600, min(3600, parallel)))


def get_gpu_memory_utilization(total_params_b: float) -> float:
    """GPU memory utilization — uniform 0.90 for fair comparison.

    Leaves ~8GB headroom per 80GB GPU for CUDA graphs + sampler.
    """
    return 0.90


def get_max_model_len() -> int:
    """Uniform max_model_len for all models.

    Returns 4096 to match MLPerf v4.0 standard context for chat workloads.
    Prevents OOM from models with large default contexts (10M+, 128K+)
    and avoids penalizing small models with excessive KV cache reservation.
    """
    return 4096


# The GemmaClaw commit the quality pillar scores against unless
# ``--gemmaclaw-commit`` overrides it.
#
# A sha and not ``main``, because this is the value almost every run uses
# and it is hashed into the quality ``scaffold_id``. A default that tracks
# a branch makes two unflagged runs a week apart two different experiments,
# and the id correctly moves to say so, which means the series breaks for a
# reason nobody chose. Pinning it here is what makes the default
# reproducible: the same bare ``gbench --quality-only`` gives the same
# scaffold_id in six months.
#
# Promoted by hand, which is the point rather than a shortcoming. Pick the
# newest ``gemmaclaw-v*`` tag, put its sha here, update the tag name and
# date below, and say so in the PR. Changing the scorer is then a
# deliberate reviewable commit instead of something that happens to you.
# ``scripts/`` has no automation for this on purpose.
#
# Currently gemmaclaw-v2026.8.3, tagged 2026-06-23.
DEFAULT_GEMMACLAW_COMMIT = "d8bc6989133e0e93535b5ad84e6acda019642839"


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution.

    Includes optimized vLLM settings for maximum throughput:
    - gpu_memory_utilization: 0.90 (uniform headroom for CUDA graphs)
    - enable_chunked_prefill: True (~20-30% throughput improvement)
    - max_num_batched_tokens: 16384 (optimal scheduler throughput)
    """

    # Iteration control for statistical reliability
    num_iterations: int = 3  # Run each config N times
    warmup_iterations: int = 1  # Warmup runs (not included in stats)
    min_acceptable_cv_percent: float = 5.0  # Max CV% for variance check

    # Batch sizes to test
    batch_sizes: list[int] = field(
        default_factory=lambda: [1, 4, 8, 16, 32]
    )

    # Input lengths for throughput benchmarks (tokens)
    input_lengths: list[int] = field(
        default_factory=lambda: [128, 512, 2048]
    )

    # Output lengths for throughput benchmarks (tokens)
    output_lengths: list[int] = field(
        default_factory=lambda: [128, 512, 1024]
    )

    # Number of images per request (multimodal only)
    images_per_request: list[int] = field(
        default_factory=lambda: [1, 2, 4]
    )

    # Base configuration
    num_prompts: int = 200
    num_prompts_throughput: int = 200
    request_rate: str = "inf"
    dataset: str = "random"
    dataset_path: Optional[str] = None
    dataset_multimodal: str = "random-mm"
    remote_endpoint: Optional[str] = None
    tokenizer: Optional[str] = None

    # Output configuration
    results_dir: Path = field(
        default_factory=lambda: Path("./results")
    )
    enable_logging: bool = True
    log_samples: bool = False

    # Resource limits and vLLM optimizations
    gpu_memory_utilization: float = 0.90
    max_num_seqs: Optional[int] = 256
    enable_chunked_prefill: bool = True
    max_num_batched_tokens: int = 16384

    # Multi-GPU configuration
    num_gpus: int = 1
    tensor_parallel_size: Optional[int] = None

    # Execution control
    dry_run: bool = False
    skip_existing: bool = False

    # Quality benchmark (gemmaclaw) configuration
    gemmaclaw_commit: str = DEFAULT_GEMMACLAW_COMMIT
    gemmaclaw_path: Optional[str] = None
    remote_endpoint: Optional[str] = None
    selected_scenarios: Optional[list[str]] = None

    # Golden Set benchmark configuration
    golden: bool = False
    golden_only: bool = False
    selected_golden_tasks: Optional[list[str]] = None
    # Model name to put in the golden request payload. Leave unset to let
    # the runner resolve it from the endpoint's /models listing.
    golden_model_id: Optional[str] = None

    # Evaluation suites configuration
    evals: Optional[list[str]] = None
    eval_thinking: bool = False
    eval_max_output_tokens: Optional[int] = None
    eval_max_soft_tokens: int = 1120
    eval_n_shot: int = 0
    #: Per-suite wall-clock budget (s). None -> per-suite defaults in evals.py.
    suite_timeout: Optional[int] = None
    eval_categories: Optional[str] = None
    eval_limit: Optional[int] = None
    sandboxes: Optional[int] = None
    temperature: float = 0.0
    eval_plugins_dir: Optional[list[str]] = None
    eval_custom_jsonl: Optional[str] = None

    # Run metadata tracking
    tags: Optional[list[str]] = None

    def __post_init__(self):
        """Validate configuration. LogManager is deferred to initialize()."""
        self.log_manager = None

    def initialize(self):
        """Create LogManager after all config fields (incl. --results-dir) are set.

        Must be called after CLI argument overrides are applied to this config.
        """
        from gbench.utils import LogManager

        self.log_manager = LogManager(self.results_dir)
        self.results_dir = self.log_manager.results_dir

    def get_serving_configs(self) -> list[dict]:
        """Generate all serving benchmark configurations."""
        configs = []
        for batch in self.batch_sizes:
            configs.append({
                "batch_size": batch,
                "num_prompts": self.num_prompts,
                "request_rate": self.request_rate,
                "dataset": self.dataset,
            })
        return configs

    def get_throughput_configs(self) -> list[dict]:
        """Generate all throughput benchmark configurations."""
        configs = []
        for input_len in self.input_lengths:
            for output_len in self.output_lengths:
                for batch in self.batch_sizes:
                    configs.append({
                        "input_length": input_len,
                        "output_length": output_len,
                        "batch_size": batch,
                        "num_prompts": self.num_prompts_throughput,
                    })
        return configs

    def get_multimodal_configs(self) -> list[dict]:
        """Generate multimodal-specific configurations."""
        configs = []
        for num_images in self.images_per_request:
            for batch in self.batch_sizes:
                configs.append({
                    "num_images": num_images,
                    "batch_size": batch,
                    "num_prompts": self.num_prompts,
                    "dataset": self.dataset_multimodal,
                })
        return configs


@dataclass
class PerformanceTargets:
    """Target performance metrics for validation."""

    ttft_p50_target_ms: float = 200.0
    ttft_p95_target_ms: float = 500.0
    itl_p50_target_ms: float = 50.0
    itl_p95_target_ms: float = 100.0
    tpot_target_ms: float = 100.0
    throughput_target_tps: float = 1000.0
    memory_target_gb: float = 45.0
    gguf_quality_delta_max: float = 2.0
    repeatability_variance_max: float = 5.0

    def validate_serving_results(
        self, results: dict
    ) -> dict[str, bool]:
        """Validate serving benchmark results against targets."""
        checks = {}
        checks["ttft_p50"] = (
            results.get("ttft_p50", float("inf"))
            <= self.ttft_p50_target_ms
        )
        checks["ttft_p95"] = (
            results.get("ttft_p95", float("inf"))
            <= self.ttft_p95_target_ms
        )
        checks["itl_p50"] = (
            results.get("itl_p50", float("inf"))
            <= self.itl_p50_target_ms
        )
        checks["itl_p95"] = (
            results.get("itl_p95", float("inf"))
            <= self.itl_p95_target_ms
        )
        return checks


# Default configurations
DEFAULT_CONFIG = BenchmarkConfig()
DEFAULT_TARGETS = PerformanceTargets()

# Quick test configuration (smoke test)
QUICK_CONFIG = BenchmarkConfig(
    num_iterations=1,
    warmup_iterations=1,
    batch_sizes=[1, 50],
    input_lengths=[128],
    output_lengths=[512],
    num_prompts=200,
    num_prompts_throughput=200,
    gpu_memory_utilization=0.90,
    enable_chunked_prefill=True,
    max_num_batched_tokens=16384,
)

# Default configuration (production benchmark)
DEFAULT_CONFIG = BenchmarkConfig(
    num_iterations=3,
    warmup_iterations=1,
    batch_sizes=[1, 16, 50, 100],
    input_lengths=[128],
    output_lengths=[512],
    num_prompts=1000,
    num_prompts_throughput=1000,
    gpu_memory_utilization=0.90,
    enable_chunked_prefill=True,
    max_num_batched_tokens=16384,
)


def get_available_gpus() -> int:
    """Get the number of available NVIDIA GPUs.

    Respects CUDA_VISIBLE_DEVICES if set.
    """
    import subprocess

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible:
        return len([x.strip() for x in cuda_visible.split(",") if x.strip()])

    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return len([l for l in result.stdout.splitlines() if "GPU" in l])
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return 0


def validate_gpu_config(
    num_gpus: int, total_params_b: float
) -> tuple[bool, str]:
    """Validate GPU configuration for a model.

    Args:
        num_gpus: Number of GPUs requested.
        total_params_b: Total model parameters in billions.

    Returns:
        Tuple of (is_valid, message).
    """
    available = get_available_gpus()
    recommended = get_num_gpus(total_params_b)

    if available == 0:
        return False, "No NVIDIA GPUs detected"

    if num_gpus > available:
        return False, f"Requested {num_gpus} GPUs but only {available} available"

    VALID_TP_SIZES = {1, 2, 4, 8, 16}
    if num_gpus not in VALID_TP_SIZES:
        return False, (
            f"GPU count {num_gpus} not supported for tensor parallel. "
            f"Valid sizes: {sorted(VALID_TP_SIZES)}"
        )

    if num_gpus < recommended:
        return False, (
            f"Model ({total_params_b:.0f}B params) requires at least "
            f"{recommended} GPU(s)"
        )

    if num_gpus > recommended * 2:
        return True, (
            f"Warning: Using {num_gpus} GPUs for {total_params_b:.0f}B model "
            f"may not provide linear scaling. Recommended: {recommended}"
        )

    return True, (
        f"Using {num_gpus} GPU(s) for {total_params_b:.0f}B model"
    )


def check_gpu_ready(min_free_memory_gb: float = 70.0) -> tuple[bool, str]:
    """Check if GPU is ready for benchmarking (no conflicting processes).

    Uses nvidia-smi to detect other processes and insufficient free memory.
    Respects CUDA_VISIBLE_DEVICES to only check assigned GPUs.
    """
    import subprocess

    gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES", "")

    try:
        proc_cmd = [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory,gpu_uuid",
            "--format=csv,noheader,nounits",
        ]
        result = subprocess.run(
            proc_cmd, capture_output=True, text=True, timeout=10,
        )

        if result.returncode != 0:
            return True, "nvidia-smi unavailable, skipping GPU check"

        if not gpu_ids:
            processes = [
                line.strip()
                for line in result.stdout.strip().splitlines()
                if line.strip()
            ]
            if processes:
                process_info = "; ".join(processes[:3])
                return False, (
                    f"GPU is currently in use by other processes.\n"
                    f"Active GPU processes: {process_info}\n"
                    f"Please terminate these processes before running benchmarks.\n"
                    f"Tip: Use 'nvidia-smi' to view processes and 'kill <PID>' to stop them."
                )

        mem_cmd = [
            "nvidia-smi",
            "--query-gpu=memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ]
        if gpu_ids:
            mem_cmd.extend(["--id=" + gpu_ids])

        mem_result = subprocess.run(
            mem_cmd, capture_output=True, text=True, timeout=10,
        )

        if mem_result.returncode == 0:
            lines = mem_result.stdout.strip().splitlines()
            if lines:
                min_free_gb = float("inf")
                total_gb_first = 0
                for line in lines:
                    parts = line.split(",")
                    if len(parts) >= 2:
                        free_mb = float(parts[0].strip())
                        total_mb = float(parts[1].strip())
                        free_gb = free_mb / 1024
                        if total_gb_first == 0:
                            total_gb_first = total_mb / 1024
                        min_free_gb = min(min_free_gb, free_gb)

                if min_free_gb < min_free_memory_gb:
                    return False, (
                        f"Insufficient GPU memory.\n"
                        f"Free: {min_free_gb:.1f}GB / {total_gb_first:.1f}GB "
                        f"total (min across {len(lines)} GPU(s))\n"
                        f"Required: {min_free_memory_gb:.1f}GB minimum\n"
                        f"Try waiting 30 seconds or restart the terminal."
                    )

                return True, (
                    f"GPU ready: {min_free_gb:.1f}GB / {total_gb_first:.1f}GB "
                    f"free ({len(lines)} GPU(s))"
                )

        return True, "GPU check passed"

    except FileNotFoundError:
        return True, "nvidia-smi not found, skipping GPU check"
    except subprocess.TimeoutExpired:
        return True, "nvidia-smi timed out, skipping GPU check"
    except Exception as e:
        return True, f"GPU check failed: {e}, continuing anyway"
