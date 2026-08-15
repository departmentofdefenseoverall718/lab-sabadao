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

"""Throughput benchmark runner for vLLM bench throughput command."""

import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Optional

from ..core.config import BenchmarkConfig, get_max_model_len, get_gpu_memory_utilization
from ..core.models import ModelConfig, ModelFormat
from ..analysis.statistics import (
    aggregate_benchmark_results,
    format_statistics_summary,
    validate_repeatability,
)

logger = logging.getLogger(__name__)


class ThroughputBenchmarkRunner:
    """Runner for vLLM throughput benchmarks."""

    def __init__(self, config: BenchmarkConfig):
        """Initialize the throughput benchmark runner.

        Args:
            config: Benchmark configuration
        """
        self.config = config
        self._first_run = True  # Track if this is first run (no wait needed)
    
    def _wait_for_gpu_memory(self, min_free_gb: float = 70.0, max_wait_seconds: int = 120):
        """Wait for GPU memory to be reclaimed by CUDA driver.
        
        Uses nvidia-smi to actively poll until sufficient memory is free.
        Respects CUDA_VISIBLE_DEVICES to only check the assigned GPUs.
        
        Args:
            min_free_gb: Minimum free memory required in GB
            max_wait_seconds: Maximum time to wait before giving up
        """
        if self._first_run:
            self._first_run = False
            return  # No wait needed for first run
        
        import subprocess
        
        logger.info(f"Waiting for GPU memory to be reclaimed (need {min_free_gb:.0f}GB free)...")
        
        # Build nvidia-smi command targeting only our assigned GPUs
        gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        smi_cmd = ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"]
        if gpu_ids:
            smi_cmd.extend(["--id=" + gpu_ids])
        
        for elapsed in range(0, max_wait_seconds, 5):
            try:
                result = subprocess.run(
                    smi_cmd,
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    # Check ALL assigned GPUs have enough free memory
                    free_values = [float(x.strip()) for x in result.stdout.strip().split('\n') if x.strip()]
                    if free_values:
                        min_free_mb = min(free_values)
                        free_gb = min_free_mb / 1024
                        if free_gb >= min_free_gb:
                            logger.info(f"GPU memory ready: {free_gb:.1f}GB free (min across {len(free_values)} GPU(s))")
                            return
                        logger.info(f"  GPU memory: {free_gb:.1f}GB free (min), waiting... ({elapsed}s)")
            except Exception:
                pass  # nvidia-smi failed, fall back to fixed wait
            time.sleep(5)
        
        logger.warning(f"GPU memory wait timed out after {max_wait_seconds}s, proceeding anyway")

    def run(
        self,
        model: ModelConfig,
        format: ModelFormat,
        input_length: int,
        output_length: int,
        batch_size: int,
        num_prompts: Optional[int] = None,
        dataset: str = "random",
    ) -> dict:
        """Run throughput benchmark for a specific configuration.

        Args:
            model: Model configuration
            format: Model format (HF or GGUF)
            input_length: Input prompt length in tokens
            output_length: Output generation length in tokens
            batch_size: Batch size
            num_prompts: Number of prompts to test (optional)
            dataset: Dataset name (random or random-mm)

        Returns:
            Dictionary with benchmark results

        Raises:
            RuntimeError: If benchmark execution fails
        """
        if getattr(self.config, "remote_endpoint", None):
            logger.info("Skipping throughput benchmark for remote endpoint (throughput benchmarks require local offline vLLM GPU engine).")
            return {
                "skipped": True, 
                "reason": "remote_endpoint", 
                "model": model.short_name, 
                "format": format.value,
                "batch_size": batch_size
            }

        num_prompts = num_prompts or self.config.num_prompts_throughput
        lm = self.config.log_manager

        # Generate output filename
        output_file = lm.get_throughput_result_path(
            model.short_name, format.value, input_length, output_length, batch_size
        )

        # Build command
        cmd = self._build_command(
            model,
            format,
            input_length,
            output_length,
            batch_size,
            num_prompts,
            output_file,
            dataset,
        )

        # Log clear benchmark configuration banner
        # Note: batch_size is not shown because vllm bench throughput doesn't use it
        logger.info(
            f"\n{'='*60}\n"
            f"  BENCHMARK CONFIGURATION\n"
            f"{'='*60}\n"
            f"  Type:           Throughput\n"
            f"  Mode:           Text\n"
            f"  Model:          {model.short_name} ({format.value})\n"
            f"  Input length:   {input_length}\n"
            f"  Output length:  {output_length}\n"
            f"  Num prompts:    {num_prompts}\n"
            f"{'='*60}"
        )

        if self.config.dry_run:
            logger.info(f"[DRY RUN] Would execute: {' '.join(cmd)}")
            return {"dry_run": True, "command": cmd}

        from gbench.utils import require_vllm_engine
        require_vllm_engine("Offline throughput benchmarks")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )

            # Parse plain text output from vllm bench throughput
            # Output format: "Throughput: X requests/s, Y total tokens/s, Z output tokens/s"
            benchmark_result = self._parse_throughput_output(result.stdout)

            # Save results to file
            lm.save_result(output_file, benchmark_result)

            # Log key metrics
            req_tput = benchmark_result.get("request_throughput", "N/A")
            total_tps = benchmark_result.get("total_tokens_per_second", "N/A")
            out_tps = benchmark_result.get("output_tokens_per_second", "N/A")
            logger.info(
                f"Results: {req_tput:.2f} req/s, "
                f"{total_tps:.2f} total tok/s, {out_tps:.2f} output tok/s"
            )

            # Log output if enabled
            if self.config.enable_logging:
                log_file = lm.get_throughput_log_path(
                    model.short_name, format.value,
                    input_length, output_length, batch_size
                )
                lm.save_log(log_file, result.stdout, result.stderr)

            return benchmark_result

        except subprocess.CalledProcessError as e:
            logger.error(f"Benchmark failed: {e}")
            logger.error(f"STDOUT: {e.stdout}")
            logger.error(f"STDERR: {e.stderr}")
            raise RuntimeError(f"Throughput benchmark failed: {e}")


    def run_with_iterations(
        self,
        model: ModelConfig,
        format: ModelFormat,
        input_length: int,
        output_length: int,
        batch_size: int,
        num_prompts: Optional[int] = None,
        dataset: str = "random",
    ) -> dict:
        """Run benchmark with multiple iterations and statistical analysis.

        Args:
            model: Model configuration
            format: Model format (HF or GGUF)
            input_length: Input prompt length in tokens
            output_length: Output generation length in tokens
            batch_size: Batch size
            num_prompts: Number of prompts to test (optional)
            dataset: Dataset name (random or random-mm)

        Returns:
            Dictionary with aggregated results and statistics
        """
        if self.config.num_iterations == 1:
            result = self.run(
                model,
                format,
                input_length,
                output_length,
                batch_size,
                num_prompts,
                dataset,
            )
            # Add configuration metadata for single-iteration case
            result["model"] = model.short_name
            result["format"] = format.value
            result["input_length"] = input_length
            result["output_length"] = output_length
            result["batch_size"] = batch_size
            return result

        logger.info(
            f"Running {self.config.num_iterations} iterations "
            f"(+{self.config.warmup_iterations} warmup)"
        )

        # Warmup iterations
        for i in range(self.config.warmup_iterations):
            logger.info(f"Warmup iteration {i+1}")
            try:
                self.run(
                    model,
                    format,
                    input_length,
                    output_length,
                    batch_size,
                    num_prompts,
                    dataset,
                )
            except Exception as e:
                logger.warning(f"Warmup iteration {i+1} failed: {e}")

        # Actual benchmark iterations
        results = []
        for i in range(self.config.num_iterations):
            logger.info(
                f"Iteration {i+1}/{self.config.num_iterations}"
            )
            try:
                result = self.run(
                    model,
                    format,
                    input_length,
                    output_length,
                    batch_size,
                    num_prompts,
                    dataset,
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Iteration {i+1} failed: {e}")
                continue

        if not results:
            raise RuntimeError("All iterations failed")

        # Aggregate statistics using field names from _parse_throughput_output
        metrics = [
            "request_throughput",
            "total_tokens_per_second",
            "output_tokens_per_second",
        ]

        aggregated = aggregate_benchmark_results(results, metrics)

        # Add configuration metadata
        aggregated["model"] = model.short_name
        aggregated["format"] = format.value
        aggregated["input_length"] = input_length
        aggregated["output_length"] = output_length
        aggregated["batch_size"] = batch_size

        # Validate repeatability
        for metric in ["output_tokens_per_second"]:
            is_valid, msg = validate_repeatability(
                aggregated,
                metric,
                self.config.min_acceptable_cv_percent,
            )
            logger.info(f"{metric}: {msg}")
            aggregated[f"{metric}_repeatability_valid"] = is_valid

            summary = format_statistics_summary(aggregated, metric)
            logger.info(summary)

        return aggregated

    def run_all(
        self,
        model: ModelConfig,
        format: ModelFormat,
        multimodal: bool = False,
    ) -> list[dict]:
        """Run all throughput benchmark configurations for a model.

        Args:
            model: Model configuration
            format: Model format
            multimodal: If True, run multimodal throughput using offline inference

        Returns:
            List of aggregated result dictionaries
        """
        if getattr(self.config, "remote_endpoint", None):
            mode_str = "multimodal " if multimodal else ""
            logger.info(f"Skipping {mode_str}throughput benchmark for remote endpoint (throughput benchmarks require local offline vLLM GPU engine).")
            return [{
                "skipped": True,
                "reason": "remote_endpoint",
                "model": model.short_name,
                "format": format.value,
                "batch_size": "N/A"
            }]

        results = []
        configs = self.config.get_throughput_configs()

        # Deduplicate configs because batch_size is NOT used by vllm bench throughput.
        # The batch_size parameter was removed from the CLI in newer vLLM versions.
        # Without deduplication, we'd run the exact same benchmark multiple times.
        if multimodal:
            # Multimodal: dedupe by (output_length, num_prompts)
            seen_configs = set()
            deduplicated_configs = []
            for cfg in configs:
                key = (cfg["output_length"], cfg["num_prompts"])
                if key not in seen_configs:
                    seen_configs.add(key)
                    deduplicated_configs.append(cfg)
            configs = deduplicated_configs
        else:
            # Text: dedupe by (input_length, output_length, num_prompts)
            seen_configs = set()
            deduplicated_configs = []
            for cfg in configs:
                key = (cfg["input_length"], cfg["output_length"], cfg["num_prompts"])
                if key not in seen_configs:
                    seen_configs.add(key)
                    deduplicated_configs.append(cfg)
            configs = deduplicated_configs

        for cfg in configs:
            try:
                # Wait for GPU memory from previous run to be reclaimed
                self._wait_for_gpu_memory()
                
                if multimodal:
                    # Multi-iteration MM throughput (same as text path)
                    logger.info(
                        f"Running {self.config.num_iterations} MM iterations "
                        f"(+{self.config.warmup_iterations} warmup)"
                    )
                    # Warmup iterations
                    for wi in range(self.config.warmup_iterations):
                        logger.info(f"MM warmup iteration {wi+1}")
                        try:
                            self._run_multimodal_throughput(
                                model, format,
                                num_prompts=cfg["num_prompts"],
                                output_length=cfg["output_length"],
                            )
                        except Exception as we:
                            logger.warning(f"MM warmup {wi+1} failed: {we}")
                    # Actual iterations
                    mm_iter_results = []
                    for mi in range(self.config.num_iterations):
                        logger.info(f"MM iteration {mi+1}/{self.config.num_iterations}")
                        try:
                            mm_r = self._run_multimodal_throughput(
                                model, format,
                                num_prompts=cfg["num_prompts"],
                                output_length=cfg["output_length"],
                            )
                            mm_iter_results.append(mm_r)
                        except Exception as ie:
                            logger.error(f"MM iteration {mi+1} failed: {ie}")
                            continue
                    if not mm_iter_results:
                        raise RuntimeError("All MM iterations failed")
                    # Aggregate with same metrics as text throughput
                    mm_metrics = [
                        "request_throughput",
                        "total_tokens_per_second",
                        "output_tokens_per_second",
                    ]
                    result = aggregate_benchmark_results(mm_iter_results, mm_metrics)
                    # Validate repeatability
                    for mm_m in ["output_tokens_per_second"]:
                        is_valid, msg = validate_repeatability(
                            result, mm_m, self.config.min_acceptable_cv_percent,
                        )
                        logger.info(f"{mm_m}: {msg}")
                        result[f"{mm_m}_repeatability_valid"] = is_valid
                    # Add config values for display
                    result["batch_size"] = "N/A"  # Not applicable for offline inference
                    result["input_length"] = "img"  # Image input, not token length
                    result["output_length"] = cfg["output_length"]
                else:
                    # Use standard CLI-based throughput
                    result = self.run_with_iterations(
                        model,
                        format,
                        input_length=cfg["input_length"],
                        output_length=cfg["output_length"],
                        batch_size=cfg["batch_size"],  # Kept for display, but not used by vLLM
                        num_prompts=cfg["num_prompts"],
                        dataset="random",
                    )
                    # Mark batch_size as N/A in results since it's not actually used
                    result["batch_size"] = "N/A"
                result["multimodal"] = multimodal
                results.append(result)
            except Exception as e:
                logger.error(
                    f"Failed benchmark for {model.short_name}: {e}"
                )
                # Track failure in results
                results.append({
                    "failed": True,
                    "error": str(e),
                    "model": model.short_name,
                    "format": format.value,
                    "input_length": cfg["input_length"],
                    "output_length": cfg["output_length"],
                    "batch_size": cfg["batch_size"],
                    "multimodal": multimodal,
                })

        return results

    def _run_multimodal_throughput(
        self,
        model: ModelConfig,
        format: ModelFormat,
        num_prompts: int,
        output_length: int,
    ) -> dict:
        """Run multimodal throughput benchmark using vllm-chat backend.
        
        Uses vllm bench throughput with --backend vllm-chat and a custom
        ShareGPT-format dataset containing synthetic images.
        
        Args:
            model: Model configuration
            format: Model format
            num_prompts: Number of prompts to process
            output_length: Maximum output tokens per prompt
            
        Returns:
            Dictionary with throughput metrics
        """
        import tempfile
        import base64
        import io
        from PIL import Image
        import numpy as np
        
        # Log clear benchmark configuration banner
        logger.info(
            f"\n{'='*60}\n"
            f"  BENCHMARK CONFIGURATION\n"
            f"{'='*60}\n"
            f"  Type:           Throughput\n"
            f"  Mode:           Multimodal\n"
            f"  Model:          {model.short_name} ({format.value})\n"
            f"  Output length:  {output_length}\n"
            f"  Num prompts:    {num_prompts}\n"
            f"{'='*60}"
        )
        
        # Check for dry run
        if self.config.dry_run:
            logger.info("[DRY RUN] Would execute multimodal throughput benchmark")
            return {"dry_run": True}
            
        if getattr(self.config, "remote_endpoint", None):
            logger.info("Skipping multimodal throughput benchmark for remote endpoint (throughput benchmarks require local offline vLLM GPU engine).")
            return {
                "skipped": True, 
                "reason": "remote_endpoint", 
                "model": model.short_name, 
                "format": format.value,
                "batch_size": "N/A"
            }
        
        # Create temp directory for images
        import os
        img_dir = tempfile.mkdtemp()
        
        # Create ShareGPT-format dataset with synthetic images
        # Format required by vLLM:
        # - conversations: list with 2+ turns (human prompt, assistant response)
        # - image: path or URL to image file
        dataset = []
        for i in range(num_prompts):
            # Create and save synthetic 256x256 RGB image
            img_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            image = Image.fromarray(img_array)
            img_path = os.path.join(img_dir, f"img_{i}.jpg")
            image.save(img_path, format='JPEG')
            
            # ShareGPT format: value is string, image is separate field
            dataset.append({
                "conversations": [
                    {
                        "from": "human",
                        "value": "Describe this image in detail."
                    },
                    {
                        "from": "gpt", 
                        "value": "This is a synthetic image with random pixel values."
                    }
                ],
                "image": img_path  # Image path as separate field
            })
        
        # Write dataset to temp file
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        ) as f:
            json.dump(dataset, f)
            dataset_path = f.name
        
        try:
            # Build command using vllm-chat backend
            cmd = [
                "vllm", "bench", "throughput",
                "--model", model.get_model_path(format),
                "--backend", "vllm-chat",
                "--dataset-name", "sharegpt",
                "--dataset-path", dataset_path,
                "--num-prompts", str(num_prompts),
                "--output-json", "/tmp/mm_throughput_result.json",
                # Allow loading local images from temp directory
                "--allowed-local-media-path", img_dir,
            ]
            
            # Add GPU memory utilization
            mm_gpu_mem = get_gpu_memory_utilization(model.total_params_b)
            if mm_gpu_mem:
                cmd.extend([
                    "--gpu-memory-utilization",
                    str(mm_gpu_mem),
                ])
            
            # GGUF models need explicit tokenizer
            if format == ModelFormat.GGUF:
                cmd.extend(["--tokenizer", model.hf_model_id])
            
            # Multi-GPU: tensor parallel (critical for models requiring TP>1)
            num_gpus = self.config.tensor_parallel_size or 1
            if num_gpus > 1:
                cmd.extend(["--tensor-parallel-size", str(num_gpus)])
            
            # Uniform context length for fair comparison
            max_model_len = get_max_model_len()
            if max_model_len:
                cmd.extend(["--max-model-len", str(max_model_len)])
            
            # Limit max concurrent sequences for uniform memory usage
            if self.config.max_num_seqs:
                cmd.extend(["--max-num-seqs", str(self.config.max_num_seqs)])
            
            # Performance optimization flags (parity with text throughput path)
            if self.config.enable_chunked_prefill:
                cmd.append("--enable-chunked-prefill")
            
            if self.config.max_num_batched_tokens:
                cmd.extend([
                    "--max-num-batched-tokens",
                    str(self.config.max_num_batched_tokens),
                ])
            
            # Output length control
            if output_length:
                cmd.extend(["--output-len", str(output_length)])
            
            # Reproducibility seed (matches text throughput path)
            cmd.extend(["--seed", "83"])
            
            logger.info(f"Command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800,  # 30 minute timeout
            )
            
            if result.returncode != 0:
                logger.error(f"Multimodal throughput STDOUT: {result.stdout}")
                logger.error(f"Multimodal throughput STDERR: {result.stderr}")
                raise RuntimeError(
                    f"Multimodal throughput failed: {result.stderr}"
                )
            
            # Log stdout for debugging
            logger.debug(f"Throughput STDOUT: {result.stdout[-1000:]}")
            
            # Parse output JSON if it exists
            import os
            json_path = "/tmp/mm_throughput_result.json"
            if os.path.exists(json_path):
                logger.info(f"Reading results from {json_path}")
                with open(json_path) as f:
                    metrics = json.load(f)
                logger.info(f"Throughput metrics: {metrics}")
            else:
                # Parse from stdout as fallback
                logger.warning(f"No JSON output file, parsing stdout")
                logger.info(f"STDOUT (last 500 chars): {result.stdout[-500:]}")
                metrics = self._parse_throughput_output(result.stdout)
            
            # Map field names for CLI compatibility
            metrics["model_short"] = model.short_name
            metrics["model"] = model.short_name
            metrics["format"] = format.value
            # Map vLLM's 'tokens_per_second' to expected field name
            if "tokens_per_second" in metrics:
                metrics["output_tokens_per_second"] = metrics["tokens_per_second"]
            return metrics
            
        finally:
            import os
            import shutil
            os.unlink(dataset_path)
            if os.path.exists("/tmp/mm_throughput_result.json"):
                os.unlink("/tmp/mm_throughput_result.json")
            # Clean up temp image directory
            if os.path.exists(img_dir):
                shutil.rmtree(img_dir)

    def _build_command(
        self,
        model: ModelConfig,
        format: ModelFormat,
        input_length: int,
        output_length: int,
        batch_size: int,
        num_prompts: int,
        output_file: Path,
        dataset: str = "random",
    ) -> list[str]:
        """Build vllm bench throughput command."""
        # For multimodal, use HF dataset with VQA
        # random-mm is not available for throughput in current vLLM
        is_multimodal = dataset == "random-mm"
        
        cmd = [
            "vllm",
            "bench",
            "throughput",
            "--model",
            model.get_model_path(format),
        ]
        
        if is_multimodal:
            # Use HuggingFace VQA dataset for multimodal throughput
            cmd.extend([
                "--dataset-name", "hf",
                "--dataset", "lmms-lab/textvqa",
                "--hf-split", "validation",
            ])
        else:
            cmd.extend([
                "--dataset-name", dataset,
                "--input-len", str(input_length),
                "--output-len", str(output_length),
                "--random-range-ratio", "0.5",  # ±50% variance for realistic workload
                "--seed", "83",  # Reproducible randomness
            ])
        
        cmd.extend([
            "--num-prompts", str(num_prompts),
        ])

        # Add GPU memory utilization
        gpu_mem = get_gpu_memory_utilization(model.total_params_b)
        if gpu_mem:
            cmd.extend(
                [
                    "--gpu-memory-utilization",
                    str(gpu_mem),
                ]
            )

        # Add performance optimization flags
        if self.config.enable_chunked_prefill:
            cmd.append("--enable-chunked-prefill")
        
        if self.config.max_num_batched_tokens:
            cmd.extend([
                "--max-num-batched-tokens",
                str(self.config.max_num_batched_tokens),
            ])

        # GGUF models need explicit tokenizer from HF model ID
        if format == ModelFormat.GGUF:
            cmd.extend(["--tokenizer", model.hf_model_id])
        
        # Multi-GPU: tensor parallel
        num_gpus = self.config.tensor_parallel_size or 1
        if num_gpus > 1:
            cmd.extend(["--tensor-parallel-size", str(num_gpus)])
        
        # Uniform context length for fair comparison
        max_model_len = get_max_model_len()
        if max_model_len:
            cmd.extend(["--max-model-len", str(max_model_len)])

        # Limit max concurrent sequences for uniform memory usage
        if self.config.max_num_seqs:
            cmd.extend(["--max-num-seqs", str(self.config.max_num_seqs)])

        return cmd

    def _parse_throughput_output(self, stdout: str) -> dict:
        """Parse plain text output from vllm bench throughput.
        
        Expected format:
        Throughput: X requests/s, Y total tokens/s, Z output tokens/s
        Total num prompt tokens: NNN
        Total num output tokens: MMM
        
        Args:
            stdout: Raw stdout from vllm bench throughput
            
        Returns:
            Parsed metrics as dictionary
        """
        import re
        
        result = {}
        
        # Parse throughput line
        throughput_match = re.search(
            r'Throughput:\s+([\d.]+)\s+requests/s,\s+([\d.]+)\s+total tokens/s,\s+([\d.]+)\s+output tokens/s',
            stdout
        )
        if throughput_match:
            result["request_throughput"] = float(throughput_match.group(1))
            result["total_tokens_per_second"] = float(throughput_match.group(2))
            result["output_tokens_per_second"] = float(throughput_match.group(3))
        
        # Parse prompt tokens
        prompt_match = re.search(r'Total num prompt tokens:\s+(\d+)', stdout)
        if prompt_match:
            result["total_prompt_tokens"] = int(prompt_match.group(1))
        
        # Parse output tokens
        output_match = re.search(r'Total num output tokens:\s+(\d+)', stdout)
        if output_match:
            result["total_output_tokens"] = int(output_match.group(1))
        
        if not result:
            raise RuntimeError(f"Could not parse throughput output: {stdout[-500:]}")
        
        return result
