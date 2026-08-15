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

"""Serving benchmark runner using vLLM Python API directly."""

import asyncio
import atexit
import json
import logging
import os
import requests
import shutil
import signal
import subprocess
import tempfile
import time
import numpy as np
from pathlib import Path
from typing import Optional, Any

from ..core.config import BenchmarkConfig, get_max_model_len, get_gpu_memory_utilization, get_server_timeout
from ..core.models import ModelConfig, ModelFormat
from ..analysis.statistics import (
    aggregate_benchmark_results,
    format_statistics_summary,
    validate_repeatability,
)

# vLLM benchmark imports will be loaded dynamically in active methods

logger = logging.getLogger(__name__)


class ServingBenchmarkRunner:
    """Runner for vLLM serving benchmarks using Python API."""

    def __init__(self, config: BenchmarkConfig):
        """Initialize the serving benchmark runner.

        Args:
            config: Benchmark configuration
        """
        self.config = config
        self.server_process: Optional[subprocess.Popen] = None
        self._server_log_file = None  # File handle for server logs
        self.server_port = int(os.environ.get("GBENCH_SERVER_PORT", "8000"))
        self._mm_image_dir = None  # Temp dir for generated images
        self._mm_image_paths = []  # Generated image file paths
        # Register cleanup on exit
        atexit.register(self._cleanup_server)

    def _resolve_model_id(self, model: ModelConfig, format: ModelFormat) -> str:
        """Resolve the model ID to use for API requests."""
        if not self.config.remote_endpoint:
            return model.get_model_path(format)
            
        base_url = self.config.remote_endpoint.rstrip("/")
        try:
            import requests
            resp = requests.get(f"{base_url}/v1/models", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                if "data" in data and len(data["data"]) > 0:
                    remote_ids = [m["id"] for m in data["data"]]
                    for candidate in [model.hf_model_id, model.name, model.short_name]:
                        if candidate in remote_ids:
                            return candidate
                    return remote_ids[0]
        except Exception:
            pass
            
        return model.hf_model_id


    def _wait_for_gpu_memory(self, min_free_gb: float = 70.0, max_wait_seconds: int = 120):
        """Wait for GPU memory to be reclaimed by CUDA driver.
        
        Uses nvidia-smi to actively poll until sufficient memory is free.
        Respects CUDA_VISIBLE_DEVICES to only check the assigned GPUs.
        
        Args:
            min_free_gb: Minimum free memory required in GB
            max_wait_seconds: Maximum time to wait before giving up
        """
        if self.config.remote_endpoint:
            return

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

    def _setup_mm_images(self, num_images: int = 50) -> str:
        """Generate synthetic images for multimodal serving benchmark.

        Creates a temp directory with random JPEG images.

        Args:
            num_images: Number of images to generate

        Returns:
            Path to the temp directory containing images
        """
        import numpy as np
        from PIL import Image

        img_dir = tempfile.mkdtemp(prefix="serving_mm_images_")
        self._mm_image_paths = []

        for i in range(num_images):
            # 256x256 random images — small enough for fast I/O,
            # large enough to exercise the vision encoder
            img_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img_path = os.path.join(img_dir, f"serve_img_{i:03d}.jpg")
            img.save(img_path, format="JPEG", quality=85)
            self._mm_image_paths.append(img_path)

        logger.info(f"Generated {num_images} synthetic images in {img_dir}")
        self._mm_image_dir = img_dir
        return img_dir

    def _cleanup_mm_images(self):
        """Clean up temp image directory."""
        if self._mm_image_dir and os.path.exists(self._mm_image_dir):
            shutil.rmtree(self._mm_image_dir, ignore_errors=True)
            logger.info(f"Cleaned up MM image directory: {self._mm_image_dir}")
            self._mm_image_dir = None
            self._mm_image_paths = []

    def _generate_sample_requests(
        self,
        model: ModelConfig,
        format: ModelFormat,
        num_prompts: int,
        input_len: int = 512,
        output_len: int = 128,
    ) -> tuple[list[Any], Any]:
        """Generate sample requests for benchmarking using vLLM's RandomDataset.

        Args:
            model: Model configuration
            format: Model format
            num_prompts: Number of prompts to generate
            input_len: Input token length
            output_len: Output token length

        Returns:
            Tuple of (list of SampleRequest, tokenizer)
        """
        from gbench.utils import check_vllm_available, require_vllm_engine

        if not check_vllm_available():
            if not self.config.remote_endpoint:
                require_vllm_engine("Serving benchmark datasets")

            in_len = self.config.input_lengths[0] if self.config.input_lengths else input_len
            out_len = self.config.output_lengths[0] if self.config.output_lengths else output_len

            from dataclasses import dataclass
            @dataclass
            class FallbackSampleRequest:
                prompt: str
                prompt_len: int
                expected_output_len: int
                multi_modal_data: Optional[dict] = None

            sample_text = "The quick brown fox jumps over the lazy dog. " * (in_len // 8 + 1)
            requests = [
                FallbackSampleRequest(
                    prompt=sample_text,
                    prompt_len=in_len,
                    expected_output_len=out_len,
                )
                for _ in range(num_prompts)
            ]
            return requests, None

        if self.config.dataset == "sharegpt":
            from vllm.benchmarks.datasets import ShareGPTDataset
            from gbench.utils import safe_get_tokenizer
            tokenizer_path = model.hf_model_id if (self.config.remote_endpoint or format == ModelFormat.GGUF) else model.get_model_path(format)
            tokenizer = safe_get_tokenizer(tokenizer_path, custom_tokenizer=self.config.tokenizer)
            
            # Use ShareGPT default path convention or download
            sharegpt_path = getattr(self.config, "dataset_path", None) or os.environ.get("SHAREGPT_PATH", os.path.expanduser("~/.cache/gbench/ShareGPT_V3_unfiltered_cleaned_split.json"))
            if not os.path.exists(sharegpt_path):
                # Download just-in-time if missing
                os.makedirs(os.path.dirname(sharegpt_path), exist_ok=True)
                import urllib.request
                logger.info(f"Downloading ShareGPT dataset to {sharegpt_path}...")
                urllib.request.urlretrieve("https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json", sharegpt_path)
                
            dataset = ShareGPTDataset(dataset_path=sharegpt_path)
            requests = dataset.sample(
                tokenizer=tokenizer,
                num_requests=num_prompts,
            )
            return requests, tokenizer
        elif self.config.dataset == "custom":
            from vllm.benchmarks.datasets import CustomDataset
            from gbench.utils import safe_get_tokenizer
            tokenizer_path = model.hf_model_id if (self.config.remote_endpoint or format == ModelFormat.GGUF) else model.get_model_path(format)
            tokenizer = safe_get_tokenizer(tokenizer_path, custom_tokenizer=self.config.tokenizer)
            
            dataset_path = getattr(self.config, "dataset_path", None)
            if not dataset_path:
                raise ValueError("--dataset-path is required when using custom dataset")
                
            dataset = CustomDataset(dataset_path=dataset_path)
            out_len = self.config.output_lengths[0] if self.config.output_lengths else 128
            requests = dataset.sample(
                tokenizer=tokenizer,
                num_requests=num_prompts,
                output_len=out_len,
            )
            return requests, tokenizer
        elif self.config.dataset == "hf":
            from vllm.benchmarks.datasets import get_samples
            from gbench.utils import safe_get_tokenizer
            import argparse
            tokenizer_path = model.hf_model_id if (self.config.remote_endpoint or format == ModelFormat.GGUF) else model.get_model_path(format)
            tokenizer = safe_get_tokenizer(tokenizer_path, custom_tokenizer=self.config.tokenizer)
            
            dataset_path = getattr(self.config, "dataset_path", None)
            if not dataset_path:
                raise ValueError("--dataset-path is required when using hf dataset")
                
            args = argparse.Namespace(
                dataset_name="hf",
                dataset_path=dataset_path,
                hf_name=dataset_path,
                hf_split="train",
                hf_subset=None,
                disable_shuffle=False,
                seed=83,
                num_prompts=num_prompts,
                custom_output_len=self.config.output_lengths[0] if self.config.output_lengths else 128,
                skip_chat_template=False,
                request_id_prefix="",
                no_oversample=False,
            )
            requests = get_samples(args, tokenizer)
            return requests, tokenizer
        else:
            from vllm.benchmarks.datasets import RandomDataset
            from gbench.utils import safe_get_tokenizer
            
            tokenizer_path = model.hf_model_id if (self.config.remote_endpoint or format == ModelFormat.GGUF) else model.get_model_path(format)
            tokenizer = safe_get_tokenizer(tokenizer_path, custom_tokenizer=self.config.tokenizer)
            
            # Use config's input/output lengths
            in_len = self.config.input_lengths[0] if self.config.input_lengths else input_len
            out_len = self.config.output_lengths[0] if self.config.output_lengths else output_len

            dataset = RandomDataset(random_seed=83)
            requests = dataset.sample(
                tokenizer=tokenizer,
                num_requests=num_prompts,
                input_len=in_len,
                output_len=out_len,
                range_ratio=0.5,
            )
            return requests, tokenizer

    async def _run_mm_serving_benchmark(
        self,
        model: ModelConfig,
        format: ModelFormat,
        max_concurrency: int,
        num_prompts: int,
    ) -> dict:
        """Run multimodal serving benchmark via aiohttp + SSE.

        Sends chat completion requests with images, measures TTFT, TPOT, ITL
        from SSE streaming response tokens.

        Args:
            model: Model configuration
            format: Model format
            max_concurrency: Maximum concurrent requests (from batch_size)
            num_prompts: Number of requests to send

        Returns:
            Dictionary with serving metrics matching text serving output format
        """
        import aiohttp

        model_path = self._resolve_model_id(model, format)
        base_url = self.config.remote_endpoint or f"http://127.0.0.1:{self.server_port}"
        url = base_url.rstrip("/")
        api_url = f"{url}/chat/completions" if url.endswith("/v1") else f"{url}/v1/chat/completions"

        semaphore = asyncio.Semaphore(max_concurrency)
        all_ttfts = []  # Time to first token (ms)
        all_tpots = []  # Time per output token (ms)
        all_itls = []   # Inter-token latencies (ms)
        all_e2els = []  # End-to-end latencies (ms)
        completed = 0
        failed = 0
        total_output_tokens = 0

        async def send_mm_request(session, img_path, request_id, pbar=None):
            nonlocal completed, failed, total_output_tokens
            if self.config.remote_endpoint:
                import base64
                with open(img_path, "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                image_url = f"data:image/jpeg;base64,{encoded_string}"
            else:
                image_url = f"file://{img_path}"

            payload = {
                "model": model_path,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image in detail."},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url},
                        },
                    ],
                }],
                "max_tokens": 128,
                "stream": True,
            }
            async with semaphore:
                start = time.perf_counter()
                token_times = []  # Timestamps for each token
                try:
                    async with session.post(api_url, json=payload) as resp:
                        if resp.status != 200:
                            error_text = await resp.text()
                            logger.debug(
                                f"MM request {request_id} HTTP {resp.status}: "
                                f"{error_text[:200]}"
                            )
                            failed += 1
                            return
                        # Parse SSE stream — record each token's arrival time
                        async for raw_line in resp.content:
                            line = raw_line.decode("utf-8", errors="ignore").strip()
                            if (
                                line.startswith("data:")
                                and "[DONE]" not in line
                            ):
                                token_times.append(time.perf_counter())

                    end = time.perf_counter()

                    if not token_times:
                        failed += 1
                        return

                    # Compute metrics from token timestamps
                    ttft_ms = (token_times[0] - start) * 1000
                    e2el_ms = (end - start) * 1000
                    num_tokens = len(token_times)
                    total_output_tokens += num_tokens

                    all_ttfts.append(ttft_ms)
                    all_e2els.append(e2el_ms)

                    if num_tokens > 1:
                        # TPOT = (last_token - first_token) / (num_tokens - 1)
                        generation_time_ms = (token_times[-1] - token_times[0]) * 1000
                        tpot_ms = generation_time_ms / (num_tokens - 1)
                        all_tpots.append(tpot_ms)

                        # ITL = inter-token latencies
                        for j in range(1, len(token_times)):
                            itl_ms = (token_times[j] - token_times[j-1]) * 1000
                            all_itls.append(itl_ms)

                    completed += 1
                except Exception as e:
                    logger.debug(f"MM request {request_id} error: {e}")
                    failed += 1
                finally:
                    if pbar:
                        pbar.update(1)

        connector = aiohttp.TCPConnector(limit=max_concurrency + 20)
        timeout = aiohttp.ClientTimeout(total=600)
        from tqdm import tqdm
        with tqdm(total=num_prompts, desc="MM Serving") as pbar:
            async with aiohttp.ClientSession(
                connector=connector, timeout=timeout
            ) as session:
                overall_start = time.perf_counter()
                tasks = [
                    send_mm_request(
                        session,
                        self._mm_image_paths[i % len(self._mm_image_paths)],
                        i,
                        pbar,
                    )
                    for i in range(num_prompts)
                ]
                await asyncio.gather(*tasks)
                overall_elapsed = time.perf_counter() - overall_start

        if not all_ttfts:
            raise RuntimeError(
                f"All {num_prompts} MM serving requests failed "
                f"({failed} errors)"
            )

        # Compute percentile metrics matching text serving output format
        import numpy as np
        ttft_arr = np.array(all_ttfts)
        e2el_arr = np.array(all_e2els)
        request_throughput = completed / overall_elapsed
        output_throughput = total_output_tokens / overall_elapsed

        result = {
            "request_throughput": request_throughput,
            "output_throughput": output_throughput,
            "total_output_tokens": int(total_output_tokens),
            "completed": completed,
            "failed": failed,
            "total": num_prompts,
            "duration": overall_elapsed,
            # TTFT metrics
            "mean_ttft_ms": float(np.mean(ttft_arr)),
            "median_ttft_ms": float(np.median(ttft_arr)),
            "p99_ttft_ms": float(np.percentile(ttft_arr, 99)),
            # E2EL metrics
            "mean_e2el_ms": float(np.mean(e2el_arr)),
            "median_e2el_ms": float(np.median(e2el_arr)),
            "p99_e2el_ms": float(np.percentile(e2el_arr, 99)),
        }

        # TPOT metrics (need >1 token per request)
        if all_tpots:
            tpot_arr = np.array(all_tpots)
            result["mean_tpot_ms"] = float(np.mean(tpot_arr))
            result["median_tpot_ms"] = float(np.median(tpot_arr))
            result["p99_tpot_ms"] = float(np.percentile(tpot_arr, 99))

        # ITL metrics
        if all_itls:
            itl_arr = np.array(all_itls)
            result["mean_itl_ms"] = float(np.mean(itl_arr))
            result["median_itl_ms"] = float(np.median(itl_arr))
            result["p99_itl_ms"] = float(np.percentile(itl_arr, 99))

        return result

    async def _run_benchmark_async(
        self,
        model: ModelConfig,
        format: ModelFormat,
        batch_size: int,
        num_prompts: int,
        input_requests: list[Any],
        tokenizer,
    ) -> dict:
        """Run serving benchmark using vLLM's benchmark() API directly.

        Args:
            model: Model configuration
            format: Model format
            batch_size: Max concurrency
            num_prompts: Number of prompts
            input_requests: Pre-generated sample requests
            tokenizer: Tokenizer

        Returns:
            Dictionary with benchmark results
        """
        from gbench.utils import check_vllm_available, require_vllm_engine

        model_path = self._resolve_model_id(model, format)
        base_url = self.config.remote_endpoint or f"http://127.0.0.1:{self.server_port}"

        if self.config.remote_endpoint or not check_vllm_available():
            if not self.config.remote_endpoint:
                require_vllm_engine("Serving benchmark execution")
            return await self._run_lightweight_remote_benchmark(
                base_url=base_url,
                model_id=model_path,
                input_requests=input_requests,
                batch_size=batch_size,
            )

        from vllm.benchmarks.serve import benchmark, TaskType
        api_url = f"{base_url}/v1/completions"

        # Parse request rate
        request_rate = float("inf") if self.config.request_rate == "inf" else float(self.config.request_rate)

        result = await benchmark(
            task_type=TaskType.GENERATION,
            endpoint_type="openai",  # Maps to async_request_openai_completions
            api_url=api_url,
            base_url=base_url,
            model_id=model_path,
            model_name=model_path,
            tokenizer=tokenizer,
            input_requests=input_requests,
            logprobs=None,
            request_rate=request_rate,
            burstiness=1.0,
            max_concurrency=batch_size,
            disable_tqdm=False,
            num_warmups=0,  # We handle warmups ourselves
            profile=False,
            selected_percentile_metrics=["ttft", "tpot", "itl", "e2el"],
            selected_percentiles=[50.0, 99.0],
            ignore_eos=False,
            goodput_config_dict={},
            lora_modules=None,
            extra_headers=None,
            extra_body=None,
        )

        return result

    async def _run_lightweight_remote_benchmark(
        self,
        base_url: str,
        model_id: str,
        input_requests: list,
        batch_size: int,
    ) -> Any:
        """Pure-Python async streaming benchmark runner for remote HTTP endpoints."""
        import time
        import urllib.request
        import json
        import asyncio
        from dataclasses import dataclass

        @dataclass
        class MetricResult:
            metrics: dict

        url = f"{base_url.rstrip('/')}/chat/completions"
        if not url.startswith("http://") and not url.startswith("https://"):
            url = f"http://{url}"

        semaphore = asyncio.Semaphore(batch_size)
        ttfts = []
        tpots = []
        latencies = []
        completed_requests = 0
        total_output_tokens = 0
        start_time = time.perf_counter()

        def _send_single_request(req):
            payload = json.dumps({
                "model": model_id,
                "messages": [{"role": "user", "content": req.prompt}],
                "max_tokens": getattr(req, "expected_output_len", 128),
                "stream": True,
            }).encode("utf-8")

            request = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
            )

            req_start = time.perf_counter()
            first_token_time = None
            tokens_count = 0

            req_timeout = getattr(self.config, "timeout", 300)
            with urllib.request.urlopen(request, timeout=req_timeout) as resp:
                for line in resp:
                    line_str = line.decode("utf-8").strip()
                    if line_str.startswith("data: ") and line_str != "data: [DONE]":
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                        try:
                            data = json.loads(line_str[6:])
                            delta = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            if delta:
                                tokens_count += 1
                        except Exception:
                            pass

            req_end = time.perf_counter()
            if first_token_time is None:
                first_token_time = req_end

            ttft = first_token_time - req_start
            latency = req_end - req_start
            tpot = (req_end - first_token_time) / max(tokens_count - 1, 1) if tokens_count > 1 else latency

            return ttft, tpot, latency, tokens_count

        async def _worker(req):
            nonlocal completed_requests, total_output_tokens
            async with semaphore:
                try:
                    ttft, tpot, latency, tokens = await asyncio.to_thread(_send_single_request, req)
                    ttfts.append(ttft)
                    tpots.append(tpot)
                    latencies.append(latency)
                    completed_requests += 1
                    total_output_tokens += tokens
                except Exception as e:
                    logger.warning(f"Remote benchmark request error: {e}")

        tasks = [_worker(req) for req in input_requests]
        await asyncio.gather(*tasks)

        end_time = time.perf_counter()
        dur = end_time - start_time

        def p50(lst):
            return float(np.percentile(lst, 50)) if lst else 0.0
        def p99(lst):
            return float(np.percentile(lst, 99)) if lst else 0.0

        metrics = {
            "mean_ttft_ms": float(np.mean(ttfts)) * 1000 if ttfts else 0.0,
            "median_ttft_ms": p50(ttfts) * 1000,
            "p99_ttft_ms": p99(ttfts) * 1000,
            "mean_tpot_ms": float(np.mean(tpots)) * 1000 if tpots else 0.0,
            "median_tpot_ms": p50(tpots) * 1000,
            "p99_tpot_ms": p99(tpots) * 1000,
            "mean_itl_ms": float(np.mean(tpots)) * 1000 if tpots else 0.0,
            "median_itl_ms": p50(tpots) * 1000,
            "p99_itl_ms": p99(tpots) * 1000,
            "mean_e2el_ms": float(np.mean(latencies)) * 1000 if latencies else 0.0,
            "median_e2el_ms": p50(latencies) * 1000,
            "p99_e2el_ms": p99(latencies) * 1000,
            "request_throughput": completed_requests / dur if dur > 0 else 0.0,
            "output_throughput": total_output_tokens / dur if dur > 0 else 0.0,
            "completed_requests": completed_requests,
            "duration": dur,
        }

        return metrics

    def run(
        self,
        model: ModelConfig,
        format: ModelFormat,
        batch_size: int,
        num_prompts: Optional[int] = None,
        dataset: Optional[str] = None,
        multimodal: bool = False,
    ) -> dict:
        """Run serving benchmark for a specific configuration (text only).

        For multimodal, use run_with_iterations() which routes to
        _run_mm_with_iterations() for real image requests.

        Args:
            model: Model configuration
            format: Model format (HF or GGUF)
            batch_size: Number of concurrent requests
            num_prompts: Number of prompts to test (optional)
            dataset: Dataset to use (optional, ignored - we use RandomDataset)
            multimodal: If True, delegates to _run_mm_with_iterations()

        Returns:
            Dictionary with benchmark results

        Raises:
            RuntimeError: If benchmark execution fails
        """
        num_prompts = num_prompts or self.config.num_prompts

        # Route multimodal to the proper MM implementation
        if multimodal:
            return self._run_mm_with_iterations(
                model, format, batch_size, num_prompts
            )

        lm = self.config.log_manager

        # Generate output filename
        output_file = lm.get_serving_result_path(
            model.short_name, format.value, batch_size, multimodal
        )

        # Start vLLM server
        if not self.config.dry_run and not self.config.remote_endpoint:
            if not self._start_server(model, format):
                raise RuntimeError("Failed to start vLLM server")

        try:
            if self.config.dry_run:
                logger.info(f"[DRY RUN] Would run benchmark for {model.short_name}")
                return {"dry_run": True}

            # Log clear benchmark configuration banner
            logger.info(
                f"\n{'='*60}\n"
                f"  BENCHMARK CONFIGURATION\n"
                f"{'='*60}\n"
                f"  Type:           Serving\n"
                f"  Mode:           Text\n"
                f"  Model:          {model.short_name} ({format.value})\n"
                f"  Batch size:     {batch_size}\n"
                f"  Num prompts:    {num_prompts}\n"
                f"{'='*60}"
            )

            # Generate sample requests
            logger.info(f"Generating {num_prompts} sample requests...")
            input_requests, tokenizer = self._generate_sample_requests(
                model, format, num_prompts
            )

            # Run benchmark
            result = asyncio.run(self._run_benchmark_async(
                model, format, batch_size, num_prompts, input_requests, tokenizer
            ))

            # Save result to file
            result["model"] = model.short_name
            result["model_short"] = model.short_name
            result["model_name"] = model.name
            result["format"] = format.value
            result["batch_size"] = batch_size
            result["multimodal"] = multimodal
            result["modality"] = "multimodal" if multimodal else "text"
            result["benchmark_type"] = "serving"
            result["output_token_throughput"] = result.get("output_throughput", result.get("output_token_throughput", 0.0))
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)

            # Log key metrics
            req_tput = result.get("request_throughput", "N/A")
            ttft = result.get("mean_ttft_ms", "N/A")
            tpot = result.get("mean_tpot_ms", "N/A")
            logger.info(
                f"Results: request_throughput={req_tput:.2f} req/s, "
                f"TTFT={ttft:.2f}ms, TPOT={tpot:.2f}ms"
            )

            return result

        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            raise RuntimeError(f"Serving benchmark failed: {e}")
        
        finally:
            # Always cleanup server
            if not self.config.dry_run and not self.config.remote_endpoint:
                self._cleanup_server()

    def _get_output_path(
        self,
        model: ModelConfig,
        format: ModelFormat,
        batch_size: int,
        multimodal: bool = False,
    ) -> Path:
        """Generate output path for benchmark results.
        
        Args:
            model: Model configuration
            format: Model format
            batch_size: Batch size used
            multimodal: Whether this is a multimodal benchmark
            
        Returns:
            Path to the output JSON file
        """
        lm = self.config.log_manager
        return lm.get_serving_result_path(
            model.short_name, format.value, batch_size, multimodal
        )

    def run_with_iterations(
        self,
        model: ModelConfig,
        format: ModelFormat,
        batch_size: int,
        num_prompts: Optional[int] = None,
        dataset: Optional[str] = None,
        multimodal: bool = False,
    ) -> dict:
        """Run benchmark with multiple iterations and statistical analysis.

        For text: uses vLLM's benchmark() API with RandomDataset.
        For multimodal: uses aiohttp + SSE with real images via /v1/chat/completions.

        Args:
            model: Model configuration
            format: Model format (HF or GGUF)
            batch_size: Number of concurrent requests
            num_prompts: Number of prompts to test (optional)
            dataset: Dataset to use (optional)
            multimodal: If True, send real image requests

        Returns:
            Dictionary with aggregated results and statistics
        """
        # For single iteration with no warmup on text-only, shortcut
        if not multimodal and self.config.num_iterations == 1 and self.config.warmup_iterations == 0:
            return self.run(
                model, format, batch_size, num_prompts, dataset, multimodal=False,
            )

        # Log clear benchmark configuration banner
        mode_str = "Multimodal" if multimodal else "Text"
        logger.info(
            f"\n{'='*60}\n"
            f"  BENCHMARK CONFIGURATION\n"
            f"{'='*60}\n"
            f"  Type:           Serving\n"
            f"  Mode:           {mode_str}\n"
            f"  Model:          {model.short_name} ({format.value})\n"
            f"  Batch size:     {batch_size}\n"
            f"  Num prompts:    {num_prompts or self.config.num_prompts}\n"
            f"{'='*60}"
        )

        logger.info(
            f"Running {self.config.num_iterations} iterations "
            f"(+{self.config.warmup_iterations} warmup)"
        )

        num_prompts = num_prompts or self.config.num_prompts

        if multimodal:
            return self._run_mm_with_iterations(
                model, format, batch_size, num_prompts
            )
        else:
            return self._run_text_with_iterations(
                model, format, batch_size, num_prompts
            )

    def _run_text_with_iterations(
        self,
        model: ModelConfig,
        format: ModelFormat,
        batch_size: int,
        num_prompts: int,
    ) -> dict:
        """Run text serving benchmark with multiple iterations."""
        # Start server once for all iterations
        if not self.config.dry_run and not self.config.remote_endpoint:
            if not self._start_server(model, format):
                raise RuntimeError("Failed to start vLLM server")

        try:
            # Generate sample requests once (reused for all iterations)
            logger.info(f"Generating {num_prompts} sample requests...")
            input_requests, tokenizer = self._generate_sample_requests(
                model, format, num_prompts
            )

            # Warmup iterations
            warmup_total = self.config.warmup_iterations
            for i in range(warmup_total):
                progress = "█" * (i + 1) + "░" * (warmup_total - i - 1)
                logger.info(f"Warmup [{progress}] {i+1}/{warmup_total} starting...")
                iter_start = time.time()
                try:
                    warmup_reqs = input_requests[:min(2, len(input_requests))] if self.config.remote_endpoint else input_requests
                    asyncio.run(self._run_benchmark_async(
                        model, format, min(batch_size, len(warmup_reqs)), len(warmup_reqs), warmup_reqs, tokenizer
                    ))
                    elapsed = time.time() - iter_start
                    logger.info(f"  Warmup {i+1} completed in {elapsed:.1f}s")
                except Exception as e:
                    elapsed = time.time() - iter_start
                    logger.warning(f"Warmup iteration {i+1} failed after {elapsed:.1f}s: {e}")

            # Actual benchmark iterations
            results = []
            iter_total = self.config.num_iterations
            for i in range(iter_total):
                progress = "█" * (i + 1) + "░" * (iter_total - i - 1)
                logger.info(f"Benchmark [{progress}] {i+1}/{iter_total} starting...")
                iter_start = time.time()
                try:
                    result = asyncio.run(self._run_benchmark_async(
                        model, format, batch_size, num_prompts, input_requests, tokenizer
                    ))
                    elapsed = time.time() - iter_start
                    logger.info(f"  Benchmark {i+1} completed in {elapsed:.1f}s")
                    results.append(result)
                except Exception as e:
                    elapsed = time.time() - iter_start
                    logger.error(f"Iteration {i+1} failed after {elapsed:.1f}s: {e}")
                    continue

            return self._aggregate_serving_results(
                results, model, format, batch_size, multimodal=False
            )

        finally:
            if not self.config.dry_run and not self.config.remote_endpoint:
                self._cleanup_server()

    def _run_mm_with_iterations(
        self,
        model: ModelConfig,
        format: ModelFormat,
        batch_size: int,
        num_prompts: int,
    ) -> dict:
        """Run multimodal serving benchmark with multiple iterations.

        Sets up images, starts server with --allowed-local-media-path,
        runs warmup + benchmark iterations using aiohttp + SSE,
        then aggregates and cleans up.
        """
        # Generate synthetic images
        mm_media_path = self._setup_mm_images(num_images=50)

        # Start server with allowed media path
        if not self.config.dry_run and not self.config.remote_endpoint:
            if not self._start_server(model, format, allowed_local_media_path=mm_media_path):
                self._cleanup_mm_images()
                raise RuntimeError("Failed to start vLLM server for MM serving")

        try:
            # Warmup iterations
            warmup_total = self.config.warmup_iterations
            for i in range(warmup_total):
                progress = "█" * (i + 1) + "░" * (warmup_total - i - 1)
                logger.info(f"MM Warmup [{progress}] {i+1}/{warmup_total} starting...")
                iter_start = time.time()
                try:
                    asyncio.run(self._run_mm_serving_benchmark(
                        model, format, max_concurrency=batch_size, num_prompts=num_prompts
                    ))
                    elapsed = time.time() - iter_start
                    logger.info(f"  MM Warmup {i+1} completed in {elapsed:.1f}s")
                except Exception as e:
                    elapsed = time.time() - iter_start
                    logger.warning(f"MM Warmup {i+1} failed after {elapsed:.1f}s: {e}")

            # Actual benchmark iterations
            results = []
            iter_total = self.config.num_iterations
            for i in range(iter_total):
                progress = "█" * (i + 1) + "░" * (iter_total - i - 1)
                logger.info(f"MM Benchmark [{progress}] {i+1}/{iter_total} starting...")
                iter_start = time.time()
                try:
                    result = asyncio.run(self._run_mm_serving_benchmark(
                        model, format, max_concurrency=batch_size, num_prompts=num_prompts
                    ))
                    elapsed = time.time() - iter_start
                    logger.info(f"  MM Benchmark {i+1} completed in {elapsed:.1f}s")
                    results.append(result)
                except Exception as e:
                    elapsed = time.time() - iter_start
                    logger.error(f"MM Iteration {i+1} failed after {elapsed:.1f}s: {e}")
                    continue

            return self._aggregate_serving_results(
                results, model, format, batch_size, multimodal=True
            )

        finally:
            if not self.config.dry_run and not self.config.remote_endpoint:
                self._cleanup_server()
            self._cleanup_mm_images()

    def _aggregate_serving_results(
        self,
        results: list[dict],
        model: ModelConfig,
        format: ModelFormat,
        batch_size: int,
        multimodal: bool,
    ) -> dict:
        """Aggregate multi-iteration serving results with statistics."""
        if not results:
            raise RuntimeError("All iterations failed")

        # Define serving metrics to aggregate
        serving_metrics = [
            "mean_ttft_ms",
            "median_ttft_ms",
            "p99_ttft_ms",
            "mean_tpot_ms",
            "median_tpot_ms",
            "p99_tpot_ms",
            "mean_itl_ms",
            "median_itl_ms",
            "p99_itl_ms",
            "request_throughput",
            "output_throughput",
        ]

        # Aggregate results
        aggregated = aggregate_benchmark_results(results, serving_metrics)

        # Add validation info for key metrics
        for metric in ["request_throughput", "mean_ttft_ms"]:
            is_valid, msg = validate_repeatability(
                aggregated, metric, self.config.min_acceptable_cv_percent
            )
            logger.info(f"{metric}: {msg}")
            aggregated[f"{metric}_repeatability_valid"] = is_valid

        # Log statistics summary for key metrics
        for metric in ["request_throughput", "mean_ttft_ms", "mean_tpot_ms"]:
            summary = format_statistics_summary(aggregated, metric)
            logger.info(summary)

        # Save aggregated results to file
        aggregated["model"] = model.short_name
        aggregated["model_short"] = model.short_name
        aggregated["model_name"] = model.name
        aggregated["format"] = format.value
        aggregated["batch_size"] = batch_size
        aggregated["multimodal"] = multimodal
        aggregated["modality"] = "multimodal" if multimodal else "text"
        aggregated["output_token_throughput"] = aggregated.get("output_throughput_mean", aggregated.get("output_throughput", 0.0))
        aggregated["request_throughput"] = aggregated.get("request_throughput_mean", aggregated.get("request_throughput", 0.0))
        aggregated["mean_ttft_ms"] = aggregated.get("mean_ttft_ms_mean", aggregated.get("mean_ttft_ms", 0.0))
        aggregated["mean_tpot_ms"] = aggregated.get("mean_tpot_ms_mean", aggregated.get("mean_tpot_ms", 0.0))
        aggregated["mean_itl_ms"] = aggregated.get("mean_itl_ms_mean", aggregated.get("mean_itl_ms", 0.0))
        output_file = self._get_output_path(
            model, format, batch_size, multimodal
        )
        with open(output_file, 'w') as f:
            json.dump(aggregated, f, indent=2)
        logger.info(f"Results saved to: {output_file}")

        return aggregated


    def run_all(
        self,
        model: ModelConfig,
        format: ModelFormat,
        multimodal: bool = False,
    ) -> list[dict]:
        """Run all serving benchmark configurations for a model.

        Args:
            model: Model configuration
            format: Model format
            multimodal: If True, use random-mm dataset for image+text

        Returns:
            List of aggregated result dictionaries (includes failed runs with failed=True)
        """
        results = []
        configs = self.config.get_serving_configs()

        for cfg in configs:
            try:
                # Use iteration-aware runner
                result = self.run_with_iterations(
                    model,
                    format,
                    batch_size=cfg["batch_size"],
                    num_prompts=cfg["num_prompts"],
                    dataset=cfg.get("dataset"),
                    multimodal=multimodal,
                )
                # Add config metadata to result
                result["batch_size"] = cfg["batch_size"]
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
                    "batch_size": cfg["batch_size"],
                    "multimodal": multimodal,
                })

        return results

    def _start_server(
        self,
        model: ModelConfig,
        format: ModelFormat,
        allowed_local_media_path: Optional[str] = None,
    ) -> bool:
        """Start vLLM server for serving benchmarks.

        Args:
            model: Model configuration
            format: Model format
            allowed_local_media_path: If set, allow loading local media from this path

        Returns:
            True if server started successfully, False otherwise
        """
        if self.server_process is not None:
            logger.warning("Server already running, cleaning up first")
            self._cleanup_server()

        logger.info(f"Starting vLLM server for {model.name} ({format.value})")

        # Build server command
        cmd = [
            "vllm",
            "serve",
            model.get_model_path(format),
            "--port",
            str(self.server_port),
        ]

        # Add performance optimization flags
        gpu_mem = get_gpu_memory_utilization(model.total_params_b)
        if gpu_mem:
            cmd.extend([
                "--gpu-memory-utilization",
                str(gpu_mem),
            ])
        
        if self.config.enable_chunked_prefill:
            cmd.append("--enable-chunked-prefill")
        
        if self.config.max_num_batched_tokens:
            cmd.extend([
                "--max-num-batched-tokens",
                str(self.config.max_num_batched_tokens),
            ])
        
        # Limit max concurrent sequences for uniform memory usage
        if self.config.max_num_seqs:
            cmd.extend([
                "--max-num-seqs",
                str(self.config.max_num_seqs),
            ])
        
        # Multi-GPU: tensor parallel
        if self.config.tensor_parallel_size and self.config.tensor_parallel_size > 1:
            cmd.extend([
                "--tensor-parallel-size",
                str(self.config.tensor_parallel_size),
            ])

        # GGUF models need explicit tokenizer from HF model ID
        if format == ModelFormat.GGUF:
            cmd.extend(["--tokenizer", model.hf_model_id])
        
        # Uniform context length for fair comparison
        max_model_len = get_max_model_len()
        if max_model_len:
            logger.info(f"Setting max_model_len={max_model_len} for uniform benchmark comparison")
            cmd.extend(["--max-model-len", str(max_model_len)])

        # Allow loading local media files (needed for multimodal benchmarks)
        if allowed_local_media_path:
            cmd.extend(["--allowed-local-media-path", allowed_local_media_path])

        if getattr(self.config, "eval_max_soft_tokens", None) is not None:
            cmd.extend([
                "--hf-overrides",
                json.dumps({"max_soft_tokens": self.config.eval_max_soft_tokens}),
            ])

        try:
            # Redirect server output to log file to avoid pipe buffer deadlock.
            # When using PIPE, the OS buffer (~64KB) can fill up during model
            # loading and CUDA graph capture, causing the server to block.
            # Writing to a file avoids this issue while preserving logs.
            lm = self.config.log_manager
            server_log_path = lm.get_server_log_path(model.short_name, format.value)
            logger.info(f"Server logs will be written to: {server_log_path}")
            
            self._server_log_file = open(server_log_path, "a")
            self.server_process = subprocess.Popen(
                cmd,
                stdout=self._server_log_file,
                stderr=subprocess.STDOUT,  # Combine stderr with stdout
                start_new_session=True,  # New process group for clean killpg
            )


            # Set timeout based on model size and format
            # - 27b: ~2-3 min for torch.compile + CUDA graphs on first run
            # - GGUF: slower loading (~5-6 min)
            # - Others: 2 min is usually sufficient
            timeout = get_server_timeout(model.total_params_b, model.is_moe)
            if format == ModelFormat.GGUF:
                timeout = max(timeout, 600)
            
            # Wait for server to be ready
            if self._wait_for_server_ready(timeout):
                logger.info(f"vLLM server started successfully on port {self.server_port}")
                from gbench.utils import verify_endpoint_functional
                is_up, err_msg, max_model_len = verify_endpoint_functional(f"http://127.0.0.1:{self.server_port}")
                if not is_up:
                    logger.error(f"❌ Local server on port {self.server_port} is not functional: {err_msg}")
                    self._cleanup_server()
                    return False
                logger.info(f"✅ Local server on port {self.server_port} is functional and answering requests.")
                if max_model_len:
                    logger.info(f"   Server reported max_model_len: {max_model_len} tokens")
                return True
            else:
                logger.error(f"Server failed to become ready. Check logs at: {server_log_path}")
                self._cleanup_server()
                return False

        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            self._cleanup_server()
            return False

    def _wait_for_server_ready(self, timeout: int = 120) -> bool:
        """Wait for vLLM server to be ready.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if server is ready, False if timeout
        """
        url = f"http://127.0.0.1:{self.server_port}/health"
        start_time = time.time()

        logger.info("Waiting for server to be ready...")

        while time.time() - start_time < timeout:
            try:
                response = requests.get(url, timeout=1)
                if response.status_code == 200:
                    return True
            except requests.exceptions.RequestException:
                pass

            # Check if process died
            if self.server_process and self.server_process.poll() is not None:
                logger.error("Server process died during startup")
                # Capture and log server output
                stdout, stderr = self.server_process.communicate()
                if stdout:
                    logger.error(f"Server STDOUT:\n{stdout[-2000:]}")
                if stderr:
                    logger.error(f"Server STDERR:\n{stderr[-2000:]}")
                return False

            time.sleep(2)

        return False

    def _cleanup_server(self):
        """Clean up vLLM server process.
        
        Note: After killing the process, we wait for GPU memory to be 
        reclaimed by the CUDA driver. This typically takes 5-15 seconds.
        """
        if self.server_process is None:
            return

        logger.info("Cleaning up vLLM server...")

        try:
            # Kill the entire process group (API server + EngineCore + workers)
            # vLLM spawns EngineCore as a grandchild process that terminate()
            # cannot reach, leaving zombie GPU-holding processes.
            pgid = os.getpgid(self.server_process.pid)
            os.killpg(pgid, signal.SIGTERM)
            try:
                self.server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning("Server did not terminate gracefully, forcing kill")
                os.killpg(pgid, signal.SIGKILL)
                self.server_process.wait(timeout=5)

            logger.info("vLLM server stopped")
            
            # Wait for GPU memory to be reclaimed by CUDA driver
            # Use active polling instead of fixed wait for reliability
            self._wait_for_gpu_memory()
            
        except Exception as e:
            logger.error(f"Error cleaning up server: {e}")
        finally:
            self.server_process = None
            # Close the log file handle
            if self._server_log_file is not None:
                try:
                    self._server_log_file.close()
                except Exception:
                    pass
                self._server_log_file = None
