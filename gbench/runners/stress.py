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

"""Stress test runner for finding maximum sustainable throughput.

Uses binary search approach: starts at a high concurrency, then searches
up or down to find the exact threshold where P99 TTFT exceeds the limit.
Each concurrency level is tested 3 times (median P99 used for robustness).
"""

import asyncio
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Optional

from ..core.config import BenchmarkConfig
from ..core.models import ModelConfig, ModelFormat, ModelCategory
from .serving import ServingBenchmarkRunner

# vLLM benchmark imports will be loaded dynamically in active methods

logger = logging.getLogger(__name__)


class StressTestRunner:
    """Runner for stress testing to find max sustainable throughput.
    
    Uses binary search approach: starts at a high concurrency, then searches
    up or down to find the maximum concurrency where P99 TTFT stays within threshold.
    Supports both text-only and multimodal stress tests.
    """

    # Binary search parameters
    INITIAL_CONCURRENCY = 200  # Start below max_num_seqs=256 scheduler cap
    MIN_CONCURRENCY = 1  # Lower bound (allows search down to 1 concurrent request)
    SEARCH_PRECISION = 2  # Stop when search range is smaller than 2
    
    # Default parameters
    DEFAULT_TTFT_THRESHOLD_MS = 5000  # P99 TTFT threshold (can be overridden by CLI)
    # Note: Prompts are dynamically scaled per test: min(max(200, 2x concurrency), 1000)

    def __init__(self, config: BenchmarkConfig, ttft_threshold_ms: int = None):
        """Initialize stress test runner.

        Args:
            config: Benchmark configuration
            ttft_threshold_ms: P99 TTFT threshold in ms (overrides default)
        """
        self.config = config
        self.ttft_threshold_ms = ttft_threshold_ms or self.DEFAULT_TTFT_THRESHOLD_MS
        self.initial_concurrency = 16 if config.remote_endpoint else self.INITIAL_CONCURRENCY
        self._serving_runner: Optional[ServingBenchmarkRunner] = None
        self._tested_levels = {}  # Cache tested concurrency levels
        self._tokenizer = None  # Cached tokenizer
        self._model_path = None  # Cached model path
        self._multimodal = False  # Whether current run is multimodal
        self._mm_image_dir = None  # Temp dir for generated images
        self._mm_image_paths = []  # Generated image file paths

    def _generate_requests(self, num_prompts: int) -> list:
        """Generate sample requests with realistic length variance."""
        try:
            from vllm.benchmarks.datasets import RandomDataset
            dataset = RandomDataset()
            return dataset.sample(
                tokenizer=self._tokenizer,
                num_requests=num_prompts,
                prefix_len=0,
                input_len=128,
                output_len=128,
                range_ratio=0.5,
            )
        except ImportError:
            from dataclasses import dataclass
            @dataclass
            class SampleRequest:
                prompt: str
                expected_output_len: int = 128
            
            prompt = "Explain quantum computing in simple terms. " * 5
            return [SampleRequest(prompt=prompt, expected_output_len=128) for _ in range(num_prompts)]

    def _setup_mm_images(self, num_images: int = 50) -> str:
        """Generate synthetic images for multimodal stress test.
        
        Creates a temp directory with random JPEG images.
        
        Args:
            num_images: Number of images to generate
            
        Returns:
            Path to the temp directory containing images
        """
        import numpy as np
        from PIL import Image
        
        img_dir = tempfile.mkdtemp(prefix="stress_mm_images_")
        self._mm_image_paths = []
        
        for i in range(num_images):
            # 256x256 random images — small enough for fast I/O,
            # large enough to exercise the vision encoder
            img_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img_path = os.path.join(img_dir, f"stress_img_{i:03d}.jpg")
            img.save(img_path, format="JPEG", quality=85)
            self._mm_image_paths.append(img_path)
        
        logger.info(f"Generated {num_images} synthetic images in {img_dir}")
        self._mm_image_dir = img_dir
        return img_dir

    def _cleanup_mm_images(self):
        """Clean up temp image directory."""
        if self._mm_image_dir and os.path.exists(self._mm_image_dir):
            import shutil
            shutil.rmtree(self._mm_image_dir, ignore_errors=True)
            logger.info(f"Cleaned up MM image directory: {self._mm_image_dir}")
            self._mm_image_dir = None
            self._mm_image_paths = []

    async def _run_benchmark_at_concurrency(
        self,
        model: ModelConfig,
        format: ModelFormat,
        input_requests: list,
        tokenizer,
        max_concurrency: int,
    ) -> dict:
        return await self._serving_runner._run_benchmark_async(
            model=model,
            format=format,
            batch_size=max_concurrency,
            num_prompts=len(input_requests),
            input_requests=input_requests,
            tokenizer=tokenizer,
        )

    async def _run_mm_benchmark_at_concurrency(
        self,
        model: ModelConfig,
        format: ModelFormat,
        max_concurrency: int,
        num_prompts: int,
    ) -> dict:
        """Run multimodal benchmark at a specific concurrency level.
        
        Sends chat completion requests with images via aiohttp,
        measures TTFT from SSE streaming response.
        """
        import aiohttp
        
        model_path = self._serving_runner._resolve_model_id(model, format)
        base_url = self.config.remote_endpoint or f"http://127.0.0.1:{self._serving_runner.server_port}"
        api_url = f"{base_url}/v1/chat/completions"
        
        semaphore = asyncio.Semaphore(max_concurrency)
        ttft_list = []
        completed = 0
        failed = 0
        
        async def send_mm_request(session, img_path, request_id):
            nonlocal completed, failed
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
                        {"type": "text", "text": "Describe this image briefly."},
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
                first_token_time = None
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
                        # Parse SSE stream — TTFT = time to first data chunk
                        async for raw_line in resp.content:
                            line = raw_line.decode("utf-8", errors="ignore").strip()
                            if (
                                line.startswith("data:")
                                and "[DONE]" not in line
                            ):
                                if first_token_time is None:
                                    first_token_time = time.perf_counter()
                    end = time.perf_counter()
                    ttft_ms = (
                        (first_token_time or end) - start
                    ) * 1000
                    ttft_list.append(ttft_ms)
                    completed += 1
                except Exception as e:
                    logger.debug(f"MM request {request_id} error: {e}")
                    failed += 1
        
        connector = aiohttp.TCPConnector(limit=max_concurrency + 20)
        timeout = aiohttp.ClientTimeout(total=300)
        async with aiohttp.ClientSession(
            connector=connector, timeout=timeout
        ) as session:
            overall_start = time.perf_counter()
            tasks = [
                send_mm_request(
                    session,
                    self._mm_image_paths[i % len(self._mm_image_paths)],
                    i,
                )
                for i in range(num_prompts)
            ]
            await asyncio.gather(*tasks)
            overall_elapsed = time.perf_counter() - overall_start
        
        if not ttft_list:
            raise RuntimeError(
                f"All {num_prompts} MM stress requests failed "
                f"({failed} errors)"
            )
        
        ttft_list.sort()
        p99_idx = int(len(ttft_list) * 0.99)
        p99_ttft = ttft_list[min(p99_idx, len(ttft_list) - 1)]
        request_throughput = completed / overall_elapsed
        median_idx = len(ttft_list) // 2
        median_ttft = ttft_list[median_idx]
        mean_ttft = sum(ttft_list) / len(ttft_list)
        
        logger.info(
            f"\n============ Serving Benchmark Result ============\n"
            f"Successful requests:                     {completed}\n"
            f"Failed requests:                         {failed}\n"
            f"Maximum request concurrency:             {max_concurrency}\n"
            f"Benchmark duration (s):                  {overall_elapsed:.2f}\n"
            f"Request throughput (req/s):              {request_throughput:.2f}\n"
            f"---------------Time to First Token----------------\n"
            f"Mean TTFT (ms):                          {mean_ttft:.2f}\n"
            f"Median TTFT (ms):                        {median_ttft:.2f}\n"
            f"P99 TTFT (ms):                           {p99_ttft:.2f}\n"
            f"=================================================="
        )
        
        return {
            "p99_ttft_ms": p99_ttft,
            "request_throughput": request_throughput,
            "completed": completed,
            "failed": failed,
            "total": num_prompts,
        }

    def _test_concurrency(
        self,
        concurrency: int,
        model: ModelConfig,
        format: ModelFormat,
    ) -> tuple[bool, float, float]:
        """Test a specific concurrency level with multiple iterations for robustness.
        
        Runs 3 iterations and uses median P99 TTFT to reduce noise.
        Supports both text and multimodal modes via self._multimodal flag.
        
        Returns:
            Tuple of (passed, median_p99_ttft, median_throughput)
        """
        # Check cache first
        if concurrency in self._tested_levels:
            cached = self._tested_levels[concurrency]
            return cached['passed'], cached['p99_ttft_ms'], cached['request_throughput']
        
        # Generate enough prompts: at least 2x concurrency to build up queue
        min_p = 20 if self.config.remote_endpoint else 200
        max_p = 500 if self.config.remote_endpoint else 1000
        num_prompts = min(max(min_p, concurrency * 2), max_p)
        
        # Run multiple iterations for robustness (1 iteration for remote endpoints for fast response)
        NUM_ITERATIONS = 1 if self.config.remote_endpoint else 3
        p99_ttft_samples = []
        throughput_samples = []
        
        for i in range(NUM_ITERATIONS):
            if self._multimodal:
                # Multimodal: send chat completion requests with images
                result = asyncio.run(self._run_mm_benchmark_at_concurrency(
                    model=model,
                    format=format,
                    max_concurrency=concurrency,
                    num_prompts=num_prompts,
                ))
            else:
                # Text: use standard vLLM benchmark function
                input_requests = self._generate_requests(num_prompts)
                result = asyncio.run(self._run_benchmark_at_concurrency(
                    model=model,
                    format=format,
                    input_requests=input_requests,
                    tokenizer=self._tokenizer,
                    max_concurrency=concurrency,
                ))
            
            p99_ttft_samples.append(result.get("p99_ttft_ms", 0))
            throughput_samples.append(result.get("request_throughput", 0))
        
        # Use median for robustness
        p99_ttft_samples.sort()
        throughput_samples.sort()
        median_idx = NUM_ITERATIONS // 2
        p99_ttft = p99_ttft_samples[median_idx]
        throughput = throughput_samples[median_idx]
        
        passed = p99_ttft <= self.ttft_threshold_ms
        
        # Cache result with all samples for analysis
        self._tested_levels[concurrency] = {
            'concurrency': concurrency,
            'p99_ttft_ms': p99_ttft,
            'request_throughput': throughput,
            'passed': passed,
            'p99_ttft_samples': p99_ttft_samples,
            'throughput_samples': throughput_samples,
        }
        
        # Log with sample range for transparency
        status = "✓ PASSED" if passed else "⚠ DEGRADED"
        range_str = f"[{min(p99_ttft_samples):.0f}-{max(p99_ttft_samples):.0f}]"
        logger.info(f"{concurrency:<12} {p99_ttft:>8.1f}ms {range_str:>12}   {throughput:>10.1f} req/s   {status}")
        
        return passed, p99_ttft, throughput

    def run(
        self,
        model: ModelConfig,
        format: ModelFormat,
        multimodal: bool = False,
    ) -> dict:
        """Run stress test using binary search to find max sustainable concurrency.

        Args:
            model: Model configuration
            format: Model format (hf, gguf)
            multimodal: Whether to test multimodal (sends real images)

        Returns:
            Dictionary with stress test results
        """
        mode_str = "Multimodal" if multimodal else "Text"
        self._tested_levels = {}  # Reset cache
        self._multimodal = multimodal

        logger.info(
            f"\n{'='*60}\n"
            f"  STRESS TEST CONFIGURATION (Binary Search)\n"
            f"{'='*60}\n"
            f"  Mode:           {mode_str}\n"
            f"  Model:          {model.short_name} ({format.value})\n"
            f"  Start at:       {self.INITIAL_CONCURRENCY} concurrent requests\n"
            f"  TTFT threshold: {self.ttft_threshold_ms}ms (P99)\n"
            f"{'='*60}"
        )

        if self.config.dry_run:
            logger.info("[DRY RUN] Would run stress test with binary search")
            return {
                "dry_run": True,
                "stress_test": True,
                "model": model.short_name,
                "format": format.value,
                "multimodal": multimodal,
                "initial_concurrency": self.INITIAL_CONCURRENCY,
                "ttft_threshold_ms": self.ttft_threshold_ms,
            }

        try:
            # Set up multimodal images if needed
            mm_media_path = None
            if multimodal:
                mm_media_path = self._setup_mm_images(num_images=50)

            # Start vLLM server (with media path for MM)
            self._serving_runner = ServingBenchmarkRunner(self.config)
            if not self.config.remote_endpoint:
                self._serving_runner._start_server(
                    model, format,
                    allowed_local_media_path=mm_media_path,
                )

            # Initialize tokenizer for request generation (text mode)
            self._api_model_id = self._serving_runner._resolve_model_id(model, format)
            tokenizer_path = model.hf_model_id if (self.config.remote_endpoint or format == ModelFormat.GGUF) else model.get_model_path(format)
            if not multimodal:
                from gbench.utils import safe_get_tokenizer
                self._tokenizer = safe_get_tokenizer(tokenizer_path, custom_tokenizer=self.config.tokenizer)
                logger.info(f"Tokenizer loaded safely for: {self.config.tokenizer or tokenizer_path}")

            # Log clear benchmark configuration banner
            logger.info(
                f"\n{'='*60}\n"
                f"  BENCHMARK CONFIGURATION\n"
                f"{'='*60}\n"
                f"  Type:           Stress Test\n"
                f"  Mode:           {mode_str}\n"
                f"  Model:          {model.short_name} ({format.value})\n"
                f"  P99 TTFT Threshold: {self.ttft_threshold_ms}ms\n"
                f"  Initial Concurrency: {self.initial_concurrency}\n"
                f"{'='*60}"
            )

            # Binary search for max sustainable concurrency
            min_p = 20 if self.config.remote_endpoint else 200
            max_p = 500 if self.config.remote_endpoint else 1000
            logger.info(f"(Prompts scaled per concurrency: min {min_p}, up to 2x concurrency, max {max_p})")
            logger.info(f"{'Concurrency':<12} {'Median P99':<10} {'Range':<14} {'Throughput':<12} {'Status'}")
            logger.info("-" * 55)
            
            # Phase 1: Test initial concurrency to determine search direction
            passed, p99, tp = self._test_concurrency(
                self.initial_concurrency, model, format
            )
            
            if passed:
                # Initial passed - search UPWARD for the limit
                logger.info("-" * 55)
                logger.info(f"Initial {self.initial_concurrency} passed - searching upward...")
                logger.info("-" * 55)
                
                # Exponentially increase until we find a failure
                low = self.initial_concurrency
                high = self.initial_concurrency * 2
                
                while True:
                    passed, p99, tp = self._test_concurrency(
                        high, model, format
                    )
                    if not passed:
                        break  # Found upper bound
                    low = high
                    high = high * 2  # Double each time
                    if high > 10000:  # Safety limit
                        logger.info(f"Reached safety limit at {high} - model handles extreme load!")
                        break
            else:
                # Initial failed - search DOWNWARD for passing level
                logger.info("-" * 55)
                logger.info(f"Initial {self.initial_concurrency} failed - testing minimum concurrency {self.MIN_CONCURRENCY}...")
                logger.info("-" * 55)
                
                min_passed, p99, tp = self._test_concurrency(
                    self.MIN_CONCURRENCY, model, format
                )
                if not min_passed:
                    logger.info(f"Minimum concurrency {self.MIN_CONCURRENCY} failed threshold.")
                    low = 0
                    high = self.MIN_CONCURRENCY
                else:
                    low = self.MIN_CONCURRENCY
                    high = self.initial_concurrency
            
            # Phase 2: Binary search between low and high
            if low > 0:
                logger.info("-" * 55)
                logger.info(f"Binary search between {low} and {high}...")
                logger.info("-" * 55)
                
                while high - low >= self.SEARCH_PRECISION and low > 0:
                    mid = (low + high) // 2
                    if mid == low or mid == high:
                        break
                    passed, p99, tp = self._test_concurrency(
                        mid, model, format
                    )
                    
                    if passed:
                        low = mid
                    else:
                        high = mid
            
            # The max sustainable concurrency is 'low' (last known passing value)
            max_sustainable = low
            max_result = self._tested_levels.get(max_sustainable, {})
            max_throughput = max_result.get('request_throughput', 0.0)
            
            logger.info("-" * 55)
            logger.info(
                f"✅ Maximum sustainable: {max_sustainable} concurrent requests "
                f"(~{max_throughput:.1f} req/s)"
            )

            # Build final result
            final_result = {
                "stress_test": True,
                "binary_search": True,
                "ttft_threshold_ms": self.ttft_threshold_ms,
                "max_sustainable_concurrency": max_sustainable,
                "max_sustainable_throughput": max_throughput,
                "search_results": list(self._tested_levels.values()),
                "multimodal": multimodal,
                "model": model.short_name,
                "format": format.value,
                "p99_ttft_within_threshold": max_sustainable >= self.MIN_CONCURRENCY,
            }
            
            # Add the max result's full metrics
            if max_result:
                final_result["p99_ttft_ms"] = max_result.get('p99_ttft_ms', 0)
                final_result["request_throughput"] = max_result.get('request_throughput', 0)

            # Save result
            lm = self.config.log_manager
            results_dir = lm.results_dir / "performance"
            results_dir.mkdir(parents=True, exist_ok=True)
            mode_suffix = "mm" if multimodal else "text"
            output_file = results_dir / f"stress_{model.short_name}_{format.value}_{mode_suffix}.json"
            with open(output_file, "w") as f:
                json.dump(final_result, f, indent=2, default=str)
            logger.info(f"Results saved to: {output_file}")

            return final_result

        except Exception as e:
            logger.error(f"Stress test failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                "stress_test": True,
                "failed": True,
                "error": str(e),
                "model": model.short_name,
                "format": format.value,
            }

        finally:
            if self._serving_runner and not self.config.remote_endpoint:
                self._serving_runner._cleanup_server()
            # Clean up MM images
            if multimodal:
                self._cleanup_mm_images()

    def run_all(
        self,
        model: ModelConfig,
        format: ModelFormat,
    ) -> list[dict]:
        """Run stress tests — text always, multimodal for MM-capable models.
        
        Text stress test uses vLLM's built-in benchmark function with RandomDataset.
        MM stress test sends real image+text chat requests via aiohttp with
        TTFT measured from SSE streaming.
        """
        results = []

        logger.info("Running text stress test...")
        text_result = self.run(model, format, multimodal=False)
        results.append(text_result)

        if model.category == ModelCategory.MULTIMODAL:
            logger.info("Running multimodal stress test...")
            mm_result = self.run(model, format, multimodal=True)
            results.append(mm_result)

        return results
