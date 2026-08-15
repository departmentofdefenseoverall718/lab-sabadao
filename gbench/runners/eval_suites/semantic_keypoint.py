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

"""Native Semantic Keypoint spatial pointing and coordinate localization evaluation suite."""

import base64
import io
import json
import logging
import math
import re
from typing import Any, Dict, List, Optional, Tuple
from .base import run_eval_suite
from .sampling import limit_dataset

logger = logging.getLogger(__name__)


def _load_semantic_keypoint_samples(
    enable_thinking: bool = False,
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], Any, Dict[str, Any]]]:
    """Load semantic keypoint spatial grounding samples directly from canonical HF Hub dataset."""
    from datasets import load_dataset

    ds = load_dataset("HongxinLi/ScreenSpot_v2", split="test")
    # Stratified, not a contiguous head (audit RC-1).
    ds = limit_dataset(ds, limit, 'data_type', seed="semantic_keypoint")
    raw_samples = list(ds)
    logger.info(f"Loaded {len(raw_samples)} Semantic Keypoint samples from HF Hub ('HongxinLi/ScreenSpot_v2').")

    samples = []
    for item in raw_samples:
        instruction = item.get("instruction", "")
        img_obj = item.get("image")
        bbox = item.get("bbox", [0, 0, 0, 0])  # [x0, y0, x1, y1] normalized to [0, 1] or [0, 1000]

        if img_obj is None:
            continue

        # Compute gold center keypoint in normalized [0, 1000] space
        if all(0.0 <= float(c) <= 1.0 for c in bbox):
            gold_cx = (float(bbox[0]) + float(bbox[2])) / 2.0 * 1000.0
            gold_cy = (float(bbox[1]) + float(bbox[3])) / 2.0 * 1000.0
        else:
            gold_cx = (float(bbox[0]) + float(bbox[2])) / 2.0
            gold_cy = (float(bbox[1]) + float(bbox[3])) / 2.0

        gold_point = {"x": gold_cx, "y": gold_cy, "bbox": bbox}
        data_type = item.get("data_type", "general")

        if hasattr(img_obj, "mode") and img_obj.mode != "RGB":
            img_obj = img_obj.convert("RGB")
        buf = io.BytesIO()
        img_obj.save(buf, format="PNG")
        b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")

        if enable_thinking:
            prompt = (
                f"Target Element: {instruction}\n"
                "Locate the precise semantic keypoint/center coordinate for the target element in the image. "
                "Analyze the visual boundaries step by step, then output the final point in the format: Point: (x, y) where x and y are in [0, 1000]."
            )
        else:
            prompt = (
                f"Target Element: {instruction}\n"
                "Output the normalized keypoint coordinate [x, y] within [0, 1000] corresponding to the center of the target element. "
                "Format: Point: (x, y)"
            )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_str}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        samples.append((messages, gold_point, {"category": data_type}))

    return samples


def _eval_semantic_keypoint(response_text: str, gold_point: Any) -> bool:
    """Check if predicted (x, y) coordinate is within Euclidean error threshold (<= 5% of viewport)."""
    if not response_text or not isinstance(gold_point, dict):
        return False

    gold_cx = float(gold_point.get("x", 0.0))
    gold_cy = float(gold_point.get("y", 0.0))

    # Pattern matching for (x, y), [x, y], or {"x": x, "y": y}
    pred_x, pred_y = None, None

    # Check Point: (x, y) or (x, y)
    pt_matches = re.findall(r"\(?\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*([0-9]+(?:\.[0-9]+)?)\s*\)?", response_text)
    if pt_matches:
        pred_x, pred_y = float(pt_matches[-1][0]), float(pt_matches[-1][1])
    else:
        # Check json style {"x": ..., "y": ...}
        x_match = re.search(r'"x"\s*:\s*([0-9]+(?:\.[0-9]+)?)', response_text)
        y_match = re.search(r'"y"\s*:\s*([0-9]+(?:\.[0-9]+)?)', response_text)
        if x_match and y_match:
            pred_x = float(x_match.group(1))
            pred_y = float(y_match.group(1))

    if pred_x is None or pred_y is None:
        return False

    # Normalize if in [0, 1] range instead of [0, 1000]
    if pred_x <= 1.0 and pred_y <= 1.0 and gold_cx > 1.0:
        pred_x *= 1000.0
        pred_y *= 1000.0

    # Euclidean distance check (50px / 5% normalized radius tolerance)
    dist = math.sqrt((pred_x - gold_cx) ** 2 + (pred_y - gold_cy) ** 2)
    return dist <= 50.0


def run_semantic_keypoint(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run native Semantic Keypoint pointing evaluation suite."""
    samples = _load_semantic_keypoint_samples(
        enable_thinking=enable_thinking,
        limit=kwargs.get("limit"),
    )
    return run_eval_suite(
        eval_name="semantic_keypoint",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_semantic_keypoint,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens"),
    )
