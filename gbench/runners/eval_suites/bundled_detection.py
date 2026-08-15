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

"""Native Bundled Object Detection & Grounding evaluation suite."""

import base64
import io
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from .base import run_eval_suite

logger = logging.getLogger(__name__)

COCO_CATEGORIES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def _compute_iou(box1: List[float], box2: List[float]) -> float:
    """Compute Intersection-over-Union between two boxes in [ymin, xmin, ymax, xmax] format."""
    ymin1, xmin1, ymax1, xmax1 = box1
    ymin2, xmin2, ymax2, xmax2 = box2

    inter_ymin = max(ymin1, ymin2)
    inter_xmin = max(xmin1, xmin2)
    inter_ymax = min(ymax1, ymax2)
    inter_xmax = min(xmax1, xmax2)

    inter_w = max(0.0, inter_xmax - inter_xmin)
    inter_h = max(0.0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h

    area1 = max(0.0, xmax1 - xmin1) * max(0.0, ymax1 - ymin1)
    area2 = max(0.0, xmax2 - xmin2) * max(0.0, ymax2 - ymin2)
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def _load_bundled_detection_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], Any, Dict[str, Any]]]:
    """Load object detection samples from canonical HF Hub dataset ('detection-datasets/coco')."""
    from datasets import load_dataset

    ds = load_dataset("detection-datasets/coco", split="val", streaming=True)
    samples = []
    count = 0

    for item in ds:
        objects = item.get("objects", {})
        bboxes = objects.get("bbox", [])
        categories = objects.get("category", [])
        if not bboxes:
            continue

        width = item.get("width", 640)
        height = item.get("height", 480)

        # Convert gold bboxes [x, y, w, h] to normalized [ymin, xmin, ymax, xmax] in [0, 1000]
        gold_objects = []
        labels_present = set()
        for bbox, cat_id in zip(bboxes, categories):
            if cat_id < len(COCO_CATEGORIES):
                cat_name = COCO_CATEGORIES[cat_id]
            else:
                cat_name = str(cat_id)
            labels_present.add(cat_name)

            # detection-datasets/coco stores bboxes as [xmin, ymin, xmax, ymax], NOT
            # [x, y, w, h]. Treating them as xywh added xmax to xmin (and ymax to ymin),
            # inflating every gold box and corrupting all IoU comparisons.
            x0, y0, x1, y1 = bbox
            xmin = (x0 / width) * 1000.0
            ymin = (y0 / height) * 1000.0
            xmax = (x1 / width) * 1000.0
            ymax = (y1 / height) * 1000.0
            gold_objects.append({
                "label": cat_name,
                "box_2d": [ymin, xmin, ymax, xmax]
            })

        # Image encode
        img_obj = item["image"]
        if hasattr(img_obj, "mode") and img_obj.mode != "RGB":
            img_obj = img_obj.convert("RGB")
        buf = io.BytesIO()
        img_obj.save(buf, format="PNG")
        b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")

        prompt = (
            f"Detect all objects in this image for categories: {', '.join(sorted(labels_present))}.\n"
            "Output a JSON array of objects, where each object has 'box_2d' ([ymin, xmin, ymax, xmax] "
            "normalized to [0, 1000]) and 'label'."
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

        primary_cat = sorted(labels_present)[0] if labels_present else "object"
        samples.append((messages, gold_objects, {"category": primary_cat}))
        count += 1
        if limit is not None and count >= limit:
            break

    logger.info(f"Loaded {len(samples)} Bundled Detection samples from HF Hub ('detection-datasets/coco').")
    return samples


def _eval_bundled_detection(response_text: str, gold_objects: Any) -> bool:
    """Evaluate predicted bounding boxes against gold objects using IoU >= 0.5."""
    if not response_text:
        return False

    pred_objects = []
    # Try parsing JSON array from response
    try:
        # Extract JSON code blocks or bracketed content
        json_match = re.search(r"\[\s*\{.*?\}\s*\]", response_text, re.DOTALL)
        if json_match:
            pred_objects = json.loads(json_match.group(0))
        else:
            pred_objects = json.loads(response_text)
    except Exception:
        # Fallback regex extraction of coordinates [ymin, xmin, ymax, xmax]
        coord_matches = re.findall(
            r"\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]",
            response_text
        )
        for m in coord_matches:
            pred_objects.append({
                "label": "object",
                "box_2d": [float(c) for c in m]
            })

    if not pred_objects or not isinstance(pred_objects, list):
        return False

    # Check for at least 1 true positive with IoU >= 0.5
    for pred in pred_objects:
        if not isinstance(pred, dict):
            continue
        p_box = pred.get("box_2d") or pred.get("bbox") or pred.get("box")
        if not p_box or len(p_box) != 4:
            continue
        try:
            p_box = [float(x) for x in p_box]
            # Normalize if in [0, 1] range instead of [0, 1000]
            if max(p_box) <= 1.0:
                p_box = [x * 1000.0 for x in p_box]
        except (ValueError, TypeError):
            continue

        p_label = str(pred.get("label", "")).strip().lower()

        for gold in gold_objects:
            g_box = gold.get("box_2d")
            g_label = str(gold.get("label", "")).strip().lower()

            # Class match check (or loose match if pred label missing)
            if p_label and g_label and p_label != g_label and p_label != "object":
                continue

            iou = _compute_iou(p_box, g_box)
            if iou >= 0.5:
                return True

    return False


def run_bundled_detection(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run native Bundled Object Detection evaluation suite."""
    samples = _load_bundled_detection_samples(limit=kwargs.get("limit"))
    return run_eval_suite(
        eval_name="bundled_detection",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_bundled_detection,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens"),
    )
