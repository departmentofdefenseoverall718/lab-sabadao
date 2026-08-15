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

"""Native MedXpertQA (Medical Multimodal Expert Exam & Diagnostic QA) evaluation suite."""

import base64
import io
import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from .base import run_eval_suite
from .sampling import limit_dataset

logger = logging.getLogger(__name__)

OPTION_LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H"]


def _load_medxpertqa_samples(
    enable_thinking: bool = False,
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Load MedXpertQA medical exam samples from canonical HF Hub dataset ('TsinghuaC3I/MedXpertQA')."""
    from datasets import load_dataset

    try:
        ds = load_dataset("TsinghuaC3I/MedXpertQA", "MM", split="test")
    except Exception:
        ds = load_dataset("wish6424/MedXpertQA-Diagnosis", split="test")

    # Stratified, not a contiguous head (audit RC-1).
    ds = limit_dataset(ds, limit, "medical_task", seed="medxpertqa")
    raw_samples = list(ds)
    logger.info(f"Loaded {len(raw_samples)} MedXpertQA samples from HF Hub.")

    samples = []
    for item in raw_samples:
        q_text = item.get("question", "")
        options = item.get("options", {})
        gold_ans = str(item.get("label", item.get("answer", ""))).strip().upper()
        specialty = item.get("medical_task", item.get("body_system", item.get("specialty", "Clinical")))

        # Format multiple choice options
        if isinstance(options, dict):
            options_lines = [f"({k.upper()}) {v}" for k, v in sorted(options.items())]
        elif isinstance(options, list):
            options_lines = [f"({OPTION_LETTERS[i]}) {opt}" for i, opt in enumerate(options)]
        else:
            options_lines = []

        options_str = "\n".join(options_lines)
        prompt_text = (
            f"Medical Question ({specialty}):\n{q_text}\n\n"
            f"Options:\n{options_str}\n\n"
        )
        if enable_thinking:
            prompt_text += "Let's reason carefully step by step and output the correct option letter in the format: 'Answer: (X)'."
        else:
            prompt_text += "Answer directly with the correct option letter in the format: 'Answer: (X)'."

        # MedXpertQA "MM" exposes `images` (a LIST); reading the singular `image` returned
        # None for every row, so the multimodal split was silently evaluated text-only.
        img_obj = item.get("image")
        if img_obj is None:
            imgs = item.get("images")
            if isinstance(imgs, list) and imgs:
                img_obj = imgs[0]
            elif imgs is not None and not isinstance(imgs, list):
                img_obj = imgs
        b64_str = None
        if img_obj is not None:
            if isinstance(img_obj, bytes):
                b64_str = base64.b64encode(img_obj).decode("utf-8")
            elif isinstance(img_obj, dict) and "bytes" in img_obj and img_obj["bytes"]:
                b64_str = base64.b64encode(img_obj["bytes"]).decode("utf-8")
            elif hasattr(img_obj, "save"):
                if hasattr(img_obj, "mode") and img_obj.mode != "RGB":
                    img_obj = img_obj.convert("RGB")
                buf = io.BytesIO()
                img_obj.save(buf, format="PNG")
                b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")

        if b64_str:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_str}"}},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]
        else:
            messages = [{"role": "user", "content": prompt_text}]

        samples.append((messages, gold_ans, {"category": specialty}))
    return samples


def _eval_medxpertqa(response_text: str, gold_letter: str) -> bool:
    """Extract predicted option letter and compare with gold answer."""
    if not response_text or not gold_letter:
        return False

    gold = gold_letter.strip().upper()
    if len(gold) > 1:
        match = re.search(r"\b([A-H])\b", gold)
        if match:
            gold = match.group(1)

    resp = response_text.strip()
    match = re.search(r"(?:answer|correct\s+option|choice)\s*(?:is|:)?\s*\(?([A-H])\)?", resp, re.IGNORECASE)
    if match:
        return match.group(1).upper() == gold

    tokens = re.findall(r"\b([A-H])\b", resp)
    if tokens:
        return tokens[-1].upper() == gold

    return False


def run_medxpertqa(
    model_name: str,
    base_url: str,
    concurrency: int = 1,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run native MedXpertQA medical multimodal evaluation suite."""
    samples = _load_medxpertqa_samples(
        enable_thinking=enable_thinking,
        limit=kwargs.get("limit"),
    )
    return run_eval_suite(
        eval_name="medxpertqa",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_medxpertqa,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens", 8192 if enable_thinking else 2048),
        temperature=kwargs.get("temperature", 0.0),
    )
