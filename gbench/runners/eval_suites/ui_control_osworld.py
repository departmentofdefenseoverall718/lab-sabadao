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

"""Native OSWorld Desktop UI Control evaluation suite with prerequisite validation."""

import logging
import os
import shutil
from typing import Any, Dict, List, Optional, Tuple
from .base import run_eval_suite
from .sampling import limit_dataset

logger = logging.getLogger(__name__)

DOCS_URL = "docs/evals/ui_control_osworld.md"


def check_ui_control_osworld_prerequisites() -> Tuple[bool, str]:
    """Check if required packages (datasets) are available for ui_control_osworld."""
    try:
        import datasets  # noqa: F401
    except ImportError:
        return False, "Python package 'datasets' is not installed."
    return True, ""


def _load_osworld_dataset(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Load canonical desktop OSWorld benchmark tasks (369 items) from Hugging Face Hub."""
    import json
    from datasets import load_dataset

    ds = load_dataset("hud-evals/OSWorld-Verified", split="train")
    # Stratified, not a contiguous head (audit RC-1).
    ds = limit_dataset(ds, limit, "category", seed="ui_control_osworld")
    samples = []
    for item in ds:
        task_id = item.get("id", "")
        prompt_text = item.get("prompt", "")

        st = item.get("setup_tool")
        if isinstance(st, str):
            try:
                st = json.loads(st)
            except Exception:
                st = {}
        args = st.get("arguments", {}) if isinstance(st, dict) else {}
        task_config = args.get("task_config", {}) if isinstance(args, dict) else {}

        instruction = task_config.get("instruction", prompt_text)
        evaluator = task_config.get("evaluator", item.get("evaluate_tool", {}))

        # Category extraction & normalization
        related_apps = task_config.get("related_apps", [])
        metadata = item.get("metadata", {})
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except Exception:
                metadata = {}
        tags = metadata.get("tags", []) if isinstance(metadata, dict) else []

        raw_cat = ""
        if related_apps and isinstance(related_apps, list):
            raw_cat = str(related_apps[0]).lower().strip()
        elif tags and isinstance(tags, list):
            raw_cat = str(tags[0]).lower().strip()

        # Normalize domain names
        if "calc" in raw_cat:
            category = "libreoffice_calc"
        elif "writer" in raw_cat:
            category = "libreoffice_writer"
        elif "impress" in raw_cat:
            category = "libreoffice_impress"
        elif "code" in raw_cat:
            category = "vscode"
        elif raw_cat in ("os", "desktop", "terminal"):
            category = "os"
        elif raw_cat:
            category = raw_cat
        else:
            category = "os"

        prompt = (
            f"You are a computer-use desktop GUI AI agent operating an Ubuntu desktop environment.\n\n"
            f"Target Application Domain: {category}\n"
            f"User Task Instruction: {instruction}\n\n"
            "Output the precise bash terminal commands or Python GUI automation actions to solve this task."
        )
        messages = [{"role": "user", "content": prompt}]
        samples.append((messages, json.dumps(evaluator) if isinstance(evaluator, (dict, list)) else str(evaluator), {"category": category, "id": str(task_id)}))

    return samples


def _eval_osworld(response_text: str, expected_evaluator_json: str) -> bool:
    """Validate candidate desktop actions against ground-truth OSWorld evaluator rules."""
    if not response_text or not expected_evaluator_json:
        return False

    import json
    try:
        evaluator = json.loads(expected_evaluator_json)
    except Exception:
        return False

    text = response_text.lower()
    
    # 1. Infeasible tasks: check if model recognizes task cannot/should not be done
    func_str = ""
    if isinstance(evaluator, dict):
        func_str = str(evaluator.get("func", "")).lower()
    elif isinstance(evaluator, list) and evaluator and isinstance(evaluator[0], dict):
        func_str = str(evaluator[0].get("func", "")).lower()

    if "infeasible" in func_str:
        infeasible_keywords = ["infeasible", "impossible", "cannot", "not possible", "unable", "not supported", "unsupported", "can't"]
        return any(kw in text for kw in infeasible_keywords)

    expected_list = []
    include_list = []
    exclude_list = []

    def _walk(obj, key_name=None):
        if isinstance(obj, dict):
            for k, v in obj.items():
                _walk(v, k)
        elif isinstance(obj, list):
            for item in obj:
                _walk(item, key_name)
        elif isinstance(obj, (str, int, float)):
            s = str(obj).strip()
            if not s or s.startswith("http"):
                return
            if key_name == "exclude":
                exclude_list.append(s.lower())
            elif key_name in ("include", "expected", "dest", "ref", "value", "sheet_idx0", "sheet_idx1", "query"):
                base = os.path.basename(s).lower()
                if base:
                    expected_list.append(base)
            elif key_name == "command":
                for token in s.split():
                    if len(token) > 2 and not token.startswith("-"):
                        expected_list.append(token.lower())

    _walk(evaluator)

    if exclude_list and any(exc in text for exc in exclude_list):
        return False

    if expected_list:
        return any(exp in text for exp in expected_list)

    return False


def run_ui_control_osworld(
    model_name: str,
    base_url: str,
    concurrency: int = 1,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run OSWorld Desktop UI Control evaluation suite or skip if prerequisites are missing."""
    ok, reason = check_ui_control_osworld_prerequisites()
    if not ok:
        msg = f"[SKIP] ui_control_osworld skipped: {reason} See '{DOCS_URL}' for setup instructions."
        logger.warning(msg)
        print(f"\n{msg}")
        return {
            "benchmark_type": "eval",
            "eval_name": "ui_control_osworld",
            "model_name": model_name,
            "status": "skipped",
            "total_questions": 0,
            "correct_answers": 0,
            "accuracy": 0.0,
            "skip_reason": f"{reason} (See {DOCS_URL})",
        }

    samples = _load_osworld_dataset(limit=kwargs.get("limit"))
    extra_payload = kwargs.get("extra_payload") or {}
    eval_max_soft_tokens = kwargs.get("eval_max_soft_tokens")
    if eval_max_soft_tokens is not None and "mm_processor_kwargs" not in extra_payload:
        extra_payload["mm_processor_kwargs"] = {"max_soft_tokens": eval_max_soft_tokens}

    return run_eval_suite(
        eval_name="ui_control_osworld",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_osworld,
        thinking=enable_thinking,
        extra_payload=extra_payload,
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens"),
        temperature=kwargs.get("temperature", 0.0),
    )
