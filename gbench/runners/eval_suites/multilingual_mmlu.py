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

"""Native Multilingual MMLU (14-Language Reasoning and Knowledge) evaluation suite."""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from .base import run_eval_suite
from .sampling import stratified_sample

logger = logging.getLogger(__name__)

OPTION_LETTERS = ["A", "B", "C", "D"]

# alexandrainst/m_mmlu is per-language (there is no 'all' config). Columns:
# 'instruction', 'option_a'..'option_d', 'answer' (letter A-D). We evaluate a
# fixed multilingual set of 14 non-English languages. Use --eval-limit to cap;
# rows are interleaved across languages so a limit still spans languages.
MULTILINGUAL_MMLU_LANGS = [
    "ar", "bn", "de", "es", "fr", "hi", "id",
    "it", "nl", "pt", "ru", "sv", "vi", "zh",
]


def _load_multilingual_mmlu_samples(
    enable_thinking: bool = False,
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Load Multilingual MMLU from HF Hub (alexandrainst/m_mmlu, per-language configs).

    Raises on load/schema failure (no fabricated fallback, no silent fallback to
    English cais/mmlu).
    """
    from datasets import load_dataset

    per_lang: List[Tuple[str, List[Dict[str, Any]]]] = []
    for lang in MULTILINGUAL_MMLU_LANGS:
        try:
            ds = load_dataset("alexandrainst/m_mmlu", lang, split="test")
        except Exception as e:
            logger.error(f"Failed to load m_mmlu language '{lang}': {e}")
            raise RuntimeError(
                f"Could not load multilingual_mmlu language '{lang}': {e}"
            ) from e
        per_lang.append((lang, list(ds)))

    # Round-robin interleave so a --eval-limit still spans multiple languages.
    lang_rows: List[Tuple[str, Dict[str, Any]]] = []
    max_len = max((len(rows) for _, rows in per_lang), default=0)
    for i in range(max_len):
        for lang, rows in per_lang:
            if i < len(rows):
                lang_rows.append((lang, rows[i]))

    if not lang_rows:
        raise RuntimeError("Dataset for multilingual_mmlu returned empty rows")

    # Stratified, not a contiguous head (audit RC-1).
    lang_rows = stratified_sample(lang_rows, limit, lambda r: r[0], seed="multilingual_mmlu")
    logger.info(f"Loaded {len(lang_rows)} Multilingual MMLU samples from HF Hub.")

    samples = []
    for lang, item in lang_rows:
        q_text = item.get("instruction")
        opts = [item.get(f"option_{c}") for c in ("a", "b", "c", "d")]
        ans = item.get("answer")
        if (not q_text or any(o is None for o in opts)
                or not (isinstance(ans, str) and ans.strip().upper() in OPTION_LETTERS)):
            raise RuntimeError(
                "multilingual_mmlu: unexpected dataset schema "
                "(need 'instruction', 'option_a'..'option_d', 'answer' letter); "
                "refusing to fabricate sample data"
            )
        gold_letter = ans.strip().upper()
        options_str = "\n".join(
            f"({OPTION_LETTERS[i]}) {opt}" for i, opt in enumerate(opts)
        )
        prompt = f"Question:\n{q_text}\n\nOptions:\n{options_str}\n\n"
        if enable_thinking:
            prompt += "Let's think step by step and output the correct option letter in the format: 'Answer: (X)'."
        else:
            prompt += "Output only the correct option letter in the format: 'Answer: (X)'."

        messages = [{"role": "user", "content": prompt}]
        samples.append((messages, gold_letter, {"category": lang}))

    return samples


def _eval_multilingual_mmlu(response_text: str, gold_letter: str) -> bool:
    """Extract predicted option letter and compare with gold answer."""
    if not response_text or not gold_letter:
        return False

    gold = gold_letter.strip().upper()
    resp = response_text.strip()

    match = re.search(r"(?:answer|choice|option|resposta|réponse|antwort)\s*(?:is|:)?\s*\(?([A-D])\)?", resp, re.IGNORECASE)
    if match:
        return match.group(1).upper() == gold

    tokens = re.findall(r"\b([A-D])\b", resp)
    if tokens:
        return tokens[-1].upper() == gold

    return False


def run_multilingual_mmlu(
    model_name: str,
    base_url: str,
    concurrency: int = 1,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run native Multilingual MMLU evaluation suite."""
    samples = _load_multilingual_mmlu_samples(
        enable_thinking=enable_thinking,
        limit=kwargs.get("limit"),
    )
    return run_eval_suite(
        eval_name="multilingual_mmlu",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_multilingual_mmlu,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens", 4096 if enable_thinking else 512),
        temperature=kwargs.get("temperature", 0.0),
    )
