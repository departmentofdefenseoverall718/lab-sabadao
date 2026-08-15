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

"""Native Multilingual Translation Quality (WMT) evaluation suite."""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from .base import run_eval_suite
from .sampling import stratified_sample

logger = logging.getLogger(__name__)

LANG_PAIRS = [
    ("de-en", "German", "English", "de", "en"),
    ("zh-en", "Chinese", "English", "zh", "en"),
    ("ru-en", "Russian", "English", "ru", "en"),
    ("cs-en", "Czech", "English", "cs", "en"),
]


def _load_i18n_translate_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Load translation evaluation pairs from canonical WMT dataset ('wmt/wmt19')."""
    from datasets import load_dataset

    samples = []
    per_pair = (limit // len(LANG_PAIRS)) + 1 if limit is not None else 25

    for pair_name, src_lang, tgt_lang, src_code, tgt_code in LANG_PAIRS:
        try:
            ds = load_dataset("wmt/wmt19", pair_name, split="validation", streaming=True)
            count = 0
            for item in ds:
                tr = item.get("translation", {})
                src_text = tr.get(src_code, "").strip()
                tgt_text = tr.get(tgt_code, "").strip()
                if not src_text or not tgt_text or len(src_text) < 10:
                    continue

                prompt = f"Translate the following text from {src_lang} into {tgt_lang}. Output only the translation:\n\n{src_text}"
                messages = [{"role": "user", "content": prompt}]
                samples.append((messages, tgt_text, {"category": pair_name}))
                count += 1
                if count >= per_pair:
                    break
        except Exception as e:
            logger.warning(f"Could not load WMT pair {pair_name}: {e}")

    # Stratified, not a contiguous head (audit RC-1).
    samples = stratified_sample(samples, limit, None, seed="i18n_translate")

    logger.info(f"Loaded {len(samples)} multilingual translation samples from WMT.")
    return samples


def _eval_i18n_translate(response_text: str, gold_text: str) -> bool:
    """Evaluate translation quality using token F1 overlap against gold reference."""
    if not response_text or not gold_text:
        return False

    pred = response_text.strip().lower()
    gold = gold_text.strip().lower()

    # Remove enclosing quotes or formatting
    pred = re.sub(r'^["\'`](.*)["\'`]$', r'\1', pred).strip()

    return chrf_score(pred, gold) >= 40.0


def chrf_score(pred: str, gold: str) -> float:
    """chrF (character n-gram F-score, 0-100) - a canonical MT metric.

    Replaces a set-based word F1 with a 0.45 threshold. That measure was order- and
    repetition-insensitive - a bag of the right words in any order scored the same as a
    correct translation, and a translation repeating one word scored the same as one
    using it once - and it discarded morphology entirely, which matters most for exactly
    the languages here. chrF is character-level, so it credits partial word matches and is
    the standard choice when COMET is unavailable.

    `sacrebleu` is used when installed; the fallback is an equivalent local implementation
    (character 1-6 grams, beta=2) so the suite does not silently change metric.
    """
    pred, gold = str(pred or "").strip(), str(gold or "").strip()
    if not pred or not gold:
        return 0.0
    try:
        import sacrebleu
        return float(sacrebleu.sentence_chrf(pred, [gold]).score)
    except ImportError:
        pass

    from collections import Counter
    beta, total_p, total_r, orders = 2.0, 0.0, 0.0, 0
    p_chars = re.sub(r"\s+", "", pred)
    g_chars = re.sub(r"\s+", "", gold)
    for n in range(1, 7):
        p_ngrams = Counter(p_chars[i:i + n] for i in range(len(p_chars) - n + 1))
        g_ngrams = Counter(g_chars[i:i + n] for i in range(len(g_chars) - n + 1))
        if not p_ngrams or not g_ngrams:
            continue
        overlap = sum((p_ngrams & g_ngrams).values())
        total_p += overlap / sum(p_ngrams.values())
        total_r += overlap / sum(g_ngrams.values())
        orders += 1
    if not orders:
        return 0.0
    precision, recall = total_p / orders, total_r / orders
    if precision + recall == 0:
        return 0.0
    return 100.0 * (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)


def run_i18n_translate(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run native multilingual translation quality evaluation suite.

    Translation quality is continuous, so the headline is the **mean chrF**; the pass rate
    at the chrF>=40 threshold is kept alongside it. Scope: WMT19 into-English only
    (de/zh/ru/cs -> en), so this does not measure generation *into* those languages.
    """
    samples = _load_i18n_translate_samples(limit=kwargs.get("limit"))
    result = run_eval_suite(
        eval_name="i18n_translate",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_i18n_translate,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens"),
    )
    scores = []
    for trace in result.get("sample_traces", []):
        if trace.get("response_text") is None:
            continue
        s = chrf_score(trace["response_text"], str(trace.get("gold_answer") or ""))
        trace["chrf"] = round(s, 2)
        scores.append(s)
    result["pass_rate_chrf40"] = result.get("accuracy")
    result["metric"] = "mean chrF (into-English only: de/zh/ru/cs -> en)"
    if scores:
        result["accuracy"] = round(sum(scores) / len(scores), 2)
        result["mean_chrf"] = result["accuracy"]
    return result
