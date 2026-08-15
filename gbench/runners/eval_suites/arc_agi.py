# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# Evaluation Suite: arc_agi
# Description: ARC-AGI-1 (Abstraction & Reasoning Corpus - grid-transformation puzzles)

"""gbench native built-in runner for arc_agi (Abstract Reasoning).

Canonical ARC-AGI-1 (Francois Chollet). Loads the public 400-task evaluation
corpus of 2D integer-grid puzzles, serialises each task's train demonstrations +
test input with the official ARC-Prize prompt, and scores by EXACT grid match
(pass@1). This is NOT allenai/ai2_arc (a different grade-school-science MCQ set).
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from .base import run_eval_suite
from .sampling import stratified_sample

logger = logging.getLogger(__name__)

PILLAR = "Language, Knowledge & Reasoning"

_PROMPT_TEMPLATE = (
    "You are participating in a puzzle solving competition. You are an expert at "
    "solving puzzles.\n\n"
    "Below is a list of input and output pairs with a pattern. Your goal is to "
    "identify the pattern or transformation in the training examples that maps the "
    "input to the output, then apply that pattern to the test input to give a final "
    "output.\n\n"
    "Respond in the format of the training output examples\n\n"
    "--Training Examples--\n{training_examples}--End of Training Examples--\n\n"
    "--Test Input--\n{test_input}\n--End of Test Input--\n\n"
    "Your response:"
)


def _grid_ok(g: Any) -> bool:
    return (
        isinstance(g, list) and len(g) > 0
        and all(isinstance(r, list) and len(r) > 0 for r in g)
        and all(isinstance(c, int) for r in g for c in r)
    )


def _load_arc_agi_samples(
    enable_thinking: bool = False,
    limit: Optional[int] = None,
    split: str = "evaluation",
    dataset_id: str = "dataartist/arc-agi",
) -> List[Tuple[List[Dict[str, Any]], Any, Dict[str, Any]]]:
    """Load ARC-AGI-1 (evaluation split) from HF Hub; raises on load/schema failure."""
    try:
        from datasets import load_dataset
        ds = load_dataset(dataset_id, split=split)
        rows = list(ds)
    except Exception as e:
        logger.error(f"Failed to load dataset for arc_agi: {e}")
        raise RuntimeError(f"Could not load dataset for arc_agi: {e}") from e

    if not rows:
        raise RuntimeError("Dataset for arc_agi returned empty rows")

    samples: List[Tuple[List[Dict[str, Any]], Any, Dict[str, Any]]] = []
    for idx, item in enumerate(rows):
        train_pairs = item.get("train")
        test_pairs = item.get("test")
        task_id = item.get("id") or f"task_{idx}"
        if (not isinstance(train_pairs, list) or not train_pairs
                or not isinstance(test_pairs, list) or not test_pairs):
            raise RuntimeError(
                "arc_agi: unexpected dataset schema (missing 'train'/'test'); "
                "refusing to fabricate sample data"
            )

        demos = ""
        for i, pair in enumerate(train_pairs):
            inp, out = pair.get("input"), pair.get("output")
            if not _grid_ok(inp) or not _grid_ok(out):
                raise RuntimeError(f"arc_agi: bad train grid in task {task_id}")
            demos += (
                f"--Example {i}-- \n\n INPUT: \n\n{json.dumps(inp)}"
                f"\n\nOUTPUT: \n\n{json.dumps(out)}\n\n"
            )

        for k, tpair in enumerate(test_pairs):
            t_in, t_out = tpair.get("input"), tpair.get("output")
            if not _grid_ok(t_in) or not _grid_ok(t_out):
                raise RuntimeError(f"arc_agi: bad/missing test grid in task {task_id}")
            prompt = _PROMPT_TEMPLATE.format(
                training_examples=demos, test_input=json.dumps(t_in)
            )
            messages = [{"role": "user", "content": prompt}]
            samples.append((
                messages, t_out,
                {"category": "grid_transformation", "task_id": task_id, "pair_index": k},
            ))

    # Stratified, not a contiguous head (audit RC-1).
    samples = stratified_sample(samples, limit, None, seed="arc_agi")
    logger.info(f"Loaded {len(samples)} arc_agi samples from HF Hub ('{dataset_id}').")
    return samples


def _parse_grid(text: str) -> Optional[List[List[int]]]:
    """Extract the LAST valid 2D int grid (JSON [[...]]) from the model output."""
    if not text:
        return None
    cleaned = re.sub(r"```[a-zA-Z]*", "", text).replace("```", "")
    candidates: List[List[List[int]]] = []
    n = len(cleaned)
    i = 0
    while i < n:
        if cleaned[i] == "[":
            depth = 0
            j = i
            while j < n:
                if cleaned[j] == "[":
                    depth += 1
                elif cleaned[j] == "]":
                    depth -= 1
                    if depth == 0:
                        frag = cleaned[i:j + 1]
                        try:
                            val = json.loads(frag)
                        except Exception:
                            val = None
                        if _grid_ok(val):
                            candidates.append(val)
                        i = j
                        break
                j += 1
        i += 1
    return candidates[-1] if candidates else None


def _eval_arc_agi(response_text: str, gold: Any) -> bool:
    """Exact grid match: predicted 2D int grid == gold 2D int grid."""
    pred = _parse_grid(response_text)
    if pred is None:
        return False
    try:
        g = [[int(c) for c in row] for row in gold]
        p = [[int(c) for c in row] for row in pred]
    except Exception:
        return False
    return p == g


def run_arc_agi(
    model_name: str,
    base_url: str,
    concurrency: int = 1,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run native ARC-AGI-1 grid-transformation evaluation suite."""
    samples = _load_arc_agi_samples(
        enable_thinking=enable_thinking, limit=kwargs.get("limit")
    )
    return run_eval_suite(
        eval_name="arc_agi",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_arc_agi,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens", 8192 if enable_thinking else 4096),
        temperature=kwargs.get("temperature", 0.0),
    )
