# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# Evaluation Suite: gaia2
# Description: GAIA2 / Meta Agents Research Environments (ARE) - stateful multi-turn agentic benchmark

"""gbench native built-in runner for gaia2 (Tool Use & Agentic).

Canonical GAIA2 (`meta-agents-research-environments/gaia2`) is a stateful,
multi-turn, time-sensitive agentic benchmark: the model must act as an agent inside
Meta's ARE simulator, issuing tool calls over several turns against evolving app
state, and is scored by the environment's oracle (final-state checks + timing) plus
an LLM judge - not by a single model response.

gbench drives one request per sample and cannot host that agent loop, so a
single-turn proxy (e.g. fuzzy function-name matching) would not measure the
benchmark. gaia2 is therefore **blocked_external**: it skips cleanly until the ARE
simulator is wired in as a Pattern-B harness (like terminal_bench). No approximate
score is emitted. See docs/evals/gaia2.md.
"""

import importlib.util
import logging
import os
from typing import Any, Dict, Tuple
from .swebench_common import skipped_result

logger = logging.getLogger(__name__)

PILLAR = "Tool Use & Agentic"
DOCS_URL = "docs/evals/gaia2.md"


def check_gaia2_prerequisites() -> Tuple[bool, str]:
    """The Meta ARE simulator must be installed AND explicitly opted into.

    Even when both hold, gbench does not yet host the ARE agent loop, so this
    returns a 'harness not wired' reason - the suite always skips today. The gate is
    kept so the message is accurate about what is missing.
    """
    if importlib.util.find_spec("are") is None:
        return False, ("GAIA2 requires the Meta ARE simulator "
                       "(pip install meta-agents-research-environments); it is not installed.")
    if os.getenv("GAIA2_RUN") != "1":
        return False, "GAIA2 is gated (multi-turn ARE simulator). Set GAIA2_RUN=1 to enable."
    return False, ("GAIA2 is blocked_external: the ARE agent-in-the-loop harness is not yet wired into "
                   "gbench, so no faithful score can be produced (a single-turn proxy is not the benchmark).")


def run_gaia2(
    model_name: str,
    base_url: str,
    concurrency: int = 4,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """GAIA2 is blocked_external; skip cleanly (no approximate scoring)."""
    _ok, reason = check_gaia2_prerequisites()
    return skipped_result("gaia2", model_name, reason, DOCS_URL)
