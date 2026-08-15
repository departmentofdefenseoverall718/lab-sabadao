# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# Evaluation Suite: mcp_atlas
# Description: MCP-Atlas (Scale AI Multi-Turn Agentic MCP Tool-Use Evaluation across 20+ MCP Servers)

"""gbench native built-in runner for mcp_atlas (Tool Use & Function Calling)."""

import ast
import csv
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from .base import run_eval_suite, DEFAULT_JUDGE_MODEL
from .sampling import stratified_sample

logger = logging.getLogger(__name__)

PILLAR = "Tool Use & Function Calling"


def _parse_list_field(raw: Any) -> List[str]:
    """MCP-Atlas stores `ENABLED_TOOLS` / `GTFA_CLAIMS` as a *string* holding a list.

    The literal uses Python quoting (mixed `'`/`"`), so `json.loads` alone is not enough.
    """
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    text = str(raw or "").strip()
    if not text:
        return []
    for loader in (json.loads, ast.literal_eval):
        try:
            value = loader(text)
        except Exception:
            continue
        if isinstance(value, list):
            return [str(x).strip() for x in value if str(x).strip()]
    return [text]


def _load_mcp_atlas_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Load MCP-Atlas benchmark dataset directly from HF Hub (ScaleAI/MCP-Atlas)."""
    rows = []
    try:
        from datasets import load_dataset
        ds = load_dataset("ScaleAI/MCP-Atlas", split="train")
        rows = list(ds)
    except Exception as e:
        logger.error(f"Failed to load dataset for mcp_atlas: {e}")
        raise RuntimeError(f"Could not load dataset for mcp_atlas: {e}") from e

    if not rows:
        raise RuntimeError("Dataset for mcp_atlas returned empty rows")

    # Stratified, not a contiguous head (audit RC-1).
    rows = stratified_sample(rows, limit, None, seed="mcp_atlas")

    samples = []
    for item in rows:
        prompt_text = str(item.get("PROMPT") or "").strip()
        enabled_tools = _parse_list_field(item.get("ENABLED_TOOLS"))
        gtfa_claims = _parse_list_field(item.get("GTFA_CLAIMS"))
        if not prompt_text or not gtfa_claims:
            continue

        # Both columns are *strings* holding a list literal. Slicing the raw value took the
        # first ten CHARACTERS and joined them, so the prompt advertised the tool list as
        # `[, ", f, e, t, c, h, _, f, e`; and the claims never split, so the whole literal
        # (brackets and quotes included) was one "claim".
        tools_str = ", ".join(enabled_tools) if enabled_tools else "none"
        server = enabled_tools[0].split("_")[0] if enabled_tools else "mcp"

        prompt = (
            f"MCP tools available on this task: {tools_str}\n\n"
            f"User Task: {prompt_text}\n\n"
            "Answer the task. State every fact your answer depends on explicitly, and "
            "finish with your conclusion on the last line as: Final Answer: <response>"
        )
        messages = [{"role": "user", "content": prompt}]
        # Declare the named tools on the request. The prompt already advertises them, so
        # the model tries to call one; gemma-4 then emits `<|tool_response>` (a stop token)
        # and vLLM's tool-call parser lifts the call out of `content`. With no `tools` on
        # the request the extraction is DISCARDED - measured live as completion_tokens=34,
        # content=null, tool_calls=[], stop_reason=50, i.e. 13/20 "empty responses" that
        # were really answers thrown away. MCP-Atlas ships names but no JSON schemas, so
        # these are permissive stubs: enough for the parser to bind the call.
        meta = {"category": server, "claims": gtfa_claims}
        if enabled_tools:
            meta["tools"] = [{"type": "function",
                              "function": {"name": t,
                                           "description": f"MCP tool {t}.",
                                           "parameters": {"type": "object",
                                                          "properties": {},
                                                          "additionalProperties": True}}}
                             for t in enabled_tools if isinstance(t, str) and t.strip()]
        samples.append((messages, json.dumps(gtfa_claims), meta))

    logger.info(f"Loaded {len(samples)} mcp_atlas samples.")
    return samples


_CLAIM_JUDGE_PROMPT = """You are verifying a model's answer against the ground-truth claims for a task.

# TASK GIVEN TO THE MODEL
{task}

# GROUND-TRUTH CLAIMS
{claims}

# MODEL ANSWER
{answer}

For each numbered claim, decide whether the MODEL ANSWER asserts the same fact. Wording may
differ; the fact must match. A claim the answer does not address, contradicts, or hedges
about is NOT supported.

Reply with ONLY a JSON object: {{"supported": [<claim numbers that are supported>]}}"""


def _eval_mcp_atlas(response_text: str, gold_target: str) -> bool:
    """Deterministic floor for MCP-Atlas: every ground-truth claim stated verbatim.

    NOT the scoring authority - `_async_judge_mcp_atlas` is. MCP-Atlas grades an answer
    against `GTFA_CLAIMS`, natural-language facts the answer must assert ("The domain
    registration year of the AssaultCube's official site is 2006."), which is a semantic
    judgement. The previous scorer accepted a claim when 75% of its 4+ character tokens
    appeared anywhere in the response, so echoing the task's own nouns largely satisfied it
    while the decisive value (the year) was never checked. With no judge available this
    falls back to exact containment of each claim, which under-credits paraphrase but
    cannot invent a pass.
    """
    claims = _parse_list_field(gold_target)
    if not response_text or not claims:
        return False
    resp_lower = response_text.lower()
    return all(c.lower() in resp_lower for c in claims)


async def _async_judge_mcp_atlas(
    sample_traces: List[Dict[str, Any]],
    judge_model: str = DEFAULT_JUDGE_MODEL,
    judge_concurrency: int = 32,
) -> None:
    """Verify each GTFA claim against the model's answer (the canonical MCP-Atlas metric).

    A sample is correct when EVERY claim is supported. `claim_recall` is recorded per
    sample as a secondary, finer-grained signal.
    """
    import asyncio
    from tqdm import tqdm

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.warning("GEMINI_API_KEY not set; MCP-Atlas falls back to verbatim claim "
                       "containment, which under-credits correct paraphrases.")
        for trace in sample_traces:
            trace["is_correct"] = _eval_mcp_atlas(str(trace.get("response_text") or ""),
                                                  trace.get("gold_answer"))
            trace["judge_grade"] = "containment_fallback"
        return

    try:
        from google import genai
        client = genai.Client(api_key=api_key)
    except Exception as e:
        logger.error(f"Failed to initialize google-genai client: {e}")
        for trace in sample_traces:
            trace["is_correct"] = False
            trace["judge_grade"] = "judge_error"
        return

    semaphore = asyncio.Semaphore(judge_concurrency)

    async def _judge_single(trace: Dict[str, Any], pbar: tqdm) -> None:
        claims = _parse_list_field(trace.get("gold_answer"))
        answer = str(trace.get("response_text") or "").strip()
        if not answer or not claims:
            trace["is_correct"] = False
            trace["judge_grade"] = "no_response" if not answer else "no_claims"
            pbar.update(1)
            return

        messages = trace.get("messages") or []
        task = messages[0].get("content", "") if messages else ""
        prompt = _CLAIM_JUDGE_PROMPT.format(
            task=task[:4000],
            claims="\n".join(f"{i + 1}. {c}" for i, c in enumerate(claims)),
            answer=answer[:8000])

        supported: Optional[List[int]] = None
        async with semaphore:
            for attempt in range(3):
                try:
                    res = await client.aio.models.generate_content(model=judge_model,
                                                                   contents=prompt)
                    m = re.search(r'"supported"\s*:\s*\[([^\]]*)\]', res.text or "")
                    if m:
                        supported = [int(x) for x in re.findall(r"\d+", m.group(1))]
                        break
                except Exception as e:
                    if attempt == 2:
                        logger.warning(f"MCP-Atlas judge failed after 3 attempts: {e}")
                await asyncio.sleep(1.0 * (attempt + 1))

        if supported is None:
            # An unparseable verdict is not a pass, and must be distinguishable from one.
            trace["is_correct"] = False
            trace["judge_grade"] = "judge_error"
        else:
            hit = len({n for n in supported if 1 <= n <= len(claims)})
            trace["claim_recall"] = round(hit / len(claims), 4)
            trace["is_correct"] = hit == len(claims)
            trace["judge_grade"] = f"{hit}/{len(claims)} claims"
        pbar.update(1)

    with tqdm(total=len(sample_traces), desc="Judging [MCP_ATLAS]") as pbar:
        await asyncio.gather(*[_judge_single(t, pbar) for t in sample_traces])


def run_mcp_atlas(
    model_name: str,
    base_url: str,
    limit: Optional[int] = None,
    concurrency: int = 4,
    enable_thinking: bool = False,
    results_dir: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Run native MCP_ATLAS evaluation benchmark.

    NB this is a **closed-book** run. MCP-Atlas tasks need live MCP servers (GitHub API,
    whois, fetch) to reach their answers; gbench lists the enabled tool names in the prompt
    but executes nothing, so the score is a lower bound on agentic performance and is not
    comparable with a figure produced by a real MCP harness. `closed_book: true` is
    recorded on the result to keep that visible.
    """
    samples = _load_mcp_atlas_samples(limit=limit)
    result = run_eval_suite(
        eval_name="mcp_atlas",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        async_eval_fn=_async_judge_mcp_atlas,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens", 4096),
    )
    recalls = [t["claim_recall"] for t in result.get("sample_traces", [])
               if isinstance(t.get("claim_recall"), (int, float))]
    result["closed_book"] = True
    result["metric"] = "all GTFA claims supported (closed-book, no MCP execution)"
    if recalls:
        result["mean_claim_recall"] = round(sum(recalls) / len(recalls) * 100.0, 2)
    return result
