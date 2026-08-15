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

"""Base async execution engine for self-contained evaluation benchmark suites.

Handles concurrent HTTP dispatching to OpenAI-compatible server, real-time tqdm
progress logging, error retries, and standardized metric calculations.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

from .sampling import stratified_sample


class Reply(NamedTuple):
    """One model response, with the metadata needed to tell WHY it looks the way it does."""
    text: Optional[str]
    tool_calls: Optional[List[Dict[str, Any]]]
    finish_reason: Optional[str] = None
    reasoning: Optional[str] = None
    error: Optional[str] = None
    completion_tokens: int = 0
    stop_reason: Optional[Any] = None


def classify_reply(reply: "Reply") -> str:
    """`ok` | `request_failed` | `truncated` | `empty_reasoned` | `empty`.

    Without this every one of these looked like an ordinary wrong answer:
      * `truncated`       - hit the output-token budget mid-answer. In the 2026-08-15
                            sweep this alone was codeforces' IOI 0/6 and 10/20 of
                            swe_bench_multilingual.
      * `empty_reasoned`  - the model spent its whole budget in the reasoning channel and
                            emitted no `content` (gemma-4 with `--reasoning-parser gemma4`).
      * `empty`           - the server returned 200 with nothing at all.
    """
    if reply.text is None:
        return "request_timeout" if (reply.error or "").startswith("timeout") else "request_failed"
    has_output = bool(str(reply.text).strip()) or bool(reply.tool_calls)
    if reply.finish_reason == "length":
        return "truncated"
    if not has_output:
        if (reply.reasoning or "").strip():
            return "empty_reasoned"
        # The server generated tokens and returned none of them. On gemma-4 this is the
        # model emitting a tool call followed by `<|tool_response>` (token 50, a configured
        # stop token): vLLM's tool-call parser lifts the call out of `content`, and when the
        # request declared no `tools` there is nowhere to put it, so it is DISCARDED.
        # Measured live: completion_tokens=34, content=null, tool_calls=[], stop_reason=50.
        # That is a suite/serving mismatch, not a model that said nothing - scoring it as an
        # ordinary wrong answer hid 28 samples across bfcl_v3_live / mcp_atlas / skillsbench.
        if reply.completion_tokens:
            return "output_discarded"
        return "empty"
    return "ok"

#: Judge / grader model for every LLM-graded built-in suite, overridable with
#: `GBENCH_JUDGE_MODEL`. Previously each suite hardcoded its own: 13 sat on
#: `gemini-2.5-flash`, 4 on `gemini-3.5-flash`, and the plugin engine on
#: `gemini-3.6-flash` - so "the judge" meant three different models depending on which
#: suite you looked at, and no single knob moved them.
DEFAULT_JUDGE_MODEL = os.environ.get("GBENCH_JUDGE_MODEL", "gemini-3.6-flash")

#: Run-level generation knobs, set once by the runner and consulted by `run_eval_suite`
#: for anything a suite did not pass explicitly.
#:
#: `evals.py` hands every suite the same kwargs dict, but a suite only honours a knob if
#: it forwards it - and 107 of 122 never forwarded `temperature`, so `--temperature`
#: (documented as applying "across all benchmarks") silently did nothing for them
#: (audit RC-2). Resolving here means a knob cannot be lost by omission; a suite that
#: passes a value explicitly still wins.
_RUN_KNOBS: Dict[str, Any] = {}

#: Output-token FLOOR for suites whose answers are long by nature.
#:
#: Sized against the serving envelope rather than guessed. On the 8xA100 box the model is
#: served DP=4 x TP=2 at max_model_len=262144. Gemma-4 26B-A4B keeps 5 full-attention
#: layers and 25 sliding (window 1024), 8 KV heads x 256 dim in bf16, so KV per sequence is
#:     L x 40 KiB   (the 5 full layers)  +  200 MiB   (the 25 sliding layers, capped)
#: With ~92 GiB of KV per engine that sustains roughly
#:     ~32.5k tokens/sequence at --batch-sizes 256
#:     ~70.2k tokens/sequence at --batch-sizes 128   (the non-thinking campaign)
#:     ~70.2k tokens/sequence at --batch-sizes 128   (32 per engine, thinking)
#: before vLLM starts preempting. Exceeding it costs throughput, not correctness, and
#: max_tokens is only a ceiling - most responses are far shorter - so these are set
#: generously against that envelope.
#:
#: The 2026-08-15 sweep truncated 16 suites, 10 of which had no entry here at all and ran
#: at the old 8192 default (copilot_bench_swe lost 8/20 even at 32768).
#:
#: Precedence: an explicit --max-output-tokens always wins; otherwise the suite's own value
#: is raised to this floor. A floor rather than a default because every one of these suites
#: passes a hardcoded number, which would shadow a plain default (audit P1-7).
SUITE_MIN_OUTPUT_TOKENS = {
    # whole-file patches and multi-file diffs: measured up to 114 k chars, still cut at 65536
    "swe_bench_pro": 65536, "swe_bench_live": 65536, "swe_bench_multilingual": 65536,
    "copilot_bench_swe": 65536, "multi_swe_bench": 65536, "swe_lancer": 65536,
    # programs, proofs and long chains of reasoning. Raised 32768 -> 65536 after the
    # 2026-08-15 run still truncated arc_agi 3/20 and livebench 6/20 at 32768; the
    # no-think campaign moved to --batch-sizes 128, whose ~70.2k envelope makes room.
    "codeforces": 65536, "lcb": 65536, "scicode": 65536, "multipl_e": 65536,
    "bigcodebench": 65536, "aider_polyglot": 65536, "skillsbench": 65536,
    "ojbench": 65536, "cybergym": 65536,
    "putnam": 65536, "putnam_formal": 65536, "imo_answer_bench": 65536,
    "humanitys_last_exam": 65536, "hmmt": 65536, "aime": 65536, "new_amc_aime": 65536,
    "arc_agi": 65536, "livebench": 65536, "gpqa": 65536, "gpqa_diamond": 65536,
    # long-form generation that also hit the ceiling
    "browsecomp": 16384, "culer": 16384, "ifeval": 16384, "loft_x_arxiv": 16384,
    "gaia": 16384, "deepsearch_qa": 16384,
}


def set_run_knobs(**knobs: Any) -> None:
    """Record run-level generation knobs (called by the runner, once per suite)."""
    _RUN_KNOBS.update({k: v for k, v in knobs.items() if v is not None})


def get_run_knob(name: str, default: Any = None) -> Any:
    return _RUN_KNOBS.get(name, default)


#: Pessimistic per-sequence decode rate used to size the HTTP timeout. Under heavy
#: concurrency each sequence gets a small share of aggregate throughput, so a long
#: generation legitimately takes a long time.
MIN_DECODE_TOK_S = float(os.environ.get("GBENCH_MIN_DECODE_TOK_S", "8"))
REQUEST_TIMEOUT_FLOOR_S = int(os.environ.get("GBENCH_REQUEST_TIMEOUT_S", "1200"))


def request_timeout_s(max_output_tokens: Optional[int]) -> int:
    """How long one request may take, given how much it is allowed to generate.

    A fixed 1200 s was fine at 8192 tokens and wrong at 65536: at 256-way concurrency each
    sequence gets a small slice of aggregate decode throughput, so a long answer needs well
    over an hour. The request would time out, retry three times and land as a bare
    `request_failed` - i.e. the campaign would silently drop exactly its longest answers,
    which are the ones the big budgets exist to capture.

    Scales with the token budget rather than the concurrency because the budget is what
    bounds the work; `GBENCH_MIN_DECODE_TOK_S` tunes the assumed floor rate.
    """
    budget = max_output_tokens or 8192
    return max(REQUEST_TIMEOUT_FLOOR_S,
               int(REQUEST_TIMEOUT_FLOOR_S + budget / max(1.0, MIN_DECODE_TOK_S)))


#: chars-per-token used only to pre-clamp `max_tokens`; deliberately low (i.e. it
#: OVER-estimates the prompt) so the estimate errs toward a smaller, safe request.
_CHARS_PER_TOKEN = 3.0


def clamp_to_context(messages: Any, max_tokens: Optional[int],
                     max_model_len: Optional[int] = None) -> Tuple[Optional[int], Optional[str]]:
    """Shrink `max_tokens` so `prompt + max_tokens` fits the server's context window.

    vLLM rejects a request whose prompt+max_tokens exceeds max_model_len. The old code
    only recovered *reactively*, by parsing the 400 and retrying - which costs a round
    trip per request and, when the prompt alone is over the limit, simply retried three
    times and recorded `request_failed` with no reason at all. On the 2026-08-15 sweep
    that was mrcr: 6 rows with 1.3-4.2 MB prompts (~330k-1M tokens) against a 262144
    window, reported as unexplained failures.

    Returns `(clamped_max_tokens, reason_if_unanswerable)`. A non-None reason means the
    prompt cannot fit at all and the request should not be sent.
    """
    if not max_model_len:
        max_model_len = get_run_knob("max_model_len")
    if not max_model_len:
        return max_tokens, None
    try:
        import json as _json
        prompt_chars = len(_json.dumps(messages)) if not isinstance(messages, str) else len(messages)
    except Exception:
        return max_tokens, None
    est_prompt = int(prompt_chars / _CHARS_PER_TOKEN)
    headroom = max_model_len - est_prompt - 64        # 64 tokens of slack for the template
    if headroom < 256:
        return max_tokens, (
            f"prompt is ~{est_prompt:,} tokens, which does not fit the model's "
            f"{max_model_len:,}-token context window (no room left to answer)")
    if max_tokens is None or max_tokens > headroom:
        return headroom, None
    return max_tokens, None

_GRADE_VERDICT_RE = re.compile(r"GRADE\s*:\s*(NOT_ATTEMPTED|INCORRECT|CORRECT)", re.IGNORECASE)


def parse_grade_verdict(grade_text: str) -> bool:
    """Is the judge's verdict CORRECT? Reads the verdict, not the whole reply.

    The suites all asked for `Grade: CORRECT / INCORRECT` and then tested
    ``"CORRECT" in grade_str and "INCORRECT" not in grade_str`` over the entire response.
    Judges routinely ignore the one-word instruction and return several paragraphs of
    analysis, so a reply reasoning "the patch is not incorrect ... GRADE: CORRECT" scored
    as wrong: the word `INCORRECT` appeared somewhere in the prose.

    The last explicit `GRADE:` line wins (a judge that revises itself means the later
    one); only when the reply contains no explicit verdict at all do we fall back to the
    old substring behaviour.
    """
    text = str(grade_text or "")
    verdicts = _GRADE_VERDICT_RE.findall(text)
    if verdicts:
        return verdicts[-1].upper() == "CORRECT"
    upper = text.upper()
    return "CORRECT" in upper and "INCORRECT" not in upper
try:
    import aiohttp
except ImportError:
    aiohttp = None

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else None

logger = logging.getLogger(__name__)
logging.getLogger("huggingface_hub.repocard").setLevel(logging.ERROR)
for noisy_logger in ["google_genai", "google", "grpc", "httpx", "httpcore", "urllib3"]:
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def strip_thinking_tags(text: Optional[str]) -> str:
    """Strip <thought>...</thought> or <reasoning>...</reasoning> blocks if present."""
    if not text:
        return ""
    cleaned = re.sub(r"<thought>.*?</thought>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<reasoning>.*?</reasoning>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    # If output was entirely wrapped in thought tags, return stripped raw text
    return cleaned.strip() if cleaned.strip() else text.strip()


def _sanitize_for_trace(obj: Any) -> Any:
    """Recursively truncate bulky payloads for human-readable JSON traces."""
    if isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            if isinstance(v, str) and v.startswith("data:image/") and len(v) > 64:
                res[k] = v[:48] + "..."
            elif k == "tools" and isinstance(v, list) and len(v) > 4:
                # A tool-use suite may offer 100+ schemas, identical on every sample; kept
                # verbatim they dominate the result file.
                # Names are what a failure analysis needs, so keep only those.
                res[k] = [t.get("function", {}).get("name", t) if isinstance(t, dict) else t
                          for t in v]
            else:
                res[k] = _sanitize_for_trace(v)
        return res
    elif isinstance(obj, list):
        return [_sanitize_for_trace(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(_sanitize_for_trace(item) for item in obj)
    elif isinstance(obj, str) and obj.startswith("data:image/") and len(obj) > 64:
        return obj[:48] + "..."
    return obj


async def _send_single_request(
    session: aiohttp.ClientSession,
    api_url: str,
    model_name: str,
    messages: List[Dict[str, Any]],
    extra_payload: Dict[str, Any],
    semaphore: asyncio.Semaphore,
    pbar: tqdm,
    max_output_tokens: int = 8192,
    temperature: float = 0.0,
    thinking: bool = False,
) -> "Reply":
    """Send a single chat completion request.

    Returns a `Reply`: rendered text, raw `tool_calls`, `finish_reason` and `reasoning`.

    The raw `tool_calls` array is kept because the text rendering is lossy (it stringifies
    every argument, hiding an INTEGER answered as "2010"). `finish_reason` and `reasoning`
    are kept because without them a TRUNCATED answer and an answer that lived entirely in
    the model's reasoning channel are both indistinguishable from an ordinary wrong answer
    (audit RC-3: 22 truncated, 32 empty in the 2026-08-15 sweep, none of it visible).
    """
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_output_tokens,
    }
    ALLOWED_API_KEYS = {
        "tools", "tool_choice", "response_format", "top_p", "top_k", "presence_penalty",
        "frequency_penalty", "stop", "seed", "stream", "n", "logit_bias",
        # CC3: let suites pass native reasoning toggle, vision soft-token control, and
        # per-sample generation overrides through to the server (previously silently dropped).
        "chat_template_kwargs", "mm_processor_kwargs", "extra_body", "max_tokens", "temperature",
    }
    if extra_payload:
        for k, v in extra_payload.items():
            if k in ALLOWED_API_KEYS:
                payload[k] = v
    # CC2: actually toggle the model's native reasoning channel to match --eval-thinking.
    # Only inject when a suite hasn't already set enable_thinking explicitly (e.g. via a
    # sample's extra_payload), so per-suite intent still wins.
    ctk = payload.get("chat_template_kwargs")
    if not (isinstance(ctk, dict) and "enable_thinking" in ctk):
        payload["chat_template_kwargs"] = {
            **(ctk if isinstance(ctk, dict) else {}),
            "enable_thinking": bool(thinking),
        }

    last_error: Optional[str] = None
    async with semaphore:
        for attempt in range(3):
            try:
                async with session.post(api_url, json=payload) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        choice = data["choices"][0]
                        msg = choice["message"]
                        content = msg.get("content") or ""
                        tool_calls = msg.get("tool_calls")
                        finish = choice.get("finish_reason")
                        stop_reason = choice.get("stop_reason")
                        usage_tokens = int((data.get("usage") or {}).get("completion_tokens") or 0)
                        # vLLM exposes the gemma-4 reasoning channel under either key.
                        reasoning = msg.get("reasoning_content") or msg.get("reasoning") or None
                        if tool_calls:
                            call_strs = []
                            for tc in tool_calls:
                                fn = tc.get("function", {})
                                fname = fn.get("name", "")
                                fargs_str = fn.get("arguments", "{}")
                                py_kwargs_clean = ""
                                try:
                                    import json
                                    args_dict = json.loads(fargs_str) if isinstance(fargs_str, str) else fargs_str
                                    if isinstance(args_dict, dict):
                                        py_kwargs_clean = fname + "(" + ", ".join(f"{k}={v}" for k, v in args_dict.items()) + ")"
                                except Exception:
                                    py_kwargs_clean = ""
                                call_strs.append(f"{fname}({fargs_str}) {py_kwargs_clean}")
                            return Reply("\n".join(call_strs) + "\n" + str(content),
                                         tool_calls, finish, reasoning, None,
                                         usage_tokens, stop_reason)
                        return Reply(str(content), None, finish, reasoning, None,
                                     usage_tokens, stop_reason)
                    else:
                        err = await resp.text()
                        if resp.status == 400 and ("maximum context length" in err or "input tokens" in err):
                            match_input = re.search(r"contains at least (\d+) input tokens", err)
                            match_max = re.search(r"maximum context length is (\d+) tokens", err)
                            if match_input and match_max:
                                inp_tokens = int(match_input.group(1))
                                max_len = int(match_max.group(1))
                                clamped_tokens = max(64, max_len - inp_tokens - 16)
                                if clamped_tokens < payload.get("max_tokens", max_output_tokens):
                                    payload["max_tokens"] = clamped_tokens
                                    continue
                        logger.warning(f"HTTP {resp.status} on eval request: {err[:200]}")
            except asyncio.TimeoutError:
                # Distinct from any other failure: a timeout means the generation was still
                # running, so it is the LONG answers that get dropped - exactly the ones the
                # large budgets exist to capture. Recorded so it is never a mystery failure.
                last_error = f"timeout after {request_timeout_s(max_output_tokens)}s"
                if attempt == 2:
                    logger.warning("Request timed out after %ss (max_tokens=%s). Raise "
                                   "GBENCH_REQUEST_TIMEOUT_S / lower --batch-sizes, or the "
                                   "longest answers will be lost.",
                                   request_timeout_s(max_output_tokens), max_output_tokens)
                await asyncio.sleep(0.5 * (attempt + 1))
            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                if attempt == 2:
                    logger.warning(f"Request exception on attempt {attempt+1}: {e}")
                await asyncio.sleep(0.5 * (attempt + 1))
        return Reply(None, None, None, None, last_error)


async def _run_suite_async(
    eval_name: str,
    model_name: str,
    base_url: str,
    concurrency: int,
    samples: List[Tuple[List[Dict[str, Any]], Any, Dict[str, Any]]],
    eval_fn: Optional[Callable[[str, Any], bool]] = None,
    async_eval_fn: Optional[Callable[[List[Dict[str, Any]]], Any]] = None,
    thinking: bool = False,
    extra_payload: Optional[Dict[str, Any]] = None,
    limit: Optional[int] = None,
    max_output_tokens: Optional[int] = None,
    temperature: float = 0.0,
    tool_executor: Optional[Callable[[str, Dict[str, Any]], Any]] = None,
    max_tool_rounds: int = 8,
) -> Dict[str, Any]:
    """Execute evaluation suite samples concurrently with tqdm progress bar."""
    api_url = f"{base_url.rstrip('/')}/chat/completions"
    
    # Enforce limit if provided. Stratified across the samples' own `category` metadata,
    # not a contiguous head: benchmark rows are stored grouped by category, so `[:limit]`
    # returned one category (audit RC-1 - 56 of 93 scored suites collapsed to a single
    # category at --eval-limit 20). Deterministic: seeded on the suite name.
    if limit and limit > 0 and len(samples) > limit:
        samples = stratified_sample(
            samples, limit,
            key_fn=lambda s: (s[2] or {}).get("category") if len(s) > 2 and isinstance(s[2], dict) else None,
            seed=eval_name)

    # Global fallback for suites with no floor entry. Raised from 8192/16384: the sweep
    # truncated 10 suites that ran at the old non-thinking default, and a thinking run
    # spends most of its budget in the reasoning channel before the answer starts (the
    # whole completion shares one max_tokens).
    effective_max_tokens = (
        max_output_tokens
        if max_output_tokens is not None
        else (32768 if thinking else 16384)
    )

    tool_rounds_used: Dict[int, int] = {}
    semaphore = asyncio.Semaphore(concurrency)
    if aiohttp is None:
        raise RuntimeError(
            "The 'aiohttp' package is required for native asynchronous evaluation runners.\n"
            "Install dependencies via: pip install aiohttp (or pip install -e .)"
        )
    connector = aiohttp.TCPConnector(limit=concurrency + 20)
    timeout_s = request_timeout_s(effective_max_tokens)
    if timeout_s > REQUEST_TIMEOUT_FLOOR_S:
        logger.info("[%s] per-request timeout %ds (max_tokens=%s at >=%.0f tok/s)",
                    eval_name, timeout_s, effective_max_tokens, MIN_DECODE_TOK_S)
    timeout = aiohttp.ClientTimeout(total=timeout_s)

    correct_count = 0
    total_count = len(samples)
    failed_requests = 0
    empty_responses = 0
    truncated_responses = 0
    judge_failures = 0
    health_counts: Dict[str, int] = {}
    category_stats = {}

    with tqdm(total=total_count, desc=f"Eval [{eval_name.upper()}]") as pbar:
        async def _fetch_sample(idx, messages, gold_answer, sample_extra_payload):
            # Merge suite-level extra_payload with sample-level category/extra payload
            request_payload = {}
            if extra_payload:
                request_payload.update(extra_payload)
            sample_cat = None
            if isinstance(sample_extra_payload, dict):
                request_payload.update({k: v for k, v in sample_extra_payload.items() if k != "category"})
                sample_cat = sample_extra_payload.get("category")
            elif isinstance(sample_extra_payload, str):
                sample_cat = sample_extra_payload

            # Fit the request to the context window before sending, rather than paying a
            # 400 + retry (or three silent failures when the prompt alone is over the limit).
            sample_max_tokens, oversize = clamp_to_context(messages, effective_max_tokens)
            if oversize:
                if pbar is not None:
                    pbar.update(1)
                return (idx, Reply(None, None, None, None), gold_answer, sample_cat,
                        messages, sample_extra_payload, oversize)

            # Agentic loop. A web-research suite cannot be answered in one shot: the model
            # asks for a search, reads the results, and asks again. Without it `gaia` and
            # `deepsearch_qa` reply "I do not have access to the internet" and score a
            # structural 0. Only runs when the suite supplies an executor; single-turn
            # suites take the same path they always did (one request, no extra state).
            convo = list(messages)
            rounds = 0
            while True:
                reply = await _send_single_request(
                    session=session,
                    api_url=api_url,
                    model_name=model_name,
                    messages=convo,
                    extra_payload=request_payload,
                    semaphore=semaphore,
                    pbar=pbar if rounds == 0 else None,
                    max_output_tokens=sample_max_tokens,
                    temperature=temperature,
                    thinking=thinking,
                )
                if (tool_executor is None or not reply.tool_calls
                        or rounds >= max_tool_rounds):
                    break
                rounds += 1
                convo.append({"role": "assistant", "content": reply.text or "",
                              "tool_calls": reply.tool_calls})
                for call in reply.tool_calls:
                    fn = (call.get("function") or {})
                    raw_args = fn.get("arguments")
                    try:
                        parsed = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                    except Exception:
                        parsed = {}
                    try:
                        out = await tool_executor(fn.get("name", ""), parsed)
                    except Exception as e:                     # a broken tool is reported,
                        out = json.dumps({"error": f"tool raised {type(e).__name__}: {e}"})
                    convo.append({"role": "tool", "tool_call_id": call.get("id", ""),
                                  "name": fn.get("name", ""), "content": str(out)})
                # the follow-up must fit too, now that the transcript has grown
                sample_max_tokens, oversize = clamp_to_context(convo, effective_max_tokens)
                if oversize:
                    break
            if rounds:
                tool_rounds_used[idx] = rounds
            return (idx, reply, gold_answer, sample_cat, messages, sample_extra_payload, None)

        sample_traces = []
        completed = 0
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            coroutines = [
                _fetch_sample(idx, sample[0], sample[1], sample[2] if len(sample) > 2 else {})
                for idx, sample in enumerate(samples)
            ]
            for future in asyncio.as_completed(coroutines):
                (idx, reply, gold_answer, category, messages,
                 sample_extra_payload, skip_reason) = await future
                resp_text, tool_calls = reply.text, reply.tool_calls
                health = classify_reply(reply)
                health_counts[health] = health_counts.get(health, 0) + 1
                completed += 1
                pbar.update(1)

                if resp_text is None:
                    failed_requests += 1
                    is_correct = False
                    # Record WHY. A bare `request_failed` with `error: None` is what mrcr's
                    # six over-context rows looked like on the 2026-08-15 sweep.
                    if skip_reason:
                        status = "OVER_CONTEXT"
                    elif health == "request_timeout":
                        status = "TIMEOUT"
                        skip_reason = reply.error
                    else:
                        status = "FAILED"
                        skip_reason = reply.error
                elif health == "truncated":
                    # The generation hit the output-token budget mid-answer. Scored on
                    # what arrived (it may still be right), but counted so the run says so
                    # instead of reporting a quietly depressed accuracy.
                    truncated_responses += 1
                    cleaned_pred = strip_thinking_tags(resp_text)
                    if async_eval_fn is not None:
                        is_correct, status = False, "PENDING_JUDGE"
                    else:
                        is_correct = eval_fn(cleaned_pred, gold_answer) if eval_fn else False
                        if is_correct:
                            correct_count += 1
                        status = "TRUNCATED"
                elif not str(resp_text).strip() and not tool_calls:
                    # The server answered 200 with empty `content`. That is not a wrong
                    # answer, it is a request that produced nothing - most often the
                    # generation hitting the output-token budget mid-reasoning. Scored as
                    # incorrect either way, but counted so the run reports
                    # `completed_with_errors` instead of a quietly depressed accuracy.
                    empty_responses += 1
                    failed_requests += 1
                    is_correct = False
                    status = "EMPTY_RESPONSE"
                else:
                    cleaned_pred = strip_thinking_tags(resp_text)
                    if async_eval_fn is not None:
                        is_correct = False
                        status = "PENDING_JUDGE"
                    else:
                        is_correct = eval_fn(cleaned_pred, gold_answer) if eval_fn else False
                        if is_correct:
                            correct_count += 1
                        status = "OK"

                if category and async_eval_fn is None:
                    if category not in category_stats:
                        category_stats[category] = {"correct": 0, "total": 0}
                    category_stats[category]["total"] += 1
                    if is_correct:
                        category_stats[category]["correct"] += 1

                sample_traces.append({
                    "sample_idx": idx,
                    "category": category,
                    "messages": _sanitize_for_trace(messages),
                    "extra_payload": _sanitize_for_trace(sample_extra_payload) if sample_extra_payload else None,
                    "gold_answer": gold_answer,
                    "response_text": resp_text,
                    "tool_calls": tool_calls,
                    "finish_reason": reply.finish_reason,
                    "health": health,
                    "reasoning_chars": len(reply.reasoning or ""),
                    "completion_tokens": reply.completion_tokens,
                    "stop_reason": reply.stop_reason,
                    "error": skip_reason,
                    "is_correct": is_correct,
                    "status": status,
                })

                if async_eval_fn is None:
                    pbar.set_postfix(
                        correct=f"{correct_count}/{completed} ({correct_count / completed * 100.0:.1f}%)"
                    )

    # Phase 2: Post-generation batch/async judging if provided
    if async_eval_fn is not None:
        await async_eval_fn(sample_traces)
        correct_count = 0
        category_stats = {}
        # A grader that never returned a verdict is a harness failure, not a "no". Without
        # this the run still reported `success` and the unjudged samples simply counted as
        # incorrect, so a judge outage looked like a low score.
        # Only an explicit grader marker counts. `status == "FAILED"` must NOT be treated as
        # a judge failure: execution-based suites use it for an ordinary wrong answer (lcb
        # sets it whenever a solution does not pass its tests), so inferring from it counted
        # every incorrect sample as a harness error and downgraded clean runs to
        # `completed_with_errors`.
        JUDGE_FAILURE_GRADES = ("judge_error", "unparsed", "JUDGE_FAILED")
        judge_failures = sum(
            1 for t in sample_traces
            if str(t.get("judge_grade") or "") in JUDGE_FAILURE_GRADES)
        failed_requests += judge_failures
        for trace in sample_traces:
            is_corr = bool(trace.get("is_correct", False))
            if is_corr:
                correct_count += 1
            cat = trace.get("category")
            if cat:
                if cat not in category_stats:
                    category_stats[cat] = {"correct": 0, "total": 0}
                category_stats[cat]["total"] += 1
                if is_corr:
                    category_stats[cat]["correct"] += 1

    sample_traces.sort(key=lambda x: x["sample_idx"])

    accuracy = (correct_count / total_count * 100.0) if total_count > 0 else 0.0

    category_accuracy = {}
    for cat, stats in category_stats.items():
        cat_acc = (stats["correct"] / stats["total"] * 100.0) if stats["total"] > 0 else 0.0
        category_accuracy[cat] = {
            "correct": stats["correct"],
            "total": stats["total"],
            "accuracy": round(cat_acc, 2),
        }

    # CC6: never emit a clean "success" 0% when nothing was actually scored. Zero samples
    # (loader produced nothing / dataset failed to materialize) is an error, not a 0% pass.
    if total_count == 0:
        status_str = "error"
    elif failed_requests == total_count:
        status_str = "failed"
    elif failed_requests == 0 and truncated_responses * 10 <= total_count:
        status_str = "success"
    else:
        # More than 10% of answers cut off at the token budget is a measurement problem,
        # not a model result: those samples were never given the chance to be right.
        status_str = "completed_with_errors"
    # A suite that fell back to substring matching because the judge was unavailable must
    # not report its number as the canonical metric (audit RC-5).
    fallback = sum(1 for t in sample_traces if t.get("scoring_mode") == "judge_fallback")
    if fallback:
        scoring_mode = "judge_fallback"
        logger.warning(
            "[%s] %d/%d samples were graded by the no-judge fallback (substring match), "
            "not the canonical judge. This is a lower bound, not the benchmark's metric.",
            eval_name, fallback, total_count)
    elif async_eval_fn is not None:
        scoring_mode = "judge"
    else:
        scoring_mode = "deterministic"

    over_context = sum(1 for t in sample_traces if t.get("status") == "OVER_CONTEXT")
    discarded = sum(1 for t in sample_traces if t.get("health") == "output_discarded")
    if discarded:
        logger.warning(
            "[%s] %d/%d responses were GENERATED BUT DISCARDED by the server: the model "
            "emitted a tool call and the tool-call parser removed it, but the request "
            "declared no `tools` so there was nowhere to put it. These are not empty "
            "answers. Declare the suite's tool schemas in the sample meta "
            "(`{\"tools\": [...]}`) to capture them.", eval_name, discarded, total_count)
    timed_out = sum(1 for t in sample_traces if t.get("status") == "TIMEOUT")
    if timed_out:
        logger.warning(
            "[%s] %d/%d requests timed out at %ds. Timeouts drop the LONGEST answers, so "
            "the accuracy is biased downward. Raise GBENCH_REQUEST_TIMEOUT_S or lower "
            "--batch-sizes before quoting this.",
            eval_name, timed_out, total_count, request_timeout_s(effective_max_tokens))
    if over_context:
        logger.warning(
            "[%s] %d/%d prompts exceed the model's context window and could not be sent. "
            "These are a dataset/serving mismatch, not wrong answers - they are excluded "
            "from nothing, so the accuracy denominator still counts them.",
            eval_name, over_context, total_count)

    if truncated_responses:
        logger.warning(
            "[%s] %d/%d responses hit the output-token budget (max_tokens=%d) and were "
            "scored on a partial answer. Raise --max-output-tokens before quoting this.",
            eval_name, truncated_responses, total_count, effective_max_tokens)

    return {
        "benchmark_type": "eval",
        "eval_name": eval_name,
        "model_name": model_name,
        "thinking": thinking,
        "total_questions": total_count,
        "correct_answers": correct_count,
        "failed_requests": failed_requests,
        "empty_responses": empty_responses,
        "truncated_responses": truncated_responses,
        "over_context_prompts": over_context,
        "timed_out_requests": timed_out,
        "discarded_outputs": discarded,
        "request_timeout_s": request_timeout_s(effective_max_tokens),
        "tool_rounds": {"samples_using_tools": len(tool_rounds_used),
                        "total_rounds": sum(tool_rounds_used.values())} if tool_rounds_used else None,
        "scoring_mode": scoring_mode,
        "judge_failures": judge_failures,
        "response_health": health_counts,
        "accuracy": round(accuracy, 2),
        "category_accuracy": category_accuracy,
        "sample_traces": sample_traces,
        "status": status_str,
    }


def run_eval_suite(
    eval_name: str,
    model_name: str,
    base_url: str,
    concurrency: int,
    samples: List[Tuple[List[Dict[str, Any]], Any, Dict[str, Any]]],
    eval_fn: Optional[Callable[[str, Any], bool]] = None,
    async_eval_fn: Optional[Callable[[List[Dict[str, Any]]], Any]] = None,
    thinking: bool = False,
    extra_payload: Optional[Dict[str, Any]] = None,
    limit: Optional[int] = None,
    max_output_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    tool_executor: Optional[Callable[[str, Dict[str, Any]], Any]] = None,
    max_tool_rounds: int = 8,
) -> Dict[str, Any]:
    """Synchronous entry point to run an evaluation suite.

    `temperature` and `max_output_tokens` fall back to the run-level knobs when the suite
    does not pass them, so `--temperature` reaches every suite rather than only the 15
    that happened to forward it (audit RC-2).
    """
    if temperature is None:
        temperature = get_run_knob("temperature", 0.0)
    knob = get_run_knob("max_output_tokens")
    floor = SUITE_MIN_OUTPUT_TOKENS.get(eval_name)
    if knob:
        max_output_tokens = knob          # operator was explicit; never override it
    elif floor and (max_output_tokens is None or max_output_tokens < floor):
        if max_output_tokens is not None:
            logger.info("[%s] raising max_output_tokens %d -> %d (long-answer suite floor)",
                        eval_name, max_output_tokens, floor)
        max_output_tokens = floor
    start_time = time.time()
    result = asyncio.run(
        _run_suite_async(
            eval_name=eval_name,
            model_name=model_name,
            base_url=base_url,
            concurrency=concurrency,
            samples=samples,
            eval_fn=eval_fn,
            async_eval_fn=async_eval_fn,
            thinking=thinking,
            extra_payload=extra_payload,
            limit=limit,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            tool_executor=tool_executor,
            max_tool_rounds=max_tool_rounds,
        )
    )
    result["duration_s"] = round(time.time() - start_time, 2)
    return result


def gemini_required_skip(eval_name: str, model_name: str) -> Optional[Dict[str, Any]]:
    """Return a standardized 'skipped' result if GEMINI_API_KEY is absent, else None.

    LLM-judge suites use this to fail hard (skip) rather than silently downgrade to
    a heuristic scorer when the judge model is unavailable. The skip prints an
    on-screen pointer to the suite's setup doc (docs/evals/<eval_name>.md).
    """
    import logging
    import os
    if not os.environ.get("GEMINI_API_KEY"):
        docs_url = f"docs/evals/{eval_name}.md"
        reason = "requires GEMINI_API_KEY for canonical LLM-judge grading"
        msg = f"[SKIP] {eval_name} skipped: {reason}. See '{docs_url}' for setup instructions."
        logging.getLogger(__name__).warning(msg)
        print(f"\n{msg}")
        return {
            "benchmark_type": "eval",
            "eval_name": eval_name,
            "model_name": model_name,
            "status": "skipped",
            "total_questions": 0,
            "correct_answers": 0,
            "accuracy": 0.0,
            "skip_reason": f"{reason} (See {docs_url})",
        }
    return None
