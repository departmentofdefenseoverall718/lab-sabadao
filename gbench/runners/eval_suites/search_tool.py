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

"""A `web_search` tool the agentic suites can actually call, backed by Gemini grounding.

`gaia` and `deepsearch_qa` are web-research benchmarks: their questions are answerable only
by looking things up. Run without any tool they score a structural 0 - on the 2026-08-15
sweep GAIA replied "I do not have access to the internet. Therefore, I cannot watch the
video" and both suites returned 0/20. That is not a measurement of the model.

This offers the same Google-Search grounding `bfcl_v4_agentic` already uses (same
`GEMINI_API_KEY`, same endpoint), exposed as an OpenAI-style tool so the evaluated model
drives the search itself.

Comparability: canonical GAIA leaderboard entries use a full browsing agent (page fetch,
file download, code execution). This is search-only, so a gbench GAIA number is **not**
leaderboard-comparable; the result records the backend it used.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

#: How many grounding chunks to hand back per query.
MAX_RESULTS = int(os.environ.get("GBENCH_SEARCH_MAX_RESULTS", "8"))

#: The tool as the evaluated model sees it.
WEB_SEARCH_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web and return ranked results with titles, URLs and snippets. "
            "Use it whenever the answer depends on information you do not already know, "
            "and call it repeatedly to refine or cross-check. Prefer specific queries."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."},
            },
            "required": ["query"],
        },
    },
}


def search_available() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY"))


def unavailable_reason() -> str:
    return ("web search requires GEMINI_API_KEY (Google-Search grounding); without it "
            "every question that needs a lookup is unanswerable and the suite would "
            "report a structural 0%")


def gemini_search(query: str, max_results: int = MAX_RESULTS) -> List[Dict[str, str]]:
    """Google-Search-grounded results as [{title, url, snippet}].

    Gemini returns grounding *chunks* (title = source domain, uri = a redirecting link)
    plus grounding *supports* (the answer spans each chunk backs); the supports are the
    closest analogue of a SERP snippet.
    """
    import urllib.request

    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        return [{"title": "error", "url": "", "snippet": "GEMINI_API_KEY is not set"}]
    from .base import DEFAULT_JUDGE_MODEL
    model = os.environ.get("GBENCH_SEARCH_MODEL", DEFAULT_JUDGE_MODEL)
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent?key={key}")
    payload = {"contents": [{"parts": [{"text": str(query)}]}],
               "tools": [{"google_search": {}}]}
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            data = json.loads(r.read().decode("utf-8"))
    except Exception as e:
        # A failed lookup is reported to the model as a failed lookup, not as "no results" -
        # otherwise the model concludes the fact does not exist.
        return [{"title": "error", "url": "", "snippet": f"search backend failed: {e}"}]

    cand = (data.get("candidates") or [{}])[0]
    gm = cand.get("groundingMetadata") or {}
    chunks = gm.get("groundingChunks") or []

    bodies: Dict[int, List[str]] = {}
    for sup in gm.get("groundingSupports") or []:
        text = ((sup.get("segment") or {}).get("text") or "").strip()
        for idx in sup.get("groundingChunkIndices") or []:
            if text:
                bodies.setdefault(int(idx), []).append(text)

    answer = "".join(p.get("text", "") for p in (cand.get("content") or {}).get("parts", []))
    results: List[Dict[str, str]] = []
    for i, ch in enumerate(chunks[:max_results]):
        web = ch.get("web") or {}
        results.append({
            "title": web.get("title") or web.get("domain") or "result",
            "url": web.get("uri") or "",
            "snippet": " ".join(bodies.get(i, [])) or answer[:500],
        })
    if not results and answer:
        results = [{"title": "grounded-answer", "url": "", "snippet": answer[:1500]}]
    if not results:
        results = [{"title": "no results", "url": "", "snippet": "The search returned nothing."}]
    return results


async def execute_tool(name: str, args: Dict[str, Any]) -> str:
    """Tool executor for `run_eval_suite(tool_executor=...)`. Returns the tool message."""
    import asyncio
    if name != "web_search":
        return json.dumps({"error": f"unknown tool '{name}'"})
    query = str((args or {}).get("query") or "").strip()
    if not query:
        return json.dumps({"error": "web_search requires a non-empty 'query'"})
    results = await asyncio.to_thread(gemini_search, query)
    return json.dumps({"query": query, "results": results}, ensure_ascii=False)


def search_backend_name() -> Optional[str]:
    return "gemini-google-search-grounding" if search_available() else None
