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

"""The web-research suites must be able to actually search.

`gaia` and `deepsearch_qa` scored 0/20 on the 2026-08-15 sweep because they were run with
no tool at all - GAIA literally replied "I do not have access to the internet". These tests
cover the `web_search` tool, the agentic loop that drives it, and the refusal to report a
structural zero when no search backend is configured. Nothing here touches the network.
"""

import asyncio
import json
import os
from unittest import mock

import pytest

from gbench.runners.eval_suites import base, search_tool


# --------------------------------------------------------------------------- #
# the tool itself
# --------------------------------------------------------------------------- #
def test_tool_schema_is_a_valid_openai_function():
    fn = search_tool.WEB_SEARCH_TOOL
    assert fn["type"] == "function"
    assert fn["function"]["name"] == "web_search"
    params = fn["function"]["parameters"]
    assert params["type"] == "object"
    assert params["properties"]["query"]["type"] == "string"
    assert params["required"] == ["query"]


def _grounding_payload():
    return {"candidates": [{
        "content": {"parts": [{"text": "Three species appear on camera."}]},
        "groundingMetadata": {
            "groundingChunks": [{"web": {"title": "birds.org", "uri": "https://birds.org/a"}},
                                {"web": {"domain": "audubon.org", "uri": "https://a.org/b"}}],
            "groundingSupports": [
                {"segment": {"text": "Three species appear"}, "groundingChunkIndices": [0]}],
        }}]}


def _with_urlopen(payload):
    class _Resp:
        def read(self): return json.dumps(payload).encode()
        def __enter__(self): return self
        def __exit__(self, *a): return False
    return mock.patch("urllib.request.urlopen", return_value=_Resp())


def test_search_maps_grounding_chunks_to_results():
    with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "k"}), _with_urlopen(_grounding_payload()):
        got = search_tool.gemini_search("how many bird species")
    assert [r["title"] for r in got] == ["birds.org", "audubon.org"]
    assert got[0]["snippet"] == "Three species appear"      # from groundingSupports
    assert got[1]["snippet"].startswith("Three species")     # falls back to the answer text
    assert got[0]["url"] == "https://birds.org/a"


def test_search_with_no_chunks_falls_back_to_the_grounded_answer():
    payload = {"candidates": [{"content": {"parts": [{"text": "The answer is 42."}]}}]}
    with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "k"}), _with_urlopen(payload):
        got = search_tool.gemini_search("q")
    assert len(got) == 1 and "42" in got[0]["snippet"]


def test_backend_failure_is_reported_as_a_failure_not_as_no_results():
    """"No results" tells the model the fact does not exist; that is a different claim."""
    with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "k"}), \
         mock.patch("urllib.request.urlopen", side_effect=OSError("boom")):
        got = search_tool.gemini_search("q")
    assert got[0]["title"] == "error" and "failed" in got[0]["snippet"]


def test_executor_rejects_unknown_tools_and_empty_queries():
    assert "unknown tool" in asyncio.run(search_tool.execute_tool("rm_rf", {"query": "x"}))
    assert "non-empty" in asyncio.run(search_tool.execute_tool("web_search", {"query": "  "}))


def test_executor_returns_json_the_model_can_read():
    with mock.patch.object(search_tool, "gemini_search",
                           return_value=[{"title": "t", "url": "u", "snippet": "s"}]):
        out = json.loads(asyncio.run(search_tool.execute_tool("web_search", {"query": "q"})))
    assert out["query"] == "q" and out["results"][0]["title"] == "t"


# --------------------------------------------------------------------------- #
# the agentic loop
# --------------------------------------------------------------------------- #
def _run_loop(replies, executor, **kw):
    seen = {"n": 0, "convos": []}

    async def _fake_send(**k):
        seen["convos"].append(list(k["messages"]))
        r = replies[min(seen["n"], len(replies) - 1)]
        seen["n"] += 1
        return r

    with mock.patch.object(base, "_send_single_request", _fake_send):
        res = base.run_eval_suite(
            eval_name="gaia", model_name="m", base_url="http://x", concurrency=1,
            samples=[([{"role": "user", "content": "q"}], "3", {})],
            eval_fn=lambda resp, g: g in (resp or ""), tool_executor=executor, **kw)
    return res, seen


_ASK = base.Reply(text="", tool_calls=[{"id": "c1", "function": {
    "name": "web_search", "arguments": '{"query":"birds"}'}}], finish_reason="tool_calls")
_ANSWER = base.Reply(text="FINAL ANSWER: 3", tool_calls=None, finish_reason="stop")


def test_tool_call_is_executed_and_fed_back():
    ran = []

    async def ex(name, args):
        ran.append((name, args))
        return '{"results":[{"snippet":"three"}]}'

    res, seen = _run_loop([_ASK, _ANSWER], ex)
    assert ran == [("web_search", {"query": "birds"})]
    assert res["correct_answers"] == 1
    assert res["tool_rounds"] == {"samples_using_tools": 1, "total_rounds": 1}
    # the follow-up request carried the assistant turn AND the tool result
    roles = [m["role"] for m in seen["convos"][1]]
    assert roles == ["user", "assistant", "tool"]


def test_loop_is_bounded():
    """A model that only ever asks to search must not spin forever."""
    async def ex(name, args):
        return "{}"
    res, seen = _run_loop([_ASK], ex, max_tool_rounds=3)
    assert seen["n"] == 4                       # initial + 3 tool rounds
    assert res["tool_rounds"]["total_rounds"] == 3


def test_a_raising_tool_is_reported_to_the_model_not_crashed_on():
    async def ex(name, args):
        raise RuntimeError("network down")
    res, seen = _run_loop([_ASK, _ANSWER], ex)
    tool_msg = [m for m in seen["convos"][1] if m["role"] == "tool"][0]
    assert "RuntimeError" in tool_msg["content"]
    assert res["status"] != "error"


def test_single_turn_suites_are_untouched():
    """No executor -> exactly one request, no tool bookkeeping."""
    res, seen = _run_loop([_ANSWER], None)
    assert seen["n"] == 1
    assert res["tool_rounds"] is None


# --------------------------------------------------------------------------- #
# no search backend -> skip, never a structural zero
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("suite", ["gaia", "deepsearch_qa"])
def test_web_research_suites_skip_without_a_backend(suite):
    import importlib
    mod = importlib.import_module(f"gbench.runners.eval_suites.{suite}")
    runner = getattr(mod, f"run_{suite}")
    key = os.environ.pop("GEMINI_API_KEY", None)
    try:
        res = runner("m", "http://x", concurrency=1, limit=2)
    finally:
        if key is not None:
            os.environ["GEMINI_API_KEY"] = key
    assert res["status"] == "skipped"
    assert res["accuracy"] == 0.0
    assert "GEMINI_API_KEY" in res["skip_reason"]


@pytest.mark.parametrize("suite", ["gaia", "deepsearch_qa"])
def test_web_research_suites_offer_the_tool_and_disclaim_comparability(suite):
    import inspect, importlib
    mod = importlib.import_module(f"gbench.runners.eval_suites.{suite}")
    src = inspect.getsource(getattr(mod, f"run_{suite}"))
    assert "WEB_SEARCH_TOOL" in src and "tool_executor=execute_tool" in src
    # search-only is not the browsing agent the public leaderboard uses
    assert 'result["leaderboard_comparable"] = False' in src
