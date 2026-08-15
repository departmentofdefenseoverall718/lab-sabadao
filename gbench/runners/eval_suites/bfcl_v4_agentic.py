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

"""BFCL v4 **Agentic** track, driven through Berkeley's own `bfcl-eval` harness.

The v4 agentic track is not an answer-matching dataset: it is a stateful tool-execution
benchmark (agents call memory / web-search tools across turns), so it cannot be scored by
loading rows and comparing strings. gbench therefore wraps the canonical harness the same
way the tau2/tau3 suites wrap tau2-bench:

* categories (verified from `bfcl_eval`): ``memory_kv``, ``memory_vector``,
  ``memory_rec_sum``, ``web_search_base``, ``web_search_no_snippet``;
* the v4 data ships inside the `bfcl-eval` package (BFCL_v4_memory.json /
  BFCL_v4_web_search.json) - nothing is downloaded from HF;
* generation runs against the gbench endpoint (`REMOTE_OPENAI_BASE_URL` + the harness'
  ``--skip-server-setup``), so no extra GPU server is started.

Search backend
--------------
Canonical BFCL queries DuckDuckGo **via SerpAPI** (`SERPAPI_API_KEY`). gbench defaults to
**Gemini Google-Search grounding** using the same `GEMINI_API_KEY` the other judged suites
use, by replacing the harness' `search_engine_query`. This removes the paid dependency but
changes what evidence is retrievable, so `web_search_*` numbers are internally consistent
and reproducible yet **not** comparable to the public leaderboard. The backend actually
used is recorded in the result payload. `memory_*` does not search and stays canonical.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .swebench_common import skipped_result

logger = logging.getLogger(__name__)

DOCS_URL = "docs/evals/bfcl_v4_agentic.md"
PILLAR = "Tool Use & Agentic Workflows"

#: Categories that issue live web searches (the rest are memory-only, no search backend).
WEB_SEARCH_CATEGORIES = ("web_search_base", "web_search_no_snippet")

DEFAULT_PROJECT_ROOT = os.path.expanduser("~/.cache/gbench/bfcl")


# --------------------------------------------------------------------------- #
# prerequisites
# --------------------------------------------------------------------------- #
def _agentic_categories() -> List[str]:
    from bfcl_eval.constants.category_mapping import TEST_COLLECTION_MAPPING
    return list(TEST_COLLECTION_MAPPING["agentic"])


def search_backend() -> Tuple[Optional[str], str]:
    """Pick the web-search backend: (name, reason-if-unavailable)."""
    forced = os.environ.get("BFCL_SEARCH_BACKEND", "").strip().lower()
    if forced == "serpapi":
        return ("serpapi", "") if os.environ.get("SERPAPI_API_KEY") else \
               (None, "BFCL_SEARCH_BACKEND=serpapi but SERPAPI_API_KEY is unset")
    if forced == "gemini":
        return ("gemini", "") if os.environ.get("GEMINI_API_KEY") else \
               (None, "BFCL_SEARCH_BACKEND=gemini but GEMINI_API_KEY is unset")
    if forced == "ddg":
        try:
            import ddgs  # noqa: F401
            return "ddg", ""
        except ImportError:
            try:
                import duckduckgo_search  # noqa: F401
                return "ddg", ""
            except ImportError:
                return None, "BFCL_SEARCH_BACKEND=ddg but no duckduckgo client is installed"
    # auto: prefer canonical SerpAPI when a key exists, else Gemini grounding
    if os.environ.get("SERPAPI_API_KEY"):
        return "serpapi", ""
    if os.environ.get("GEMINI_API_KEY"):
        return "gemini", ""
    return None, ("no web-search backend: set GEMINI_API_KEY (Gemini grounding) or "
                  "SERPAPI_API_KEY (canonical DuckDuckGo-via-SerpAPI)")


def check_bfcl_prerequisites(base_url: str) -> Tuple[bool, str, List[str]]:
    """Validate the harness, endpoint, project root and search backend.

    Returns (ok, reason, categories_to_run). A missing search backend does NOT fail the
    whole suite: the three memory categories still run and the result says so.
    """
    try:
        import bfcl_eval  # noqa: F401
    except ImportError:
        return False, ("the canonical BFCL harness is not installed - "
                       "`pip install bfcl-eval` (not the unrelated `bfcl` package)"), []

    try:
        categories = _agentic_categories()
    except Exception as e:
        return False, f"bfcl-eval is installed but its category mapping is unreadable ({e})", []

    root = Path(os.environ.get("BFCL_PROJECT_ROOT") or DEFAULT_PROJECT_ROOT)
    try:
        root.mkdir(parents=True, exist_ok=True)
        probe = root / ".gbench_write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except Exception as e:
        return False, f"BFCL_PROJECT_ROOT {root} is not writable ({e})", []

    try:
        import urllib.request
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/models", timeout=10) as r:
            if r.status != 200:
                return False, f"model endpoint {base_url} returned HTTP {r.status}", []
    except Exception as e:
        return False, f"model endpoint {base_url} is not reachable ({e})", []

    backend, why = search_backend()
    if backend is None:
        categories = [c for c in categories if c not in WEB_SEARCH_CATEGORIES]
        logger.warning("bfcl_v4_agentic: %s -> running memory categories only (%s)",
                       why, ", ".join(categories))
        if not categories:
            return False, why, []
    return True, "", categories


# --------------------------------------------------------------------------- #
# model registration + search backend patch
# --------------------------------------------------------------------------- #
def _patched_openai_handler():
    """`OpenAICompletionsHandler` with its FC decoders fixed for prose (no-tool-call) turns.

    Upstream bug (bfcl-eval, `api_inference/openai_completion.py`): in FC mode
    `_parse_query_response_FC` normally returns ``[{name: arguments_json}]``, but when the
    model answers WITHOUT calling a tool it falls back to
    ``model_responses = api_response.choices[0].message.content`` - a plain string.
    `decode_ast`/`decode_execute` then iterate that string CHARACTER BY CHARACTER and call
    ``.items()`` on each character, raising ``'str' object has no attribute 'items'``,
    which the harness logs as "Failed to decode the model response. Proceed to next turn."

    Answering in prose is legitimate and frequent on the memory tracks (the task often
    asks a question the agent should answer from memory). Measured on the memory_kv run:
    1225 of ~1425 decode errors were this one path. It is model-independent - any model
    driven through this handler hits it - so the resulting score was a harness artifact.

    Here a string response is routed to the prompting decoder, exactly as the non-FC
    branch does; if it contains no call the turn decodes to "no calls", which the harness
    already handles as an ordinary empty-response turn.
    """
    from bfcl_eval.model_handler.api_inference.openai_completion import OpenAICompletionsHandler
    from bfcl_eval.model_handler.utils import (
        default_decode_ast_prompting, default_decode_execute_prompting)

    class _GbenchOpenAICompletionsHandler(OpenAICompletionsHandler):
        def decode_ast(self, result, language, has_tool_call_tag):
            if self.is_fc_model and not isinstance(result, list):
                try:
                    return default_decode_ast_prompting(result, language, has_tool_call_tag)
                except Exception:
                    return []
            return super().decode_ast(result, language, has_tool_call_tag)

        def decode_execute(self, result, has_tool_call_tag):
            if self.is_fc_model and not isinstance(result, list):
                try:
                    return default_decode_execute_prompting(result)
                except Exception:
                    return []          # prose with no call: not a decode failure
            return super().decode_execute(result, has_tool_call_tag)

    return _GbenchOpenAICompletionsHandler


def _register_model(model_name: str) -> str:
    """Register the served model with BFCL's handler registry (it ships no gemma-4 entry).

    Returns the key to pass as ``--model``. Reuses the local-inference Gemma handler, which
    already knows the Gemma chat format; the served model id is used verbatim so the
    harness' OpenAI client requests the right model.
    """
    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig
    if model_name in MODEL_CONFIG_MAPPING:
        return model_name
    # Use the OpenAI *chat completions* handler in function-calling mode, not the local
    # GemmaHandler. GemmaHandler hand-builds a raw gemma-3 prompt string and then parses
    # prompt-style `[func(a=1)]` text back out; against a vLLM-served gemma-4 with native
    # tool calling that decode fails on every row ("Failed to decode the model response").
    # OpenAICompletionsHandler talks OpenAI tool-calling and honours OPENAI_BASE_URL, which
    # we point at the gbench endpoint.
    MODEL_CONFIG_MAPPING[model_name] = ModelConfig(
        model_name=model_name,
        display_name=f"{model_name} (gbench)",
        url="https://ai.google.dev/gemma",
        org="Google",
        license="gemma-terms-of-use",
        model_handler=_patched_openai_handler(),
        input_price=None,
        output_price=None,
        is_fc_model=True,       # native tool calling via /v1/chat/completions
        underscore_to_dot=False,
    )
    logger.info("bfcl_v4_agentic: registered '%s' (OpenAICompletionsHandler, FC mode)", model_name)
    return model_name


def _gemini_search(keywords: str, region: str = "wt-wt", max_results: int = 10) -> Any:
    """Google-Search-grounded results shaped like BFCL expects: [{title, href, body}].

    Gemini returns grounding *chunks* (title = source domain, uri = a redirecting
    vertexaisearch link) plus grounding *supports* (the answer spans each chunk backs). We
    use the supports as each result's snippet, which is the closest analogue of a SERP body.
    """
    import urllib.error
    import urllib.request

    key = os.environ.get("GEMINI_API_KEY", "")
    from .base import DEFAULT_JUDGE_MODEL
    model = os.environ.get("BFCL_SEARCH_GEMINI_MODEL", DEFAULT_JUDGE_MODEL)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    payload = {
        "contents": [{"parts": [{"text": str(keywords)}]}],
        "tools": [{"google_search": {}}],
    }
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            data = json.loads(r.read().decode("utf-8"))
    except Exception as e:
        return {"error": f"gemini search failed: {e}"}

    cand = (data.get("candidates") or [{}])[0]
    gm = cand.get("groundingMetadata") or {}
    chunks = gm.get("groundingChunks") or []

    # chunk index -> supporting answer spans (used as the snippet/body)
    bodies: Dict[int, List[str]] = {}
    for sup in gm.get("groundingSupports") or []:
        text = ((sup.get("segment") or {}).get("text") or "").strip()
        for idx in sup.get("groundingChunkIndices") or []:
            if text:
                bodies.setdefault(int(idx), []).append(text)

    answer = "".join(p.get("text", "") for p in (cand.get("content") or {}).get("parts", []))
    results = []
    for i, ch in enumerate(chunks[:max_results]):
        web = ch.get("web") or {}
        results.append({
            "title": web.get("title") or web.get("domain") or "result",
            "href": web.get("uri") or "",
            "body": " ".join(bodies.get(i, [])) or answer[:500],
        })
    if not results and answer:
        results = [{"title": "gemini-grounded-answer", "href": "", "body": answer[:1000]}]
    return results


def _ddg_search(keywords: str, region: str = "wt-wt", max_results: int = 10) -> Any:
    """Direct DuckDuckGo (same engine as canonical, native {title, href, body} shape)."""
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS  # type: ignore
        with DDGS() as d:
            return list(d.text(keywords, region=region, max_results=max_results))
    except Exception as e:
        return {"error": f"duckduckgo search failed: {e}"}


def _patch_search_backend(backend: str) -> None:
    """Swap the harness' SerpAPI call for the selected backend (no vendor edits)."""
    if backend == "serpapi":
        return  # canonical path, nothing to patch
    from bfcl_eval.eval_checker.multi_turn_eval.func_source_code import web_search as ws
    impl = _gemini_search if backend == "gemini" else _ddg_search
    for cls_name in dir(ws):
        cls = getattr(ws, cls_name)
        if isinstance(cls, type) and hasattr(cls, "search_engine_query"):
            setattr(cls, "search_engine_query",
                    staticmethod(lambda keywords, region="wt-wt", _impl=impl, **kw: _impl(keywords, region)))
    if hasattr(ws, "search_engine_query"):
        ws.search_engine_query = lambda keywords, region="wt-wt", **kw: impl(keywords, region)
    logger.info("bfcl_v4_agentic: web-search backend = %s", backend)


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
#: Bootstrap executed INSIDE the harness subprocess. Registering the model and swapping the
#: search backend in the parent process would have no effect on the child, so the child
#: re-applies both before typer dispatches the command.
_CHILD_BOOTSTRAP = """
import sys
from gbench.runners.eval_suites.bfcl_v4_agentic import _register_model, _patch_search_backend
_register_model({model!r})
_backend = {backend!r}
if _backend and _backend != "serpapi":
    _patch_search_backend(_backend)
sys.argv = ["bfcl"] + {args!r}
from bfcl_eval.__main__ import cli
cli()
"""


def _run_cli(args: List[str], env: Dict[str, str], timeout: int,
             model: str = "", backend: Optional[str] = None) -> Tuple[int, str]:
    code = _CHILD_BOOTSTRAP.format(model=model, backend=backend, args=args)
    proc = subprocess.run([sys.executable, "-c", code],
                          capture_output=True, text=True, env=env, timeout=timeout)
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def _parse_scores(score_root: Path, model_name: str, categories: List[str]) -> Dict[str, Any]:
    """Read the harness' per-category score files into gbench's category_accuracy shape."""
    per_cat: Dict[str, Dict[str, Any]] = {}
    model_dir = score_root / model_name.replace("/", "_")
    if not model_dir.is_dir():
        alt = [p for p in score_root.glob(f"**/{model_name.split('/')[-1]}*") if p.is_dir()]
        model_dir = alt[0] if alt else model_dir
    # Scores are nested by track, e.g. <model>/agentic/memory/kv/BFCL_v4_memory_kv_score.json,
    # so this must recurse - a flat glob finds nothing.
    for path in sorted(model_dir.rglob("*_score.json")) if model_dir.is_dir() else []:
        cat = next((c for c in categories if c in path.name), path.stem.replace("_score", ""))
        try:
            first = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        except Exception:
            continue
        acc = first.get("accuracy")
        total = first.get("total_count") or first.get("total") or 0
        correct = first.get("correct_count") or (round(acc * total) if isinstance(acc, (int, float)) else 0)
        if acc is None and total:
            acc = correct / total
        per_cat[cat] = {"correct": int(correct or 0), "total": int(total or 0),
                        "accuracy": round(float(acc or 0.0) * 100.0, 2)}
    return per_cat


def _select_limited_ids(categories: List[str], limit: int,
                        seed: str = "bfcl_v4_agentic") -> Dict[str, List[str]]:
    """Pick `limit` scored questions spread across `categories`, plus their prerequisites.

    `bfcl generate` has no --limit: it runs the whole category. On the 2026-08-15 sweep
    that made `--eval-limit 20` generate all 665 agentic questions. The harness does take
    an id file (`--run-ids`), so gbench builds one.

    Two things the caller should know about the count:

    * Selection is round-robin across categories, and stratified by scenario *within* a
      category, so 20 over 5 categories is 4 each rather than 20 from `memory_kv` - whose
      first 12 ids are all the `customer` scenario out of five (audit RC-1).
    * Memory questions `depends_on` `*_prereq` entries that install the memory state. Those
      are pulled in transitively and are NOT scored questions, so the number of generated
      entries exceeds `limit`. Dropping them would score the model on memory it was never
      given.
    """
    from bfcl_eval.utils import load_dataset_entry

    from .sampling import stratified_sample

    entries = {c: load_dataset_entry(c) for c in categories}
    by_id = {c: {e["id"]: e for e in rows} for c, rows in entries.items()}
    pools = {c: [e["id"] for e in rows if "_prereq" not in e["id"]] for c, rows in entries.items()}

    def _scenario(cat: str, sid: str) -> str:
        """`memory_kv_7-healthcare-3` -> `healthcare`; `web_search_base_7` -> the category."""
        tail = sid[len(cat) + 1:] if sid.startswith(f"{cat}_") else sid
        parts = tail.split("-")
        return parts[1] if len(parts) >= 2 else cat

    # Round-robin the quota across categories, then stratify by scenario inside each.
    quota = {c: 0 for c in categories}
    taken, room = 0, {c: len(pools[c]) for c in categories}
    while taken < limit and any(quota[c] < room[c] for c in categories):
        for c in categories:
            if taken >= limit or quota[c] >= room[c]:
                continue
            quota[c] += 1
            taken += 1

    chosen = {c: stratified_sample(pools[c], quota[c],
                                   key_fn=lambda i, _c=c: _scenario(_c, i),
                                   seed=f"{seed}:{c}") if quota[c] else []
              for c in categories}

    out: Dict[str, List[str]] = {}
    for c, picked in chosen.items():
        if not picked:
            continue
        closure, stack = set(picked), list(picked)
        while stack:
            for dep in (by_id[c].get(stack.pop(), {}).get("depends_on") or []):
                if dep not in closure:
                    closure.add(dep)
                    stack.append(dep)
        out[c] = sorted(closure)
    return out


def run_bfcl_v4_agentic(
    model_name: str,
    base_url: str,
    concurrency: int = 4,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run the BFCL v4 agentic track via the canonical `bfcl-eval` harness."""
    ok, reason, categories = check_bfcl_prerequisites(base_url)
    if not ok:
        return skipped_result("bfcl_v4_agentic", model_name, reason, DOCS_URL)

    backend, _ = search_backend()
    project_root = Path(os.environ.get("BFCL_PROJECT_ROOT") or DEFAULT_PROJECT_ROOT)

    try:
        model_key = _register_model(model_name)
        if backend:
            _patch_search_backend(backend)
    except Exception as e:
        return skipped_result("bfcl_v4_agentic", model_name,
                              f"could not configure the BFCL harness ({e})", DOCS_URL)

    env = dict(os.environ)
    env.update({
        "BFCL_PROJECT_ROOT": str(project_root),
        "REMOTE_OPENAI_BASE_URL": base_url,
        "REMOTE_OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", "EMPTY"),
        # OpenAICompletionsHandler builds its client from these two:
        "OPENAI_BASE_URL": base_url,
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY") or "EMPTY",
        "BFCL_SEARCH_BACKEND": backend or "none",
    })
    env.setdefault("REMOTE_OPENAI_TOKENIZER_PATH", model_name)

    cats = ",".join(categories)
    threads = str(max(1, int(concurrency)))
    timeout = int(os.environ.get("BFCL_TIMEOUT_S", "10800"))

    # `bfcl generate` defaults to --allow-overwrite=False, which does NOT mean "skip the
    # run": it loads whatever is already in <BFCL_PROJECT_ROOT>/result/, subtracts those
    # ids from the work list, and generates nothing when the file is complete
    # (_llm_response_generation.py:133-160). The harness then re-scores those stale
    # generations and reports them as this run's result - a 665-question track "completed"
    # in 19 seconds against output from two days earlier. That cache is BFCL's own, keyed
    # on its project root, so gbench's --skip-existing has no bearing on it.
    #
    # An eval run should measure the model as it is now, so regenerate by default. Set
    # BFCL_REUSE_GENERATIONS=1 to deliberately re-score existing generations instead.
    reuse = os.environ.get("BFCL_REUSE_GENERATIONS") == "1"
    score_root = project_root / "score"
    generate_cmd = ["generate", "--model", model_key, "--test-category", cats,
                    "--num-threads", threads, "--skip-server-setup"]
    evaluate_cmd = ["evaluate", "--model", model_key, "--test-category", cats]

    # --eval-limit: restrict to an id file rather than running all 665 questions.
    limit = kwargs.get("limit")
    limited_ids: Dict[str, List[str]] = {}
    if limit and int(limit) > 0:
        try:
            limited_ids = _select_limited_ids(categories, int(limit))
        except Exception as e:
            logger.warning("bfcl_v4_agentic: could not build the id file for --eval-limit "
                           "%s (%s); running the full track.", limit, e)
        if limited_ids:
            (project_root / "test_case_ids_to_generate.json").write_text(
                json.dumps(limited_ids, indent=2), encoding="utf-8")
            # Separate result/score dirs: --run-ids updates result files in place, so
            # sharing the full-run directory would leave earlier questions in the file and
            # the harness would score those too (audit RC-4). These paths are resolved
            # relative to BFCL_PROJECT_ROOT by the harness CLI.
            rdir, sdir = f"result_limit{int(limit)}", f"score_limit{int(limit)}"
            generate_cmd += ["--run-ids", "--result-dir", rdir]
            # --partial-eval: the categories are only partly present by construction, and
            # without it the harness raises on the missing ids.
            evaluate_cmd += ["--result-dir", rdir, "--score-dir", sdir, "--partial-eval"]
            score_root = project_root / sdir
            n_prereq = sum(1 for ids in limited_ids.values() for i in ids if "_prereq" in i)
            logger.info("bfcl_v4_agentic: --eval-limit %s -> %d scored questions across "
                        "%d categories (+%d prerequisite entries) in %s",
                        limit, sum(len(v) for v in limited_ids.values()) - n_prereq,
                        len(limited_ids), n_prereq, rdir)
    if not reuse:
        generate_cmd.append("--allow-overwrite")
    else:
        logger.warning("bfcl_v4_agentic: BFCL_REUSE_GENERATIONS=1 - re-scoring existing "
                       "generations in %s; the result reflects whenever those were "
                       "produced, not this run.", env.get("BFCL_PROJECT_ROOT", "the project root"))

    rc, out = _run_cli(generate_cmd, env, timeout, model=model_key, backend=backend)
    if rc != 0:
        logger.error("bfcl generate failed (rc=%s):\n%s", rc, out[-1500:])
        return skipped_result("bfcl_v4_agentic", model_name,
                              f"`bfcl generate` failed (rc={rc}): {out.strip()[-300:]}", DOCS_URL)

    rc, out = _run_cli(evaluate_cmd, env, timeout, model=model_key, backend=backend)
    if rc != 0:
        logger.error("bfcl evaluate failed (rc=%s):\n%s", rc, out[-1500:])
        return skipped_result("bfcl_v4_agentic", model_name,
                              f"`bfcl evaluate` failed (rc={rc}): {out.strip()[-300:]}", DOCS_URL)

    per_cat = _parse_scores(score_root, model_key, categories)
    if not per_cat:
        return skipped_result("bfcl_v4_agentic", model_name,
                              f"the harness produced no score files under {score_root}",
                              DOCS_URL)

    total = sum(c["total"] for c in per_cat.values())
    correct = sum(c["correct"] for c in per_cat.values())
    ran_web = [c for c in categories if c in WEB_SEARCH_CATEGORIES]
    return {
        "benchmark_type": "eval",
        "eval_name": "bfcl_v4_agentic",
        "model_name": model_name,
        "mode": "bfcl_eval_harness",
        "total_questions": total,
        "correct_answers": correct,
        "accuracy": round(correct / total * 100.0, 2) if total else 0.0,
        "category_accuracy": per_cat,
        "bfcl_report": {
            "categories_run": categories,
            "search_backend": backend or "none (memory-only)",
            "leaderboard_comparable": backend == "serpapi",
            "web_search_categories_run": ran_web,
            "project_root": str(project_root),
            # Whether the model was actually queried, or a cached generation re-scored.
            "generations": "reused (BFCL_REUSE_GENERATIONS=1)" if reuse else "regenerated",
            "eval_limit": int(limit) if limited_ids else None,
            "limited_via": "test_case_ids_to_generate.json + --run-ids" if limited_ids else None,
            "score_dir": str(score_root),
        },
        "status": "success",
    }
