# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
"""Shared gated full-environment harness for tau2 / tau3 (Sierra tau2-bench).

The canonical tau-bench is a multi-turn *environment* benchmark: an LLM user and the
agent converse over several turns, tools mutate a domain database, and each task is
scored by the simulator's oracle (final DB-state check) plus an nl_assertion LLM
judge, yielding a per-task reward. It cannot be measured from a single response, so
gbench's default tau2/tau3 runners skip; this module runs the REAL simulator when the
operator opts in (TAU2_ENV_RUN=1) and the `tau2` package is importable.

Design (mirrors the reference tau-bench setup):
  * Three distinct LLM roles:
      - agent  = the model under test, routed to the gbench endpoint via LiteLLM's
                 `openai/<model>` with `api_base`/`api_key` passed PER-CALL;
      - user   = user simulator (TAU2_USER_LLM, defaults to GBENCH_JUDGE_MODEL);
      - judge  = nl_assertion evaluator (TAU2_EVAL_LLM, same default).
    Neither is the model under test: the agent is the local endpoint. A 503 from
    these two is upstream capacity, not an agent failure.
  * We deliberately DO NOT set OPENAI_API_BASE globally: tau2's evaluator/user also
    issue LLM calls, and redirecting all OpenAI traffic to the endpoint would corrupt
    grading. Only the agent is routed, per-call.
  * Best-effort robustness patches (empty/malformed-response retries, markdown-fence
    stripping) matching the reference, applied defensively so tau2 version drift can
    never crash the run.

Sampling/run knobs default to the reference values and are overridable via
env (TAU2_TEMPERATURE / TAU2_TOP_K / TAU2_TOP_P / TAU2_NUM_TRIALS / TAU2_MAX_STEPS /
TAU2_MAX_ERRORS / TAU2_SEED / TAU2_USER_LLM / TAU2_EVAL_LLM / TAU2_USER_TEMPERATURE).
"""

import importlib
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

from .swebench_common import skipped_result

logger = logging.getLogger(__name__)

_PATCHED = False   # robustness patches are process-global; apply once
_QUIETED = False   # cosmetic-log suppression; apply once
_TAU_PROGRESS_DESC = "tau"   # tqdm bar label; run_tau_env sets it per domain


def _quiet_tau2_noise() -> None:
    """Quiet tau2's redundant output so gbench logs stay readable. Two parts:

    1. Silence two cosmetic ERROR logs *at the source* (a loguru handler-filter does not
       survive: tau2's batch runner calls `logger.remove(); logger.add(...)` when it runs,
       wiping any filter). We no-op the emitting functions - neither value affects scoring:
         - get_response_cost: litellm can't price a custom model name -> 0.0;
         - get_commit_hash: `git rev-parse HEAD` fails off a repo -> "unknown".
    2. Mute tau2's rich console (`ConsoleDisplay.console`) - the per-task "Simulation
       Overview" panels and the live "Status: X/N complete" progress reprints, which are
       redundant because gbench builds its own summary from results.simulations, and which
       spam the log when stdout is redirected to a file (rich can't update in place). Set
       TAU2_VERBOSE=1 to keep the full tau2 panels. Real errors still surface via loguru.

    Best-effort across tau2 versions; patches the by-value import in runner.helpers too.
    """
    global _QUIETED
    if _QUIETED:
        return
    _QUIETED = True
    try:
        import tau2.utils.llm_utils as _llm            # called module-locally (llm_utils:420)
        _llm.get_response_cost = lambda *a, **k: 0.0
    except Exception as e:
        logger.debug("tau2: could not patch get_response_cost: %s", e)
    try:
        import tau2.utils.utils as _u
        _u.get_commit_hash = lambda *a, **k: "unknown"
        import tau2.runner.helpers as _h               # `from ...utils import get_commit_hash`
        if hasattr(_h, "get_commit_hash"):
            _h.get_commit_hash = lambda *a, **k: "unknown"
    except Exception as e:
        logger.debug("tau2: could not patch get_commit_hash: %s", e)
    if os.getenv("TAU2_VERBOSE") != "1":
        try:
            from rich.console import Console
            import tau2.utils.display as _disp
            _disp.ConsoleDisplay.console = Console(quiet=True)   # all panels route through this
        except Exception as e:
            logger.debug("tau2: could not mute rich console: %s", e)
        # litellm logs a Gemini-3+ sampling DeprecationWarning on every user-sim/judge call
        # (twice per call, via two loggers) - raise its loggers to ERROR so they don't flood.
        try:
            os.environ.setdefault("LITELLM_LOG", "ERROR")
            for _n in ("LiteLLM", "litellm", "LiteLLM Router", "LiteLLM Proxy"):
                logging.getLogger(_n).setLevel(logging.ERROR)
            import litellm as _ll
            _ll.suppress_debug_info = True
        except Exception as e:
            logger.debug("tau2: could not quiet litellm logging: %s", e)
        # The openai/httpx/httpcore HTTP clients log routine "Retrying request .../
        # HTTP Request: POST ..." at INFO for every agent call - pure noise here (gbench
        # builds its own summary). Raise them to WARNING; genuine failures still surface.
        try:
            for _n in ("openai", "openai._base_client", "httpx", "httpcore"):
                logging.getLogger(_n).setLevel(logging.WARNING)
        except Exception as e:
            logger.debug("tau2: could not quiet http client logging: %s", e)
        # Muting the console also killed tau2's useful 30s "Status: X/N complete" heartbeat
        # (StatusMonitor._monitor prints it through the same console). Redirect that heartbeat
        # to gbench's logger so the run shows live per-task progress like every other eval,
        # WITHOUT the per-task panels. Reuses tau2's own counters/reward math.
        _redirect_tau2_progress()


def _tau_progress_postfix(self, _time) -> Dict[str, Any]:
    """Build the tqdm postfix (avg reward, in-flight count, oldest elapsed) from a monitor."""
    post: Dict[str, Any] = {}
    try:
        results = self._simulation_results
        if results is not None:
            rewards = [s.reward_info.reward for s in results.simulations
                       if getattr(s, "reward_info", None) is not None]
            if rewards:
                post["reward"] = f"{sum(rewards) / len(rewards):.3f}"
        with self._lock:
            running = list(self.running_tasks.values())
        post["running"] = len(running)
        if running:
            now = _time.time()
            post["oldest"] = f"{max(now - i['start_time'] for i in running):.0f}s"
    except Exception:
        pass
    return post


def _redirect_tau2_progress() -> None:
    """Drive a tqdm progress bar from tau2's StatusMonitor - like every other gbench eval.

    tau2's native progress ("Status: X/N complete ...") prints through the console we mute.
    Instead we attach a tqdm bar to StatusMonitor (the same `tqdm(total=...)` mechanism
    `run_eval_suite` uses for every other suite): it advances per completed task and its
    postfix (avg reward, in-flight count, oldest elapsed) refreshes on a fixed cadence
    (min(TAU2_PROGRESS_SECS, 5)s) so it stays live between completions. On a TTY / `tail -f`
    this renders as one updating line, exactly like the other evals' `Eval [X]` bars.

    Falls back to periodic gbench-logger lines if tqdm is unavailable. Never fatal.
    """
    try:
        import time as _time
        from tau2.runner.progress import StatusMonitor
    except Exception as e:
        logger.debug("tau2: could not import StatusMonitor for progress: %s", e)
        return

    interval = max(1.0, float(os.getenv("TAU2_PROGRESS_SECS", "30")))

    try:
        from tqdm import tqdm
    except Exception:
        tqdm = None

    if tqdm is None:
        # --- fallback: one gbench-logger line per interval (no tqdm available) ---
        def _logger_monitor(self) -> None:
            while not self._stop_event.wait(timeout=interval):
                try:
                    with self._lock:
                        completed, total = self.completed_count, self.total_count
                    post = _tau_progress_postfix(self, _time)
                    logger.info("%s progress: %d/%d done | reward %s | %s running (oldest %s)",
                                _TAU_PROGRESS_DESC, completed, total,
                                post.get("reward", "n/a"), post.get("running", 0),
                                post.get("oldest", "-"))
                except Exception:
                    pass
        StatusMonitor._monitor = _logger_monitor
        return

    # --- tqdm bar, updated per-task; postfix refreshed by the monitor thread ---
    bar_interval = min(interval, 5.0)
    _orig_start = StatusMonitor.start
    _orig_finished = StatusMonitor.task_finished
    _orig_stop = StatusMonitor.stop

    def start(self) -> None:
        try:
            self._pbar = tqdm(total=self.total_count, initial=self.completed_count,
                              desc=f"Eval [{_TAU_PROGRESS_DESC}]", unit="task")
        except Exception:
            self._pbar = None
        _orig_start(self)                       # starts the (patched) _monitor thread

    def task_finished(self, task_key) -> None:
        _orig_finished(self, task_key)          # increments completed_count, pops running
        pb = getattr(self, "_pbar", None)
        if pb is not None:
            try:
                pb.update(1)                    # tqdm.update is thread-safe across workers
            except Exception:
                pass

    def stop(self) -> None:
        _orig_stop(self)
        pb = getattr(self, "_pbar", None)
        if pb is not None:
            try:
                pb.set_postfix(_tau_progress_postfix(self, _time), refresh=True)
                pb.close()
            except Exception:
                pass
            self._pbar = None

    def _monitor(self) -> None:                 # keep postfix + elapsed live between completions
        while not self._stop_event.wait(timeout=bar_interval):
            pb = getattr(self, "_pbar", None)
            if pb is None:
                continue
            try:
                pb.set_postfix(_tau_progress_postfix(self, _time), refresh=True)
            except Exception:
                pass

    StatusMonitor.start = start
    StatusMonitor.task_finished = task_finished
    StatusMonitor.stop = stop
    StatusMonitor._monitor = _monitor


def env_requested() -> bool:
    """True iff the operator explicitly opted into the full tau2 simulator."""
    return os.getenv("TAU2_ENV_RUN") == "1"


def _user_llm() -> str:
    from .base import DEFAULT_JUDGE_MODEL
    return os.getenv("TAU2_USER_LLM", f"gemini/{DEFAULT_JUDGE_MODEL}")


def _eval_llm() -> str:
    from .base import DEFAULT_JUDGE_MODEL
    return os.getenv("TAU2_EVAL_LLM", f"gemini/{DEFAULT_JUDGE_MODEL}")


def _import_tau2() -> Tuple[bool, str]:
    """Attempt a real `import tau2`, adding TAU2_BENCH_SRC to sys.path first if set.

    We import (not just find_spec) because tau2 can be *findable* yet not *importable* -
    e.g. on Python 3.13 its voice module imports the removed stdlib `audioop`. Callers
    need the real failure reason, so we return (ok, error_message).
    """
    src = os.getenv("TAU2_BENCH_SRC")
    if src and os.path.isdir(src) and src not in sys.path:
        sys.path.insert(0, src)
    # tau2 emits import-time loguru noise before it ever does useful work: a DEBUG
    # "Registry info: {...}" block (domains/agents/users/task_sets) and a WARNING
    # "No .env file found" (it looks for an optional .env to load keys; we pass keys via
    # the environment, so a missing .env is irrelevant). Configure loguru *before*
    # importing: raise the threshold to WARNING (kills the DEBUG dump) and add a filter
    # that drops the benign .env line while keeping real warnings (e.g. task retries).
    # TAU2_VERBOSE=1 keeps everything.
    if os.getenv("TAU2_VERBOSE") != "1":
        try:
            from loguru import logger as _loguru

            def _drop_import_noise(record):
                return "No .env file found" not in record["message"]

            _loguru.remove()
            _loguru.add(sys.stderr, level="WARNING", filter=_drop_import_noise)
        except Exception:
            pass
    try:
        import tau2  # noqa: F401
        return True, ""
    except Exception as e:  # ImportError + anything tau2.__init__ raises
        return False, f"{type(e).__name__}: {e}"


def check_tau_env_prerequisites() -> Tuple[bool, str]:
    """tau2-bench actually importable + a judge/user key for the gemini defaults."""
    ok, err = _import_tau2()
    if not ok:
        if "No module named 'tau2'" in err:
            return False, ("tau2-bench is not importable. It is not on PyPI - clone it and "
                           "install: `git clone https://github.com/sierra-research/tau2-bench && "
                           "pip install -e ./tau2-bench --no-deps`, or set TAU2_BENCH_SRC to a "
                           "tau2-bench/src checkout.")
        hint = ""
        if "audioop" in err:
            hint = (" tau2's voice module imports the stdlib `audioop`, removed in Python 3.13 - "
                    "install the backport: `pip install audioop-lts`.")
        return False, f"tau2-bench is installed but failed to import ({err}).{hint}"
    if (_user_llm().startswith("gemini/") or _eval_llm().startswith("gemini/")) \
            and not os.getenv("GEMINI_API_KEY"):
        return False, ("tau2 environment needs GEMINI_API_KEY for the user simulator and the "
                       "nl-assertion judge (or set TAU2_USER_LLM / TAU2_EVAL_LLM to another "
                       "LiteLLM provider you have credentials for).")
    return True, ""


def check_banking_prerequisites() -> Tuple[bool, str]:
    """Extra prerequisites for the banking_knowledge (tau3) RAG domain.

    banking_knowledge is a knowledge-retrieval domain, so on top of the shared tau2 env
    prereqs it needs a BM25 backend (`rank-bm25`) for the default/alltools retrieval configs,
    and - for the default Gemini-embedding dense retrieval - an embeddings key (the same
    GEMINI_API_KEY works via Google's OpenAI-compatible endpoint). Mirrors the house-style
    prereq skips (audioop, GEMINI key) with a docs pointer.
    """
    try:
        import rank_bm25  # noqa: F401
    except Exception:
        return False, ("banking_knowledge (tau3) needs the BM25 retrieval backend - install it: "
                       "`pip install rank-bm25`. See 'docs/evals/tau3.md' for setup.")
    # The default retrieval config (no TAU2_RETRIEVAL_CONFIG override) uses Gemini embeddings
    # for dense retrieval; that needs a key. A custom config may use other creds, so only
    # enforce this for the default path.
    if not os.getenv("TAU2_RETRIEVAL_CONFIG") and not (
        os.getenv("TAU2_EMBED_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
    ):
        return False, ("banking_knowledge (tau3) default dense retrieval uses Gemini embeddings and "
                       "needs GEMINI_API_KEY (or TAU2_EMBED_API_KEY / OPENAI_API_KEY). See "
                       "'docs/evals/tau3.md'.")
    return True, ""


# Google's OpenAI-compatible base URL - lets the OpenAI-SDK embedder hit Gemini embeddings.
_GEMINI_EMBED_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"


def _setup_banking_retrieval() -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Resolve the banking_knowledge retrieval config, wiring Gemini embeddings by default.

    Returns (retrieval_config_name, retrieval_config_kwargs) to set on TextRunConfig.

    Default (no TAU2_RETRIEVAL_CONFIG): a canonical `alltools` variant whose dense half uses
    a Gemini embedding model (TAU2_EMBED_MODEL, default `gemini-embedding-001`) via Google's
    OpenAI-compatible endpoint - so no OpenAI/OpenRouter key is required. We register a
    dedicated `alltools-gemini` variant (embedder_type "openai" -> OpenAI SDK -> Gemini) and
    point the OpenAI SDK env at Gemini. Set TAU2_RETRIEVAL_CONFIG to any stock tau2-bench
    variant (e.g. `bm25`, `grep_only`, or `alltools` with your own OpenAI/OpenRouter creds)
    to bypass all of this.
    """
    override = os.getenv("TAU2_RETRIEVAL_CONFIG")
    if override:
        return override, None

    # `gemini-embedding-001` is the published embeddings model id. The previous default,
    # `gemini-embedding-2`, is not a served id: the KB warm-up 404s, retrieval never
    # builds, and tau3 banking_knowledge skips - which reads as "harness unavailable"
    # rather than "one env var is wrong".
    embed_model = os.getenv("TAU2_EMBED_MODEL", "gemini-embedding-001")
    embed_key = os.getenv("TAU2_EMBED_API_KEY") or os.getenv("GEMINI_API_KEY")
    embed_base = os.getenv("TAU2_EMBED_BASE_URL", _GEMINI_EMBED_BASE)
    # Route the OpenAI-SDK embedder to Gemini unless real OpenAI creds are already present.
    if embed_key and not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = embed_key
        os.environ.setdefault("OPENAI_BASE_URL", embed_base)
    # Gemini's OpenAI-compat embeddings endpoint caps a batch at 100 inputs; tau2-bench's
    # OpenAIEmbedder sends all ~700 docs in one request (400 on Gemini). Chunk embed().
    _patch_openai_embedder_batch(int(os.getenv("TAU2_EMBED_BATCH", "100")))

    variant_name = "alltools-gemini"
    try:
        from tau2.domains.banking_knowledge import retrieval as _r
        if variant_name not in _r.RETRIEVAL_VARIANTS:
            _r.RETRIEVAL_VARIANTS[variant_name] = _r.all_tools_variant(
                name=variant_name, embedder_type="openai", embedder_model=embed_model)
        # The KB-cache warmer resolves embedder configs from a hardcoded table keyed by
        # config name (embeddings_cache.get_unique_embedder_configs_for_retrieval_configs),
        # which doesn't know our custom variant. Wrap it so the doc embeddings are pre-warmed
        # with our Gemini model instead of being skipped (which would leave dense retrieval
        # to embed lazily or, worse, fall back to a wrong model).
        _patch_embedder_config_resolver(variant_name, embed_model)
        logger.info("tau3: banking_knowledge retrieval = %s (embedder=openai:%s via %s)",
                    variant_name, embed_model,
                    os.getenv("OPENAI_BASE_URL", "OpenAI default"))
        return variant_name, None
    except Exception as e:
        logger.warning("tau3: could not register gemini alltools variant (%s); using stock 'alltools'", e)
        return "alltools", None


def _patch_openai_embedder_batch(max_batch: int = 100) -> None:
    """Chunk OpenAIEmbedder.embed() into <=max_batch inputs per request (idempotent).

    tau2-bench sends every document in one embeddings.create() call, which is fine for
    OpenAI (batch limit ~2048) but 400s on Gemini's OpenAI-compatible endpoint (max 100
    per BatchEmbedContents). We wrap embed() to split large inputs and concatenate.
    """
    if max_batch < 1:
        return
    try:
        import numpy as _np
        from tau2.knowledge.embedders.openai_embedder import OpenAIEmbedder
        if getattr(OpenAIEmbedder.embed, "_gbench_chunked", False):
            return
        _orig = OpenAIEmbedder.embed

        def _chunked(self, texts):
            if not texts or len(texts) <= max_batch:
                return _orig(self, texts)
            parts = [_orig(self, texts[i:i + max_batch])
                     for i in range(0, len(texts), max_batch)]
            return _np.concatenate(parts, axis=0)

        _chunked._gbench_chunked = True
        OpenAIEmbedder.embed = _chunked
    except Exception as e:
        logger.debug("tau3: could not patch OpenAIEmbedder batch size: %s", e)


def _patch_embedder_config_resolver(variant_name: str, embed_model: str) -> None:
    """Teach the KB-cache warmer about our Gemini variant (idempotent, best-effort)."""
    try:
        from tau2.knowledge import embeddings_cache as _ec
        if getattr(_ec.get_unique_embedder_configs_for_retrieval_configs, "_gbench_wrapped", False):
            return
        _orig = _ec.get_unique_embedder_configs_for_retrieval_configs

        def _wrapped(names, retrieval_config_kwargs=None):
            out = list(_orig(names, retrieval_config_kwargs) or [])
            if variant_name in (names or []):
                cfg = ("openai", {"model": embed_model})
                if cfg not in out:
                    out.append(cfg)
            return out

        _wrapped._gbench_wrapped = True
        _ec.get_unique_embedder_configs_for_retrieval_configs = _wrapped
    except Exception as e:
        logger.debug("tau3: could not patch embedder-config resolver: %s", e)


def _strip_md_fences(text: Optional[str]) -> Optional[str]:
    if not text:
        return text
    c = text.strip()
    if c.startswith("```"):
        lines = c.split("\n")
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines)
    return text


def _install_robustness_patches(retries: int = 3) -> None:
    """Retry empty/malformed LLM responses and strip stray markdown fences.

    All patches are best-effort: any import/attribute drift in a given tau2 version is
    caught and logged, never fatal (the run proceeds with whatever could be patched).
    """
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    # --- litellm.completion: retry when the first response is empty/malformed ---
    try:
        import json as _json
        import litellm
        _orig_completion = litellm.completion

        def _completion_retry(*args, **kwargs):
            resp = _orig_completion(*args, **kwargs)
            for attempt in range(retries - 1):
                try:
                    msg = resp.choices[0].message
                    tool_calls = getattr(msg, "tool_calls", None)
                    if not msg.content and not tool_calls:
                        raise ValueError("empty response")
                    if tool_calls:
                        for tc in tool_calls:
                            a = tc.function.arguments
                            if not a:
                                raise ValueError("empty tool arguments")
                            _json.loads(a)  # unparseable -> retry
                    return resp
                except (IndexError, AttributeError):
                    return resp  # unexpected shape; hand back untouched
                except (ValueError, TypeError):
                    logger.debug("tau2: empty/malformed response, retry %d/%d", attempt + 2, retries)
                    resp = _orig_completion(*args, **kwargs)
            return resp

        litellm.completion = _completion_retry
        try:
            import litellm.main
            litellm.main.completion = _completion_retry
        except Exception:
            pass
        try:
            import tau2.utils.llm_utils as _llm
            _llm.completion = _completion_retry
        except Exception:
            pass
    except Exception as e:
        logger.warning("tau2: could not install litellm retry patch: %s", e)

    # --- tau2 generate(): retry empty content + strip markdown fences ---
    try:
        import tau2.utils.llm_utils as _llm
        _orig_generate = _llm.generate

        def _generate_retry(*args, **kwargs):
            last = None
            for attempt in range(retries):
                res = _orig_generate(*args, **kwargs)
                last = res
                content = getattr(res, "content", None)
                tool_calls = getattr(res, "tool_calls", None)
                if content is not None:
                    stripped = _strip_md_fences(content)
                    if stripped != content:
                        try:
                            res.content = stripped
                        except Exception:
                            pass
                if (content or tool_calls) and (content is None or str(content).strip() or tool_calls):
                    return res
                logger.debug("tau2: empty generate(), retry %d/%d", attempt + 2, retries)
            return last

        for modname in ("tau2.utils.llm_utils", "tau2.agent.llm_agent",
                        "tau2.user.user_simulator", "tau2.environment.utils.interface_agent",
                        "tau2.evaluator.evaluator_nl_assertions",
                        "tau2.evaluator.hallucination_reviewer"):
            try:
                mod = importlib.import_module(modname)
                if hasattr(mod, "generate"):
                    mod.generate = _generate_retry
            except Exception:
                pass
    except Exception as e:
        logger.warning("tau2: could not install generate() retry patch: %s", e)


def _configure_evaluator_llm(eval_model: str) -> None:
    """Point tau2's nl-assertion evaluator + env-interface LLMs at `eval_model`.

    `from X import Y` copies values at import time, so we patch both the config module
    and the already-imported evaluator module. Best-effort across versions.
    """
    eval_args = {"temperature": 0.0}
    try:
        import tau2.config as cfg
        for attr, val in (("DEFAULT_LLM_NL_ASSERTIONS", eval_model),
                          ("DEFAULT_LLM_NL_ASSERTIONS_ARGS", eval_args),
                          ("DEFAULT_LLM_ENV_INTERFACE", eval_model),
                          ("DEFAULT_LLM_ENV_INTERFACE_ARGS", eval_args)):
            if hasattr(cfg, attr):
                setattr(cfg, attr, val)
    except Exception as e:
        logger.warning("tau2: could not set evaluator config: %s", e)
    try:
        import tau2.evaluator.evaluator_nl_assertions as nl
        if hasattr(nl, "DEFAULT_LLM_NL_ASSERTIONS"):
            nl.DEFAULT_LLM_NL_ASSERTIONS = eval_model
            nl.DEFAULT_LLM_NL_ASSERTIONS_ARGS = eval_args
    except Exception:
        pass


def _build_agent_args(base_url: str, enable_thinking: bool) -> Dict[str, Any]:
    """LiteLLM args routing ONLY the agent to the gbench endpoint, per-call."""
    args: Dict[str, Any] = {
        "temperature": float(os.getenv("TAU2_TEMPERATURE", "1.0")),
        "api_base": base_url,
        "api_key": os.getenv("OPENAI_API_KEY", "EMPTY"),
        "top_p": float(os.getenv("TAU2_TOP_P", "0.95")),
    }
    extra_body: Dict[str, Any] = {"top_k": int(os.getenv("TAU2_TOP_K", "64"))}
    if enable_thinking:
        extra_body["chat_template_kwargs"] = {"enable_thinking": True}
    args["extra_body"] = extra_body
    return args


def _run_one_domain(domain, model_name, base_url, concurrency, limit, enable_thinking):
    """Run a single tau2 domain via the Python API; return (total, reward_sum, perfect, infra)."""
    from tau2.data_model.simulation import TextRunConfig
    from tau2.run import run_domain

    # Opt-in: persist tau2's full SimulationResults (per-task messages + reward breakdown)
    # so a run can be audited afterwards. By default gbench keeps only the aggregate.
    save_to = None
    trace_dir = os.getenv("TAU2_SAVE_TRACES")
    if trace_dir:
        try:
            os.makedirs(trace_dir, exist_ok=True)
            save_to = os.path.join(trace_dir, f"tau2_{domain}_traces.json")
        except Exception as e:
            logger.warning("tau2: could not prepare TAU2_SAVE_TRACES dir %r: %s", trace_dir, e)

    # banking_knowledge (tau3) is a RAG domain: select/wire its retrieval backend. Other
    # domains (airline/retail/telecom) ignore retrieval_config.
    retrieval_config = None
    retrieval_config_kwargs = None
    if domain == "banking_knowledge":
        retrieval_config, retrieval_config_kwargs = _setup_banking_retrieval()

    config = TextRunConfig(
        domain=domain,
        agent="llm_agent",
        user="user_simulator",
        llm_agent=f"openai/{model_name}",
        llm_args_agent=_build_agent_args(base_url, enable_thinking),
        llm_user=_user_llm(),
        llm_args_user={"temperature": float(os.getenv("TAU2_USER_TEMPERATURE", "0.0"))},
        num_trials=int(os.getenv("TAU2_NUM_TRIALS", "1")),
        max_steps=int(os.getenv("TAU2_MAX_STEPS", "200")),
        max_errors=int(os.getenv("TAU2_MAX_ERRORS", "10")),
        max_concurrency=max(1, concurrency),
        seed=int(os.getenv("TAU2_SEED", "300")),
        log_level="ERROR",
        task_ids=None,
        num_tasks=(limit if (limit and limit > 0) else None),
        retrieval_config=retrieval_config,
        retrieval_config_kwargs=retrieval_config_kwargs,
        save_to=save_to,
        verbose_logs=False,
        auto_resume=False,
        hallucination_retries=0,
    )
    results = run_domain(config)
    sims = list(results.simulations)
    reward_sum = 0.0
    perfect = 0
    infra = 0
    for s in sims:
        ri = getattr(s, "reward_info", None)
        if ri is None:                       # infra error == reward 0 (not excluded)
            infra += 1
            continue
        r = float(ri.reward or 0.0)
        reward_sum += r
        if r >= 1.0:
            perfect += 1
    return len(sims), reward_sum, perfect, infra


def run_tau_env(
    eval_name: str,
    domains: List[str],
    model_name: str,
    base_url: str,
    concurrency: int,
    limit: Optional[int],
    docs_url: str,
    enable_thinking: bool = False,
) -> Dict[str, Any]:
    """Run the real tau2 simulator per domain and aggregate into a standard result dict.

    Accuracy is the canonical tau2 mean reward; correct_answers
    is the perfect-task (reward>=1) count. Skips cleanly if prerequisites are missing or
    no domain produced results.
    """
    ok, reason = check_tau_env_prerequisites()
    if not ok:
        return skipped_result(eval_name, model_name, reason, docs_url)
    if "banking_knowledge" in domains:
        ok_b, reason_b = check_banking_prerequisites()
        if not ok_b:
            return skipped_result(eval_name, model_name, reason_b, docs_url)

    _quiet_tau2_noise()
    _install_robustness_patches()
    _configure_evaluator_llm(_eval_llm())

    category_accuracy: Dict[str, Any] = {}
    total = 0
    reward_sum_all = 0.0
    perfect_all = 0
    infra_all = 0
    ran_any = False

    # tau2's own per-task console is muted by default (it spams a file-redirected log), so
    # progress is shown via a tqdm bar driven by StatusMonitor - one per domain, exactly like
    # every other eval's `Eval [X]` bar. TAU2_VERBOSE=1 restores tau2's live per-task panels.
    global _TAU_PROGRESS_DESC
    scope = f"{limit} tasks/domain" if (limit and limit > 0) else "all tasks"
    trials = int(os.getenv("TAU2_NUM_TRIALS", "1"))
    logger.info("%s: running %d domain(s) %s via tau2 simulator (%s, concurrency=%d, trials=%d); "
                "each task is a multi-turn agent<->user-simulator conversation scored by an LLM "
                "judge, so a domain runs for several minutes - a tqdm progress bar follows "
                "(set TAU2_VERBOSE=1 for per-task detail)",
                eval_name, len(domains), domains, scope, max(1, concurrency), trials)

    for i, domain in enumerate(domains, 1):
        _TAU_PROGRESS_DESC = f"{eval_name} {domain}" if len(domains) > 1 else eval_name
        logger.info("%s: [%d/%d] starting domain '%s' ...", eval_name, i, len(domains), domain)
        try:
            d_total, d_reward, d_perfect, d_infra = _run_one_domain(
                domain, model_name, base_url, concurrency, limit, enable_thinking)
        except Exception as e:
            logger.error("%s env: domain %s failed: %s", eval_name, domain, e, exc_info=True)
            continue
        if d_total == 0:
            logger.error("%s env: domain %s produced no simulations", eval_name, domain)
            continue
        ran_any = True
        total += d_total
        reward_sum_all += d_reward
        perfect_all += d_perfect
        infra_all += d_infra
        category_accuracy[domain] = {
            "correct": d_perfect,
            "total": d_total,
            "accuracy": round(d_reward / d_total * 100.0, 2),  # mean reward for the domain
        }
        logger.info("%s: [%d/%d] domain '%s' done - mean reward %.3f (%d/%d perfect, %d infra errors)",
                    eval_name, i, len(domains), domain,
                    d_reward / d_total, d_perfect, d_total, d_infra)

    if not ran_any:
        return skipped_result(
            eval_name, model_name,
            "tau2 simulator produced no results for any domain (check tau2 install, "
            "GEMINI_API_KEY, and the model endpoint)",
            docs_url)

    accuracy = (reward_sum_all / total * 100.0) if total > 0 else 0.0
    return {
        "benchmark_type": "eval",
        "eval_name": eval_name,
        "model_name": model_name,
        "mode": "full_environment",
        "total_questions": total,
        "correct_answers": perfect_all,
        "accuracy": round(accuracy, 2),          # canonical tau2 mean reward
        "category_accuracy": category_accuracy,
        "tau2_report": {
            "mean_reward": round(reward_sum_all / total, 4) if total else 0.0,
            "perfect_tasks": perfect_all,
            "total_simulations": total,
            "infra_errors": infra_all,
            "num_trials": int(os.getenv("TAU2_NUM_TRIALS", "1")),
            "agent_llm": f"openai/{model_name}",
            "user_llm": _user_llm(),
            "eval_llm": _eval_llm(),
        },
        "status": "success",
    }
