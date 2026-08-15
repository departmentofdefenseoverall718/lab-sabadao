# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
"""Tier-4 agentic / execution-environment suites: swe_lancer, gaia2, tau2/tau3 env.

- swe_lancer: gated OpenAI SWELancer Docker harness (substring heuristic removed).
- gaia2: blocked_external - a single-turn proxy is not the ARE benchmark, so it skips
  and emits no approximate score.
- tau2/tau3: a plain single-turn /v1 endpoint CANNOT score tau-bench (assertion-only
  tasks + environment-dependent gold args), so the default reports a clean skip instead
  of a misleading 0%; the only scored path is the gated full environment (TAU2_ENV_RUN=1
  + the tau2-bench simulator), which itself skips cleanly when opted-in-but-unavailable.
- gdpval: grades produced file deliverables, so a text endpoint can't be scored -> skips.

All harnesses are absent on CI, so these lock in the gates, skip dicts, loaders (no
fabrication), and the env result-parsing/aggregation that runs fully offline.
"""

import json
import os
import tempfile
from unittest.mock import patch, MagicMock

import pytest

from gbench.runners.eval_suites import SUITES


# --------------------------------------------------------------------------- #
# registration + sandbox wiring
# --------------------------------------------------------------------------- #
def test_tier4_suites_registered():
    for name in ("swe_lancer", "gaia2", "tau2", "tau3"):
        assert name in SUITES, f"{name} not registered in SUITES"


@pytest.mark.parametrize("name", ["swe_lancer", "tau2", "tau3"])
def test_tier4_execution_suites_are_sandbox_evals(name):
    from gbench.runners.evals import EvalsBenchmarkRunner
    from gbench.core.config import BenchmarkConfig

    config = BenchmarkConfig()
    config.sandboxes = 5
    runner = EvalsBenchmarkRunner(config)
    with patch.dict("gbench.runners.eval_suites.SUITES",
                    {name: MagicMock(return_value={"status": "success"})}) as mock_suites:
        runner._run_single_eval(name, "gemma-4-E4B-it", "http://localhost:8000/v1", num_threads=128)
        assert mock_suites[name].call_args[1]["concurrency"] == 5


# --------------------------------------------------------------------------- #
# swe_lancer: gated Docker harness (no heuristic scoring)
# --------------------------------------------------------------------------- #
def test_swe_lancer_skip_on_missing_harness():
    from gbench.runners.eval_suites.swe_lancer import run_swe_lancer
    res = run_swe_lancer("gemma-4-E4B-it", "http://localhost:8000/v1", 2)
    assert res["status"] == "skipped"
    assert "docs/evals/swe_lancer.md" in res["skip_reason"]
    assert res["accuracy"] == 0.0


def test_swe_lancer_gated_behind_opt_in():
    from gbench.runners.eval_suites import swe_lancer as SL
    with tempfile.TemporaryDirectory() as hd:
        open(os.path.join(hd, "run_swelancer_eval.py"), "w").close()
        with patch("shutil.which", return_value="/usr/bin/docker"), \
             patch("subprocess.run", return_value=MagicMock(returncode=0)), \
             patch.dict("sys.modules", {"docker": MagicMock()}), \
             patch.dict(os.environ, {"SWELANCER_HARNESS_DIR": hd}, clear=False):
            os.environ.pop("SWELANCER_RUN", None)
            ok, reason = SL.check_swe_lancer_prerequisites()
            assert ok is False and "SWELANCER_RUN=1" in reason


def test_swe_lancer_loader_and_patch_extractor():
    from gbench.runners.eval_suites import swe_lancer as SL
    with tempfile.TemporaryDirectory() as d:
        inst = os.path.join(d, "instruction.md")
        with open(inst, "w") as f:
            f.write("Fix the login bug.")

        class _Api:
            def list_repo_files(self, *a, **k):
                return ["task_42/instruction.md", "task_42/solution/solve.sh"]

        with patch("huggingface_hub.HfApi", _Api), \
             patch("huggingface_hub.hf_hub_download", return_value=inst):
            samples = SL._load_swe_lancer_samples()
    assert len(samples) == 1 and samples[0][1] == "task_42"
    assert samples[0][2]["task_id"] == "task_42"
    assert "Fix the login bug." in samples[0][0][0]["content"]
    assert SL._extract_patch("```diff\ndiff --git a/x b/x\n@@\n```").startswith("diff --git")
    assert SL._extract_patch("nothing here") == ""


def test_swe_lancer_results_parser():
    from gbench.runners.eval_suites import swe_lancer as SL
    with tempfile.TemporaryDirectory() as out:
        with open(os.path.join(out, "r.json"), "w") as f:
            json.dump({"resolved_ids": ["task_42"]}, f)
        assert SL._parse_results(out) == {"task_42": True}
    with tempfile.TemporaryDirectory() as out:
        with open(os.path.join(out, "r.json"), "w") as f:
            json.dump({"task_1": True, "task_2": False}, f)
        assert SL._parse_results(out) == {"task_1": True, "task_2": False}


# --------------------------------------------------------------------------- #
# gaia2: blocked_external (no approximate score)
# --------------------------------------------------------------------------- #
def test_gaia2_is_blocked_external_and_skips():
    from gbench.runners.eval_suites.gaia2 import run_gaia2
    res = run_gaia2("gemma-4-E4B-it", "http://localhost:8000/v1", 2)
    assert res["status"] == "skipped"
    assert "docs/evals/gaia2.md" in res["skip_reason"]
    assert res["accuracy"] == 0.0


def test_gaia2_reports_harness_not_wired_even_when_present():
    from gbench.runners.eval_suites import gaia2 as G2
    with patch("importlib.util.find_spec", return_value=MagicMock()), \
         patch.dict(os.environ, {"GAIA2_RUN": "1"}, clear=False):
        ok, reason = G2.check_gaia2_prerequisites()
        assert ok is False and "blocked_external" in reason


# --------------------------------------------------------------------------- #
# tau2 / tau3: proxy default + gated full environment
# --------------------------------------------------------------------------- #
def test_tau_env_not_requested_by_default():
    from gbench.runners.eval_suites import tau_common as TC
    old = os.environ.pop("TAU2_ENV_RUN", None)
    try:
        assert TC.env_requested() is False
    finally:
        if old is not None:
            os.environ["TAU2_ENV_RUN"] = old


def test_tau2_and_tau3_default_skip_not_a_misleading_zero():
    """Without the env, tau2/tau3 must SKIP (not score 0) since single-turn can't measure them."""
    from gbench.runners.eval_suites import tau2 as T2, tau3 as T3
    old = os.environ.pop("TAU2_ENV_RUN", None)
    try:
        for run, name in ((T2.run_tau2, "tau2"), (T3.run_tau3, "tau3")):
            r = run("m", "http://x", 2)
            assert r["status"] == "skipped", r
            assert r["accuracy"] == 0.0 and r["total_questions"] == 0
            assert f"docs/evals/{name}.md" in r["skip_reason"]
            assert "TAU2_ENV_RUN=1" in r["skip_reason"]
    finally:
        if old is not None:
            os.environ["TAU2_ENV_RUN"] = old


def test_gdpval_skips_against_text_endpoint():
    """gdpval grades file deliverables; a text /v1 endpoint can't produce them -> skip."""
    from gbench.runners.eval_suites.gdpval import run_gdpval
    r = run_gdpval("m", "http://x", 2)
    assert r["status"] == "skipped"
    assert r["accuracy"] == 0.0 and r["total_questions"] == 0
    assert "docs/evals/gdpval.md" in r["skip_reason"]


def test_aider_java_probe_requires_full_toolchain():
    """Java must be skipped (not scored 0) unless javac AND gradle are present."""
    from unittest.mock import patch as _patch
    from gbench.runners.eval_suites import aider_polyglot as AP

    def fake_which(name):
        return {"java": "/j", "javac": "/jc"}.get(name)  # gradle absent
    with _patch.object(AP.shutil, "which", side_effect=fake_which):
        ok, why = AP._lang_toolchain_ok("java")
    assert ok is False and "gradle" in why

    def all_present(name):
        return "/" + name
    with _patch.object(AP.shutil, "which", side_effect=all_present):
        ok, why = AP._lang_toolchain_ok("java")
    assert ok is True


def test_tau2_and_tau3_env_opt_in_but_missing_skips_cleanly():
    from gbench.runners.eval_suites import tau2 as T2, tau3 as T3
    from gbench.runners.eval_suites import tau_common as TC
    with patch.dict(os.environ, {"TAU2_ENV_RUN": "1"}, clear=False), \
         patch.object(TC, "_import_tau2",
                      return_value=(False, "ModuleNotFoundError: No module named 'tau2'")):
        r2 = T2.run_tau2("m", "http://x", 2)
        assert r2["status"] == "skipped" and "docs/evals/tau2.md" in r2["skip_reason"]
        r3 = T3.run_tau3("m", "http://x", 2)
        assert r3["status"] == "skipped" and "docs/evals/tau3.md" in r3["skip_reason"]


def test_tau_env_prereq_requires_gemini_key():
    """With tau2 importable but no GEMINI_API_KEY, the gemini user/judge defaults -> skip reason."""
    from gbench.runners.eval_suites import tau_common as TC
    old = os.environ.pop("GEMINI_API_KEY", None)
    try:
        with patch.object(TC, "_import_tau2", return_value=(True, "")):
            ok, why = TC.check_tau_env_prerequisites()
        assert ok is False and "GEMINI_API_KEY" in why
    finally:
        if old is not None:
            os.environ["GEMINI_API_KEY"] = old


def test_tau_env_prereq_surfaces_import_error_with_audioop_hint():
    """A findable-but-broken tau2 (e.g. py3.13 audioop) yields the real reason + backport hint."""
    from gbench.runners.eval_suites import tau_common as TC
    with patch.object(TC, "_import_tau2",
                      return_value=(False, "ModuleNotFoundError: No module named 'audioop'")), \
         patch.dict(os.environ, {"GEMINI_API_KEY": "x"}, clear=False):
        ok, why = TC.check_tau_env_prerequisites()
    assert ok is False and "audioop-lts" in why and "audioop" in why


def test_tau_common_run_env_aggregates_reward_across_domains():
    """run_tau_env aggregates per-domain (total, reward_sum, perfect) into the result dict.

    Mocks the tau2 Python-API layer (_run_one_domain) so the gbench-side aggregation +
    result schema are what's under test, not the external simulator.
    """
    from gbench.runners.eval_suites import tau_common as TC
    canned = {"airline": (2, 1.0, 1, 0), "retail": (2, 2.0, 2, 0)}  # (total, reward_sum, perfect, infra)

    with patch.object(TC, "check_tau_env_prerequisites", return_value=(True, "")), \
         patch.object(TC, "_install_robustness_patches", lambda *a, **k: None), \
         patch.object(TC, "_configure_evaluator_llm", lambda *a, **k: None), \
         patch.object(TC, "_run_one_domain", side_effect=lambda d, *a, **k: canned[d]):
        r = TC.run_tau_env("tau2", ["airline", "retail"], "m", "http://x", 4, None, "docs/evals/tau2.md")

    assert r["status"] == "success" and r["mode"] == "full_environment"
    # total=4, reward_sum=3.0 -> mean reward 0.75 -> accuracy 75.0; perfect=3
    assert r["total_questions"] == 4 and r["correct_answers"] == 3 and r["accuracy"] == 75.0
    assert r["category_accuracy"]["airline"]["accuracy"] == 50.0    # 1.0/2
    assert r["category_accuracy"]["retail"]["accuracy"] == 100.0    # 2.0/2
    assert r["tau2_report"]["mean_reward"] == 0.75 and r["tau2_report"]["perfect_tasks"] == 3


# --------------------------------------------------------------------------- #
# tau3 == τ³-bench banking_knowledge (RAG) repurpose
# --------------------------------------------------------------------------- #
def test_tau3_default_skip_names_banking_knowledge():
    """tau3's default skip should describe the τ³ banking_knowledge RAG domain, not telecom."""
    from gbench.runners.eval_suites import tau3 as T3
    old = os.environ.pop("TAU2_ENV_RUN", None)
    try:
        r = T3.run_tau3("m", "http://x", 2)
        assert r["status"] == "skipped"
        assert "banking_knowledge" in r["skip_reason"]
        assert "telecom" not in r["skip_reason"].lower()
    finally:
        if old is not None:
            os.environ["TAU2_ENV_RUN"] = old


def test_tau3_env_runs_banking_knowledge_domain():
    """With the env opt-in, tau3 must drive the banking_knowledge domain (not telecom)."""
    from gbench.runners.eval_suites import tau3 as T3
    with patch.dict(os.environ, {"TAU2_ENV_RUN": "1"}, clear=False), \
         patch.object(T3, "run_tau_env", return_value={"status": "success"}) as m:
        T3.run_tau3("m", "http://x", 8, enable_thinking=False, limit=5)
    args, kwargs = m.call_args
    assert args[0] == "tau3" and args[1] == ["banking_knowledge"]


def test_check_banking_prerequisites_requires_rank_bm25():
    """Missing rank-bm25 -> clean skip with the pip hint + docs pointer (like audioop)."""
    import sys as _sys
    from gbench.runners.eval_suites import tau_common as TC
    with patch.dict(_sys.modules, {"rank_bm25": None}):   # makes `import rank_bm25` raise
        ok, why = TC.check_banking_prerequisites()
    assert ok is False and "rank-bm25" in why and "docs/evals/tau3.md" in why


def test_check_banking_prerequisites_default_needs_embed_key():
    """rank-bm25 present but no key and no config override -> skip about the embeddings key."""
    from gbench.runners.eval_suites import tau_common as TC
    with patch.dict(os.environ, {}, clear=False):
        for k in ("TAU2_RETRIEVAL_CONFIG", "GEMINI_API_KEY", "TAU2_EMBED_API_KEY", "OPENAI_API_KEY"):
            os.environ.pop(k, None)
        ok, why = TC.check_banking_prerequisites()
        assert ok is False and "GEMINI_API_KEY" in why
        os.environ["GEMINI_API_KEY"] = "k"
        ok2, _ = TC.check_banking_prerequisites()
        assert ok2 is True


def test_check_banking_prerequisites_override_bypasses_key_check():
    """A TAU2_RETRIEVAL_CONFIG override (e.g. bm25, no embeddings) shouldn't demand a Gemini key."""
    from gbench.runners.eval_suites import tau_common as TC
    with patch.dict(os.environ, {"TAU2_RETRIEVAL_CONFIG": "bm25"}, clear=False):
        for k in ("GEMINI_API_KEY", "TAU2_EMBED_API_KEY", "OPENAI_API_KEY"):
            os.environ.pop(k, None)
        ok, _ = TC.check_banking_prerequisites()
    assert ok is True


def test_run_tau_env_gates_on_banking_prereq():
    """run_tau_env must apply the banking prereq gate for the banking_knowledge domain."""
    from gbench.runners.eval_suites import tau_common as TC
    with patch.object(TC, "check_tau_env_prerequisites", return_value=(True, "")), \
         patch.object(TC, "check_banking_prerequisites",
                      return_value=(False, "needs rank-bm25 ... docs/evals/tau3.md")):
        r = TC.run_tau_env("tau3", ["banking_knowledge"], "m", "http://x", 4, None, "docs/evals/tau3.md")
    assert r["status"] == "skipped" and "rank-bm25" in r["skip_reason"]


def test_setup_banking_retrieval_wires_gemini_and_honors_override():
    """Default banking retrieval registers a Gemini alltools variant + points OpenAI SDK at Gemini;
    an explicit TAU2_RETRIEVAL_CONFIG override is used verbatim."""
    pytest.importorskip("tau2.domains.banking_knowledge.retrieval")
    from gbench.runners.eval_suites import tau_common as TC
    with patch.dict(os.environ, {"GEMINI_API_KEY": "k"}, clear=False):
        for k in ("TAU2_RETRIEVAL_CONFIG", "OPENAI_API_KEY", "OPENAI_BASE_URL"):
            os.environ.pop(k, None)
        name, kw = TC._setup_banking_retrieval()
        assert name == "alltools-gemini" and kw is None
        assert "generativelanguage.googleapis.com" in os.environ.get("OPENAI_BASE_URL", "")
        assert os.environ.get("OPENAI_API_KEY") == "k"
    with patch.dict(os.environ, {"TAU2_RETRIEVAL_CONFIG": "bm25"}, clear=False):
        name2, _ = TC._setup_banking_retrieval()
        assert name2 == "bm25"


def _fake_tau2_modules():
    """Fake tau2 submodules that _quiet_tau2_noise patches (functions + rich console)."""
    import types
    llm = types.ModuleType("tau2.utils.llm_utils")
    llm.get_response_cost = lambda r: 999.0
    utils = types.ModuleType("tau2.utils.utils")
    utils.get_commit_hash = lambda: "realhash"
    helpers = types.ModuleType("tau2.runner.helpers")
    helpers.get_commit_hash = lambda: "realhash"
    display = types.ModuleType("tau2.utils.display")

    class ConsoleDisplay:
        console = "loud-console"  # a placeholder tau2 would set to a rich Console()
    display.ConsoleDisplay = ConsoleDisplay
    return {
        "tau2": types.ModuleType("tau2"),
        "tau2.utils": types.ModuleType("tau2.utils"),
        "tau2.utils.llm_utils": llm,
        "tau2.utils.utils": utils,
        "tau2.utils.display": display,
        "tau2.runner": types.ModuleType("tau2.runner"),
        "tau2.runner.helpers": helpers,
    }


def test_tau_common_quiet_noops_functions_and_mutes_console():
    """_quiet_tau2_noise no-ops the two noisy functions (incl. the by-value import in
    runner.helpers) AND mutes tau2's rich console by default."""
    import sys
    from gbench.runners.eval_suites import tau_common as TC
    fake = _fake_tau2_modules()
    TC._QUIETED = False
    old = os.environ.pop("TAU2_VERBOSE", None)
    try:
        with patch.dict(sys.modules, fake):
            TC._quiet_tau2_noise()
        assert fake["tau2.utils.llm_utils"].get_response_cost("x") == 0.0
        assert fake["tau2.utils.utils"].get_commit_hash() == "unknown"
        assert fake["tau2.runner.helpers"].get_commit_hash() == "unknown"
        # rich console replaced with a quiet one
        console = fake["tau2.utils.display"].ConsoleDisplay.console
        assert getattr(console, "quiet", False) is True
        # litellm's noisy loggers raised to ERROR (kills the per-call Gemini deprecation spam)
        import logging as _logging
        assert _logging.getLogger("LiteLLM").level == _logging.ERROR
    finally:
        TC._QUIETED = False
        if old is not None:
            os.environ["TAU2_VERBOSE"] = old


def test_tau_common_verbose_env_keeps_console():
    """TAU2_VERBOSE=1 leaves tau2's rich console untouched (full panels)."""
    import sys
    from gbench.runners.eval_suites import tau_common as TC
    fake = _fake_tau2_modules()
    TC._QUIETED = False
    try:
        with patch.dict(sys.modules, fake), patch.dict(os.environ, {"TAU2_VERBOSE": "1"}):
            TC._quiet_tau2_noise()
        assert fake["tau2.utils.display"].ConsoleDisplay.console == "loud-console"
    finally:
        TC._QUIETED = False


def test_tau_common_skips_when_no_domain_produces_results():
    from gbench.runners.eval_suites import tau_common as TC
    with patch.object(TC, "check_tau_env_prerequisites", return_value=(True, "")), \
         patch.object(TC, "_install_robustness_patches", lambda *a, **k: None), \
         patch.object(TC, "_configure_evaluator_llm", lambda *a, **k: None), \
         patch.object(TC, "_run_one_domain", side_effect=RuntimeError("simulator boom")):
        r = TC.run_tau_env("tau2", ["airline"], "m", "http://x", 2, None, "docs/evals/tau2.md")
    assert r["status"] == "skipped" and "docs/evals/tau2.md" in r["skip_reason"]
