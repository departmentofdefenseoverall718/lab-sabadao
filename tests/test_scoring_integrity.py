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

"""Contract: a reported number must say how it was produced, and knobs must arrive.

Three defects from the 2026-08-15 sweep are locked in here:

* RC-5 - seven suites quietly swapped the canonical LLM judge for a substring match when
  `GEMINI_API_KEY` was missing and reported the result as the benchmark's metric.
* RC-2 - `--temperature` was forwarded by 15 of 122 suites; the other 107 dropped it, so
  the run's headline knob silently did nothing.
* P1-7 - long-answer suites hardcoded 2048-8192 output tokens and were scored on truncated
  patches (swe_bench_multilingual 10/20, codeforces 5/6, copilot_bench_swe 5/20).
"""

import re
from pathlib import Path
from unittest import mock

import pytest

from gbench.runners.eval_suites import base

SUITE_DIR = Path(__file__).resolve().parent.parent / "gbench" / "runners" / "eval_suites"

#: Suites with a documented no-judge fallback path (they degrade to substring matching).
FALLBACK_SUITES = ["aa_lcr", "beam_128k", "frames", "simpleqa",
                   "cybergym", "skillsbench", "wildclawbench", "cimemories"]


# --------------------------------------------------------------------------- #
# RC-5: a fallback must never be presented as canonical
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("suite", FALLBACK_SUITES)
def test_fallback_branch_marks_itself(suite):
    """Every no-key branch tags its traces, so the aggregate can tell them apart."""
    src = (SUITE_DIR / f"{suite}.py").read_text(encoding="utf-8")
    assert 'GEMINI_API_KEY' in src, f"{suite}: expected a judge-availability check"
    assert 'trace["scoring_mode"] = "judge_fallback"' in src, (
        f"{suite}: the no-judge branch scores with a substring match but does not mark "
        f'the traces. Set trace["scoring_mode"] = "judge_fallback" so the result cannot '
        f"claim to be the canonical metric.")


def _run_with_traces(traces, **kw):
    """Drive run_eval_suite over canned replies and return its result dict."""
    samples = [([{"role": "user", "content": "q"}], "gold", {}) for _ in traces]

    async def _fake_send(*a, **k):
        return base.Reply(text="answer", tool_calls=None, finish_reason="stop")

    def _apply(sample_traces):
        for t, patch in zip(sample_traces, traces):
            t.update(patch)

    async def _async_eval(sample_traces):
        _apply(sample_traces)

    with mock.patch.object(base, "_send_single_request", _fake_send):
        return base.run_eval_suite(
            eval_name="unit_test_suite", model_name="m", base_url="http://x",
            concurrency=1, samples=samples, async_eval_fn=_async_eval, **kw)


def test_result_reports_judge_fallback_when_any_trace_fell_back():
    res = _run_with_traces([
        {"is_correct": True, "scoring_mode": "judge_fallback"},
        {"is_correct": True},
    ])
    assert res["scoring_mode"] == "judge_fallback", (
        "one fallback-scored sample is enough: the suite's number is a lower bound, "
        "not the canonical metric")


def test_result_reports_judge_when_nothing_fell_back():
    res = _run_with_traces([{"is_correct": True}, {"is_correct": False}])
    assert res["scoring_mode"] == "judge"


def test_result_reports_deterministic_without_a_judge():
    samples = [([{"role": "user", "content": "q"}], "answer", {})]

    async def _fake_send(*a, **k):
        return base.Reply(text="answer", tool_calls=None, finish_reason="stop")

    with mock.patch.object(base, "_send_single_request", _fake_send):
        res = base.run_eval_suite(
            eval_name="unit_test_suite", model_name="m", base_url="http://x",
            concurrency=1, samples=samples, eval_fn=lambda r, g: r == g)
    assert res["scoring_mode"] == "deterministic"


def test_fallback_is_logged_loudly(caplog):
    with caplog.at_level("WARNING"):
        _run_with_traces([{"is_correct": True, "scoring_mode": "judge_fallback"}])
    assert any("fallback" in r.message.lower() or "fallback" in str(r.args).lower()
               for r in caplog.records), "the downgrade must be visible in the run log"


def test_cimemories_has_no_length_proxy():
    """`len(resp) >= 30` scored 20/20 including an answer about a different person."""
    src = (SUITE_DIR / "cimemories.py").read_text(encoding="utf-8")
    code = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
    assert not re.search(r"len\(\s*resp\w*\s*\)\s*[<>]=?\s*\d", code), (
        "cimemories must not grade response length; contextual integrity needs a judge")
    assert "_async_judge_cimemories" in src


# --------------------------------------------------------------------------- #
# RC-2: run-level knobs must reach every suite
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _clear_knobs():
    base._RUN_KNOBS.clear()
    yield
    base._RUN_KNOBS.clear()


def _capture_payload(**suite_kw):
    """Run one sample and return the payload run_eval_suite would have sent."""
    seen = {}
    samples = [([{"role": "user", "content": "q"}], "gold", {})]

    async def _fake_send(**k):
        seen.update(k)
        return base.Reply(text="x", tool_calls=None, finish_reason="stop")

    with mock.patch.object(base, "_send_single_request", _fake_send):
        base.run_eval_suite(
            eval_name=suite_kw.pop("eval_name", "unit_test_suite"),
            model_name="m", base_url="http://x", concurrency=1, samples=samples,
            eval_fn=lambda r, g: True, **suite_kw)
    return seen


def test_temperature_knob_reaches_a_suite_that_never_forwarded_it():
    base.set_run_knobs(temperature=0.7)
    seen = _capture_payload()
    assert seen.get("temperature") == 0.7, (
        "a suite that does not forward --temperature must still receive it from the "
        "run-level knobs (audit RC-2)")


def test_suite_temperature_wins_over_the_knob():
    """An explicit per-suite temperature is a measurement decision; do not override it."""
    base.set_run_knobs(temperature=0.7)
    seen = _capture_payload(temperature=0.0)
    assert seen.get("temperature") == 0.0


def test_set_run_knobs_ignores_none():
    base.set_run_knobs(temperature=0.7)
    base.set_run_knobs(temperature=None)
    assert base.get_run_knob("temperature") == 0.7, (
        "an unset CLI flag arrives as None and must not erase a knob")


# --------------------------------------------------------------------------- #
# P1-7: long-answer suites get an output-token floor
# --------------------------------------------------------------------------- #
def test_floor_raises_a_hardcoded_suite_default():
    """multipl_e hardcodes 2048; a program does not fit in 2048 tokens."""
    seen = _capture_payload(eval_name="multipl_e", max_output_tokens=2048)
    got = seen.get("max_output_tokens")
    assert got == base.SUITE_MIN_OUTPUT_TOKENS["multipl_e"], (
        f"expected the floor {base.SUITE_MIN_OUTPUT_TOKENS['multipl_e']}, got {got}: a "
        "suite-hardcoded default must not undercut the floor, or the truncation stays")


def test_explicit_operator_knob_beats_the_floor():
    base.set_run_knobs(max_output_tokens=512)
    seen = _capture_payload(eval_name="multipl_e", max_output_tokens=2048)
    got = seen.get("max_output_tokens")
    assert got == 512, "--max-output-tokens is an explicit choice and always wins"


def test_floor_does_not_lower_a_generous_suite():
    seen = _capture_payload(eval_name="multipl_e", max_output_tokens=65536)
    got = seen.get("max_output_tokens")
    assert got == 65536


def test_every_floor_names_a_real_suite():
    names = {p.stem for p in SUITE_DIR.glob("*.py")}
    unknown = sorted(set(base.SUITE_MIN_OUTPUT_TOKENS) - names)
    assert not unknown, f"floor table names suites that do not exist: {unknown}"


def test_truncation_prone_suites_are_covered():
    """The suites the sweep measured truncating must all carry a floor."""
    measured = ["swe_bench_multilingual", "codeforces", "copilot_bench_swe"]
    missing = [s for s in measured if s not in base.SUITE_MIN_OUTPUT_TOKENS]
    assert not missing, f"measured truncation but no output floor: {missing}"


# --------------------------------------------------------------------------- #
# P1-5: no dead knobs. Every eval flag reaches a suite, or says whose it is.
# --------------------------------------------------------------------------- #
#: flag -> (kwargs key, who consumes it). A knob nothing reads is a lie in --help.
EVAL_KNOBS = {
    "--eval-thinking": ("enable_thinking", None),          # every suite takes it
    "--max-output-tokens": ("max_output_tokens", None),    # resolved centrally in base
    "--temperature": ("temperature", None),                # resolved centrally in base
    "--eval-limit": ("limit", None),
    "--eval-n-shot": ("eval_n_shot", "MMLU-Pro"),
    "--eval-categories": ("eval_categories", "BFCL"),
    "--eval-max-soft-tokens": ("eval_max_soft_tokens", "vision"),
    "--sandboxes": ("sandboxes", "sandboxed"),
    "--suite-timeout": ("suite_timeout", None),
}


def _cli_source():
    return (Path(__file__).resolve().parent.parent / "gbench" / "cli.py").read_text("utf-8")


@pytest.mark.parametrize("flag,spec", sorted(EVAL_KNOBS.items()))
def test_every_eval_knob_is_actually_consumed(flag, spec):
    key, _ = spec
    cli, runner = _cli_source(), (
        Path(__file__).resolve().parent.parent / "gbench" / "runners" / "evals.py"
    ).read_text("utf-8")
    assert f'"{flag}"' in cli, f"{flag} is documented here but no longer exists"

    if key in ("temperature", "max_output_tokens"):
        # Not forwarded by most suites; base.run_eval_suite resolves them from the
        # run knobs, which is the whole point of the RC-2 fix.
        base_src = (SUITE_DIR / "base.py").read_text("utf-8")
        assert f'get_run_knob("{key}"' in base_src
        assert "set_run_knobs(" in runner
        return
    if key == "suite_timeout":
        assert "suite_timeout" in runner
        return
    assert f'"{key}"' in runner or f"'{key}'" in runner, (
        f"{flag} is parsed but never put in the suite kwargs")

    if key in ("eval_n_shot", "eval_categories", "eval_max_soft_tokens"):
        consumers = [p.stem for p in SUITE_DIR.glob("*.py")
                     if key in p.read_text("utf-8")]
        assert consumers, f"{flag} reaches no suite at all"


@pytest.mark.parametrize("flag,spec", sorted(
    (f, s) for f, s in EVAL_KNOBS.items() if s[1]))
def test_narrow_knobs_say_so_in_help(flag, spec):
    """A flag only one suite reads must not read as global in --help."""
    _, who = spec
    src = _cli_source()
    start = src.index(f'"{flag}"')
    block = src[start:start + 700]
    assert who.lower() in block.lower(), (
        f"{flag} is only honoured by {who}; --help must say so or operators will assume "
        f"it applied to the whole run")


# --------------------------------------------------------------------------- #
# --skip-existing must stay OFF by default
# --------------------------------------------------------------------------- #
def _parse(*argv):
    import sys
    from gbench.cli import create_parser
    old = sys.argv
    sys.argv = ["gbench", *argv]
    try:
        return create_parser().parse_args()
    finally:
        sys.argv = old


def test_skip_existing_is_off_unless_asked_for():
    """Reusing a cached result silently mixes two methodologies.

    It defaulted to True, so a run that changed how a suite is scored would quietly
    report the *old* number for every suite that already had a file anywhere under
    --results-dir. Resuming is a deliberate act, not the default.
    """
    assert _parse("--evals", "all").skip_existing is False


def test_skip_existing_can_still_be_requested():
    assert _parse("--evals", "all", "--skip-existing").skip_existing is True
    assert _parse("--evals", "all", "--no-skip-existing").skip_existing is False
    # last flag wins, so an alias pair cannot leave it ambiguous
    assert _parse("--evals", "all", "--skip-existing", "--no-skip-existing").skip_existing is False


def test_skip_existing_help_warns_about_mixing():
    src = _cli_source()
    start = src.index('"--skip-existing"')
    block = src[start:start + 700]
    assert "default" not in block.lower() or "off by default" in block.lower(), (
        "the help text still advertises the old default")
    assert "resum" in block.lower(), (
        "--help must say what --skip-existing is for, or operators will treat it as a "
        "speed knob and reuse stale-methodology results")


# --------------------------------------------------------------------------- #
# context-window fit: clamp before sending, and say why when it cannot fit
# --------------------------------------------------------------------------- #
def test_generous_budget_fits_a_short_prompt_untouched():
    base.set_run_knobs(max_model_len=262144)
    assert base.clamp_to_context([{"role": "user", "content": "hi"}], 65536) == (65536, None)


def test_long_prompt_shrinks_the_budget_instead_of_400ing():
    base.set_run_knobs(max_model_len=262144)
    got, why = base.clamp_to_context([{"role": "user", "content": "x" * 600_000}], 65536)
    assert why is None
    assert 0 < got < 65536
    assert got + 600_000 / 3.0 <= 262144


def test_prompt_larger_than_the_window_is_reported_not_retried():
    """mrcr sent six 1.3-4.2 MB prompts into a 262144 window: three silent retries each,
    recorded as `request_failed` with `error: None`."""
    base.set_run_knobs(max_model_len=262144)
    _, why = base.clamp_to_context([{"role": "user", "content": "x" * 4_252_036}], 65536)
    assert why and "context window" in why and "1,417,356" in why


def test_clamp_is_a_noop_without_a_known_window():
    base._RUN_KNOBS.clear()
    assert base.clamp_to_context([{"role": "user", "content": "x" * 10_000}], 8192) == (8192, None)


def test_budget_floors_cover_every_suite_that_truncated():
    """All 16 suites the 2026-08-15 sweep truncated must now carry a floor above 8192."""
    measured = ["aime", "arc_agi", "codeforces", "copilot_bench_swe", "culer", "gpqa",
                "gpqa_diamond", "hmmt", "humanitys_last_exam", "ifeval", "imo_answer_bench",
                "livebench", "loft_x_arxiv", "new_amc_aime", "putnam", "swe_bench_multilingual"]
    missing = [s for s in measured if base.SUITE_MIN_OUTPUT_TOKENS.get(s, 0) <= 8192]
    assert not missing, f"still at or below the old default: {missing}"


def test_patch_suites_get_the_largest_budget():
    for s in ("swe_bench_multilingual", "copilot_bench_swe", "swe_bench_pro"):
        assert base.SUITE_MIN_OUTPUT_TOKENS[s] >= 65536, s


def test_global_default_was_raised():
    import inspect
    src = inspect.getsource(base._run_suite_async)
    assert "32768 if thinking else 16384" in src


# --------------------------------------------------------------------------- #
# request timeout must scale with the token budget, and be visible when it fires
# --------------------------------------------------------------------------- #
def test_timeout_scales_with_the_budget():
    """A fixed 1200 s was fine at 8192 tokens and drops the longest answers at 65536."""
    t8, t64 = base.request_timeout_s(8192), base.request_timeout_s(65536)
    assert t64 > t8 > 0
    assert t64 >= 65536 / base.MIN_DECODE_TOK_S
    assert base.request_timeout_s(None) >= base.REQUEST_TIMEOUT_FLOOR_S


def test_timeout_never_drops_below_the_floor():
    assert base.request_timeout_s(1) >= base.REQUEST_TIMEOUT_FLOOR_S


def test_a_timeout_is_classified_apart_from_other_failures():
    """Otherwise a dropped long answer is indistinguishable from a dead endpoint."""
    assert base.classify_reply(base.Reply(None, None, None, None, "timeout after 9392s")) \
        == "request_timeout"
    assert base.classify_reply(base.Reply(None, None, None, None, "OSError: refused")) \
        == "request_failed"
    assert base.classify_reply(base.Reply(None, None, None, None)) == "request_failed"


def test_reply_error_field_is_optional_for_existing_callers():
    r = base.Reply(text="x", tool_calls=None, finish_reason="stop")
    assert r.error is None and base.classify_reply(r) == "ok"


def test_timeouts_are_counted_and_warned_about(caplog):
    from unittest import mock

    async def _timing_out(**k):
        return base.Reply(None, None, None, None, "timeout after 9392s")

    samples = [([{"role": "user", "content": "q"}], "g", {}) for _ in range(2)]
    with mock.patch.object(base, "_send_single_request", _timing_out), \
         caplog.at_level("WARNING"):
        res = base.run_eval_suite(eval_name="unit_test_suite", model_name="m",
                                  base_url="http://x", concurrency=1, samples=samples,
                                  eval_fn=lambda r, g: True)
    assert res["timed_out_requests"] == 2
    assert res["request_timeout_s"] > 0
    assert all(t["status"] == "TIMEOUT" for t in res["sample_traces"])
    assert all("timeout" in (t["error"] or "") for t in res["sample_traces"])
    assert any("timed out" in r.message.lower() for r in caplog.records)
