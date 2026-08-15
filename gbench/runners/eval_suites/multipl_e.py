# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# Evaluation Suite: multipl_e
# Description: MultiPL-E - execution-based pass@1 across languages (HumanEval family)

"""gbench native built-in runner for multipl_e (Polyglot Coding & Software).

Canonical MultiPL-E (nuprl/MultiPL-E): translate-and-execute HumanEval. Program =
prompt + completion + tests, compiled/run per language; pass@1 = fraction with
status OK. This runner uses HOST toolchains (like aider_polyglot) and loads only
languages whose toolchain is present (others are logged + skipped, never scored
0). For the full 23-language set incl. JS/TS, use the official container
(ghcr.io/nuprl/multipl-e-evaluation) - see docs. SANDBOX_EVAL.
"""

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Tuple
from .sampling import stratified_sample
from .base import run_eval_suite, strip_thinking_tags
from .sandbox import run_sandboxed

logger = logging.getLogger(__name__)

PILLAR = "Coding & Software Engineering"
DOCS_URL = "docs/evals/multipl_e.md"

# MultiPL-E config language code -> host toolchain probe (single-file compile+run).
_LANG_TOOLCHAINS = {
    "cpp": lambda: shutil.which("g++"),
    "rs": lambda: shutil.which("rustc"),
    "go": lambda: shutil.which("go"),
    # java needs the javatuples jar (not vendored); js/ts need node -> use the container.
}


def _available_langs(requested: Optional[List[str]] = None) -> Tuple[List[str], List[str]]:
    langs = requested or list(_LANG_TOOLCHAINS.keys())
    avail = [l for l in langs if l in _LANG_TOOLCHAINS and _LANG_TOOLCHAINS[l]()]
    skipped = [l for l in langs if l not in avail]
    return avail, skipped


def check_multipl_e_prerequisites() -> Tuple[bool, str]:
    try:
        import datasets  # noqa: F401
    except ImportError:
        return False, "Python package 'datasets' is not installed."
    avail, _ = _available_langs()
    if not avail:
        return False, ("no MultiPL-E language toolchain available (need g++/rustc/go, or use "
                       "the ghcr.io/nuprl/multipl-e-evaluation container for all languages).")
    return True, ""


def _stop_at_stop_token(text: str, stop_tokens: List[str]) -> str:
    idxs = [text.find(t) for t in (stop_tokens or []) if t and text.find(t) != -1]
    return text[:min(idxs)] if idxs else text


def _extract_code(response_text: str) -> str:
    """Largest fenced code block, else the raw text."""
    blocks = re.findall(r"```[A-Za-z0-9+#._-]*\n([\s\S]*?)```", response_text or "")
    return max(blocks, key=len) if blocks else (response_text or "")


def _declaration_lines(prompt: str) -> List[str]:
    """Lines of the prompt that declare the target function.

    MultiPL-E prompts end in a signature - `def f(x: int) -> int:` in Python, `int f(int x) {`
    in C-like languages - which is exactly what a chat model repeats when it answers with a
    complete function.
    """
    out = []
    for line in (prompt or "").splitlines():
        s = line.strip()
        if "(" in s and (s.endswith(":") or s.endswith("{") or s.endswith(")")):
            out.append(s)
    return out


def _assemble_program(response_text: str, prompt: str, tests: str, stop_tokens: List[str]) -> str:
    """Canonical MultiPL-E assembly = prompt + completion + tests (completion = continuation).

    The re-emission check used to require the ENTIRE prompt as a verbatim substring of the
    response. A chat model asked to "complete the function" answers with the function -
    signature and body, but not the prompt's docstring/comments byte-for-byte - so the
    check failed, the whole definition was appended to the prompt, and the program had two
    definitions of the same function: a compile error scored as a wrong answer.

    Now a shared *declaration line* is enough to recognise a complete definition; the
    prompt's preamble (imports, helper types) is kept and its signature is not repeated.
    """
    code = _extract_code(strip_thinking_tags(response_text or ""))
    ps = prompt.strip()
    if ps and ps in code:  # model re-emitted the prompt verbatim -> keep the continuation
        completion = _stop_at_stop_token(code.split(ps, 1)[1], stop_tokens)
        return prompt + completion + "\n" + tests

    for decl in reversed(_declaration_lines(prompt)):
        index = code.find(decl)
        if index == -1:
            continue
        # The response contains the whole definition: keep only what precedes the
        # signature in the prompt, then the model's definition.
        preamble = prompt.split(decl, 1)[0] if decl in prompt else ""
        body = _stop_at_stop_token(code[index:], stop_tokens)
        return preamble + body + "\n" + tests

    completion = _stop_at_stop_token(code, stop_tokens)
    return prompt + completion + "\n" + tests


def _run_program(program: str, lang: str) -> str:
    """Compile+run one MultiPL-E program; return OK/SyntaxError/Exception/Timeout.

    Compilation and execution are isolated (audit CC5): the toolchain and the resulting
    binary are model-controlled input running on the host. Only `tmp` is writable.
    """
    tmp = tempfile.mkdtemp(prefix="gbench_mpe_")
    try:
        if lang == "cpp":
            src, binp = os.path.join(tmp, "prob.cpp"), os.path.join(tmp, "prob")
            with open(src, "w", encoding="utf-8") as f:
                f.write(program)
            c = run_sandboxed(["g++", src, "-o", binp, "-std=c++17"], writable=[tmp],
                              capture_output=True, text=True, timeout=60)
            if c.returncode != 0:
                return "SyntaxError"
            r = run_sandboxed([binp], writable=[tmp], capture_output=True, timeout=15)
            return "OK" if r.returncode == 0 else "Exception"
        if lang == "rs":
            src, binp = os.path.join(tmp, "prob.rs"), os.path.join(tmp, "prob")
            with open(src, "w", encoding="utf-8") as f:
                f.write(program)
            c = run_sandboxed(["rustc", src, "-o", binp, "--edition", "2021"], writable=[tmp],
                              capture_output=True, text=True, timeout=90)
            if c.returncode != 0:
                return "SyntaxError"
            r = run_sandboxed([binp], writable=[tmp], capture_output=True, timeout=15)
            return "OK" if r.returncode == 0 else "Exception"
        if lang == "go":
            with open(os.path.join(tmp, "go.mod"), "w", encoding="utf-8") as f:
                f.write("module prob\n\ngo 1.21\n")
            with open(os.path.join(tmp, "main_test.go"), "w", encoding="utf-8") as f:
                f.write(program)
            # Under a read-only root the toolchain cannot use its default caches in $HOME,
            # so point them inside the one writable directory.
            caches = {k: os.path.join(tmp, v) for k, v in
                      (("GOCACHE", ".gocache"), ("GOPATH", ".gopath"), ("GOMODCACHE", ".gomod"))}
            for path in caches.values():
                os.makedirs(path, exist_ok=True)
            env = dict(os.environ, GOPROXY="off", GOFLAGS="-mod=mod", **caches)
            r = run_sandboxed(["go", "test", "./..."], writable=[tmp], cwd=tmp, env=env,
                              capture_output=True, text=True, timeout=90)
            out = (r.stdout or "") + (r.stderr or "")
            if "build failed" in out or "setup failed" in out:
                return "SyntaxError"
            return "OK" if r.returncode == 0 else "Exception"
        return "SyntaxError"
    except subprocess.TimeoutExpired:
        return "Timeout"
    except Exception as e:
        logger.warning(f"multipl_e: run error ({lang}): {e}")
        return "Exception"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _load_multipl_e_samples(
    langs: Optional[List[str]] = None,
    benchmark: str = "humaneval",
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], Any, Dict[str, Any]]]:
    """Load MultiPL-E (HumanEval family) for available-toolchain languages; raises on failure."""
    avail, skipped = _available_langs(langs)
    if skipped:
        logger.warning(f"multipl_e: skipping languages without a host toolchain: {skipped} "
                       "(use the nuprl/multipl-e-evaluation container for full coverage).")
    if not avail:
        raise RuntimeError("multipl_e: no available language toolchain")

    from datasets import load_dataset
    samples = []
    for lang in avail:
        cfg = f"{benchmark}-{lang}"
        try:
            ds = load_dataset("nuprl/MultiPL-E", cfg, split="test")
        except Exception as e:
            raise RuntimeError(f"Could not load nuprl/MultiPL-E '{cfg}': {e}") from e
        for item in ds:
            name = item.get("name")
            prompt = item.get("prompt")
            tests = item.get("tests")
            if not name or not prompt or not tests:
                raise RuntimeError(
                    f"multipl_e: unexpected schema for {cfg} (name/prompt/tests); "
                    "refusing to fabricate sample data"
                )
            gold = json.dumps({
                "name": name, "language": lang, "prompt": prompt,
                "tests": tests, "stop_tokens": item.get("stop_tokens") or [],
            })
            content = (
                f"Complete the following {lang} function. Return ONLY the completed "
                f"function in a single fenced code block, no explanation.\n\n"
                f"```{lang}\n{prompt}\n```"
            )
            samples.append(([{"role": "user", "content": content}], gold, {"category": lang, "name": name}))

    if limit is not None and limit > 0:
        # Stratify by language: the loop above concatenates one language at a time, so a head is ONE language. Wrongly exempted from the sampling
        # contract test as a "per-language loop" - it loops to BUILD, then takes a head, so
        # `--eval-limit 20` measured a single language and reported it as MultiPL-E.
        samples = stratified_sample(
            samples, limit,
            lambda x: (x[2] or {}).get("category") if len(x) > 2 and isinstance(x[2], dict) else None,
            seed="multipl_e")
    logger.info(f"Loaded {len(samples)} multipl_e samples (langs: {avail}).")
    return samples


def _make_scorer():
    async def _score(sample_traces: List[Dict[str, Any]]) -> None:
        import asyncio

        def _grade(tr: Dict[str, Any]) -> bool:
            try:
                g = json.loads(tr.get("gold_answer") or "{}")
            except Exception:
                return False
            program = _assemble_program(
                tr.get("response_text") or "", g.get("prompt", ""),
                g.get("tests", ""), g.get("stop_tokens") or [])
            return _run_program(program, g.get("language", "")) == "OK"

        async def _grade_async(tr):
            tr["is_correct"] = await asyncio.to_thread(_grade, tr)
            tr["status"] = "OK"

        await asyncio.gather(*[_grade_async(t) for t in sample_traces])
    return _score


def run_multipl_e(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run MultiPL-E execution pass@1 (or skip if no language toolchain)."""
    ok, reason = check_multipl_e_prerequisites()
    if not ok:
        msg = f"[SKIP] multipl_e skipped: {reason} See '{DOCS_URL}' for setup instructions."
        logger.warning(msg)
        print(f"\n{msg}")
        return {
            "benchmark_type": "eval",
            "eval_name": "multipl_e",
            "model_name": model_name,
            "status": "skipped",
            "total_questions": 0,
            "correct_answers": 0,
            "accuracy": 0.0,
            "skip_reason": f"{reason} (See {DOCS_URL})",
        }
    samples = _load_multipl_e_samples(limit=kwargs.get("limit"))
    return run_eval_suite(
        eval_name="multipl_e",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        async_eval_fn=_make_scorer(),
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens", 2048),
        temperature=kwargs.get("temperature", 0.0),
    )
