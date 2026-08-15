# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# Evaluation Suite: aider_polyglot
# Description: Aider Polyglot - 225 Exercism exercises across 6 languages, execution pass@1

"""gbench native built-in runner for aider_polyglot (Coding & Algorithmic Design).

Canonical Aider polyglot benchmark (Aider-AI/polyglot-benchmark): fill in each
exercise's stub solution so its hidden unit tests pass. The model returns whole
files; correctness is execution-based pass@1 (the exercise's test command exits 0).
SANDBOX_EVAL. Requires git + per-language toolchains; only languages whose
toolchain is present are loaded (others are logged and skipped), and the whole
suite skips cleanly if git is unavailable.
"""

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple
from .base import run_eval_suite

logger = logging.getLogger(__name__)

PILLAR = "Coding & Algorithmic Design"
DOCS_URL = "docs/evals/aider_polyglot.md"
_REPO_URL = "https://github.com/Aider-AI/polyglot-benchmark.git"
_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "gbench", "polyglot-benchmark")
_TEST_TIMEOUT = 180

# test-file suffix -> (toolchain probe cmd, builder for the test command)
_INSTRUCTIONS_ADDENDUM = (
    "####\n\nUse the above instructions to modify the supplied files: {file_list}\n"
    "Don't change the names of existing functions or classes, as they may be "
    "referenced from other code like unit tests, etc.\n"
    "Only use standard libraries, don't suggest installing any packages.\n\n"
    "Return the COMPLETE contents of each file you modify in a fenced code block "
    "whose info string is the file path, e.g.:\n"
    "```{first_file}\n<full file contents>\n```"
)


def _which(name: str) -> bool:
    return shutil.which(name) is not None


def _lang_toolchain_ok(lang: str) -> Tuple[bool, str]:
    if lang == "python":
        try:
            import pytest  # noqa: F401
            return True, ""
        except ImportError:
            return False, "pytest not installed"
    if lang == "go":
        return (_which("go"), "go not installed")
    if lang == "rust":
        return (_which("cargo"), "cargo not installed")
    if lang == "cpp":
        return (_which("cmake") and _which("make") and _which("g++"), "cmake/make/g++ not installed")
    if lang == "java":
        # Java exercises compile with javac and run their tests via `./gradlew test`,
        # so a JRE-only `java` (no javac) or a box without gradle would fail EVERY
        # task and report a misleading 0%. Require the full toolchain, else skip java.
        if not (_which("java") and _which("javac")):
            return False, "javac (JDK, not just JRE) not installed"
        if not _which("gradle"):
            return False, "gradle not installed (java tests run via ./gradlew test)"
        return True, ""
    if lang == "javascript":
        return (_which("node") and _which("npm"), "node/npm not installed")
    return False, f"unknown language {lang}"


def check_aider_polyglot_prerequisites() -> Tuple[bool, str]:
    if not _which("git"):
        return False, "git is not installed (needed to fetch the polyglot benchmark)."
    return True, ""


def _ensure_repo() -> str:
    """Shallow-clone the polyglot benchmark to the cache (bypassing insteadOf rewrites)."""
    if os.path.isdir(os.path.join(_CACHE_DIR, ".git")):
        return _CACHE_DIR
    os.makedirs(os.path.dirname(_CACHE_DIR), exist_ok=True)
    env = dict(os.environ, GIT_CONFIG_GLOBAL="/dev/null", GIT_CONFIG_SYSTEM="/dev/null")
    r = subprocess.run(
        ["git", "clone", "--depth", "1", _REPO_URL, _CACHE_DIR],
        capture_output=True, text=True, env=env,
    )
    if r.returncode != 0 or not os.path.isdir(_CACHE_DIR):
        raise RuntimeError(f"aider_polyglot: git clone failed: {r.stderr[-400:]}")
    return _CACHE_DIR


def _load_aider_polyglot_samples(
    enable_thinking: bool = False,
    limit: Optional[int] = None,
    langs: Optional[List[str]] = None,
) -> List[Tuple[List[Dict[str, Any]], Any, Dict[str, Any]]]:
    """Load Exercism exercises for languages whose toolchain is present; raises on clone/schema failure."""
    repo = _ensure_repo()
    all_langs = ["cpp", "go", "java", "javascript", "python", "rust"]
    langs = langs or all_langs

    available, skipped = [], []
    for lang in langs:
        ok, why = _lang_toolchain_ok(lang)
        (available if ok else skipped).append(lang if ok else f"{lang} ({why})")
    if skipped:
        logger.warning(f"aider_polyglot: skipping languages with missing toolchains: {skipped}")
    if not available:
        raise RuntimeError(f"aider_polyglot: no language toolchains available (skipped: {skipped})")

    samples = []
    for lang in available:
        practice_dir = os.path.join(repo, lang, "exercises", "practice")
        if not os.path.isdir(practice_dir):
            raise RuntimeError(f"aider_polyglot: missing exercises dir for '{lang}'")
        for slug in sorted(os.listdir(practice_dir)):
            ex_dir = os.path.join(practice_dir, slug)
            cfg_path = os.path.join(ex_dir, ".meta", "config.json")
            instr_path = os.path.join(ex_dir, ".docs", "instructions.md")
            if not os.path.isfile(cfg_path) or not os.path.isfile(instr_path):
                continue
            with open(cfg_path, encoding="utf-8") as f:
                cfg = json.load(f)
            files = cfg.get("files", {})
            solution_files = files.get("solution") or []
            test_files = files.get("test") or []
            if not solution_files or not test_files:
                raise RuntimeError(f"aider_polyglot: bad config.json for {lang}/{slug}")

            with open(instr_path, encoding="utf-8") as f:
                instructions = f.read()
            append_path = os.path.join(ex_dir, ".docs", "instructions.append.md")
            if os.path.isfile(append_path):
                with open(append_path, encoding="utf-8") as f:
                    instructions += "\n\n" + f.read()

            stub_blocks = []
            for sf in solution_files:
                sp = os.path.join(ex_dir, sf)
                stub = ""
                if os.path.isfile(sp):
                    with open(sp, encoding="utf-8") as f:
                        stub = f.read()
                stub_blocks.append(f"```{sf}\n{stub}\n```")
            file_list = ", ".join(solution_files)
            addendum = _INSTRUCTIONS_ADDENDUM.format(file_list=file_list, first_file=solution_files[0])
            prompt = (
                f"{instructions}\n\n{addendum}\n\n"
                f"Current contents of the file(s) to modify:\n\n" + "\n\n".join(stub_blocks)
            )
            gold = json.dumps({
                "lang": lang, "slug": slug,
                "solution_files": solution_files, "test_files": test_files,
                "exercise_dir": ex_dir,
            })
            messages = [{"role": "user", "content": prompt}]
            samples.append((messages, gold, {"category": lang, "exercise": slug}))

    if not samples:
        raise RuntimeError("aider_polyglot returned empty exercise set")
    if limit is not None and limit > 0:
        samples = samples[:limit]
    logger.info(f"Loaded {len(samples)} aider_polyglot exercises (langs: {available}).")
    return samples


def _extract_files(text: str, solution_files: List[str]) -> Dict[str, str]:
    """Map fenced code blocks in the response to the solution filenames.

    Models label the fence inconsistently - bare `filename`, `lang:filename`,
    `path/to/filename`, or a leading comment. We normalise the info string (drop a
    leading `lang:` qualifier and any comment markers) and match against each solution
    file by full path or basename. As a last resort, when the number of code blocks equals
    the number of solution files, map them positionally so language-only fences (```cpp)
    on multi-file exercises still land.
    """
    if not text:
        return {}
    blocks = re.findall(r"```([^\n`]*)\n([\s\S]*?)```", text)
    out: Dict[str, str] = {}

    def _norm_info(info: str) -> str:
        info = info.strip().lstrip("#/").strip()          # drop leading comment/path markers
        if ":" in info:                                   # "cpp:all_your_base.h" -> "all_your_base.h"
            info = info.split(":", 1)[1].strip()
        return info

    # 1) blocks whose (normalised) info string names a solution file
    for info, body in blocks:
        norm = _norm_info(info)
        for sf in solution_files:
            bn = os.path.basename(sf)
            if norm in (sf, bn) or norm.endswith("/" + bn):
                out.setdefault(sf, body)

    # 2) single solution file: fall back to the largest code block
    if not out and len(solution_files) == 1 and blocks:
        out[solution_files[0]] = max((b for _, b in blocks), key=len)

    # 3) positional fallback: unlabelled/language-only fences on a multi-file exercise
    #    (only when counts line up exactly, to avoid mis-mapping)
    if not out and len(blocks) == len(solution_files) and len(solution_files) > 1:
        for sf, (_, body) in zip(solution_files, blocks):
            out[sf] = body

    return out


def _run_exercise_tests(work_dir: str, lang: str, test_files: List[str]) -> bool:
    """Run the exercise's test command in work_dir; return True iff it exits 0."""
    suffix = os.path.splitext(test_files[0])[1]
    env = dict(os.environ)
    try:
        if suffix == ".py":
            cmd = [sys.executable, "-m", "pytest", "-q"]
        elif suffix == ".go":
            cmd = ["go", "test", "./..."]
        elif suffix == ".rs":
            cmd = ["cargo", "test", "--", "--include-ignored"]
        elif suffix == ".java":
            for tf in test_files:  # strip @Disabled so disabled cases still run
                tp = os.path.join(work_dir, tf)
                if os.path.isfile(tp):
                    with open(tp, encoding="utf-8") as f:
                        src = f.read()
                    src = re.sub(r"@Disabled\s*(\([^)]*\))?", "", src)
                    with open(tp, "w", encoding="utf-8") as f:
                        f.write(src)
            cmd = ["./gradlew", "test", "--no-daemon"]
        elif suffix == ".cpp":
            cmd = ["bash", "-c",
                   'mkdir -p build && cd build && cmake -G "Unix Makefiles" '
                   '-DEXERCISM_RUN_ALL_TESTS=1 .. && make']
        elif suffix in (".js", ".mjs"):
            for tf in test_files:  # un-skip xtest/xit
                tp = os.path.join(work_dir, tf)
                if os.path.isfile(tp):
                    with open(tp, encoding="utf-8") as f:
                        src = f.read()
                    src = src.replace("xtest(", "test(").replace("xit(", "it(")
                    with open(tp, "w", encoding="utf-8") as f:
                        f.write(src)
            subprocess.run(["npm", "install"], cwd=work_dir, capture_output=True,
                           text=True, env=env, timeout=_TEST_TIMEOUT)
            cmd = ["npm", "run", "test"]
        else:
            return False
        proc = subprocess.run(cmd, cwd=work_dir, capture_output=True, text=True,
                              env=env, timeout=_TEST_TIMEOUT)
        return proc.returncode == 0
    except Exception as e:
        logger.warning(f"aider_polyglot: test run failed in {work_dir} ({lang}): {e}")
        return False


def _make_scorer():
    async def _score(sample_traces: List[Dict[str, Any]]) -> None:
        import asyncio

        def _grade(tr: Dict[str, Any]) -> bool:
            try:
                g = json.loads(tr.get("gold_answer") or "{}")
            except Exception:
                return False
            solution_files = g.get("solution_files") or []
            files = _extract_files(tr.get("response_text") or "", solution_files)
            if not files:
                return False
            work = tempfile.mkdtemp(prefix="gbench_polyglot_")
            try:
                dst = os.path.join(work, g["slug"])
                shutil.copytree(g["exercise_dir"], dst)  # pristine tests + build files
                for sf, content in files.items():        # overwrite only solution files
                    fp = os.path.join(dst, sf)
                    os.makedirs(os.path.dirname(fp) or dst, exist_ok=True)
                    with open(fp, "w", encoding="utf-8") as f:
                        f.write(content)
                return _run_exercise_tests(dst, g["lang"], g["test_files"])
            finally:
                shutil.rmtree(work, ignore_errors=True)

        async def _grade_async(tr):
            tr["is_correct"] = await asyncio.to_thread(_grade, tr)
            tr["status"] = "OK"

        await asyncio.gather(*[_grade_async(t) for t in sample_traces])
    return _score


def run_aider_polyglot(
    model_name: str,
    base_url: str,
    concurrency: int = 1,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run Aider polyglot (execution pass@1) or skip if git is unavailable."""
    ok, reason = check_aider_polyglot_prerequisites()
    if not ok:
        msg = f"[SKIP] aider_polyglot skipped: {reason} See '{DOCS_URL}' for setup instructions."
        logger.warning(msg)
        print(f"\n{msg}")
        return {
            "benchmark_type": "eval",
            "eval_name": "aider_polyglot",
            "model_name": model_name,
            "status": "skipped",
            "total_questions": 0,
            "correct_answers": 0,
            "accuracy": 0.0,
            "skip_reason": f"{reason} (See {DOCS_URL})",
        }

    samples = _load_aider_polyglot_samples(
        enable_thinking=enable_thinking, limit=kwargs.get("limit")
    )
    return run_eval_suite(
        eval_name="aider_polyglot",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        async_eval_fn=_make_scorer(),
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens", 8192 if enable_thinking else 4096),
        temperature=kwargs.get("temperature", 0.0),
    )
