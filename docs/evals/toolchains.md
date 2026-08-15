# Optional toolchains

Most suites need nothing beyond `pip install gbench`. The ones below execute code,
drive containers, or wrap a third-party harness, and **skip cleanly** when their
prerequisite is missing — the skip reason names the exact missing piece and points here.
Nothing is scored on a partial environment.

Install only what you intend to run. `gbench --list evals` names every suite;
`gbench --dry-run --evals <name>` shows what a run would execute without generating.

## Python extras

| Install | Needed by |
| --- | --- |
| `pip install gbench[evals]` | pulls `docker`, `swebench`, `datasets`, `pandas`, `rapidfuzz`, `jsonschema` — covers most of this table |
| `pip install bfcl-eval` | `bfcl_v4_agentic` (the canonical Berkeley harness; **not** the unrelated `bfcl` package) |
| `pip install multi-swe-bench` | `multi_swe_bench` |
| `pip install harbor` (or `uv tool install harbor`) | `terminal_bench` |
| `pip install rank-bm25` | `tau3` banking_knowledge RAG domain |
| `pip install audioop-lts` | `tau2` / `tau3` on Python 3.13 (stdlib `audioop` was removed) |
| `pip install pycocoevalcap` | `coco_caption` (CIDEr/SPICE; without it the suite skips rather than substituting BLEU) |
| `pip install sacrebleu` | `i18n_translate` chrF |
| `pip install rapidfuzz` | `omnidocbench` edit distance |
| `pip install playwright && playwright install chromium` | `lmarena_web_agent` |
| `pip install mcp jsonschema` | `mcp_bench` |
| `pip install semgrep` + `CYBERSECEVAL_ICD_RULES` (absolute path to PurpleLlama's `CodeShield/insecure_code_detector/rules`) | `cyberseceval`: 62/351 rows score without it, 317/351 with it — see [cyberseceval.md](cyberseceval.md) |
| `pip install google-genai` | every LLM-judged suite (also needs `GEMINI_API_KEY`) |
| `pip install -e ./tau2-bench --no-deps` | `tau2`, `tau3` — or set `TAU2_BENCH_SRC` |
| **separate venv** + `pip install -e SWE-bench-Live` | `swe_bench_live`. **Not** a plain install: it is `swebench` at an older version and would break `swe_bench_multilingual`. See [swe_bench_live.md](swe_bench_live.md) |
| `git clone He-Ren/OJBench && pip install -e .` + DMOJ `judge-server@f098cd3` | `ojbench` |

## System toolchains

| Install | Needed by |
| --- | --- |
| Docker CLI + running daemon | `swe_bench_*`, `multi_swe_bench`, `swe_lancer`, `bigcodebench`, `terminal_bench`, `putnam_formal` |
| `g++` | `multipl_e`, `aider_polyglot`, `ojbench` |
| `go` | `multipl_e`, `aider_polyglot` |
| `rustc` / `cargo` | `multipl_e`, `aider_polyglot` |
| `node` + `npm` | `aider_polyglot` (JavaScript tasks) |
| `javac` + `gradle` | `aider_polyglot` (Java tasks) |
| `git` | `aider_polyglot` |
| `pypy3` | `ojbench` |
| `bubblewrap` (`bwrap`) | recommended for every code-executing suite — see below |

A missing language toolchain does not fail `multipl_e` or `aider_polyglot`: the
languages you *can* run still run, and the result records which were skipped. Compare
only like-for-like language sets across runs.

## Sandboxing

Suites that execute model-written code run it through `bwrap` when available
(`apt install bubblewrap`). `GBENCH_SANDBOX` selects the policy:

- `auto` (default) — use `bwrap` if it works on this host, otherwise run unsandboxed
- `bwrap` — require it; suites skip if it is unavailable
- `none` — never sandbox

On a shared or untrusted host, set `GBENCH_SANDBOX=bwrap` so a missing sandbox becomes a
skip instead of unconfined execution.

## Harnesses and gates

| Variable | Purpose |
| --- | --- |
| `SWE_BENCH_PRO_HARNESS_DIR` | clone of `scaleapi/SWE-bench_Pro-os` (`swe_bench_pro_eval.py`, `run_scripts/`). The raw-sample CSV is generated for you — see [swe_bench_pro.md](swe_bench_pro.md) |
| `SWE_BENCH_PRO_RUN=1` | opt-in; a real run pulls tens-hundreds of GB of images |
| `SCICODE_TEST_DATA` | path to SciCode's ~1 GB `test_data.h5` reference outputs — see [scicode.md](scicode.md). Without it the suite skips rather than reporting a structural 0% |
| `SWELANCER_HARNESS_DIR` | clone of `openai/SWELancer-Benchmark` (+ `SWELANCER_RUN=1`) |
| `TAU2_ENV_RUN=1` | opt-in for the tau2/tau3 environment suites |
| `BFCL_PROJECT_ROOT` | where `bfcl-eval` keeps generations and scores |
| `GEMINI_API_KEY` | required by every judged suite; they skip without it rather than falling back to substring matching |
| `SERPAPI_API_KEY` | canonical web-search backend for `bfcl_v4_agentic`; otherwise Gemini grounding is used and the result says so |

## Checking what is available

There is no separate probe command: run the suite. Anything whose prerequisite is missing
returns `status: skipped` with the exact missing piece and a docs pointer, costs no
generation, and is reported as **skipped — never as 0%**. Read the run summary's skip
list and install from the tables above.
