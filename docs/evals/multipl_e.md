# multipl_e setup

Canonical MultiPL-E (`nuprl/MultiPL-E`, HumanEval family): execution-based
**pass@1**. Program = `prompt + completion + tests`, compiled/run per language;
a task passes iff its process exits 0.

> **Program assembly:** a chat model usually answers with the *whole* function, not a
> raw continuation. gbench detects a re-emitted signature and keeps only the prompt's
> preamble before it, so the assembled program does not contain two definitions of the
> same function (which used to compile-fail and score correct solutions wrong).

## Requirements (two paths)
- **Host toolchains (default, no image):** the suite loads only languages whose
  compiler is on `PATH` and skips the rest (logged, never scored 0). Present here:
  **C++** (`g++`), **Rust** (`rustc`), **Go** (`go`). Java needs a vendored
  `javatuples` jar; **JavaScript/TypeScript need Node** (not installed).
- **Full 23-language coverage (incl. JS/TS):** use the official container
  `docker pull ghcr.io/nuprl/multipl-e-evaluation:v3` and run it over the
  assembled programs (bundles every toolchain). Documented alternative; the
  in-process host path is the default.

Skips cleanly if no language toolchain is available.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals multipl_e \
       --sandboxes 16 --eval-limit 40
```
`category_accuracy` reports per-language pass@1. `--sandboxes` bounds concurrent
compile/run.
