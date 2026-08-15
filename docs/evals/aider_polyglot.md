# aider_polyglot setup

Canonical Aider polyglot benchmark
([Aider-AI/polyglot-benchmark](https://github.com/Aider-AI/polyglot-benchmark)):
225 Exercism exercises across C++, Go, Java, JavaScript, Python and Rust. The
model returns the complete solution file(s); correctness is execution-based
**pass@1** (the exercise's own unit tests exit 0).

## Requirements
- **git** (the benchmark is git-cloned to `~/.cache/gbench/polyglot-benchmark`).
- Per-language toolchains — only languages whose toolchain is present are loaded;
  the rest are logged and skipped:
  - Python → `pytest` (`pip install gbench[evals]`)
  - Go → `go`; Rust → `cargo`; C++ → `cmake` + `make` + `g++`
  - Java → `java` **+ `javac` + `gradle`** (tests run via `./gradlew test`; a JRE-only
    box or one without `gradle` would fail every Java task, so Java is skipped unless
    the full toolchain is present rather than scored a misleading 0%)
  - JavaScript → `node` + `npm` (**not installed by default**; the 49 JS exercises are skipped without it)

If `git` is unavailable the whole suite skips with `status="skipped"`.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals aider_polyglot \
       --sandboxes 16 --eval-limit 50
```
Each exercise's tests run in an isolated temp copy (pristine tests restored, only
the solution file overwritten), with a 180s timeout. `--sandboxes` bounds
concurrent test runs.
