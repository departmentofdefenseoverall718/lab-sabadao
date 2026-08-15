# codeforces

Canonical Codeforces competitive programming (`open-r1/codeforces-cots`, balanced
across CF/ICPC/IOI). The model writes a full stdin/stdout program. **Scoring: the
generated Python code is executed against the problem's test cases in a bubblewrap
jail (read-only root, no network, strict timeout); it passes iff all tests pass.**
A problem with no usable tests is reported incorrect, not auto-passed.

> **Scope:** only the public sample I/O ships with `open-r1/codeforces-cots`, so hidden
> tests are not run and stdout must match exactly — problems with multiple valid
> outputs or float tolerances are under-credited. The corpus is also CoT-SFT data, so
> treat the score as contamination-prone.

## Requirements
- Runs generated code in a **local subprocess sandbox** — no Docker, but the host must
  allow spawning Python subprocesses. `--sandboxes` bounds concurrency.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals codeforces --eval-thinking --sandboxes 8 --eval-limit 10
```
