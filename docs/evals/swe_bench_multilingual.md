# swe_bench_multilingual setup

Canonical SWE-bench Multilingual (`SWE-bench/SWE-bench_Multilingual`, 300 issues
across 9 non-Python languages) scored by execution-based **resolved rate** via the
**vanilla** swebench Docker harness (namespace `swebench`). The model emits one
unified-diff patch per issue; the harness applies it and runs
`FAIL_TO_PASS`/`PASS_TO_PASS` inside the prebuilt per-instance image.

## Requirements
- **Docker** (per-instance images pulled from DockerHub `swebench/...`).
- `swebench` (already in `pip install gbench[evals]`; vanilla 4.1.0 handles this
  dataset — no fork needed). `datasets` (base install).
- Host Node/toolchains are **not** needed — all language tests run inside the images.

If `swebench`/`datasets`/Docker are missing the suite **skips**.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals swe_bench_multilingual \
       --sandboxes 8 --eval-limit 20
```
First run pulls sizable images; use `--eval-limit` and `--sandboxes` to bound it
(`--sandboxes` → harness `--max_workers`). Patches truncate if `--max-output-tokens`
is small — keep it ≥ 8192.
