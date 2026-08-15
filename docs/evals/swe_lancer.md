# swe_lancer setup

SWE-Lancer scored by **execution**: the
model's patch is applied inside the task's Docker image and the hidden end-to-end
(Playwright/pytest) test suite is run — an IC-SWE task resolves iff those tests pass
(SWE-Manager tasks are scored by the correct proposal selection). Only the official
harness can verify this, so the earlier substring/filename heuristic was **removed**
(a substring match on a patch measures nothing).

> **Dataset provenance:** the prompts are loaded from the community mirror
> **`DCAgent2/swe-lancer`**, while the scoring harness is a clone of
> `openai/SWELancer-Benchmark`. If the two number their tasks differently every
> lookup misses; gbench detects zero task-id overlap and reports an explicit
> `task_id mismatch` error rather than a 0% resolved rate.

## Requirements
- **Docker** + the Python **`docker` SDK** (`pip install gbench[evals]`).
- **Harness checkout** — set `SWELANCER_HARNESS_DIR` to a clone of
  `openai/SWELancer-Benchmark`.
- **Runner** — gbench invokes, by default,
  `python $SWELANCER_HARNESS_DIR/run_swelancer_eval.py --predictions=<preds.jsonl>
  --output_dir=<out> --num_workers=<N>`. If your harness exposes a different
  entrypoint (e.g. the `nanoeval` runner), set **`SWELANCER_EVAL_CMD`** to a command
  template using the placeholders `{harness} {predictions} {output_dir}
  {num_workers}`. gbench then parses the first JSON report under `<out>` that is
  either `{"resolved_ids": [...]}` or a flat `{task_id: bool}` map.
- **Explicit opt-in** — set `SWELANCER_RUN=1`. Real runs pull very large per-task
  images, so the suite is **gated** and skips even when Docker + harness are present.

Skips cleanly (with the missing piece) when any of the above is absent.

## Run
```bash
SWELANCER_HARNESS_DIR=/opt/SWELancer-Benchmark SWELANCER_RUN=1 \
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals swe_lancer \
       --sandboxes 4 --eval-limit 5
```
Predictions are keyed by `task_id`; `--sandboxes` maps to `--num_workers`; the result
carries a `swe_lancer_report` (total / resolved). The model must return a single
```diff patch. Start small — image pulls dominate wall-clock.
