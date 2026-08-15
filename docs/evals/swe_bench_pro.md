# swe_bench_pro setup

Canonical SWE-bench Pro (`ScaleAI/SWE-bench_Pro`, public test split) scored by
**resolved rate**: the model's unified diff is applied inside the task's Docker
container, the `fail_to_pass` / `pass_to_pass` test suites are run, and an instance
resolves iff all required tests pass. Scoring uses the **Pro-specific harness**
(`scaleapi/SWE-bench_Pro-os`) with the `jefzda/sweap-images` container set — vanilla
`swebench` cannot score Pro (different container tags and test-command schema).

The harness reads the reference tests from its own `swe_bench_pro_full.csv`
(`--raw_sample_path`), so gbench sends it only `{instance_id, patch, prefix}` — the
model's patch is looked up by `instance_id`.

## Requirements
- **Docker** + the Python **`docker` SDK** (`pip install gbench[evals]`).
- **Harness checkout** — set `SWE_BENCH_PRO_HARNESS_DIR` to a clone of
  `scaleapi/SWE-bench_Pro-os` (must contain `swe_bench_pro_eval.py` and `run_scripts/`).
- **Raw-sample table** — nothing to do. The harness README references
  `swe_bench_pro_full.csv` but the repo does not ship it, so gbench generates it on first
  use from the canonical HF dataset (`ScaleAI/SWE-bench_Pro`, test split, 731 instances),
  which already carries every column the harness reads. It is written into the harness
  directory, or `$TMPDIR` if that is read-only. Point `SWE_BENCH_PRO_RAW_SAMPLE` at your
  own copy to override.
- **Explicit opt-in** — set `SWE_BENCH_PRO_RUN=1`. A real run pulls tens–hundreds of
  GB of per-instance images, so the suite is **gated**: without the flag it skips
  even when Docker and the harness are present.

Skips cleanly (with the missing piece) when any of the above is absent.

## Run
```bash
SWE_BENCH_PRO_HARNESS_DIR=/opt/SWE-bench_Pro-os SWE_BENCH_PRO_RUN=1 \
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals swe_bench_pro \
       --sandboxes 4 --eval-limit 5
```
gbench invokes `swe_bench_pro_eval.py … --dockerhub_username=jefzda
--use_local_docker`, parses `eval_results.json`, and maps each `instance_id` to
resolved/unresolved. `--sandboxes` maps to `--num_workers`; the run result carries a
`swe_bench_pro_report` (total / resolved). The model must return a single
```diff patch. Start small (`--eval-limit`) — image pulls dominate wall-clock.
