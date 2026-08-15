# swe_bench_live setup

Canonical SWE-bench-Live (`SWE-bench-Live/SWE-bench-Live`): the model emits a
unified-diff patch per GitHub issue and correctness is the execution-based
**resolved rate** (apply patch, run `FAIL_TO_PASS`/`PASS_TO_PASS`).

## Requirements
- **Docker** (per-instance images are pulled from DockerHub namespace `starryzhang`).
- `datasets` (in the base install).
- The **SWE-bench-Live fork** of the `swebench` harness — **in a separate virtualenv**.
  See below; this is not a plain `pip install`.

If any of the above is missing the suite skips with `status="skipped"`.

## Why this needs its own virtualenv

The fork and the upstream harness are the **same package name, `swebench`, at different
versions**, so only one can be installed at a time — and they support different suites.
Measured on the 8xA100 box, 2026-08-15, by building a `TestSpec` for one row of each
dataset:

| `swebench` version | `copilot_bench_swe` | `swe_bench_live` | `swe_bench_multilingual` |
| --- | --- | --- | --- |
| **4.1.0** (upstream, PyPI) | OK | FAIL `KeyError: 'aws-cloudformation/cfn-lint'` | **OK** |
| **4.0.3** (SWE-bench-Live fork) | OK | **OK** | FAIL `KeyError: 'parse_log_maven'` |

`pip install -e SWE-bench-Live` into the main environment therefore **silently trades
`swe_bench_multilingual` for `swe_bench_live`**. It also drags `fsspec` and `filelock`
backwards, and the latter violates `bfcl-eval`'s `filelock==3.20.0` pin.

The main gbench environment keeps **upstream 4.1.0**, so `swe_bench_multilingual` works and
`swe_bench_live` skips with an explanatory message. To measure SWE-bench-Live, build a
second environment and run only that suite from it:

```bash
# once
python -m venv ./swebench-live-env
./swebench-live-env/bin/pip install -e /path/to/SWE-bench-Live
./swebench-live-env/bin/pip install -e .

# per run - note the separate --results-dir so the two environments' results
# never land in the same tree
./swebench-live-env/bin/gbench \
    --evals-only --evals swe_bench_live \
    --remote-endpoint http://127.0.0.1:8000/v1 \
    --tokenizer google/gemma-4-26B-A4B-it \
    --eval-limit 20 --sandboxes 8 \
    --results-dir ./results/swe-bench-live
```

Both environments talk to the same model endpoint, so the model under test is identical;
only the scoring harness differs. Merge the two result trees when reporting, and say which
harness produced which number.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals swe_bench_live \
       --sandboxes 8 --eval-limit 50
```
Splits: `lite` (default, 300), `test` (1000), `verified` (500), `full`. Building
and running the Docker images is time- and disk-intensive; use `--eval-limit` and
`--sandboxes` to bound it. `--sandboxes` maps to the harness `--max_workers`.
