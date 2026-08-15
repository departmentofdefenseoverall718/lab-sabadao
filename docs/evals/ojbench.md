# ojbench setup

Canonical OJBench (`He-Ren/OJBench_testdata`, 232 NOI/ICPC competitive-programming
problems) scored by **online-judge Pass@1**: the model emits a full stdin/stdout
program, judged Accepted iff **every** testcase passes within the per-problem
time/memory limits. Judging uses the official `ojbench` library over the DMOJ
sandbox — this is the only correct scorer (a substring/heuristic check cannot
verify program behaviour).

The dataset prompt already embeds the required response format, so it is sent
**verbatim**; the response has its thinking tags stripped before judging.

## Requirements (all must be present, else skip)
- **`ojbench`** — `git clone https://github.com/He-Ren/OJBench && pip install -e .`
- **DMOJ judge-server** — `git clone https://github.com/DMOJ/judge-server`
  (pin `f098cd3`) `&& pip install -e .`
- **PyPy3** on `PATH` (runs Python-language solutions) and **`g++`** (compiles C++).
- **Testdata** — set `OJBENCH_TESTDATA` to a snapshot of `He-Ren/OJBench_testdata`
  containing `NOI/` and `ICPC/` (~7.85 GB;
  `huggingface-cli download He-Ren/OJBench_testdata --repo-type dataset`).

These are heavy/system-level deps, so they are **not** in `gbench[evals]`; install
them out-of-band. The suite skips cleanly and prints the missing piece otherwise.

## Run
```bash
OJBENCH_TESTDATA=/data/ojbench/testdata \
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals ojbench \
       --sandboxes 8 --eval-limit 20
```
`ojbench.init([NOI, ICPC])` runs once per process; `--sandboxes` maps to the judge
`num_workers`. `category_accuracy` reports per `{dataset}_{difficulty}` (e.g.
`NOI_hard`). Use `--max-output-tokens` ≥ 8192 for full solutions.
