# spider2 setup

Canonical Spider 2.0-lite (`xlangai/spider2-lite`) enterprise text-to-SQL scored
by **execution accuracy**: the model's SQL is run against the target database and
its result set is compared to the gold with the official `evaluate_utils.py`
semantics — column-subset containment (every required gold column must be matched
by *some* predicted column), numeric tolerance (`abs_tol=1e-2`), and
order-insensitive row comparison; multi-gold instances pass if any accepted
variant matches.

Only the **`local*` (SQLite) subset** runs offline. The 103 BigQuery and 93
Snowflake instances need live cloud accounts/credentials and are **excluded** (not
scored 0). The reported `total_questions` is the scorable local subset — check the
run log line `Loaded N spider2 local(SQLite) samples`.

## Requirements
- `pandas` + `sqlite3` (base install). No cloud SDKs needed for the local subset.
- **Gold** — set `SPIDER2_GOLD_DIR` to a checkout of
  `xlang-ai/Spider2/spider2-lite/evaluation_suite/gold` (must contain
  `spider2lite_eval.jsonl` and `exec_result/`).
- **Local DBs** — set `SPIDER2_LOCALDB_DIR` to the unzipped `spider2-localdb`
  directory of `*.sqlite` files.

Skips cleanly if either directory is missing. Each SQLite DB is copied into an
in-memory connection before the query runs, so the on-disk files are never mutated.

## Run
```bash
SPIDER2_GOLD_DIR=/data/spider2/gold SPIDER2_LOCALDB_DIR=/data/spider2/localdb \
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals spider2 \
       --sandboxes 8 --eval-limit 20
```
The model must return SQL in a ```sql fenced block. `--sandboxes` bounds concurrent
query execution; each query is capped at 120 s.
