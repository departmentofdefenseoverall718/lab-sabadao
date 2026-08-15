# scicode

Canonical SciCode (`SciCode1/SciCode`): research-level scientific computing problems
(physics, chemistry, math) requiring multi-step code. **Scoring: execution-based** —
the generated Python is run against the problem's unit tests in a subprocess; it passes
iff the tests pass.

## Test data (fetched automatically)

SciCode scores generated code against **per-problem reference outputs**, bound as `target`
from the benchmark's `test_data.h5` (`process_hdf5_to_tuple`). Without that file every test
raises `NameError`, so even a perfect solution scores 0 - a structural zero, not a model
result. gbench therefore **skips** the suite until `SCICODE_TEST_DATA` points at the file.

The file is ~1 GB and is *not* in the HF dataset (`SciCode1/SciCode` ships only
`problems_dev.jsonl` / `problems_test.jsonl`).

**You do not normally need to do anything**: gbench fetches it at eval time like any other
dataset, from the mirror below, and huggingface_hub caches it. The suite records the file's
`sha256` and where it came from under `test_data` on the result, and logs a warning that
the mirror is unverified against the canonical copy. Set `SCICODE_TEST_DATA` to use your
own file, or `SCICODE_TEST_DATA_REPO` to point at a different mirror. If neither the local
path nor the fetch works, the suite **skips** rather than reporting a structural 0%.

**Canonical source** - the SciCode repo README links a Google Drive folder, to be saved as
`./eval/data/test_data.h5`:
<https://drive.google.com/drive/folders/1W5GZW6_bdiDAiipuFMqdUhvUaHIj6-pR>
Google Drive folders are not scriptable without `gdown`/auth, so this is a manual download.

**Scriptable mirror** - a third-party copy on the Hub (1000.7 MiB, ~8.9k downloads).
Verify it against the canonical file before quoting a headline number:

```bash
hf download Srimadh/Scicode-test-data-h5 test_data.h5 \
    --repo-type dataset --local-dir ./scicode-data
```

Then point the suite at it:

```bash
export SCICODE_TEST_DATA=./scicode-data/test_data.h5
```

Doing this manually is only worth it if you want to pin a verified copy - the automatic
fetch pulls the same file.

## Requirements
- Executes generated code in a **local subprocess sandbox** — no Docker, but the host
  must allow spawning Python subprocesses. `--sandboxes` bounds concurrency.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals scicode --eval-thinking --sandboxes 8 --eval-limit 10
```
