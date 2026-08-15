# browsecomp

Canonical BrowseComp (`smolagents/browse_comp`, OpenAI): hard web-browsing/research
questions. The dataset stores `problem` and `answer` XOR-encrypted with a per-row
`canary` password; gbench decrypts them canonically before use. **Scoring: the
canonical BrowseComp LLM grader** decides correct (yes/no) against the decrypted gold
answer.

> **Closed book:** gbench gives the model no search tool and no retrieval corpus, so it answers BrowseComp questions from parametric memory alone. The score is a lower bound and is NOT comparable with a published number produced by a browsing/retrieval agent.

## Requirements
- **`GEMINI_API_KEY`** — the canonical grader model is required (the eval calls the
  judge to decide correctness). Set it before running.

## Run
```bash
export GEMINI_API_KEY="..."
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals browsecomp --eval-limit 20
```
