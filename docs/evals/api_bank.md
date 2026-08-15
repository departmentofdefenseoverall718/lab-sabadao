# api_bank

Canonical API-Bank (`liminghao1630/API-Bank`): multi-level API calling, tool retrieval,
and response synthesis. **Scoring: 100% deterministic parameter matching** of the
predicted API call against the gold call.

## Requirements
None beyond a running OpenAI-compatible `/v1` endpoint.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals api_bank --eval-limit 20
```
