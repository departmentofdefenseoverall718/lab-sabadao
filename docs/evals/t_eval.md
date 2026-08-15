# t_eval

Canonical T-Eval (`lovesnowbest/T-Eval`): step-by-step tool-usage evaluation —
instruction following, planning, reasoning, retrieval, understanding, and review of
tool interactions. **Scoring: the response is matched against the gold tool-usage
target** for the step.

## Requirements
None beyond a running OpenAI-compatible `/v1` endpoint.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals t_eval --eval-limit 20
```
