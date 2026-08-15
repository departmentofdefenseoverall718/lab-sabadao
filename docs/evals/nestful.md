# nestful

Canonical NESTFUL (`ibm-research/nestful`): nested function calling with dependent
parameters — the output of one call feeds the arguments of the next (a call DAG).
**Scoring: the predicted call sequence is matched against the gold nested-call
specification.**

## Requirements
None beyond a running OpenAI-compatible `/v1` endpoint.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals nestful --eval-limit 20
```
