# complexfuncbench

Canonical ComplexFuncBench (`THUDM/ComplexFuncBench`): multi-axis complex-parameter
function calling (nested/constrained arguments, multi-step). **Scoring: the predicted
call is matched against the expected function call** (name + parameters).

## Requirements
None beyond a running OpenAI-compatible `/v1` endpoint.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals complexfuncbench --eval-limit 20
```
