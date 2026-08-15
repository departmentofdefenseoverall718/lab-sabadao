# acebench

Canonical ACEBench (`chenchen0103/ACEBench`, via the MIT-licensed HF reformatting
`oliveirabruno01/acebench`): agentic function-calling evaluation. The model must emit
the correct tool call(s) for each task. **Scoring: the ACEBench checker compares the
predicted call(s) against the gold call specification.**

## Requirements
None beyond a running OpenAI-compatible `/v1` endpoint.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals acebench --eval-thinking --eval-limit 20
```
