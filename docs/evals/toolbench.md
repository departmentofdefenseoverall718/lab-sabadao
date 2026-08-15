# toolbench

Canonical ToolBench (`Yhyu13/ToolBench_toolllama_G123_dfs`): large-scale real-world
REST API tool calling over 16,000+ APIs. **Scoring: the predicted tool call is matched
against the gold target.**

## Requirements
None beyond a running OpenAI-compatible `/v1` endpoint.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals toolbench --eval-limit 20
```
