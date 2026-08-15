# lmsys_noncoding_hard

Canonical LMSYS / WildBench hard non-coding subset (`WildEval/WildBench`): complex,
multi-turn human prompts that are not coding tasks. **Scoring: the response is checked
against the item's checklist constraints and for substantive length/coverage.**

## Requirements
None beyond a running OpenAI-compatible `/v1` endpoint.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals lmsys_noncoding_hard --eval-limit 20
```
