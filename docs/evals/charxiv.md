# charxiv

Canonical CharXiv (`princeton-nlp/CharXiv`): complex academic chart reasoning and
scientific plot reading. **Scoring: the response is matched against the exact chart-
reasoning gold answer.**

## Requirements
- A **vision-language model** endpoint.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals charxiv --eval-limit 20
```
