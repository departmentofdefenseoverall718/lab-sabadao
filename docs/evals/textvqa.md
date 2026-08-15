# textvqa

Canonical TextVQA (`lmms-lab/textvqa`): visual question answering that requires reading
text (scene OCR) in natural images. Images are sent losslessly (PNG). **Scoring: the
predicted answer must match any gold answer (case-insensitive normalized matching).**

## Requirements
- A **vision-language model** endpoint.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals textvqa --eval-limit 20
```
