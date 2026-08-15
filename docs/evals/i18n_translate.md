# i18n_translate

Canonical multilingual machine-translation quality (`wmt/wmt19`): the model translates
source sentences across WMT19 language pairs. **Scoring: mean chrF** (character n-gram F-score, `sacrebleu` when installed) against
the reference; `pass_rate_chrf40` reports the share of segments at chrF >= 40.

> **Scope:** into-English only (de/zh/ru/cs -> en). This does not measure generation
> *into* those languages.

## Requirements
None beyond a running OpenAI-compatible `/v1` endpoint.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals i18n_translate --eval-limit 20
```
