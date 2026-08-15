# livebench setup

Canonical LiveBench (`livebench/*` configs, e.g. `livebench/math`, `livebench/reasoning`):
a contamination-free benchmark that is continuously refreshed so questions post-date
model training. Open-form answers are graded by an **LLM judge** against the gold
answer per task category.

## Requirements
- **`GEMINI_API_KEY`** — the judge model. The suite **skips** if it is unset.

## Run
```bash
export GEMINI_API_KEY="..."
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals livebench --eval-thinking --eval-limit 20
```
`category_accuracy` reports per LiveBench task category. `--max-output-tokens` ≥ 2048.
