# gaia setup

Canonical GAIA (General AI Assistants, `gaia-benchmark/GAIA`, 2023 validation, 165
tasks) scored with the official `question_scorer` (number/list/string normalized
exact-match on the model's `FINAL ANSWER:` line).

## Requirements
- The dataset is **gated**: log in to HF, accept the license at
  <https://huggingface.co/datasets/gaia-benchmark/GAIA>, and export `HF_TOKEN`
  (or `HUGGING_FACE_HUB_TOKEN`). Without it the suite **skips**.
- No other deps (the scorer is stdlib).

## Run
```bash
HF_TOKEN=hf_xxx gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals gaia
```

## Caveat
gbench runs GAIA **closed-book** — there is no tool/web loop, and ~half of the
validation tasks require a file attachment that is not passed to the model. Scores
will therefore be low; this suite validates the **canonical scorer**, not agentic
capability. A per-`Level` breakdown is reported. Full GAIA capability requires an
agent scaffold with tools, which is out of scope for the single-turn HTTP harness.
