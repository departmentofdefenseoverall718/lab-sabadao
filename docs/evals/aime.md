# aime

Canonical AIME (`AI-MO/aimo-validation-aime`): American Invitational Mathematics
Examination problems whose answer is an integer 0–999. **Scoring: extract the final
integer from the response and match the gold integer** — `\boxed{}` first, then an
explicit answer anchor, then the last number in the response.

> **Scope:** this dataset is AIME **2022–2024**, not 2025. The problems pre-date the
> training cutoff of most current models, so treat the score as potentially contaminated
> and not comparable with an AIME-2025 number.

## Requirements
None beyond a running OpenAI-compatible `/v1` endpoint.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals aime --eval-thinking --eval-limit 20
```
`--eval-thinking` strongly recommended; `--max-output-tokens` ≥ 8192 for full working.
