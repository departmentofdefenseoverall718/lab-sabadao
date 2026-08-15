# hmmt

Canonical HMMT (`MathArena/hmmt_feb_2025`): Harvard-MIT Mathematics Tournament
competition problems. **Scoring: 100% deterministic math-answer matching** of the
extracted final answer against gold.

## Requirements
None beyond a running OpenAI-compatible `/v1` endpoint.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals hmmt --eval-thinking --eval-limit 20
```
`--eval-thinking` recommended; `--max-output-tokens` generous for working.
