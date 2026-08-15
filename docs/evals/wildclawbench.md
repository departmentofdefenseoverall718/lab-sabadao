# wildclawbench setup

Canonical WildClawBench (`internlm/WildClawBench`): "wild" multi-turn agentic
tool-use scenarios drawn from realistic interactions. Open-form agent behavior is
graded by an **LLM judge** against the reference expectations.

## Requirements
- **`GEMINI_API_KEY`** — the judge model. The suite **skips** if it is unset.

## Run
```bash
export GEMINI_API_KEY="..."
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals wildclawbench --eval-limit 20
```
`--max-output-tokens` ≥ 4096.
