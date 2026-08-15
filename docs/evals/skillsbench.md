# skillsbench setup

Canonical SkillsBench (`benchflow/skillsbench`): multi-skill agentic capability
evaluation across diverse tasks. Open-form responses are graded by an **LLM judge**
against per-task reference criteria.

## Requirements
- **`GEMINI_API_KEY`** — the judge model. The suite **skips** if it is unset.

## Run
```bash
export GEMINI_API_KEY="..."
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals skillsbench --eval-limit 20
```
`--max-output-tokens` ≥ 4096.
