# putnam setup

Canonical PutnamBench (`amitayusht/PutnamBench`): collegiate William Lowell Putnam
Mathematical Competition problems evaluated as **informal** proofs. The model writes a
natural-language solution/proof.

**Scoring** splits on what the reference actually contains:
- a reference stating a *single* closed-form value (one `$...$` span, or a bare value)
  is compared deterministically — the answer expression is extracted from both sides
  and must match after LaTeX normalization. An explicitly anchored answer
  (`\boxed{}` / `Final Answer:`) is held to exact equality; an unanchored one is
  searched for the value with word boundaries, so `2` is not found inside `12`.
- everything else (a prose reference such as "The limit does not exist.", a case split,
  or a pure-proof problem with no reference) goes to the **LLM proof judge**, which
  receives the reference when the dataset has one. For the
machine-checked Lean 4 track, see [putnam_formal](putnam_formal.md).

## Requirements
- **`GEMINI_API_KEY`** — the proof-grading judge. The suite **skips** if it is unset
  (or pass `--gemini-api-key`).

## Run
```bash
export GEMINI_API_KEY="..."
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals putnam --eval-thinking --eval-limit 10
```
`--eval-thinking` recommended (multi-step proofs). `--max-output-tokens` ≥ 8192.
