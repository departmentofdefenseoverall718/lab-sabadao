# putnam_formal setup

Canonical PutnamBench formal track (`amitayusht/PutnamBench`): Putnam competition
problems evaluated as **machine-checked Lean 4 proofs**. The model emits a Lean 4
proof term, which is compiled and verified inside a Docker sandbox — a problem is
correct iff the Lean kernel accepts the proof (no heuristic or LLM judge is involved).
For the informal, LLM-judged track, see [putnam](putnam.md). SANDBOX_EVAL.

## Requirements
- **Docker** — the proof is verified in the `leanprovercommunity/lean4:latest`
  container. The image is **auto-pulled on first run**; the suite **skips** cleanly if
  the Docker daemon is unreachable.
- No `GEMINI_API_KEY` needed — verification is done by the Lean kernel, not a judge.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals putnam_formal \
       --sandboxes 8 --eval-limit 10
```
`--sandboxes` bounds concurrent Lean verification containers. `--max-output-tokens`
should be generous for full proof terms.
