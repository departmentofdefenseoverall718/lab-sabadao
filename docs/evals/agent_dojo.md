# agent_dojo

Canonical AgentDojo (`ffuuugor/agentdojo-dump`): security and prompt-injection defense
for tool-calling agents. The model faces tasks laced with injected malicious
instructions. **Scoring: the response is evaluated for resilience** — it must complete
the legitimate task without following the injected attack.

## Requirements
None beyond a running OpenAI-compatible `/v1` endpoint.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals agent_dojo --eval-limit 20
```
