# deepsearch_qa

Canonical DeepSearchQA (`xbench/DeepSearch-2510`): multi-step deep web-search research
questions with XOR-protected evidence/answers (decoded canonically before use).
**Scoring: the response is matched against the exact factual research answer.**

> **Closed book:** gbench gives the model no search tool and no retrieval corpus, so it answers DeepSearchQA questions from parametric memory alone. The score is a lower bound and is NOT comparable with a published number produced by a browsing/retrieval agent.

## Requirements
None beyond a running OpenAI-compatible `/v1` endpoint.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals deepsearch_qa --eval-limit 20
```
