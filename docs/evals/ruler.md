# ruler

Canonical RULER (`rayonlabs/ruler-all`): a high-entropy long-context matrix spanning
retrieval, multi-hop tracing, and aggregation across context lengths (4k, 8k, 16k,
32k, 64k, 128k). **Scoring: mean per-item needle recall** — multi-output tasks get partial credit for
the fraction of required outputs present, as canonical RULER does; `all_needles_rate`
reports the stricter all-or-nothing figure. `category_accuracy` is keyed
`<task>@<length band>`, and `length_bands` on the result lists the bands actually
present — `rayonlabs/ruler-all` does **not** reach 128k, so check it before comparing
with a published RULER number.

## Requirements
- A serving endpoint supporting the tested **context lengths** (up to 128k).

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals ruler --eval-limit 20
```
