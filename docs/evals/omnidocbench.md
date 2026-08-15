# omnidocbench

Canonical OmniDocBench v1.5 (`opendatalab/OmniDocBench`): multimodal document parsing —
layout, tables, and LaTeX formula recognition. **Scoring: mean normalized edit distance** against the page's annotations (canonical;
lower is better). `accuracy` carries the derived similarity `100 - mean_edit_distance
* 100`, `mean_edit_distance` the raw figure, and `strict_pass_rate` the share of pages
transcribed within 10% edit distance. Ground truth includes tables (`html`) and
formulas (`latex`), not just text, and pages are paired to annotations by filename.

## Requirements
- A **vision-language model** endpoint.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals omnidocbench --eval-limit 20
```
