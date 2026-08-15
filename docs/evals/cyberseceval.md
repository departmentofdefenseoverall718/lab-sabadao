# cyberseceval

Canonical CyberSecEval (`walledai/CyberSecEval`, Meta): cybersecurity risk and
safe-behavior evaluation. The model is prompted with security-sensitive scenarios and
**scored against the benchmark's security guidelines** (safe/compliant vs unsafe
response). Defensive-security evaluation.

## Scoring: the Insecure Code Detector

CyberSecEval asks the model to write code, then checks whether the **generated code**
contains the row's target weakness. Meta's Insecure Code Detector picks a detector per row
from the dataset's `analyzer` column; gbench implements both:

| analyzer | rows (python/instruct) | needs |
| --- | ---: | --- |
| `regex` | 62 of 351 | nothing — the rule ships inline in the dataset |
| `semgrep` | 255 of 351 | `pip install semgrep` **and** `CYBERSECEVAL_ICD_RULES` |
| *(no public rule)* | 34 of 351 | the `sql_injection` rows exist only in Meta's internal ruleset — always excluded |

A row scores **correct when the detector finds no instance of its CWE** in the generated
program (prose around the code block is ignored).

Rows whose detector is unavailable are **excluded** from the run, never guessed at. The
result carries `detector_backends`, `semgrep_rows_scored` and — when semgrep is not
configured — a `coverage_note` saying the number is over the regex subset rather than the
full benchmark. With no analysable rows at all the suite skips.

To score the semgrep rows too:

```bash
git clone https://github.com/meta-llama/PurpleLlama.git
pip install semgrep
# NOTE: the detector lives under CodeShield/, not CybersecurityBenchmarks/, in current
# PurpleLlama. Use an ABSOLUTE path - semgrep runs in a subprocess with its own cwd.
export CYBERSECEVAL_ICD_RULES=$PWD/PurpleLlama/CodeShield/insecure_code_detector/rules
```

With the rules configured, **317 of 351** rows are analysable (62 regex + 255 semgrep).
gbench points semgrep at the per-language subtree (`rules/semgrep/<language>`), never the
whole tree: `rules/regex/*.yaml` is a different format and semgrep would reject the run and
scan nothing, which reads as "no weakness found" for every response.

## Requirements
A running OpenAI-compatible `/v1` endpoint. The regex-analyzer rows need nothing
further; the semgrep rows need the two items above.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals cyberseceval --eval-limit 20
```
