# Evaluation plugins

`gbench` discovers custom evaluation suites dynamically from a plugin directory, allowing new suites to be added modularly without modifying the core codebase:

```bash
gbench --evals-only --evals all \
       --eval-plugins-dir /path/to/plugins \
       --remote-endpoint http://127.0.0.1:8000/v1 --tokenizer <model>
```

`--evals all` includes every discovered plugin (or use `--evals plugins` to run only plugins, or specify plugin names individually). Each plugin module exposes `run_<plugin_name>(model_name, base_url, concurrency, enable_thinking=False, **kwargs)` and returns the standard `gbench` result dictionary.

## Datasets
A plugin reads `"$GBENCH_DATA_DIR"/<plugin_name>.jsonl` (default `$HOME/gbench-data` or `./data`). Supported shapes:

| Shape | Recognised by | Notes |
|---|---|---|
| Chat messages | `messages: [...]` | OpenAI chat format |
| Prompt | `prompt` (+ optional `instruction_id_list` / `test_suite`) | IFEval and sandbox-executed variants |
| Structured textproto | `content` | Roles parsed from message blocks |
| CSV | Header row | Case-insensitive column mapping |

## Scoring
Scoring is deterministic where possible (sandbox execution, IFEval rules, function calling AST match); otherwise an **LLM judge** is used with defined evaluation schemas:

- **`violation`** — Safety/policy rubrics. Correct = *no* violation, so reported accuracy equals `1 - violation_rate`.
- **`binary`** — Correct = the rubric is satisfied.
- **`sxs`** — Side-by-side comparison against a baseline response; correct = candidate wins.

Suites whose expected output is a set of tool calls are scored **structurally** with an exact FunctionCall-AST match (`fc_common.score_exact_call_set`): matching tool name, argument keys, equivalent values, and parameter types.

## Prompt Diversity
Plugin results report prompt diversity statistics:

| Field | Meaning |
|---|---|
| `distinct_prompts` / `effective_n` | Number of distinct user prompts evaluated |
| `modal_prompt_share` | Fraction of rows sharing the single most common prompt |
| `low_diversity` | `true` when ≥10 rows contain fewer than 20 distinct prompts |

If a dataset replicates identical prompts across candidate versions, `effective_n` reflects the unique question count and a warning is surfaced in the summary report.

## Skip Handling
Missing prerequisites produce explicit skips (`skipped_result`) rather than failing or reporting synthetic scores:

| Skip Reason | Remedy |
|---|---|
| Dataset file not found / empty | Stage `<plugin_name>.jsonl` in `$GBENCH_DATA_DIR` |
| Judged suite without `GEMINI_API_KEY` | Set `GEMINI_API_KEY` in the environment |
| Missing compiler/sandbox toolchain | Install required runtime toolchain (see [`docs/evals/toolchains.md`](toolchains.md)) |

## Environment Variables
- `GBENCH_DATA_DIR` — Custom dataset directory path.
- `GEMINI_API_KEY` — API key for LLM judge evaluation.
- `GBENCH_JUDGE_MODEL` — Model override for the judge (default: `gemini-3.6-flash`).
- `GBENCH_RUBRICS_DIR` — Directory path containing custom rubric definitions.
