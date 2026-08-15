# gdpval — not scoreable against a plain text endpoint (skips)

Canonical GDPval (`openai/gdpval`): economically-valuable knowledge-work tasks whose
answer is a **produced file deliverable** (Excel `.xlsx`, PowerPoint `.pptx`, Word
`.docx`, etc.). Grading is done against **file-property rubrics** — e.g. "[+2] the
deliverable is a `.xlsx` workbook", "[+2] the presentation contains exactly five
slides", "[+1] one slide is a title slide".

## Status: skips (no misleading score)
A plain text `/v1` endpoint returns prose ("I can't produce the file, but here's the
methodology…") — it cannot emit an actual `.xlsx`/`.pptx`, and the deterministic
rubric check cannot credit prose against file-property criteria. Scoring anyway yields
a misleading ~0% (observed: 0.45%). So `gdpval` reports **`status: "skipped"`** with
that reason rather than a fake number.

## What full support requires
1. A **file-producing agent harness** (tool use / code execution that writes the actual
   deliverable files), and
2. a **rubric LLM judge** that grades the produced artifact against the GDPval rubric.

Until that harness exists, `gdpval` skips. The canonical loader and rubric parser
(`_load_gdpval_samples`, `_eval_gdpval`) are retained in the suite for that future
harness. (The `gdpval_one_sided_v4` **plugin** is a separate, text-gradeable variant and
is unaffected.)
