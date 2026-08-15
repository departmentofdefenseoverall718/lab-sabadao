# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the 7 new native evaluation suites in gbench.

The sample loaders pull their datasets from HF Hub. That download is not
exercised here, for the reason given at the top of .github/workflows/test.yml:
these tests never contact an external service, so the suite stays
deterministic and safe to require for merge. `datasets.load_dataset` is
patched with fixed rows instead, which lets each loader's own transformation
be asserted directly rather than only its row count.
"""

import base64
import io
from unittest import mock

import pytest
try:
    from PIL import Image
except ImportError:
    Image = None

from gbench.runners.eval_suites.mmlu_pro import _eval_mmlu_pro, _load_mmlu_pro_samples
from gbench.runners.eval_suites.aime import _eval_aime, _load_aime_samples
from gbench.runners.eval_suites.lcb import _verify_single_lcb_sample, _load_lcb_samples
from gbench.runners.eval_suites.codeforces import _eval_codeforces, _load_codeforces_samples
from gbench.runners.eval_suites.putnam import _eval_putnam, _load_putnam_samples
from gbench.runners.eval_suites.ruler import _eval_ruler, _load_ruler_samples
from gbench.runners.eval_suites.textvqa import _eval_textvqa, _load_textvqa_samples


def _patch_hf(rows_by_dataset):
    """Serve fixed rows in place of an HF Hub download.

    Every loader does `from datasets import load_dataset` inside its own body,
    so patching the attribute on the package is enough and nothing has to be
    imported before the loader runs. Requesting a dataset that the test did not
    supply raises, which is what catches a renamed or mistyped dataset id.
    """

    def _fake_load_dataset(path, *args, **kwargs):
        if path not in rows_by_dataset:
            raise AssertionError(f"loader asked for an unexpected dataset: {path}")
        return rows_by_dataset[path]

    return mock.patch("datasets.load_dataset", autospec=True, side_effect=_fake_load_dataset)


MMLU_PRO_ROWS = [
    # answer_index rather than a letter, which the loader has to map through
    # OPTION_LETTERS.
    {"question": "What is 2 + 2?", "options": ["3", "4", "5"], "answer_index": 1, "category": "math"},
    # answer already a letter, which the loader has to pass through untouched.
    {"question": "Which gas do plants absorb?", "options": ["Oxygen", "Carbon dioxide"], "answer": "B", "category": "biology"},
]

AIME_ROWS = [
    # answer arrives as an int and the scorer compares strings.
    {"problem": "Find the remainder.", "answer": 496, "category": "algebra"},
    {"problem": "Find n.", "answer": 84, "category": "number_theory"},
]

PUTNAM_ROWS = [
    # tags as a repr'd list, which the loader parses with ast.literal_eval.
    {"informal_statement": "Prove the identity.", "informal_solution": "1/2", "tags": "['algebra', 'number_theory']"},
    # tags as a bare string, which takes the other branch.
    {"informal_statement": "Show the bound.", "informal_solution": "100", "tags": "geometry"},
]

RULER_ROWS = [
    {"input": "Find the hidden words.", "outputs": ["appliance", "meter"], "length": 4096, "task": "niah_single"},
    {"input": "Track the variables.", "outputs": ["7"], "length": 8192, "task": "vt"},
]


def _rgb_png_bytes(b64_str):
    """Decode a base64 data URL payload back into a PIL image."""
    return Image.open(io.BytesIO(base64.b64decode(b64_str)))


# --------------------------------------------------------------------------
# Scoring functions. These take model output and a gold answer and are pure.
# --------------------------------------------------------------------------


def test_mmlu_pro_eval_fn():
    assert _eval_mmlu_pro("Answer: (B)", "B") is True
    assert _eval_mmlu_pro("Let's think... ANSWER: (J)", "J") is True
    assert _eval_mmlu_pro("Answer: (C)", "B") is False


def test_aime_eval_fn():
    assert _eval_aime("The final answer is 496.", "496") is True
    assert _eval_aime("We calculate that the final answer is 084", "84") is True
    assert _eval_aime("The final answer is 42", "43") is False


def test_lcb_eval_fn():
    # LiveCodeBench is execution-scored via _verify_single_lcb_sample(resp, gold_payload).
    # gold_payload is either a bare list of assert strings (-> generation) or a dict
    # tagged with task in {generation, execution, test_gen}. Cover each canonical path.

    # 1) generation via string assertions (bare-list payload)
    valid_code = "```python\ndef two_sum(nums, target):\n    return [0, 1]\n```"
    asserts = ["assert two_sum([2, 7, 11, 15], 9) == [0, 1]"]
    assert _verify_single_lcb_sample(valid_code, asserts) is True
    invalid_code = "```python\ndef two_sum(nums, target):\n    return [1, 2]\n```"
    assert _verify_single_lcb_sample(invalid_code, asserts) is False

    # 2) generation via a functional test case (fn_name/args/output)
    add_code = "```python\ndef add(a, b):\n    return a + b\n```"
    func_gold = {"task": "generation",
                 "tests": [{"testtype": "functional", "fn_name": "add", "args": [2, 3], "output": "5"}]}
    assert _verify_single_lcb_sample(add_code, func_gold) is True
    bad_add = "```python\ndef add(a, b):\n    return a + b + 1\n```"
    assert _verify_single_lcb_sample(bad_add, func_gold) is False

    # 3) generation via a stdin test case (input on stdin, expected on stdout)
    stdin_code = "```python\nimport sys\na, b = map(int, sys.stdin.read().split())\nprint(a + b)\n```"
    stdin_gold = {"task": "generation",
                  "tests": [{"testtype": "stdin", "input": "2 3", "output": "5"}]}
    assert _verify_single_lcb_sample(stdin_code, stdin_gold) is True
    assert _verify_single_lcb_sample(stdin_code, {"task": "generation",
        "tests": [{"testtype": "stdin", "input": "2 3", "output": "6"}]}) is False

    # 4) code execution: the model's STATED answer must match; a bare substring must NOT
    #    pass (that leniency inflated the category).
    exec_gold = {"task": "execution", "expected": "42"}
    assert _verify_single_lcb_sample("Final Output: 42", exec_gold) is True
    assert _verify_single_lcb_sample("42", exec_gold) is True          # whole-response value
    assert _verify_single_lcb_sample("Final Output: 7", exec_gold) is False
    # rambling that merely *contains* 42 without stating it as the answer -> incorrect
    assert _verify_single_lcb_sample("at step 42 we compute 7 so the result is 7", exec_gold) is False

    # 5) test generation: NOT credited by the lightweight scorer (canonical metric needs
    #    running the generated tests against a reference solution). Must not fake-pass.
    tg_gold = {"task": "test_gen", "expected": "assert f() == 1"}
    assert _verify_single_lcb_sample("Test: input=f(), output=1", tg_gold) is False
    assert _verify_single_lcb_sample("no", tg_gold) is False

    # empty response is always incorrect
    assert _verify_single_lcb_sample("", asserts) is False


def test_codeforces_eval_fn():
    script = (
        "```python\n"
        "import sys\n"
        "for line in sys.stdin:\n"
        "    a, b = map(int, line.split())\n"
        "    print(a + b)\n"
        "```"
    )
    tests = [{"stdin": "3 5\n", "stdout": "8"}, {"stdin": "-10 20\n", "stdout": "10"}]
    assert _eval_codeforces(script, tests) is True


def test_putnam_eval_fn():
    assert _eval_putnam("We conclude that Final Answer: 100", "100") is True
    assert _eval_putnam("We get 1/2 as the result", "1/2") is True
    assert _eval_putnam("Final Answer: 42", "100") is False


def test_putnam_formal_eval_fn():
    from gbench.runners.eval_suites.putnam_formal import _eval_putnam_formal, check_putnam_formal_prerequisites
    assert _eval_putnam_formal("", "theorem foo : True := sorry") is False
    assert check_putnam_formal_prerequisites()[0] in (True, False)


def test_ruler_eval_fn():
    needles = ["appliance", "meter"]
    assert _eval_ruler("The items are appliance and meter.", needles) is True
    assert _eval_ruler("Only appliance was found.", needles) is False


def test_textvqa_eval_fn():
    assert _eval_textvqa("Coca-Cola", "coca-cola") is True
    # The prompt says "Answer with only the short text string found in the image", so a
    # verbose sentence is non-compliant and canonical VQA scoring marks it wrong.
    assert _eval_textvqa("It says 42 on the jersey", "42") is False
    assert _eval_textvqa("42", "42") is True
    assert _eval_textvqa("Pepsi", "coca-cola") is False


def test_gpqa_diamond_parser():
    """Verify that the robust gpqa_diamond answer parser behaves identically to the standalone eval script."""
    from gbench.runners.eval_suites.gpqa_diamond import _eval_gpqa

    # 1. Direct letter matching
    assert _eval_gpqa("B", "B") is True
    assert _eval_gpqa("  a  ", "A") is True

    # 2. Final Answer matching patterns
    assert _eval_gpqa("The correct choice is Final Answer: B", "B") is True
    assert _eval_gpqa("Final Answer: (C)", "C") is True
    assert _eval_gpqa("Final Answer: C.", "C") is True

    # 3. Answer pattern matching
    assert _eval_gpqa("Answer: (D)", "D") is True
    assert _eval_gpqa("Answer: A", "A") is True

    # 4. Suffix / Last line matching
    assert _eval_gpqa("Let's do some math... Therefore, the response should be\n(B)\n", "B") is True
    assert _eval_gpqa("Thus, we choose option\nD.", "D") is True

    # 5. Mismatch check
    assert _eval_gpqa("Final Answer: A", "B") is False


# --------------------------------------------------------------------------
# Sample loaders. These turn a dataset row into the
# (messages, gold_answer, metadata) triple that run_eval_suite consumes.
# --------------------------------------------------------------------------


def test_mmlu_pro_loader_shape_and_gold_mapping():
    with _patch_hf({"TIGER-Lab/MMLU-Pro": MMLU_PRO_ROWS}) as loader:
        samples = _load_mmlu_pro_samples()

    assert loader.call_args_list[0].args[0] == "TIGER-Lab/MMLU-Pro"
    assert loader.call_args_list[0].kwargs["split"] == "test"
    assert len(samples) == 2

    messages, gold, meta = samples[0]
    # answer_index 1 has to become the second option letter, not the digit 1.
    assert gold == "B"
    assert meta == {"category": "math"}
    assert messages[0]["role"] == "user"
    assert "What is 2 + 2?" in messages[0]["content"]
    # Options are relabelled with their own letters rather than reused verbatim.
    assert "(A) 3" in messages[0]["content"]
    assert "(B) 4" in messages[0]["content"]

    # A row that already carries a letter is passed through untouched.
    assert samples[1][1] == "B"
    assert samples[1][2] == {"category": "biology"}


def test_mmlu_pro_loader_honours_limit():
    with _patch_hf({"TIGER-Lab/MMLU-Pro": MMLU_PRO_ROWS}):
        assert len(_load_mmlu_pro_samples(limit=1)) == 1


def test_aime_loader_stringifies_integer_gold():
    with _patch_hf({"AI-MO/aimo-validation-aime": AIME_ROWS}) as loader:
        samples = _load_aime_samples()

    assert loader.call_args_list[0].args[0] == "AI-MO/aimo-validation-aime"
    assert len(samples) == 2
    messages, gold, meta = samples[0]
    # The row holds int 496 and the scorer compares strings, so the loader owns
    # the conversion.
    assert gold == "496"
    assert isinstance(gold, str)
    assert meta == {"category": "algebra"}
    assert "Find the remainder." in messages[0]["content"]
    # The gold answer for the scorer, not the model, so it must not leak.
    assert "496" not in messages[0]["content"]


def test_putnam_loader_parses_both_tag_encodings():
    with _patch_hf({"amitayusht/PutnamBench": PUTNAM_ROWS}):
        samples = _load_putnam_samples()

    # A repr'd list becomes its first element, with the quotes stripped.
    assert samples[0][2] == {"category": "algebra"}
    assert samples[0][1] == "1/2"
    # A bare string is used as the category directly.
    assert samples[1][2] == {"category": "geometry"}


def test_ruler_loader_bands_the_category_by_task_and_length():
    with _patch_hf({"rayonlabs/ruler-all": RULER_ROWS}):
        samples = _load_ruler_samples()

    messages, gold, meta = samples[0]
    assert messages[0]["content"] == "Find the hidden words."
    # gold stays a list because the scorer recalls every needle.
    assert gold == ["appliance", "meter"]
    # The category used to be the sample's own token count (`ruler_4096len`), which is
    # near-unique per sample; it is now <task>@<band> so the per-category table shows how
    # each task degrades with context.
    assert meta == {"category": "niah_single@4k", "length": 4096, "task": "niah_single"}
    assert samples[1][2]["category"].endswith("@8k")


def test_textvqa_loader_converts_to_rgb_png_data_url():
    if Image is None:
        pytest.skip("Pillow is not installed")
    # CMYK is the case the loader's mode check exists for.
    rows = [{"question": "What brand is on the can?", "answers": ["coca-cola"], "image": Image.new("CMYK", (4, 2))}]
    with _patch_hf({"lmms-lab/textvqa": rows}):
        samples = _load_textvqa_samples()

    messages, gold, meta = samples[0]
    assert gold == ["coca-cola"]
    assert meta == {"category": "scene_ocr"}

    image_part, text_part = messages[0]["content"]
    assert text_part["type"] == "text"
    assert "What brand is on the can?" in text_part["text"]

    url = image_part["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    decoded = _rgb_png_bytes(url.split(",", 1)[1])
    # The CMYK source has to arrive as an RGB PNG, otherwise the vision model
    # gets a payload it cannot read.
    assert decoded.format == "PNG"
    assert decoded.mode == "RGB"
    assert decoded.size == (4, 2)


def test_codeforces_loader_balances_the_three_contest_types():
    rows = (
        [{"contest_type": "CF", "title": f"cf{i}", "description": "d", "examples": []} for i in range(20)]
        + [{"contest_type": "ICPC", "title": f"icpc{i}", "description": "d", "examples": []} for i in range(20)]
        + [{"contest_type": "IOI", "title": f"ioi{i}", "description": "d", "examples": []} for i in range(20)]
    )
    with _patch_hf({"open-r1/codeforces-cots": rows}):
        samples = _load_codeforces_samples(limit=10)

    # 40 percent CF, 30 percent ICPC, remainder IOI. A plain head-of-dataset
    # read would have returned 10 CF rows and no ICPC or IOI at all.
    counts = {}
    for _, _, meta in samples:
        counts[meta["category"]] = counts.get(meta["category"], 0) + 1
    assert counts == {"CF": 4, "ICPC": 3, "IOI": 3}
    assert len(samples) == 10


def test_lcb_loader_merges_public_and_private_tests():
    gen_rows = [
        {
            "question_title": "Two Sum",
            "question_content": "Return the indices.",
            "starter_code": "def two_sum(nums, target):",
            "difficulty": "easy",
            # public arrives as a JSON string and private as a real list, and
            # both have to end up in the same list.
            "public_test_cases": '[{"input": "1", "output": "2"}]',
            "private_test_cases": [{"input": "3", "output": "4"}],
        }
    ]
    with _patch_hf(
        {
            "livecodebench/code_generation": gen_rows,
            "livecodebench/execution": [{"code": "def f(): return 1", "input": "f()", "output": 1}],
            "livecodebench/test_generation": [{"question_title": "T", "question_content": "c", "function_name": "f", "starter_code": "def f():", "difficulty": "medium", "test": "assert f()"}],
        }
    ):
        samples = _load_lcb_samples()

    by_task = {}
    for _, gold, meta in samples:
        by_task.setdefault(gold["task"], []).append((gold, meta))

    # test_gen is excluded by default (can't be faithfully scored); loader = gen + execution.
    assert set(by_task) == {"generation", "execution"}
    gen_gold, gen_meta = by_task["generation"][0]
    assert len(gen_gold["tests"]) == 2
    assert gen_meta == {"category": "code_gen_easy"}
    # The execution row's int output is stringified for comparison.
    assert by_task["execution"][0][0]["expected"] == "1"


def test_lcb_test_gen_included_only_when_opted_in():
    """LCB_INCLUDE_TEST_GEN=1 re-adds the test_generation task (opt-in)."""
    import os
    from unittest.mock import patch
    rows = {
        "livecodebench/code_generation": [],
        "livecodebench/execution": [{"code": "def f(): return 1", "input": "f()", "output": 1}],
        "livecodebench/test_generation": [{"question_title": "T", "question_content": "c",
                                           "function_name": "f", "starter_code": "def f():",
                                           "difficulty": "medium", "test": "assert f()"}],
    }
    with _patch_hf(rows), patch.dict(os.environ, {"LCB_INCLUDE_TEST_GEN": "1"}, clear=False):
        tasks = {g["task"] for _, g, _ in _load_lcb_samples()}
    assert "test_gen" in tasks


def test_lcb_loader_still_returns_generation_when_the_extras_are_missing():
    gen_rows = [{"question_title": "Two Sum", "question_content": "c", "public_test_cases": [], "private_test_cases": []}]
    # execution and test_generation are absent, so _patch_hf raises for them.
    # Both are wrapped in try/except, so generation must survive alone.
    with _patch_hf({"livecodebench/code_generation": gen_rows}):
        samples = _load_lcb_samples()

    assert len(samples) == 1
    assert samples[0][1]["task"] == "generation"


def test_thinking_mode_prompts():
    """Verify reasoning suites alter prompt instructions when enable_thinking=True vs False."""
    rows = {
        "TIGER-Lab/MMLU-Pro": MMLU_PRO_ROWS,
        "AI-MO/aimo-validation-aime": AIME_ROWS,
        "amitayusht/PutnamBench": PUTNAM_ROWS,
    }
    with _patch_hf(rows):
        samples_mmlu_pro_think = _load_mmlu_pro_samples(enable_thinking=True)
        samples_mmlu_pro_nothink = _load_mmlu_pro_samples(enable_thinking=False)
        samples_aime_think = _load_aime_samples(enable_thinking=True)
        samples_aime_nothink = _load_aime_samples(enable_thinking=False)
        samples_putnam_think = _load_putnam_samples(enable_thinking=True)
        samples_putnam_nothink = _load_putnam_samples(enable_thinking=False)

    assert "think step by step" in samples_mmlu_pro_think[0][0][0]["content"].lower()
    assert "think step by step" not in samples_mmlu_pro_nothink[0][0][0]["content"].lower()

    assert "step by step" in samples_aime_think[0][0][0]["content"].lower()
    assert "step by step" not in samples_aime_nothink[0][0][0]["content"].lower()

    assert "think step by step" in samples_putnam_think[0][0][0]["content"].lower()
    assert "think step by step" not in samples_putnam_nothink[0][0][0]["content"].lower()


def test_thinking_mode_leaves_the_gold_answer_and_metadata_alone():
    """Only the instruction text may change, so a thinking run stays comparable."""
    with _patch_hf({"AI-MO/aimo-validation-aime": AIME_ROWS}):
        think = _load_aime_samples(enable_thinking=True)
        nothink = _load_aime_samples(enable_thinking=False)

    assert [gold for _, gold, _ in think] == [gold for _, gold, _ in nothink]
    assert [meta for _, _, meta in think] == [meta for _, _, meta in nothink]


@pytest.mark.parametrize(
    "loader,dataset_id",
    [
        (_load_mmlu_pro_samples, "TIGER-Lab/MMLU-Pro"),
        (_load_aime_samples, "AI-MO/aimo-validation-aime"),
        (_load_putnam_samples, "amitayusht/PutnamBench"),
        (_load_ruler_samples, "rayonlabs/ruler-all"),
    ],
)
def test_loaders_request_the_canonical_dataset_id(loader, dataset_id):
    """A renamed or mistyped dataset id is the failure these suites are most
    exposed to, and it is invisible to a row-count assertion."""
    with _patch_hf({dataset_id: []}) as patched:
        assert loader() == []
    assert patched.call_args_list[0].args[0] == dataset_id


def test_infographicvqa_eval_fn():
    from gbench.runners.eval_suites.infographicvqa import _eval_infographicvqa, _load_infographicvqa_samples
    # InfographicVQA prompts for "only the direct short answer"; ANLS compares the answer
    # itself, so a sentence wrapping it does not count.
    assert _eval_infographicvqa("The answer is 7 tips", ["7", "seven"]) is False
    assert _eval_infographicvqa("7", ["7", "seven"]) is True
    assert _eval_infographicvqa("Answer: 7", ["7", "seven"]) is True
    assert _eval_infographicvqa("Final value: 42.5%", ["42.5%"]) is True
    assert _eval_infographicvqa("The value is 100", ["7"]) is False
    samples = _load_infographicvqa_samples(limit=2)
    assert len(samples) == 2


def test_semantic_keypoint_eval_fn():
    from gbench.runners.eval_suites.semantic_keypoint import _eval_semantic_keypoint, _load_semantic_keypoint_samples
    gold = {"x": 500.0, "y": 300.0}
    assert _eval_semantic_keypoint("Point: (505, 298)", gold) is True
    assert _eval_semantic_keypoint('{"x": 510, "y": 305}', gold) is True
    assert _eval_semantic_keypoint("Point: (100, 100)", gold) is False
    samples = _load_semantic_keypoint_samples(limit=2)
    assert len(samples) == 2


def test_aa_lcr_eval_fn():
    import csv as _csv
    import os
    import tempfile
    import zipfile
    from unittest import mock
    from gbench.runners.eval_suites.aa_lcr import _eval_aa_lcr, _load_aa_lcr_samples

    assert _eval_aa_lcr("The revenue increased by 15.4% according to table 2.", "15.4%") is True
    assert _eval_aa_lcr("Completely unrelated text", "15.4%") is False

    # aa_lcr loads a CSV + a ZIP of source documents via hf_hub_download; provide
    # real temp files (conftest globally stubs hf_hub_download to a config.json).
    d = tempfile.mkdtemp()
    csv_path = os.path.join(d, "aa_lcr.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=[
            "document_category", "question_id", "question", "answer", "data_source_filenames"])
        w.writeheader()
        w.writerow({"document_category": "finance", "question_id": "q1",
                    "question": "Revenue growth?", "answer": "15.4%", "data_source_filenames": "doc1.txt"})
        w.writerow({"document_category": "finance", "question_id": "q2",
                    "question": "Net margin?", "answer": "8%", "data_source_filenames": "doc2.txt"})
    zip_path = os.path.join(d, "aa_lcr.zip")
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("doc1.txt", "Revenue increased 15.4%.")
        zf.writestr("doc2.txt", "Net margin was 8%.")

    def _fake_download(repo_id, filename, **kwargs):
        return zip_path if filename.endswith(".zip") else csv_path

    with mock.patch("huggingface_hub.hf_hub_download", side_effect=_fake_download):
        samples = _load_aa_lcr_samples(limit=2)
    assert len(samples) == 2
    assert samples[0][1] == "15.4%"


def test_beam_128k_eval_fn():
    import json as _json
    import os
    import tempfile
    from unittest import mock
    import pandas as pd
    from gbench.runners.eval_suites.beam_128k import _eval_beam_128k, _load_beam_128k_samples

    assert _eval_beam_128k("The project was launched on March 14.", "March 14") is True
    assert _eval_beam_128k("There is no information provided regarding this in the chat.", "no information") is True
    assert _eval_beam_128k("Completely irrelevant response", "March 14") is False

    # beam_128k loads a parquet via hf_hub_download; write a real one. chat is a
    # list<struct>; probing_questions is a JSON string (the loader parses both).
    d = tempfile.mkdtemp()
    pq_path = os.path.join(d, "beam.parquet")
    probing = {"factual_recall": [
        {"question": "When did we launch?", "ideal_response": "March 14", "difficulty": "easy"},
        {"question": "What was the reply?", "ideal_response": "Noted", "difficulty": "easy"},
    ]}
    df = pd.DataFrame([{
        "conversation_id": "c1",
        "chat": [{"role": "user", "content": "We launched on March 14."},
                 {"role": "assistant", "content": "Noted."}],
        "probing_questions": _json.dumps(probing),
    }])
    df.to_parquet(pq_path)

    with mock.patch("huggingface_hub.hf_hub_download", side_effect=lambda repo_id, filename, **kw: pq_path):
        samples = _load_beam_128k_samples(limit=2)
    assert len(samples) == 2


def test_wildclawbench_eval_fn():
    from gbench.runners.eval_suites.wildclawbench import _eval_wildclawbench, _load_wildclawbench_samples
    assert _eval_wildclawbench("Task completed. Output summary: CapRL Reinforcement Learning paper processed.", '{"paper": "CapRL"}') is True
    samples = _load_wildclawbench_samples(limit=2)
    assert len(samples) == 2


def test_skillsbench_eval_fn():
    from gbench.runners.eval_suites.skillsbench import _eval_skillsbench, _load_skillsbench_samples
    assert _eval_skillsbench("Execution script solve.sh generated output: 42", "42") is True
    samples = _load_skillsbench_samples(limit=2)
    assert len(samples) == 2


def test_cybergym_eval_fn():
    from gbench.runners.eval_suites.cybergym import _eval_cybergym, _load_cybergym_samples
    assert _eval_cybergym("Applied fix in magick/quantum-export.c to handle alpha.", "--- a/magick/quantum-export.c\n+++ b/magick/quantum-export.c") is True
    samples = _load_cybergym_samples(limit=2)
    assert len(samples) == 2


