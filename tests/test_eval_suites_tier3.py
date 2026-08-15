# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
"""Tier-3 execution-harness eval suites: spider2, ojbench, swe_bench_pro, multi_swe_bench.

These four are gated behind heavy external harnesses (SQLite gold DBs / the DMOJ
sandbox / the SWE-bench-Pro & Multi-SWE-bench Docker harnesses) that are absent on
CI, so they SKIP cleanly here. These tests lock in: (1) registration + SANDBOX
concurrency wiring, (2) clean skip dicts with the right docs URL, (3) loaders that
raise (never fabricate) on a bad schema and slice canonically, (4) spider2's
result-set comparison semantics, which run fully offline against a synthetic DB.
"""

import asyncio
import csv
import json
import os
import sqlite3
import tempfile
from unittest.mock import patch, MagicMock

import pytest

from gbench.runners.eval_suites import SUITES


# --------------------------------------------------------------------------- #
# registration + sandbox wiring
# --------------------------------------------------------------------------- #
def test_tier3_suites_registered():
    for name in ("spider2", "ojbench", "swe_bench_pro", "multi_swe_bench"):
        assert name in SUITES, f"{name} not registered in SUITES"


@pytest.mark.parametrize("name", ["spider2", "ojbench", "swe_bench_pro", "multi_swe_bench"])
def test_tier3_is_a_sandbox_eval(name):
    """--sandboxes must override concurrency for each Tier-3 (execution) suite."""
    from gbench.runners.evals import EvalsBenchmarkRunner
    from gbench.core.config import BenchmarkConfig

    config = BenchmarkConfig()
    config.sandboxes = 7
    runner = EvalsBenchmarkRunner(config)
    with patch.dict("gbench.runners.eval_suites.SUITES",
                    {name: MagicMock(return_value={"status": "success"})}) as mock_suites:
        runner._run_single_eval(name, "gemma-4-E4B-it", "http://localhost:8000/v1", num_threads=128)
        assert mock_suites[name].call_args[1]["concurrency"] == 7


# --------------------------------------------------------------------------- #
# clean skip dicts
# --------------------------------------------------------------------------- #
def test_spider2_skip_on_missing_prereqs():
    from gbench.runners.eval_suites.spider2 import run_spider2
    with patch("gbench.runners.eval_suites.spider2.check_spider2_prerequisites",
               return_value=(False, "gold not found")):
        res = run_spider2("gemma-4-E4B-it", "http://localhost:8000/v1", 2)
        assert res["status"] == "skipped"
        assert "docs/evals/spider2.md" in res["skip_reason"]
        assert res["accuracy"] == 0.0


def test_ojbench_skip_on_missing_prereqs():
    from gbench.runners.eval_suites.ojbench import run_ojbench
    with patch("gbench.runners.eval_suites.ojbench.check_ojbench_prerequisites",
               return_value=(False, "ojbench not installed")):
        res = run_ojbench("gemma-4-E4B-it", "http://localhost:8000/v1", 2)
        assert res["status"] == "skipped"
        assert "docs/evals/ojbench.md" in res["skip_reason"]


def test_swe_bench_pro_skip_on_missing_prereqs():
    from gbench.runners.eval_suites.swe_bench_pro import run_swe_bench_pro
    with patch("gbench.runners.eval_suites.swe_bench_pro.check_swe_bench_pro_prerequisites",
               return_value=(False, "harness not found")):
        res = run_swe_bench_pro("gemma-4-E4B-it", "http://localhost:8000/v1", 2)
        assert res["status"] == "skipped"
        assert "docs/evals/swe_bench_pro.md" in res["skip_reason"]


def test_swe_bench_pro_gated_behind_opt_in():
    """Even with docker + harness present, the run stays gated until SWE_BENCH_PRO_RUN=1."""
    from gbench.runners.eval_suites import swe_bench_pro as SP
    with tempfile.TemporaryDirectory() as hd:
        open(os.path.join(hd, "swe_bench_pro_eval.py"), "w").close()
        os.mkdir(os.path.join(hd, "run_scripts"))
        open(os.path.join(hd, "swe_bench_pro_full.csv"), "w").close()
        env = {"SWE_BENCH_PRO_HARNESS_DIR": hd}  # note: SWE_BENCH_PRO_RUN unset
        with patch.dict(os.environ, env, clear=False), \
             patch("shutil.which", return_value="/usr/bin/docker"), \
             patch("subprocess.run", return_value=MagicMock(returncode=0)), \
             patch.dict("sys.modules", {"docker": MagicMock()}):
            os.environ.pop("SWE_BENCH_PRO_RUN", None)
            ok, reason = SP.check_swe_bench_pro_prerequisites()
            assert ok is False and "SWE_BENCH_PRO_RUN=1" in reason


def test_multi_swe_bench_skip_on_missing_prereqs():
    from gbench.runners.eval_suites.multi_swe_bench import run_multi_swe_bench
    with patch("gbench.runners.eval_suites.multi_swe_bench.check_multi_swe_bench_prerequisites",
               return_value=(False, "multi_swe_bench not installed")):
        res = run_multi_swe_bench("gemma-4-E4B-it", "http://localhost:8000/v1", 2)
        assert res["status"] == "skipped"
        assert "docs/evals/multi_swe_bench.md" in res["skip_reason"]


def test_multi_swe_bench_prereq_survives_absent_parent_package():
    """find_spec RAISES (rather than returning None) when the parent package is absent.

    Hermetic: the old version asserted the real package was missing, so it silently
    became a no-op the day `multi-swe-bench` was installed on the box - and then failed
    outright. Mock the absence instead of depending on the environment.
    """
    from gbench.runners.eval_suites.multi_swe_bench import check_multi_swe_bench_prerequisites
    with patch("shutil.which", return_value="/usr/bin/docker"), \
         patch("subprocess.run", return_value=MagicMock(returncode=0)), \
         patch("importlib.util.find_spec", side_effect=ModuleNotFoundError("no parent")):
        ok, reason = check_multi_swe_bench_prerequisites()
    assert ok is False and "multi-swe-bench" in reason      # raised, and was caught


def test_multi_swe_bench_prereq_passes_when_the_package_is_present():
    from gbench.runners.eval_suites.multi_swe_bench import check_multi_swe_bench_prerequisites
    with patch("shutil.which", return_value="/usr/bin/docker"), \
         patch("subprocess.run", return_value=MagicMock(returncode=0)), \
         patch("importlib.util.find_spec", return_value=object()):
        ok, reason = check_multi_swe_bench_prerequisites()
    assert ok is True and reason == ""


# --------------------------------------------------------------------------- #
# loaders: canonical slicing + no fabrication (raise on bad schema)
# --------------------------------------------------------------------------- #
def test_swe_bench_pro_loader_conditional_fields_and_raises():
    from gbench.runners.eval_suites import swe_bench_pro as SP
    diff = "diff --git a/x.py b/x.py\n@@\n-a\n+b\n"
    rows = [
        {"instance_id": "o__r-1", "repo": "o/r", "problem_statement": "fix",
         "requirements": "reqs", "interface": None, "patch": diff, "repo_language": "Python"},
        {"instance_id": "o__r-2", "repo": "o/r", "problem_statement": "fix2",
         "requirements": None, "interface": "def f(): ...", "patch": diff, "repo_language": "Go"},
    ]
    with patch("datasets.load_dataset", return_value=rows):
        samples = SP._load_swe_bench_pro_samples()
    assert [g for _m, g, _e in samples] == [diff, diff]
    assert samples[0][2]["instance_id"] == "o__r-1" and samples[0][2]["category"] == "Python"
    assert "Requirements:" in samples[0][0][0]["content"]
    assert "Interface:" not in samples[0][0][0]["content"]
    assert "Interface:" in samples[1][0][0]["content"]
    with patch("datasets.load_dataset", return_value=[{"repo": "x"}]):
        with pytest.raises(RuntimeError, match="refusing to fabricate"):
            SP._load_swe_bench_pro_samples()


def test_ojbench_loader_verbatim_prompt_and_raises():
    from gbench.runners.eval_suites import ojbench as OJ
    with tempfile.TemporaryDirectory() as d:
        good = os.path.join(d, "full.jsonl")
        with open(good, "w") as f:
            f.write(json.dumps({"id": 101, "prompt": "solve X", "language": "cpp",
                                "dataset": "NOI", "difficulty": "hard"}) + "\n")
        with patch("huggingface_hub.hf_hub_download", return_value=good):
            samples = OJ._load_ojbench_samples()
        assert samples[0][1] == 101
        assert samples[0][0][0]["content"] == "solve X"       # prompt VERBATIM
        assert samples[0][2]["category"] == "NOI_hard"
        assert samples[0][2]["row"] == {"id": 101, "dataset": "NOI",
                                        "language": "cpp", "difficulty": "hard"}
        bad = os.path.join(d, "bad.jsonl")
        with open(bad, "w") as f:
            f.write(json.dumps({"id": 1}) + "\n")
        with patch("huggingface_hub.hf_hub_download", return_value=bad):
            with pytest.raises(RuntimeError, match="refusing to fabricate"):
                OJ._load_ojbench_samples()


def test_ojbench_scorer_maps_by_id_and_language():
    from gbench.runners.eval_suites import ojbench as OJ
    fake = MagicMock()
    fake.init = MagicMock()
    fake.judge_jsonl_data = MagicMock(return_value=[
        {"id": 101, "language": "cpp", "is_passed": True},
        {"id": 102, "language": "python", "is_passed": False},
    ])
    OJ._INITIALIZED = False
    with patch.dict("sys.modules", {"ojbench": fake}), \
         patch.dict(os.environ, {"OJBENCH_TESTDATA": tempfile.gettempdir()}):
        scorer = OJ._make_scorer(2)
        traces = [
            {"response_text": "<thought>x</thought>ans", "extra_payload": {"row": {"id": 101, "language": "cpp"}}},
            {"response_text": "ans2", "extra_payload": {"row": {"id": 102, "language": "python"}}},
            {"response_text": "ans3", "extra_payload": {"row": {"id": 999, "language": "cpp"}}},
        ]
        asyncio.run(scorer(traces))
    assert [t["is_correct"] for t in traces] == [True, False, False]
    # reasoning/thought tags are stripped before judging
    assert fake.judge_jsonl_data.call_args[0][0][0]["content"] == "ans"


def test_multi_swe_bench_loader_and_src_cache():
    from gbench.runners.eval_suites import multi_swe_bench as MS
    diff = "diff --git a/x b/x\n@@\n"
    with tempfile.TemporaryDirectory() as d:
        jl = os.path.join(d, "python_dataset.jsonl")
        with open(jl, "w") as f:
            f.write(json.dumps({"org": "o", "repo": "r", "number": 5, "title": "t",
                                "body": "b", "base": {"sha": "abc"},
                                "fix_patch": diff, "instance_id": "o__r-5"}) + "\n")

        class _Api:
            def list_repo_files(self, *a, **k):
                return ["python/python_dataset.jsonl", "README.md"]

        with patch("huggingface_hub.HfApi", _Api), \
             patch("huggingface_hub.hf_hub_download", return_value=jl):
            samples = MS._load_multi_swe_bench_samples()
    assert len(samples) == 1
    e = samples[0][2]
    assert e["instance_id"] == "o__r-5" and e["org"] == "o" and e["number"] == 5
    assert MS._SRC_FILES.get("o__r-5") == jl


# --------------------------------------------------------------------------- #
# spider2 result-set comparison (fully offline against a synthetic SQLite DB)
# --------------------------------------------------------------------------- #
@pytest.fixture
def spider2_env():
    with tempfile.TemporaryDirectory() as base:
        db = os.path.join(base, "shop.sqlite")
        con = sqlite3.connect(db)
        con.execute("CREATE TABLE sales(region TEXT, revenue REAL)")
        con.executemany("INSERT INTO sales VALUES (?,?)",
                        [("west", 100.0), ("east", 250.0), ("west", 50.0), ("north", 75.5)])
        con.commit()
        con.close()
        gold = os.path.join(base, "local001.csv")
        with open(gold, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["region", "total"])
            # deliberately different row order than a GROUP BY would emit
            w.writerow(["north", 75.5])
            w.writerow(["east", 250.0])
            w.writerow(["west", 150.0])
        yield db, [gold]


def test_spider2_execution_accuracy_semantics(spider2_env):
    from gbench.runners.eval_suites import spider2 as S
    db, golds = spider2_env

    # correct query matches order-insensitively
    good = S._get_sqlite_result(db, "SELECT region, SUM(revenue) AS total FROM sales GROUP BY region")
    assert S._score_against_golds(good, golds, [], True) == 1
    # wrong result set does not match
    bad = S._get_sqlite_result(db, "SELECT region, SUM(revenue) AS total FROM sales "
                                    "WHERE region!='north' GROUP BY region")
    assert S._score_against_golds(bad, golds, [], True) == 0
    # predicted superset of columns still matches (subset containment)
    sup = S._get_sqlite_result(db, "SELECT region, SUM(revenue) AS total, COUNT(*) AS n "
                                    "FROM sales GROUP BY region")
    assert S._score_against_golds(sup, golds, [], True) == 1


def test_spider2_numeric_tolerance_and_extract():
    import pandas as pd
    from gbench.runners.eval_suites import spider2 as S
    assert S.extract_sql_query("x\n```sql\nSELECT 1\n```\ny") == "SELECT 1"
    assert S.extract_sql_query("SELECT 2") == "SELECT 2"
    g = pd.DataFrame({"a": [150.0]})
    assert S.compare_pandas_table(pd.DataFrame({"a": [150.004]}), g, None, True) == 1
    assert S.compare_pandas_table(pd.DataFrame({"a": [150.5]}), g, None, True) == 0


def test_spider2_async_scorer_writes_is_correct(spider2_env):
    from gbench.runners.eval_suites import spider2 as S
    db, golds = spider2_env
    localdb_dir = os.path.dirname(db)
    scorer = S._make_scorer(localdb_dir)
    extra = {"category": "sqlite", "db": "shop", "condition_cols": [],
             "ignore_order": True, "gold_paths": golds}
    traces = [
        {"response_text": "```sql\nSELECT region, SUM(revenue) AS total FROM sales GROUP BY region\n```",
         "extra_payload": extra},
        {"response_text": "```sql\nSELECT region FROM sales\n```", "extra_payload": extra},
        {"response_text": "no sql here", "extra_payload": extra},
    ]
    asyncio.run(scorer(traces))
    assert [t["is_correct"] for t in traces] == [True, False, False]


# --------------------------------------------------------------------------- #
# swe_bench_pro raw-sample table (audit P2-8)
# --------------------------------------------------------------------------- #
def _fake_pro_frame():
    import pandas as pd
    from gbench.runners.eval_suites.swe_bench_pro import _RAW_SAMPLE_COLUMNS
    return pd.DataFrame([{c: f"v_{c}" for c in _RAW_SAMPLE_COLUMNS}])


def test_missing_csv_is_generated_from_the_canonical_dataset():
    """The Pro harness README names swe_bench_pro_full.csv but the clone never ships it."""
    from gbench.runners.eval_suites import swe_bench_pro as SP
    with tempfile.TemporaryDirectory() as hd:
        ds = MagicMock()
        ds.to_pandas.return_value = _fake_pro_frame()
        with patch.dict(os.environ, {"SWE_BENCH_PRO_HARNESS_DIR": hd}, clear=False), \
             patch("datasets.load_dataset", return_value=ds):
            os.environ.pop("SWE_BENCH_PRO_RAW_SAMPLE", None)
            path = SP.raw_sample_path()
        assert path == os.path.join(hd, "swe_bench_pro_full.csv")
        assert os.path.isfile(path)
        header = open(path, encoding="utf-8").readline()
        for col in SP._RAW_SAMPLE_COLUMNS:
            assert col in header, f"the harness reads {col} off this table"


def test_existing_csv_is_never_overwritten():
    from gbench.runners.eval_suites import swe_bench_pro as SP
    with tempfile.TemporaryDirectory() as hd:
        csv = os.path.join(hd, "swe_bench_pro_full.csv")
        with open(csv, "w", encoding="utf-8") as f:
            f.write("instance_id\nkeep-me\n")
        with patch.dict(os.environ, {"SWE_BENCH_PRO_HARNESS_DIR": hd}, clear=False), \
             patch("datasets.load_dataset", side_effect=AssertionError("must not download")):
            os.environ.pop("SWE_BENCH_PRO_RAW_SAMPLE", None)
            assert SP.raw_sample_path() == csv
        assert "keep-me" in open(csv, encoding="utf-8").read()


def test_unreadable_dataset_is_a_skip_not_a_guess():
    from gbench.runners.eval_suites import swe_bench_pro as SP
    with tempfile.TemporaryDirectory() as hd:
        open(os.path.join(hd, "swe_bench_pro_eval.py"), "w").close()
        os.mkdir(os.path.join(hd, "run_scripts"))
        with patch.dict(os.environ,
                        {"SWE_BENCH_PRO_HARNESS_DIR": hd, "SWE_BENCH_PRO_RUN": "1"},
                        clear=False), \
             patch("shutil.which", return_value="/usr/bin/docker"), \
             patch("subprocess.run", return_value=MagicMock(returncode=0)), \
             patch.dict("sys.modules", {"docker": MagicMock()}), \
             patch("datasets.load_dataset", side_effect=OSError("offline")):
            os.environ.pop("SWE_BENCH_PRO_RAW_SAMPLE", None)
            ok, reason = SP.check_swe_bench_pro_prerequisites()
        assert ok is False
        assert "raw-sample table" in reason and "SWE_BENCH_PRO_RAW_SAMPLE" in reason


def test_bigcodebench_workdir_is_traversable_by_the_container_uid():
    """`mkdtemp` is 0700/host-uid; the image runs as uid 1000 and cannot traverse it.

    That made every relative path resolution inside the container raise PermissionError -
    including `datasets.load_dataset`'s `Path("bigcode/bigcodebench/state.json").exists()`
    probe - so the harness produced no results and the suite scored 0. Reproduced against
    the real image: 0700 raises, 0777 resolves and is writable.
    """
    import re
    src = open(os.path.join(os.path.dirname(__file__), "..", "gbench", "runners",
                            "eval_suites", "bigcodebench.py")).read()
    body = src[src.index("def _make_scorer"):]
    mk = body.index("mkdtemp")
    chmod = body.find("os.chmod(workdir, 0o777)")
    assert chmod != -1, "the bind-mounted workdir must be made traversable by the image uid"
    assert chmod > mk, "chmod must follow mkdtemp"
    # the samples file the container reads must not inherit a 077 umask either
    assert re.search(r"os\.chmod\(samples_path,\s*0o6\d\d\)", body), \
        "samples.jsonl must be readable by the container uid"


# --------------------------------------------------------------------------- #
# scicode reference outputs are fetched at eval time, like any other dataset
# --------------------------------------------------------------------------- #
def test_scicode_prefers_an_explicit_local_copy(tmp_path):
    from gbench.runners.eval_suites import scicode
    f = tmp_path / "test_data.h5"
    f.write_bytes(b"x")
    with patch.dict(os.environ, {"SCICODE_TEST_DATA": str(f)}, clear=False), \
         patch("huggingface_hub.hf_hub_download", side_effect=AssertionError("must not fetch")):
        path, prov = scicode.resolve_test_data()
    assert path == str(f) and prov == "SCICODE_TEST_DATA"


def test_scicode_fetches_the_mirror_when_no_local_copy(tmp_path):
    from gbench.runners.eval_suites import scicode
    f = tmp_path / "test_data.h5"
    f.write_bytes(b"x")
    env = {k: v for k, v in os.environ.items() if k != "SCICODE_TEST_DATA"}
    with patch.dict(os.environ, env, clear=True), \
         patch("huggingface_hub.hf_hub_download", return_value=str(f)):
        path, prov = scicode.resolve_test_data()
    assert path == str(f)
    assert prov.startswith("hf:"), "the mirror must be recorded as the provenance"


def test_scicode_skips_when_the_data_cannot_be_obtained():
    """Every test binds `target` from this file; without it a perfect solution scores 0."""
    from gbench.runners.eval_suites.scicode import run_scicode
    env = {k: v for k, v in os.environ.items() if k != "SCICODE_TEST_DATA"}
    with patch.dict(os.environ, env, clear=True), \
         patch("huggingface_hub.hf_hub_download", side_effect=OSError("offline")):
        res = run_scicode("m", "http://x", concurrency=1, limit=2)
    assert res["status"] == "skipped"
    assert res["accuracy"] == 0.0
    assert "structural 0" in res["skip_reason"]


def test_scicode_records_the_digest_of_what_decided_the_scores():
    import inspect
    from gbench.runners.eval_suites import scicode
    src = inspect.getsource(scicode.run_scicode)
    assert 'result["test_data"]' in src and "sha256" in src
