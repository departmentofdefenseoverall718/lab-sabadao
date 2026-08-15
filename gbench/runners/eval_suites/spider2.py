# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# Evaluation Suite: spider2
# Description: Spider 2.0-lite - enterprise text-to-SQL, execution accuracy (SQLite subset)

"""gbench native built-in runner for spider2 (Data & SQL Engineering).

Canonical Spider 2.0-lite (xlangai/spider2-lite) execution accuracy: run the
model's SQL against the target database and compare the result set to the gold
(column-subset containment, numeric tolerance, order-insensitive), per the
official evaluate_utils.py. Only the 64 local* (SQLite) instances run offline;
the 103 BigQuery + 93 Snowflake instances need cloud accounts and are excluded.
Gold + local DBs are not on HF - point SPIDER2_GOLD_DIR / SPIDER2_LOCALDB_DIR at
them (see docs); skips cleanly otherwise.
"""

import glob
import json
import logging
import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from .base import run_eval_suite
from .sampling import stratified_sample

logger = logging.getLogger(__name__)

PILLAR = "Data & SQL Engineering"
DOCS_URL = "docs/evals/spider2.md"
_ABS_TOL = 1e-2


def extract_sql_query(text: str) -> str:
    """Extract SQL from a ```sql fenced block, else the raw text (canonical)."""
    m = re.search(r"```sql\n(.*?)\n```", text or "", re.DOTALL)
    return (m.group(1) if m else (text or "")).strip()


def _is_num(x: Any) -> bool:
    try:
        float(x)
        return True
    except (ValueError, TypeError):
        return False


def _norm(v: Any) -> Any:
    if v is None:
        return 0
    if isinstance(v, float) and math.isnan(v):
        return 0
    return v


def _vectors_match(gold_vec: List[Any], pred_vec: List[Any], ignore_order: bool = True) -> bool:
    if len(gold_vec) != len(pred_vec):
        return False
    a = [_norm(x) for x in gold_vec]
    b = [_norm(x) for x in pred_vec]
    if ignore_order:
        a = sorted(a, key=lambda z: str(z))
        b = sorted(b, key=lambda z: str(z))
    for x, y in zip(a, b):
        if _is_num(x) and _is_num(y):
            if not math.isclose(float(x), float(y), abs_tol=_ABS_TOL):
                return False
        elif str(x) != str(y):
            return False
    return True


def _columns(df) -> List[List[Any]]:
    return [list(df.iloc[:, i]) for i in range(df.shape[1])]


def compare_pandas_table(pred_df, gold_df, condition_cols=None, ignore_order=True) -> int:
    """1 iff every required gold column is matched by SOME predicted column (subset containment)."""
    pred_cols = _columns(pred_df)
    gold_cols = _columns(gold_df)
    idxs = condition_cols if condition_cols else list(range(len(gold_cols)))
    for gi in idxs:
        if gi >= len(gold_cols):
            return 0
        if not any(_vectors_match(gold_cols[gi], pv, ignore_order) for pv in pred_cols):
            return 0
    return 1


def _score_against_golds(pred_df, gold_paths: List[str], condition_cols, ignore_order: bool) -> int:
    """Multi-gold: pass if pred matches ANY accepted gold variant."""
    import pandas as pd
    # condition_cols may be a flat list (single) or list-of-lists (per gold variant)
    nested = bool(condition_cols) and all(isinstance(c, list) for c in condition_cols)
    for i, gp in enumerate(gold_paths):
        try:
            gold_df = pd.read_csv(gp)
        except Exception:
            continue
        cc = condition_cols[i] if nested and i < len(condition_cols) else (None if nested else condition_cols)
        if compare_pandas_table(pred_df, gold_df, cc, ignore_order) == 1:
            return 1
    return 0


def _get_sqlite_result(db_path: str, sql: str):
    """Run SQL against an in-memory copy of the SQLite DB -> DataFrame (canonical)."""
    import sqlite3
    import pandas as pd
    src = sqlite3.connect(db_path)
    mem = sqlite3.connect(":memory:")
    try:
        src.backup(mem)
    finally:
        src.close()
    try:
        return pd.read_sql_query(sql, mem)
    finally:
        mem.close()


def _gold_dir() -> Optional[str]:
    return os.getenv("SPIDER2_GOLD_DIR")


def _localdb_dir() -> Optional[str]:
    return os.getenv("SPIDER2_LOCALDB_DIR")


def check_spider2_prerequisites() -> Tuple[bool, str]:
    """pandas + local gold dir + local SQLite DB dir (only the SQLite subset runs offline)."""
    try:
        import pandas  # noqa: F401
    except ImportError:
        return False, "Python package 'pandas' is not installed."
    gd, ld = _gold_dir(), _localdb_dir()
    if not gd or not os.path.isfile(os.path.join(gd, "spider2lite_eval.jsonl")) \
            or not os.path.isdir(os.path.join(gd, "exec_result")):
        return False, ("Spider2 gold not found: set SPIDER2_GOLD_DIR to a checkout of "
                       "xlang-ai/Spider2/spider2-lite/evaluation_suite/gold "
                       "(needs spider2lite_eval.jsonl + exec_result/).")
    if not ld or not glob.glob(os.path.join(ld, "*.sqlite")):
        return False, ("Spider2 local SQLite DBs not found: set SPIDER2_LOCALDB_DIR to the "
                       "unzipped spider2-localdb (*.sqlite) directory.")
    return True, ""


def _load_gold_config(gold_dir: str) -> Dict[str, Dict[str, Any]]:
    cfg = {}
    with open(os.path.join(gold_dir, "spider2lite_eval.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                cfg[row["instance_id"]] = row
    return cfg


def _load_spider2_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], Any, Dict[str, Any]]]:
    """Load Spider2-lite local* (SQLite) instances joined with local gold; raises on failure."""
    gold_dir = _gold_dir()
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id="xlangai/spider2-lite",
                               filename="spider2-lite.jsonl", repo_type="dataset")
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    except Exception as e:
        logger.error(f"Failed to load dataset for spider2: {e}")
        raise RuntimeError(f"Could not load dataset for spider2: {e}") from e

    gold_cfg = _load_gold_config(gold_dir)
    samples = []
    for item in rows:
        iid = item.get("instance_id")
        if not iid or not iid.startswith("local"):
            continue  # offline SQLite subset only
        g = gold_cfg.get(iid)
        gold_paths = sorted(glob.glob(os.path.join(gold_dir, "exec_result", f"{iid}.csv")) or
                            glob.glob(os.path.join(gold_dir, "exec_result", f"{iid}_*.csv")))
        if g is None or not gold_paths:
            continue  # no gold available for this instance
        db = item.get("db")
        question = item.get("question")
        if not db or not question:
            raise RuntimeError("spider2: unexpected schema (db/question); refusing to fabricate")
        prompt = (
            f"Database: {db}\n\nQuestion:\n{question}\n\n"
            "Write a single complete SQLite SQL query that answers the question. "
            "Return only the SQL inside a ```sql ... ``` code block."
        )
        samples.append(([{"role": "user", "content": prompt}], iid, {
            "category": "sqlite", "db": db,
            "condition_cols": g.get("condition_cols") or [],
            "ignore_order": bool(g.get("ignore_order", True)),
            "gold_paths": gold_paths,
        }))

    if not samples:
        raise RuntimeError("spider2: no scorable local SQLite instances found")
    # Stratified, not a contiguous head (audit RC-1).
    samples = stratified_sample(samples, limit, None, seed="spider2")
    # `xlangai/spider2-lite` holds 260 instances: 103 BigQuery, 93 Snowflake, 64 local
    # SQLite. Only the local subset runs offline. The old "547" was not this dataset.
    logger.info("Loaded %d spider2 local(SQLite) samples (of 64 local in the "
                "260-instance spider2-lite set; BigQuery/Snowflake need cloud creds).",
                len(samples))
    return samples


def _make_scorer(localdb_dir: str):
    async def _score(sample_traces: List[Dict[str, Any]]) -> None:
        import asyncio

        def _grade(tr: Dict[str, Any]) -> bool:
            extra = tr.get("extra_payload") or {}
            db = extra.get("db")
            sql = extract_sql_query(tr.get("response_text") or "")
            if not sql or not db:
                return False
            db_path = os.path.join(localdb_dir, f"{db}.sqlite")
            if not os.path.isfile(db_path):
                return False
            try:
                pred_df = _get_sqlite_result(db_path, sql)
            except Exception:
                return False
            if pred_df is None:
                return False
            # An empty result set can be the RIGHT answer ("which orders shipped late?"
            # -> none did). Rejecting every empty prediction made those questions
            # impossible to answer correctly; let the gold comparison decide instead.
            return _score_against_golds(
                pred_df, extra.get("gold_paths") or [],
                extra.get("condition_cols") or [], bool(extra.get("ignore_order", True))
            ) == 1

        async def _one(tr):
            try:
                tr["is_correct"] = await asyncio.wait_for(asyncio.to_thread(_grade, tr), timeout=120)
            except Exception:
                tr["is_correct"] = False
            tr["status"] = "OK"

        await asyncio.gather(*[_one(t) for t in sample_traces])
    return _score


def run_spider2(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run Spider2-lite SQLite execution accuracy (or skip if gold/DBs absent)."""
    ok, reason = check_spider2_prerequisites()
    if not ok:
        msg = f"[SKIP] spider2 skipped: {reason} See '{DOCS_URL}' for setup instructions."
        logger.warning(msg)
        print(f"\n{msg}")
        return {
            "benchmark_type": "eval",
            "eval_name": "spider2",
            "model_name": model_name,
            "status": "skipped",
            "total_questions": 0,
            "correct_answers": 0,
            "accuracy": 0.0,
            "skip_reason": f"{reason} (See {DOCS_URL})",
        }
    samples = _load_spider2_samples(limit=kwargs.get("limit"))
    return run_eval_suite(
        eval_name="spider2",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        async_eval_fn=_make_scorer(_localdb_dir()),
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens", 4096),
        temperature=kwargs.get("temperature", 0.0),
    )
