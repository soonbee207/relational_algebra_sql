#!/usr/bin/env python3
"""Shared helpers for the 340-question slice: load questions, parse IR JSON,
execute SQL, estimate tokens. Zero LLM calls.
"""

import json
import os
import re
import sqlite3
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")


def _resolve_data_dir():
    """BIRD dev data moved from llm/data to llm/full_dev_data in this repo.
    Prefer whichever exists so the study scripts keep running either way."""
    for cand in ("data", "full_dev_data"):
        if os.path.isfile(os.path.join(ROOT, "llm", cand, "dev.json")):
            return os.path.join(ROOT, "llm", cand)
    return os.path.join(ROOT, "llm", "data")


_DATA = _resolve_data_dir()
DBR = os.path.join(_DATA, "dev_databases")
DEV = os.path.join(_DATA, "dev.json")
OUT = os.path.join(HERE, "..", "outputs")
RESULTS = os.path.join(OUT, "results_340.json")
N_SLICE = 340
EX_TIMEOUT = 30.0


def load_key():
    """AIML key from env or gitignored llm/run/.env(.local). Never print this."""
    for key_name in ("AIMLAPI_API_KEY", "AIML_API_KEY"):
        v = os.getenv(key_name)
        if v:
            return v.strip().strip('"').strip("'")
    for name in (".env.local", ".env"):
        path = os.path.join(ROOT, "llm", "run", name)
        if not os.path.isfile(path):
            continue
        for line in open(path):
            line = line.strip()
            if line.startswith("export "):
                line = line[len("export "):].lstrip()
            for key_name in ("AIMLAPI_API_KEY", "AIML_API_KEY"):
                if line.startswith(key_name):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def load_slice(n=N_SLICE):
    """Prefer the evaluation master file (has ex_kg); else BIRD dev.json."""
    if os.path.isfile(RESULTS):
        rows = json.load(open(RESULTS))[:n]
        out = []
        for r in rows:
            out.append(
                {
                    "idx": r["idx"],
                    "db_id": r["db_id"],
                    "question": r["question"],
                    "evidence": r.get("evidence") or "",
                    "gold": r["gold"],
                    "difficulty": r.get("difficulty", ""),
                    "ex_kg": r.get("ex_kg"),
                }
            )
        return out
    raw = json.load(open(DEV))[:n]
    out = []
    for i, r in enumerate(raw):
        out.append(
            {
                "idx": r.get("question_id", i),
                "db_id": r["db_id"],
                "question": r["question"],
                "evidence": r.get("evidence") or "",
                "gold": r["SQL"],
                "difficulty": r.get("difficulty", ""),
                "ex_kg": None,
            }
        )
    return out


def extract_json(text):
    if not text:
        return None
    t = re.sub(r"```(?:json)?", "", text, flags=re.I).strip()
    s = t.find("{")
    if s < 0:
        return None
    depth = 0
    for i in range(s, len(t)):
        if t[i] == "{":
            depth += 1
        elif t[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(t[s : i + 1])
                except Exception:
                    return None
    return None


def extract_sql(text):
    if not text:
        return ""
    cleaned = re.sub(r"```(?:sql)?", "", text, flags=re.I).strip()
    fences = re.findall(r"```sql\s*(.*?)```", text, re.S | re.I)
    if fences:
        return fences[-1].strip().rstrip(";")
    m = re.search(r"\bSELECT\b", cleaned, re.I)
    if m:
        return cleaned[m.start() :].strip().rstrip(";")
    return cleaned


def db_path(db_id):
    return os.path.join(DBR, db_id, db_id + ".sqlite")


def _ex_sql(pred, gold, dbp):
    conn = sqlite3.connect(dbp)
    cur = conn.cursor()
    cur.execute(pred)
    pr = cur.fetchall()
    cur.execute(gold)
    gr = cur.fetchall()
    return 1 if set(pr) == set(gr) else 0


def run_ex(db_id, pred, gold, timeout=EX_TIMEOUT):
    if not pred:
        return 0
    from func_timeout import FunctionTimedOut, func_timeout

    try:
        return func_timeout(timeout, _ex_sql, args=(pred, gold, db_path(db_id)))
    except (FunctionTimedOut, Exception):
        return 0


def run_ex_detail(db_id, pred, gold, timeout=EX_TIMEOUT):
    """EX plus executability and row counts. Timeout/error => ex=0."""
    out = {
        "ex": 0,
        "executable": False,
        "n_pred": None,
        "n_gold": None,
        "error": None,
        "timeout": False,
    }
    if not pred:
        out["error"] = "empty_sql"
        return out
    from func_timeout import FunctionTimedOut, func_timeout

    def _go():
        conn = sqlite3.connect(db_path(db_id))
        cur = conn.cursor()
        cur.execute(pred)
        pr = cur.fetchall()
        out["executable"] = True
        out["n_pred"] = len(pr)
        cur.execute(gold)
        gr = cur.fetchall()
        out["n_gold"] = len(gr)
        out["ex"] = 1 if set(pr) == set(gr) else 0
        return out

    try:
        return func_timeout(timeout, _go)
    except FunctionTimedOut:
        out["timeout"] = True
        out["error"] = "timeout"
        return out
    except Exception as e:
        out["error"] = str(e)[:200]
        return out


def sql_executes(db_id, sql, timeout=EX_TIMEOUT):
    if not sql:
        return False
    from func_timeout import FunctionTimedOut, func_timeout

    def _go():
        c = sqlite3.connect(db_path(db_id)).cursor()
        c.execute(sql)
        c.fetchall()
        return True

    try:
        return bool(func_timeout(timeout, _go))
    except (FunctionTimedOut, Exception):
        return False


def gold_categories(sql):
    """A priori syntactic labels on gold SQL. Frozen before the 340 run.
    Not an error taxonomy and not derived from model output.
    """
    u = (sql or "").upper()
    n_select = gold_n_select_items(sql)
    return {
        "join": bool(re.search(r"\bJOIN\b", u)),
        "aggregation": bool(re.search(r"\b(COUNT|SUM|AVG|MIN|MAX)\s*\(", u)),
        "filter": bool(re.search(r"\bWHERE\b", u)),
        "group_by": bool(re.search(r"\bGROUP\s+BY\b", u)),
        "order_by": bool(re.search(r"\bORDER\s+BY\b", u)),
        "limit": bool(re.search(r"\bLIMIT\b", u)),
        "distinct": bool(re.search(r"\bDISTINCT\b", u)),
        "n_select_items": n_select,
        "single_projection": n_select == 1,
    }


def gold_n_select_items(sql):
    """Count top-level SELECT items before the first FROM. None if unparseable."""
    u = (sql or "").strip()
    m = re.match(r"(?is)SELECT\s+(?:DISTINCT\s+)?(.*?)\s+FROM\s", u)
    if not m:
        return None
    sel = m.group(1)
    n = 1
    depth = 0
    for ch in sel:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif ch == "," and depth == 0:
            n += 1
    return n


def estimate_tokens(text):
    """Prefer tiktoken; otherwise ~4 chars/token."""
    if not text:
        return 0
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, (len(text) + 3) // 4)


def timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t0
