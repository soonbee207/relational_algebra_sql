#!/usr/bin/env python3
"""Generation runner for the eight-arm 2x2x2 factorial.

For each question in the 340-slice and a chosen arm, build the arm's prompt
(schema_context.build_prompt), call GPT-5.2 once, and for the IR arms parse the
JSON-formatted, relational-algebra-inspired structured IR and compile it to SQL
with ra_to_sql. Resumable checkpoint per arm.

    RawSQL / PreSQL (+ _MP) -> model emits SQL directly (extract_sql)
    RawIR  / PreIR  (+ _MP) -> model emits structured IR serialized as JSON
                                (extract_json -> ra_to_sql)

Zero-token modes:
    --dry-run          build prompts only, write outputs/factorial_dryrun_<n>.json
    --dump-smoke       (with --dry-run) also (re)write outputs/smoke_prompt_<arm>_idx2.txt

Paid mode (340 calls per arm, 2720 for all eight):
    --arm RawSQL|PreSQL|RawIR|PreIR [--n 340] [--workers 8]
    --arm RawSQL_MP|PreSQL_MP|RawIR_MP|PreIR_MP   (extension arms, same cost)

Checkpoints to outputs/factorial_<arm>_<n>.json, flushing every 20 items.
"""

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.abspath(os.path.join(HERE, "..", ".."))
ROOT = os.path.dirname(STUDY)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(STUDY, "src"))
sys.path.insert(0, os.path.join(ROOT, "llm", "src"))

from exp_common import (  # noqa: E402
    OUT,
    extract_json,
    extract_sql,
    load_key,
    load_slice,
    estimate_tokens,
)
from ra_to_sql import ra_to_sql  # noqa: E402
from schema_context import ALL_ARMS, ARMS, IR_ARMS, build_prompt, load_struct  # noqa: E402

PROTOCOL = "factorial-2x2x2-v1"
ENGINE = "gpt-5.2"
TEMPERATURE = 0
MAX_TOKENS = 900

_lock = threading.Lock()
_struct_cache = {}


def _struct(db):
    if db not in _struct_cache:
        _struct_cache[db] = load_struct(db)
    return _struct_cache[db]


def _base_record(idx, arm, db_id, prompt):
    return {
        "idx": idx,
        "arm": arm,
        "db_id": db_id,
        "prompt_chars": len(prompt),
        "prompt_tokens": estimate_tokens(prompt),
        "ir": None,
        "ir_ok": False,
        "compile_ok": False,
        "sql": "",
        "raw": "",
        "out_chars": 0,
        "out_tokens": 0,
        "latency_s": 0.0,
        "protocol_version": PROTOCOL,
        "engine": ENGINE,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }


def gen_one(arm, row):
    db_id = row["db_id"]
    prompt = build_prompt(arm, _struct(db_id), row["question"], row["evidence"])
    rec = _base_record(row["idx"], arm, db_id, prompt)

    from gpt_request import connect_gpt

    t0 = time.perf_counter()
    try:
        raw = connect_gpt(
            engine=ENGINE,
            prompt=prompt,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            stop=None,
        )
        text = raw if isinstance(raw, str) else raw["choices"][0]["text"]
    except Exception as e:  # noqa: BLE001
        rec["err"] = str(e)[:200]
        rec["latency_s"] = round(time.perf_counter() - t0, 3)
        return rec
    rec["latency_s"] = round(time.perf_counter() - t0, 3)

    rec["raw"] = text
    rec["out_chars"] = len(text or "")
    rec["out_tokens"] = estimate_tokens(text)

    if arm in IR_ARMS:
        ir = extract_json(text)
        rec["ir"] = ir
        rec["ir_ok"] = ir is not None
        if ir is not None:
            try:
                rec["sql"] = ra_to_sql(ir)
                rec["compile_ok"] = True
            except Exception as e:  # noqa: BLE001
                rec["compile_err"] = str(e)[:200]
    else:
        sql = extract_sql(text)
        rec["sql"] = sql
        rec["compile_ok"] = bool(sql)

    return rec


def ckpt_path(arm, n):
    return os.path.join(OUT, f"factorial_{arm}_{n}.json")


def run_arm(arm, n, workers):
    rows = load_slice(n)
    path = ckpt_path(arm, n)

    ckpt = {}
    if os.path.exists(path):
        ckpt = {int(k): v for k, v in json.load(open(path)).items()}
    pending = [r for r in rows if r["idx"] not in ckpt]
    print(f"[{arm}] resume {len(ckpt)}/{n} done, {len(pending)} pending", flush=True)

    def flush():
        tmp = path + ".tmp"
        json.dump(
            {str(k): ckpt[k] for k in sorted(ckpt)},
            open(tmp, "w"),
            ensure_ascii=False,
            indent=2,
        )
        os.replace(tmp, path)

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(gen_one, arm, r): r["idx"] for r in pending}
        for f in as_completed(futs):
            rec = f.result()
            with _lock:
                ckpt[rec["idx"]] = rec
                done += 1
                if done % 20 == 0 or done == len(pending):
                    flush()
                    print(f"  [{arm}] {len(ckpt)}/{n}", flush=True)
    flush()
    print(f"[{arm}] wrote {path}", flush=True)


def dry_run(n, dump_smoke):
    rows = load_slice(n)
    out = []
    for r in rows:
        for arm in ALL_ARMS:
            prompt = build_prompt(arm, _struct(r["db_id"]), r["question"], r["evidence"])
            out.append(
                {
                    "idx": r["idx"],
                    "arm": arm,
                    "db_id": r["db_id"],
                    "prompt_chars": len(prompt),
                    "prompt_tokens": estimate_tokens(prompt),
                }
            )
    path = os.path.join(OUT, f"factorial_dryrun_{n}.json")
    json.dump(out, open(path, "w"), ensure_ascii=False, indent=2)
    print(f"dry-run: {len(out)} prompts across {len(ALL_ARMS)} arms -> {path}")
    # per-arm mean prompt size
    for arm in ALL_ARMS:
        toks = [o["prompt_tokens"] for o in out if o["arm"] == arm]
        print(f"  {arm:9} mean prompt_tokens ~ {sum(toks)/len(toks):.0f}")

    if dump_smoke:
        idx2 = next(r for r in rows if r["idx"] == 2)
        for arm in ALL_ARMS:
            p = build_prompt(arm, _struct(idx2["db_id"]), idx2["question"], idx2["evidence"])
            sp = os.path.join(OUT, f"smoke_prompt_{arm}_idx2.txt")
            open(sp, "w", encoding="utf-8").write(p)
            print(f"  wrote {sp}")


def main():
    ap = argparse.ArgumentParser(description="eight-arm factorial generation runner")
    ap.add_argument("--arm", choices=ALL_ARMS, help="which arm to generate (paid)")
    ap.add_argument("--n", type=int, default=340, help="slice size")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true", help="build prompts only, zero tokens")
    ap.add_argument("--dump-smoke", action="store_true", help="rewrite smoke_prompt_<arm>_idx2.txt")
    args = ap.parse_args()

    if args.dry_run or (not args.arm and args.dump_smoke):
        dry_run(args.n, args.dump_smoke)
        return
    if not args.arm:
        ap.error("provide --arm (paid: 340 calls) or --dry-run (zero tokens)")

    key = load_key()
    if not key:
        ap.error("no AIML key found (llm/run/.env or AIMLAPI_API_KEY)")
    import openai

    openai.api_key = key
    run_arm(args.arm, args.n, args.workers)


if __name__ == "__main__":
    main()
