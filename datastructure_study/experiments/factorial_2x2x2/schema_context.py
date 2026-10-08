#!/usr/bin/env python3
"""Prompt templates for the eight-arm 2x2x2 factorial.

The experiment crosses database context, output protocol, and projection
instruction. The four combinations without the projection rule are:

    RawSQL = ddl_block           + SQL_INSTRUCTION
    PreSQL = preprocessed_block  + SQL_INSTRUCTION
    RawIR  = ddl_block           + IR_SPEC
    PreIR  = preprocessed_block  + IR_SPEC (+ preprocessing-usage hint lines)

Each combination also has a matched `_MP` arm with the minimal-projection rule,
producing eight experiments in total.

`ddl_block` and `preprocessed_block` read only the offline structures/<db>.json
artifact (built by m1_build_structures.py from SQLite; never questions or gold
SQL). PreSQL and PreIR share preprocessed_block() so the preprocessing factor is
identical across the two preprocessed arms.

Reconstructed to byte-match outputs/smoke_prompt_<arm>_idx2.txt. Zero LLM calls.
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.abspath(os.path.join(HERE, "..", ".."))
STRUCT_DIR = os.path.join(STUDY, "structures")

ARMS = ("RawSQL", "PreSQL", "RawIR", "PreIR")

# Minimal-projection level of the third factor: every base combination with the
# M3 strong minimal-projection rule added. The SQL_MP arms
# carry the same rule as the IR_MP arms, so IR vs SQL can be compared with the
# rule held fixed (RawIR_MP vs RawSQL_MP isolates the IR format given the rule).
MP_ARMS = ("RawSQL_MP", "PreSQL_MP", "RawIR_MP", "PreIR_MP")
ALL_ARMS = ARMS + MP_ARMS

IR_ARMS = ("RawIR", "PreIR", "RawIR_MP", "PreIR_MP")
PRE_ARMS = ("PreSQL", "PreIR", "PreSQL_MP", "PreIR_MP")
MINPROJ_ARMS = MP_ARMS


# ----------------------------------------------------------------------------
# structures/<db>.json access
# ----------------------------------------------------------------------------
def struct_path(db):
    return os.path.join(STRUCT_DIR, db + ".json")


def load_struct(db):
    with open(struct_path(db), encoding="utf-8") as fh:
        return json.load(fh)


# ----------------------------------------------------------------------------
# schema context blocks
# ----------------------------------------------------------------------------
def ddl_block(struct):
    """Raw context: the CREATE TABLE statements exactly as stored in SQLite."""
    return struct["ddl"]


def preprocessed_block(struct):
    """Preprocessed context: DDL plus the offline enrichment block shared by
    PreSQL and PreIR (FK graph, shortest join paths, column profiles, and
    low-cardinality categorical value hints). Deterministic given the artifact.
    """
    lines = [
        "-- Database preprocessing (offline, from SQLite metadata and data; "
        "not from evaluation questions or gold SQL)",
        "-- Foreign-key relations:",
    ]
    for fk in struct["fks"]:
        lines.append(f"-- {fk['from']} -> {fk['to']}")

    lines.append("-- Valid join paths (shortest FK-reachable table sequences):")
    for pair, path in struct["join_paths"].items():
        a, b = pair.split("|", 1)
        lines.append(f"-- {a} <-> {b} : {' - '.join(path)}")

    lines.append("-- Column profiles (type, primary-key flag, distinct-value count):")
    profile = struct["profile"]
    for key in sorted(profile):
        p = profile[key]
        pk = " pk" if p.get("pk") else ""
        lines.append(f"-- {key} {p['type']}{pk} ndistinct={p['ndistinct']}")

    lines.append("-- Categorical value hints (low-cardinality columns only):")
    for key in sorted(profile):
        enum = profile[key].get("enum")
        if enum:
            vals = ", ".join(repr(v) for v in enum)
            lines.append(f"-- {key} in {{{vals}}}")

    return struct["ddl"] + "\n\n" + "\n".join(lines)


# ----------------------------------------------------------------------------
# output protocol specs
# ----------------------------------------------------------------------------
IR_SPEC = """You must answer ONLY with a JSON object in this structured query representation (no prose, no markdown):
{
  "from": "<table>",
  "joins": [{"table":"<t>","on":["<t1.col>","<t2.col>"]}],
  "select": [ "<table.col>",
              {"fn":"COUNT","arg":"<table.col>|*","distinct":false},
              {"expr":"<raw scalar SQL>","as":"r"} ],
  "where": "<raw SQL predicate>",
  "group_by": ["<table.col or raw expr>"],
  "having": "<raw SQL>",
  "order_by": [{"by":"<table.col or raw expr>","desc":true}],
  "limit": <int>,
  "distinct": false
}
The structured representation makes query-level components explicit.
Predicates and complex expressions may remain as SQL expressions within their designated fields.
"select" is the explicit output projection: list every output column or expression.
Use the exact column names from the schema (quote-sensitive names go in expr/where as-is).
Only "from" and "select" are required; other fields are optional."""

# Extra guidance appended only for the preprocessed IR arm (PreIR): it points the
# model at the enrichment block. RawIR omits these two lines.
IR_SPEC_PRE_SUFFIX = (
    "\nUse the foreign-key relations and join-path hints for join keys."
    "\nMatch question phrases to database values via the categorical value hints."
)

# Enforced minimal-projection block (soft enforcement via instruction), appended
# only for the *_MP arms. Wording carried over from the historical M3 IR prompt
# (m3_ir_prompt.py) whose enforced-projection variant scored +9.71 EX.
IR_SPEC_MINPROJ_SUFFIX = (
    "\nMINIMAL PROJECTION (most important rule):"
    "\n- \"select\" must contain ONLY the columns or expressions the question asks"
    " for — nothing extra."
    "\n- Do NOT add id, name, key, or \"for context\" columns that were not requested."
    "\n- If the question asks for a single value, \"select\" has exactly one item."
)

# The same rule for the SQL_MP arms: identical wording, with the JSON "select"
# field renamed to the SELECT clause. Placed between the schema context and the
# question, the same position the rule takes in the IR_MP arms.
SQL_MINPROJ_BLOCK = (
    "MINIMAL PROJECTION (most important rule):"
    "\n- The SELECT clause must contain ONLY the columns or expressions the question"
    " asks for — nothing extra."
    "\n- Do NOT add id, name, key, or \"for context\" columns that were not requested."
    "\n- If the question asks for a single value, the SELECT clause has exactly one item."
)


def _sql_instruction(question, evidence):
    if evidence and evidence.strip():
        head = (
            f"-- External Knowledge: {evidence}\n"
            "-- Using valid SQLite and understanding External Knowledge, "
            "answer the following questions for the tables provided above."
        )
    else:
        head = (
            "-- Using valid SQLite, answer the following questions for the "
            "tables provided above."
        )
    return (
        head
        + f"\n-- {question}"
        + "\nGenerate the SQL after thinking step by step: "
        + "\nSELECT "
    )


def _ir_instruction(question, evidence):
    if evidence and evidence.strip():
        head = f"-- External Knowledge: {evidence}\n-- Question: {question}"
    else:
        head = f"-- Question: {question}"
    return head + "\nStructured query representation (JSON):"


# ----------------------------------------------------------------------------
# arm assembly
# ----------------------------------------------------------------------------
def build_prompt(arm, struct, question, evidence=""):
    """Assemble the full prompt string for one (arm, question)."""
    if arm not in ALL_ARMS:
        raise ValueError(f"unknown arm: {arm!r} (expected one of {ALL_ARMS})")

    context = preprocessed_block(struct) if arm in PRE_ARMS else ddl_block(struct)

    if arm in IR_ARMS:
        spec = IR_SPEC
        if arm in PRE_ARMS:
            spec += IR_SPEC_PRE_SUFFIX
        if arm in MINPROJ_ARMS:
            spec += IR_SPEC_MINPROJ_SUFFIX
        return context + "\n\n" + spec + "\n\n" + _ir_instruction(question, evidence)

    if arm in MINPROJ_ARMS:
        context += "\n\n" + SQL_MINPROJ_BLOCK
    return context + "\n\n" + _sql_instruction(question, evidence)
