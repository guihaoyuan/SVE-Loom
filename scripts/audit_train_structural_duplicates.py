#!/usr/bin/env python3
"""Audit structural duplication in NL2SVE JSONL train/dev data.

This is a lightweight AST-like audit for environments without clang/libclang.
It preserves function/control-flow/SVE intrinsic/data-access structure while
normalizing local names and literals, then groups rows by exact and structural
fingerprints.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path
from typing import Iterable


SVE_RE = re.compile(r"\bsv[a-zA-Z0-9_]*\b")
IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
NUM_RE = re.compile(r"(?<![A-Za-z_])(?:0x[0-9A-Fa-f]+|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)(?:[uUlLfF]+)?")
COMMENT_RE = re.compile(r"//.*?$|/\*.*?\*/", re.S | re.M)
STRING_RE = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'')

KEYWORDS = {
    "alignas",
    "auto",
    "bool",
    "break",
    "case",
    "char",
    "const",
    "continue",
    "double",
    "else",
    "false",
    "float",
    "for",
    "if",
    "int",
    "int8_t",
    "int16_t",
    "int32_t",
    "int64_t",
    "long",
    "return",
    "short",
    "signed",
    "size_t",
    "sizeof",
    "static",
    "std",
    "string",
    "struct",
    "switch",
    "true",
    "uint8_t",
    "uint16_t",
    "uint32_t",
    "uint64_t",
    "unsigned",
    "void",
    "while",
}

TYPE_WORDS = {
    "svbool_t",
    "svfloat16_t",
    "svfloat32_t",
    "svfloat64_t",
    "svint8_t",
    "svint16_t",
    "svint32_t",
    "svint64_t",
    "svuint8_t",
    "svuint16_t",
    "svuint32_t",
    "svuint64_t",
}


def clean_code(code: str) -> str:
    code = COMMENT_RE.sub(" ", code)
    code = STRING_RE.sub(" STR ", code)
    code = re.sub(r"^\s*#\s*include[^\n]*", " ", code, flags=re.M)
    code = re.sub(r"^\s*#\s*define[^\n]*", " ", code, flags=re.M)
    code = re.sub(r"\s+", " ", code)
    return code.strip()


def digest(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()[:16]


def get_response(row: dict) -> str:
    if isinstance(row.get("response"), str):
        return row["response"]
    msgs = row.get("messages")
    if isinstance(msgs, list):
        for msg in reversed(msgs):
            if msg.get("role") == "assistant":
                return str(msg.get("content") or "")
    return ""


def get_prompt(row: dict) -> str:
    if isinstance(row.get("prompt"), str):
        return row["prompt"]
    msgs = row.get("messages")
    if isinstance(msgs, list):
        for msg in msgs:
            if msg.get("role") == "user":
                return str(msg.get("content") or "")
    return ""


def split_params(params: str) -> list[str]:
    out: list[str] = []
    cur: list[str] = []
    depth = 0
    for ch in params:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth = max(0, depth - 1)
        elif ch == "," and depth == 0:
            out.append("".join(cur).strip())
            cur = []
            continue
        cur.append(ch)
    tail = "".join(cur).strip()
    if tail:
        out.append(tail)
    return out


def normalize_type(param: str) -> str:
    param = re.sub(r"\b(__restrict__|__restrict|restrict)\b", " restrict", param)
    param = re.sub(r"\b[A-Za-z_][A-Za-z0-9_]*\s*(?=(?:\[[^\]]*\])?\s*$)", "NAME", param.strip())
    param = re.sub(r"\s+", " ", param)
    param = param.replace(" *", "*").replace("* ", "*").replace(" &", "&").replace("& ", "&")
    return param


def signature_shape(code: str, prompt: str) -> str:
    hay = prompt + "\n" + code
    # Prefer the last function-like signature followed by an opening brace.
    matches = re.findall(
        r"([A-Za-z_][\w:<>,\s\*&~]*?)\s+([A-Za-z_][A-Za-z0-9_:]*)\s*\(([^;{}]*)\)\s*\{",
        hay,
        flags=re.S,
    )
    if not matches:
        return "sig:none"
    ret, _name, params = matches[-1]
    ret = normalize_type(ret)
    parts = [normalize_type(p) for p in split_params(params) if p and p != "void"]
    return "ret=" + ret + "|params=" + ",".join(parts)


def source_prefix(task_id: str) -> str:
    if "." in task_id:
        return task_id.split(".", 1)[0]
    m = re.match(r"([A-Za-z0-9]+)", task_id)
    return m.group(1) if m else task_id[:32]


def sve_category(name: str) -> str:
    table = [
        ("whilelt", "pred_whilelt"),
        ("cntp", "pred_count"),
        ("cnt", "int_count_or_vl"),
        ("ld1_gather", "gather"),
        ("st1_scatter", "scatter"),
        ("ld", "load"),
        ("st", "store"),
        ("addv", "reduction"),
        ("maxv", "reduction"),
        ("minv", "reduction"),
        ("orv", "reduction"),
        ("andv", "reduction"),
        ("cvt", "convert"),
        ("reinterpret", "reinterpret"),
        ("cmp", "compare"),
        ("sel", "select"),
        ("mla", "mulacc"),
        ("mad", "mulacc"),
        ("mul", "arith"),
        ("add", "arith"),
        ("sub", "arith"),
        ("div", "arith"),
        ("lsl", "shift"),
        ("lsr", "shift"),
        ("asr", "shift"),
        ("rev", "permute"),
        ("ext", "permute"),
        ("uzp", "permute"),
        ("zip", "permute"),
        ("trn", "permute"),
        ("unpk", "widen"),
        ("dup", "splat"),
        ("index", "index"),
    ]
    for needle, cat in table:
        if needle in name:
            return cat
    return "other_sv"


def structural_features(code: str, prompt: str) -> dict:
    cleaned = clean_code(code)
    sve_calls = SVE_RE.findall(cleaned)
    categories = [sve_category(x) for x in sve_calls]
    cat_counts = collections.Counter(categories)
    op_flags = {
        "for": len(re.findall(r"\bfor\s*\(", cleaned)),
        "while": len(re.findall(r"\bwhile\s*\(", cleaned)),
        "if": len(re.findall(r"\bif\s*\(", cleaned)),
        "return": len(re.findall(r"\breturn\b", cleaned)),
        "array_subscript": len(re.findall(r"\[[^\]]+\]", cleaned)),
        "ptr_arith": len(re.findall(r"\+\s*[A-Za-z_][A-Za-z0-9_]*", cleaned)),
        "ternary": cleaned.count("?"),
        "mod": cleaned.count("%"),
        "shift_op": len(re.findall(r"<<|>>", cleaned)),
        "std_vector": int("std::vector" in prompt or "std::vector" in cleaned),
        "std_string": int("std::string" in prompt or "std::string" in cleaned),
    }
    sig = signature_shape(code, prompt)
    # Normalize code into a coarse token stream: keep keywords, SVE names,
    # SVE vector types, and operators; erase local names and literal values.
    def repl_num(m: re.Match) -> str:
        return "NUM"

    norm = NUM_RE.sub(repl_num, cleaned)

    def repl_ident(m: re.Match) -> str:
        tok = m.group(0)
        if tok in KEYWORDS or tok in TYPE_WORDS or tok.startswith("sv"):
            return tok
        if tok in {"NULL", "nullptr", "NAN", "FLT_MAX", "DBL_MAX"}:
            return tok
        return "ID"

    norm = IDENT_RE.sub(repl_ident, norm)
    norm = re.sub(r"\s+", " ", norm).strip()
    feature_key = "|".join(
        [
            sig,
            "loop=f{for}:w{while}:if{if}:ret{return}".format(**op_flags),
            "mem=sub{array_subscript}:ptr{ptr_arith}:vec{std_vector}:str{std_string}".format(**op_flags),
            "ops=mod{mod}:shift{shift_op}:tern{ternary}".format(**op_flags),
            "svcats=" + ",".join(f"{k}:{v}" for k, v in sorted(cat_counts.items())),
            "svseq=" + ",".join(categories[:80]),
            "sve=" + ",".join(sve_calls[:80]),
        ]
    )
    return {
        "signature_shape": sig,
        "exact_hash": digest(cleaned),
        "struct_hash": digest(feature_key),
        "normalized_hash": digest(norm),
        "feature_key": feature_key,
        "sve_calls": ",".join(sve_calls[:80]),
        "sve_categories": ",".join(categories[:80]),
        **op_flags,
    }


def iter_rows(path: Path, split: str) -> Iterable[dict]:
    with path.open() as f:
        for idx, line in enumerate(f, 1):
            row = json.loads(line)
            prompt = get_prompt(row)
            code = get_response(row)
            feats = structural_features(code, prompt)
            yield {
                "split": split,
                "line": idx,
                "task_id": str(row.get("task_id") or ""),
                "source_type": str(row.get("source_type") or ""),
                "source_prefix": source_prefix(str(row.get("task_id") or "")),
                **feats,
            }


def write_tsv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w") as f:
        f.write("\t".join(fields) + "\n")
        for row in rows:
            f.write("\t".join(str(row.get(k, "")).replace("\t", " ").replace("\n", " ") for k in fields) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = list(iter_rows(Path(args.train), "train")) + list(iter_rows(Path(args.dev), "dev"))

    row_fields = [
        "split",
        "line",
        "task_id",
        "source_type",
        "source_prefix",
        "signature_shape",
        "exact_hash",
        "struct_hash",
        "normalized_hash",
        "for",
        "while",
        "if",
        "return",
        "array_subscript",
        "ptr_arith",
        "ternary",
        "mod",
        "shift_op",
        "std_vector",
        "std_string",
        "sve_categories",
        "sve_calls",
    ]
    write_tsv(out_dir / "row_structural_fingerprints.tsv", rows, row_fields)

    group_rows = []
    for kind in ["exact_hash", "struct_hash", "normalized_hash"]:
        groups = collections.defaultdict(list)
        for row in rows:
            groups[row[kind]].append(row)
        for h, members in groups.items():
            if len(members) < 2:
                continue
            by_split = collections.Counter(m["split"] for m in members)
            by_source = collections.Counter(m["source_type"] for m in members)
            by_prefix = collections.Counter(m["source_prefix"] for m in members)
            sample = members[0]
            group_rows.append(
                {
                    "kind": kind,
                    "hash": h,
                    "n": len(members),
                    "train": by_split.get("train", 0),
                    "dev": by_split.get("dev", 0),
                    "top_source_type": ";".join(f"{k}:{v}" for k, v in by_source.most_common(5)),
                    "top_prefix": ";".join(f"{k}:{v}" for k, v in by_prefix.most_common(5)),
                    "example_task_ids": ";".join(m["task_id"] for m in members[:12]),
                    "signature_shape": sample["signature_shape"],
                    "sve_categories": sample["sve_categories"],
                    "sve_calls": sample["sve_calls"],
                }
            )
    group_rows.sort(key=lambda r: (r["kind"], -int(r["n"]), r["hash"]))
    write_tsv(
        out_dir / "duplicate_group_summary.tsv",
        group_rows,
        [
            "kind",
            "hash",
            "n",
            "train",
            "dev",
            "top_source_type",
            "top_prefix",
            "example_task_ids",
            "signature_shape",
            "sve_categories",
            "sve_calls",
        ],
    )

    summary = {
        "total_rows": len(rows),
        "train_rows": sum(1 for r in rows if r["split"] == "train"),
        "dev_rows": sum(1 for r in rows if r["split"] == "dev"),
        "unique_exact_hash": len({r["exact_hash"] for r in rows}),
        "unique_struct_hash": len({r["struct_hash"] for r in rows}),
        "unique_normalized_hash": len({r["normalized_hash"] for r in rows}),
        "duplicate_groups": {},
        "source_type_counts": dict(collections.Counter(r["source_type"] for r in rows).most_common()),
        "source_prefix_counts_top50": dict(collections.Counter(r["source_prefix"] for r in rows).most_common(50)),
    }
    for kind in ["exact_hash", "struct_hash", "normalized_hash"]:
        c = collections.Counter(r[kind] for r in rows)
        summary["duplicate_groups"][kind] = {
            "groups_ge2": sum(v >= 2 for v in c.values()),
            "groups_ge3": sum(v >= 3 for v in c.values()),
            "groups_ge5": sum(v >= 5 for v in c.values()),
            "groups_ge10": sum(v >= 10 for v in c.values()),
            "rows_in_groups_ge2": sum(v for v in c.values() if v >= 2),
            "rows_in_groups_ge5": sum(v for v in c.values() if v >= 5),
            "max_group_size": max(c.values()) if c else 0,
            "top20_group_rows": sum(v for _, v in c.most_common(20)),
        }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
