#!/usr/bin/env python3
"""Run clang JSON AST structural duplicate audit for NL2SVE train/dev JSONL.

The script is intended to run on the AArch64/SVE remote host where clang can
parse <arm_sve.h>.  It emits per-row AST fingerprints plus duplicate groups.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any


COMMENT_RE = re.compile(r"//.*?$|/\*.*?\*/", re.S | re.M)
STRING_RE = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'')
FUNC_RE = re.compile(
    r"([A-Za-z_][\w:<>,\s\*&~]*?)\s+([A-Za-z_][A-Za-z0-9_:]*)\s*\(([^;{}]*)\)\s*\{",
    re.S,
)
SVE_RE = re.compile(r"\bsv[A-Za-z0-9_]*\b")
SVE_CALL_RE = re.compile(r"\b(sv[A-Za-z0-9_]*)\s*\(")


IMPORTANT_KINDS = {
    "ArraySubscriptExpr",
    "BinaryOperator",
    "CallExpr",
    "CompoundAssignOperator",
    "CompoundStmt",
    "ConditionalOperator",
    "CXXBoolLiteralExpr",
    "CXXForRangeStmt",
    "CXXMemberCallExpr",
    "DeclRefExpr",
    "DeclStmt",
    "FloatingLiteral",
    "ForStmt",
    "IfStmt",
    "IntegerLiteral",
    "MemberExpr",
    "ParenExpr",
    "ReturnStmt",
    "UnaryOperator",
    "VarDecl",
    "WhileStmt",
}


def digest(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()[:16]


def clean_code(code: str) -> str:
    code = COMMENT_RE.sub(" ", code)
    code = STRING_RE.sub(" STR ", code)
    code = re.sub(r"\s+", " ", code)
    return code.strip()


def get_response(row: dict[str, Any]) -> str:
    if isinstance(row.get("response"), str):
        return row["response"]
    msgs = row.get("messages")
    if isinstance(msgs, list):
        for msg in reversed(msgs):
            if msg.get("role") == "assistant":
                return str(msg.get("content") or "")
    return ""


def get_prompt(row: dict[str, Any]) -> str:
    if isinstance(row.get("prompt"), str):
        return row["prompt"]
    msgs = row.get("messages")
    if isinstance(msgs, list):
        for msg in msgs:
            if msg.get("role") == "user":
                return str(msg.get("content") or "")
    return ""


def prompt_target_name(prompt: str) -> str:
    matches = FUNC_RE.findall(prompt)
    if matches:
        return matches[-1][1].split("::")[-1]
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


def normalize_type_text(text: str) -> str:
    text = re.sub(r"\b(__restrict__|__restrict|restrict)\b", "restrict", text)
    text = re.sub(r"\b[A-Za-z_][A-Za-z0-9_]*\s*(?=(?:\[[^\]]*\])?\s*$)", "NAME", text.strip())
    text = re.sub(r"\s+", " ", text)
    return text.replace(" *", "*").replace("* ", "*").replace(" &", "&").replace("& ", "&")


def fallback_signature(prompt: str, code: str) -> str:
    matches = FUNC_RE.findall(prompt + "\n" + code)
    if not matches:
        return "sig:none"
    ret, _name, params = matches[-1]
    parts = [normalize_type_text(p) for p in split_params(params) if p and p != "void"]
    return "ret=" + normalize_type_text(ret) + "|params=" + ",".join(parts)


def source_prefix(task_id: str) -> str:
    if "." in task_id:
        return task_id.split(".", 1)[0]
    m = re.match(r"([A-Za-z0-9]+)", task_id)
    return m.group(1) if m else task_id[:32]


def guess_sve_return_type(name: str) -> str:
    if name.startswith("svwhile") or name.startswith("svcmp") or name.startswith("svptest") or name.startswith("svptrue"):
        return "svbool_t" if not name.startswith("svptest") else "bool"
    if name.startswith("svcnt"):
        return "unsigned long"
    if any(x in name for x in ("addv", "maxv", "minv", "orv", "andv")):
        if "_f64" in name:
            return "double"
        if "_f32" in name or "_f16" in name:
            return "float"
        if "_u64" in name:
            return "uint64_t"
        if "_u32" in name:
            return "uint32_t"
        if "_u16" in name:
            return "uint16_t"
        if "_u8" in name:
            return "uint8_t"
        if "_s64" in name:
            return "int64_t"
        if "_s32" in name:
            return "int32_t"
        if "_s16" in name:
            return "int16_t"
        if "_s8" in name:
            return "int8_t"
    if "_b" in name and not any(x in name for x in ("_u8", "_s8", "_f")):
        return "svbool_t"
    for suffix, typ in [
        ("_f64", "svfloat64_t"),
        ("_f32", "svfloat32_t"),
        ("_f16", "svfloat16_t"),
        ("_s64", "svint64_t"),
        ("_s32", "svint32_t"),
        ("_s16", "svint16_t"),
        ("_s8", "svint8_t"),
        ("_u64", "svuint64_t"),
        ("_u32", "svuint32_t"),
        ("_u16", "svuint16_t"),
        ("_u8", "svuint8_t"),
    ]:
        if suffix in name:
            return typ
    return "svint32_t"


def make_stub_source(code: str) -> str:
    body = re.sub(r"^\s*#\s*include[^\n]*", " ", code, flags=re.M)
    body = re.sub(r"^\s*#\s*define[^\n]*", " ", body, flags=re.M)
    names = sorted(set(SVE_CALL_RE.findall(body)))
    decls = "\n".join(f"{guess_sve_return_type(name)} {name}(...);" for name in names)
    prelude = r"""
typedef unsigned long size_t;
typedef unsigned long uintptr_t;
typedef unsigned long uint64_t;
typedef unsigned int uint32_t;
typedef unsigned short uint16_t;
typedef unsigned char uint8_t;
typedef long int64_t;
typedef int int32_t;
typedef short int16_t;
typedef signed char int8_t;
typedef float float32_t;
typedef double float64_t;
#define restrict
#define __restrict__
#define __restrict
#define NULL 0
namespace std {
using size_t = ::size_t;
template<class T> struct vector {
  size_t size() const; T* data(); const T* data() const;
  T& operator[](size_t); const T& operator[](size_t) const;
  void push_back(const T&); void resize(size_t);
};
struct string {
  size_t size() const; char* data(); const char* data() const;
  char& operator[](size_t); const char& operator[](size_t) const;
  void push_back(char);
};
template<class T> T abs(T);
}
extern "C" float sqrtf(float);
extern "C" double sqrt(double);
extern "C" float fabsf(float);
extern "C" double fabs(double);
extern "C" float floorf(float);
extern "C" float ceilf(float);
extern "C" float roundf(float);
extern "C" float expf(float);
extern "C" float logf(float);
struct svbool_t { int opaque; };
struct svfloat16_t { int opaque; };
struct svfloat32_t { int opaque; };
struct svfloat64_t { int opaque; };
struct svint8_t { int opaque; };
struct svint16_t { int opaque; };
struct svint32_t { int opaque; };
struct svint64_t { int opaque; };
struct svuint8_t { int opaque; };
struct svuint16_t { int opaque; };
struct svuint32_t { int opaque; };
struct svuint64_t { int opaque; };
"""
    return prelude + "\n" + decls + "\n" + body


def run_clang_ast(
    code: str,
    clang: str,
    tmp_dir: Path,
    idx: int,
    timeout: float,
    filter_name: str,
    stub_source: bool,
) -> tuple[bool, str, str]:
    src = tmp_dir / f"row_{idx:06d}.cpp"
    src.write_text(make_stub_source(code) if stub_source else code)
    cmd = [
        clang,
        "-std=c++17",
        "-fsyntax-only",
        "-Wno-everything",
        "-Xclang",
        "-ast-dump=json",
    ]
    if not stub_source:
        cmd.insert(2, "-march=armv8-a+sve")
    if filter_name:
        cmd += ["-Xclang", "-ast-dump-filter", "-Xclang", filter_name]
    cmd.append(str(src))
    try:
        cp = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        return False, "", "timeout"
    ok = cp.returncode == 0
    if cp.stdout.strip():
        return ok, cp.stdout, cp.stderr[-1000:]
    return False, "", cp.stderr[-1000:]


def iter_nodes(node: Any):
    if isinstance(node, dict):
        yield node
        for child in node.get("inner") or []:
            yield from iter_nodes(child)
    elif isinstance(node, list):
        for item in node:
            yield from iter_nodes(item)


def find_function_nodes(ast: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for node in iter_nodes(ast):
        if node.get("kind") in {"FunctionDecl", "CXXMethodDecl"} and node.get("inner"):
            if node.get("isImplicit"):
                continue
            # Exclude definitions expanded from system headers such as
            # arm_sve.h macro wrappers.  User code and user helpers carry a
            # concrete loc.file; header macro expansions usually only have
            # loc.includedFrom.
            loc = node.get("loc") or {}
            if loc.get("includedFrom") and not loc.get("file"):
                continue
            # A definition has a CompoundStmt child.
            if any(ch.get("kind") == "CompoundStmt" for ch in node.get("inner") or [] if isinstance(ch, dict)):
                out.append(node)
    return out


def function_signature_from_ast(fn: dict[str, Any]) -> str:
    ret = str((fn.get("type") or {}).get("qualType") or "")
    params = []
    for ch in fn.get("inner") or []:
        if isinstance(ch, dict) and ch.get("kind") == "ParmVarDecl":
            params.append(str((ch.get("type") or {}).get("qualType") or ""))
    return "ret=" + ret + "|params=" + ",".join(params)


def select_target_function(ast: dict[str, Any], target_name: str) -> tuple[dict[str, Any] | None, int]:
    funcs = find_function_nodes(ast)
    if target_name:
        for fn in funcs:
            if fn.get("name") == target_name:
                return fn, len(funcs)
    non_main = [fn for fn in funcs if fn.get("name") != "main"]
    if non_main:
        return non_main[-1], len(funcs)
    return (funcs[-1] if funcs else None), len(funcs)


def find_first_declref_name(node: dict[str, Any]) -> str:
    for sub in iter_nodes(node):
        if sub.get("kind") == "DeclRefExpr":
            ref = sub.get("referencedDecl") or {}
            if ref.get("name"):
                return str(ref.get("name"))
        if sub.get("kind") == "MemberExpr" and sub.get("name"):
            return str(sub.get("name"))
    return ""


def sve_category(name: str) -> str:
    table = [
        ("whilelt", "pred_whilelt"),
        ("cntp", "pred_count"),
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
    if name.startswith("svcnt"):
        return "int_count_or_vl"
    return "other_sv"


def ast_features(ast_text: str, prompt: str, code: str) -> dict[str, Any]:
    try:
        ast = json.loads(ast_text)
    except Exception as e:
        return {
            "ast_ok": 0,
            "ast_error": f"json_parse:{type(e).__name__}",
            "function_count": 0,
            "helper_def_count": 0,
            "signature_shape": fallback_signature(prompt, code),
            "ast_hash": "",
            "kind_hash": "",
            "call_hash": "",
        }

    target_name = prompt_target_name(prompt)
    fn, function_count = select_target_function(ast, target_name)
    if not fn:
        return {
            "ast_ok": 0,
            "ast_error": "no_function_definition",
            "function_count": function_count,
            "helper_def_count": max(0, function_count - 1),
            "signature_shape": fallback_signature(prompt, code),
            "ast_hash": "",
            "kind_hash": "",
            "call_hash": "",
        }

    kinds: list[str] = []
    ops: list[str] = []
    calls: list[str] = []
    sve_calls = SVE_RE.findall(code)
    counts = collections.Counter()
    for node in iter_nodes(fn):
        kind = node.get("kind")
        if kind:
            counts[kind] += 1
        if kind in IMPORTANT_KINDS:
            token = kind
            if kind in {"BinaryOperator", "CompoundAssignOperator", "UnaryOperator"}:
                op = str(node.get("opcode") or "")
                if op:
                    token += ":" + op
                    ops.append(op)
            elif kind in {"CallExpr", "CXXMemberCallExpr"}:
                callee = find_first_declref_name(node)
                if callee:
                    token += ":CALL"
                    calls.append(callee)
            elif kind == "DeclRefExpr":
                ref = node.get("referencedDecl") or {}
                token += ":" + str(ref.get("kind") or "ref")
            elif kind == "VarDecl":
                token += ":" + str((node.get("type") or {}).get("qualType") or "var")
            kinds.append(token)

    sve_categories = [sve_category(x) for x in sve_calls]
    sig = function_signature_from_ast(fn)
    kind_stream = ",".join(kinds)
    call_stream = ",".join("SVE:" + sve_category(c) if c.startswith("sv") else "CALL" for c in calls)
    feature = "|".join(
        [
            sig,
            "funcs=" + str(function_count),
            "helpers=" + str(max(0, function_count - 1)),
            "kinds=" + kind_stream,
            "calls=" + call_stream,
            "sve=" + ",".join(sve_categories),
        ]
    )
    return {
        "ast_ok": 1,
        "ast_error": "",
        "function_count": function_count,
        "helper_def_count": max(0, function_count - 1),
        "signature_shape": sig,
        "ast_hash": digest(feature),
        "kind_hash": digest(sig + "|" + kind_stream),
        "call_hash": digest(sig + "|" + call_stream + "|" + ",".join(sve_categories)),
        "node_count": sum(counts.values()),
        "for_count": counts.get("ForStmt", 0),
        "while_count": counts.get("WhileStmt", 0),
        "if_count": counts.get("IfStmt", 0),
        "array_subscript_count": counts.get("ArraySubscriptExpr", 0),
        "binary_operator_count": counts.get("BinaryOperator", 0),
        "call_count": counts.get("CallExpr", 0) + counts.get("CXXMemberCallExpr", 0),
        "sve_calls": ",".join(sve_calls[:80]),
        "sve_categories": ",".join(sve_categories[:80]),
    }


def read_items(paths: list[tuple[Path, str]]) -> list[dict[str, Any]]:
    items = []
    idx = 0
    for path, split in paths:
        with path.open() as f:
            for line_no, line in enumerate(f, 1):
                row = json.loads(line)
                idx += 1
                items.append(
                    {
                        "_idx": idx,
                        "split": split,
                        "line": line_no,
                        "task_id": str(row.get("task_id") or ""),
                        "source_type": str(row.get("source_type") or ""),
                        "source_prefix": source_prefix(str(row.get("task_id") or "")),
                        "prompt": get_prompt(row),
                        "code": get_response(row),
                    }
                )
    return items


def process_one(
    item: dict[str, Any],
    clang: str,
    tmp_dir: Path,
    timeout: float,
    use_ast_dump_filter: bool,
    stub_source: bool,
) -> dict[str, Any]:
    code = item["code"]
    filter_name = prompt_target_name(item["prompt"]) if use_ast_dump_filter else ""
    ok, ast_text, diag = run_clang_ast(code, clang, tmp_dir, item["_idx"], timeout, filter_name, stub_source)
    feats = ast_features(ast_text, item["prompt"], code) if ast_text else {
        "ast_ok": 0,
        "ast_error": "clang_failed",
        "function_count": 0,
        "helper_def_count": 0,
        "signature_shape": fallback_signature(item["prompt"], code),
        "ast_hash": "",
        "kind_hash": "",
        "call_hash": "",
    }
    feats["clang_rc_ok"] = int(ok)
    feats["clang_diag"] = diag.replace("\t", " ").replace("\n", " ")[:500]
    return {
        "split": item["split"],
        "line": item["line"],
        "task_id": item["task_id"],
        "source_type": item["source_type"],
        "source_prefix": item["source_prefix"],
        "exact_code_hash": digest(clean_code(code)),
        **feats,
    }


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w") as f:
        f.write("\t".join(fields) + "\n")
        for row in rows:
            f.write("\t".join(str(row.get(k, "")).replace("\t", " ").replace("\n", " ") for k in fields) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--clang", default="clang++")
    ap.add_argument("--jobs", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    ap.add_argument("--timeout", type=float, default=20.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ast-dump-filter", action="store_true")
    ap.add_argument("--stub-source", action="store_true")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    items = read_items([(Path(args.train), "train"), (Path(args.dev), "dev")])
    if args.limit:
        items = items[: args.limit]

    with tempfile.TemporaryDirectory(prefix="clang_ast_audit_") as td:
        tmp_dir = Path(td)
        rows: list[dict[str, Any]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as ex:
            futs = [
                ex.submit(
                    process_one,
                    item,
                    args.clang,
                    tmp_dir,
                    args.timeout,
                    args.ast_dump_filter,
                    args.stub_source,
                )
                for item in items
            ]
            for i, fut in enumerate(concurrent.futures.as_completed(futs), 1):
                rows.append(fut.result())
                if i % 250 == 0:
                    print(f"[progress] {i}/{len(items)}", flush=True)

    rows.sort(key=lambda r: (r["split"] != "train", int(r["line"])))
    fields = [
        "split",
        "line",
        "task_id",
        "source_type",
        "source_prefix",
        "clang_rc_ok",
        "ast_ok",
        "ast_error",
        "function_count",
        "helper_def_count",
        "signature_shape",
        "exact_code_hash",
        "ast_hash",
        "kind_hash",
        "call_hash",
        "node_count",
        "for_count",
        "while_count",
        "if_count",
        "array_subscript_count",
        "binary_operator_count",
        "call_count",
        "sve_categories",
        "sve_calls",
        "clang_diag",
    ]
    write_tsv(out_dir / "row_clang_ast_fingerprints.tsv", rows, fields)

    group_rows = []
    for kind in ["exact_code_hash", "ast_hash", "kind_hash", "call_hash"]:
        groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        for row in rows:
            h = row.get(kind) or ""
            if h:
                groups[h].append(row)
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
                    "top_source_type": ";".join(f"{k}:{v}" for k, v in by_source.most_common(6)),
                    "top_prefix": ";".join(f"{k}:{v}" for k, v in by_prefix.most_common(6)),
                    "example_task_ids": ";".join(m["task_id"] for m in members[:15]),
                    "signature_shape": sample.get("signature_shape", ""),
                    "sve_categories": sample.get("sve_categories", ""),
                    "sve_calls": sample.get("sve_calls", ""),
                }
            )
    group_rows.sort(key=lambda r: (r["kind"], -int(r["n"]), r["hash"]))
    write_tsv(
        out_dir / "clang_ast_duplicate_group_summary.tsv",
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
        "rows": len(rows),
        "train_rows": sum(r["split"] == "train" for r in rows),
        "dev_rows": sum(r["split"] == "dev" for r in rows),
        "clang_rc_ok": sum(r["clang_rc_ok"] for r in rows),
        "ast_ok": sum(r["ast_ok"] for r in rows),
        "helper_def_rows": sum(int(r.get("helper_def_count") or 0) > 0 for r in rows),
        "source_type_counts": dict(collections.Counter(r["source_type"] for r in rows).most_common()),
        "ast_error_counts": dict(collections.Counter(r["ast_error"] for r in rows).most_common(20)),
        "duplicate_groups": {},
    }
    for kind in ["exact_code_hash", "ast_hash", "kind_hash", "call_hash"]:
        c = collections.Counter(r.get(kind) for r in rows if r.get(kind))
        summary["duplicate_groups"][kind] = {
            "unique": len(c),
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
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
