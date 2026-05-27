#!/usr/bin/env python3
"""Build AST-derived semantic bootstrap records for NL2SVE repair.

The script is intentionally conservative: clang JSON AST is used to extract
loop/assignment/read-write/call structure, then a small generic classifier
chooses the bootstrap route and emits code-like semantic pseudocode.  It is
not benchmark-id based and can be used for SimdBench, VecIntrinBench, and
ARM SIMD Loop style problem files.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import tarfile
import textwrap
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


DEFAULT_REMOTE_USER = "ec2-user"
DEFAULT_REMOTE_HOST = "ec2-18-183-202-105.ap-northeast-1.compute.amazonaws.com"
DEFAULT_REMOTE_KEY = "/home/user/sve-gen.pem"
DEFAULT_REMOTE_TMP_ROOT = "~/simdbench_remote_tmp"

STD_WRAPPER = r"""
#include <algorithm>
#include <array>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <string>
#include <vector>
using std::size_t;
#ifndef restrict
#define restrict
#endif
#ifndef DCTSIZE
#define DCTSIZE 8
#endif
#ifndef MAXJSAMPLE
#define MAXJSAMPLE 255
#endif
typedef unsigned char JSAMPLE;
typedef JSAMPLE *JSAMPROW;
typedef JSAMPROW *JSAMPARRAY;
typedef JSAMPARRAY *JSAMPIMAGE;
typedef short DCTELEM;
typedef float float16_t;
typedef float FLOAT16_t;
static inline float fp16_to_native(float x) { return x; }
static inline float16_t native_to_fp16(float x) { return (float16_t)x; }
"""


FUNC_DEF_RE = re.compile(
    r"([A-Za-z_~][\w:<>,\s\*&\[\]]*?)\s+([A-Za-z_][A-Za-z0-9_:]*)\s*\(([^;{}]*)\)\s*(?:const\s*)?\{",
    re.S,
)
CALL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_:]*)\s*\(")
COMMENT_RE = re.compile(r"//.*?$|/\*.*?\*/", re.S | re.M)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sanitize_filename(text: str) -> str:
    out = re.sub(r"[^A-Za-z0-9_.-]+", "_", text or "task").strip("_")
    return (out or "task")[:140]


def strip_comments(src: str) -> str:
    return COMMENT_RE.sub(" ", src or "")


def normalize_ws(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def stable_task_id(row: dict[str, Any], idx: int) -> str:
    for key in ("task_id", "id", "name", "source_name"):
        val = str(row.get(key) or "").strip()
        if val:
            return val
    return f"row_{idx:04d}"


def infer_benchmark(path: Path, row: dict[str, Any]) -> str:
    name = str(path).lower()
    if "vecintrin" in name:
        return "vecintrinbench"
    if "arm_simd" in name or "arm-simd" in name or "arm_simd_loops" in name:
        return "arm_simd_loops"
    if "simdbench" in name:
        return "simdbench"
    source_type = str(row.get("source_type") or "").lower()
    if "vecintrin" in source_type:
        return "vecintrinbench"
    if "arm_simd" in source_type:
        return "arm_simd_loops"
    return "auto"


def first_function_name(code: str) -> str:
    for m in FUNC_DEF_RE.finditer(strip_comments(code)):
        name = m.group(2).split("::")[-1]
        if name not in {"if", "for", "while", "switch"}:
            return name
    return ""


def function_name_from_signature(sig: str) -> str:
    m = re.search(r"\b([A-Za-z_][A-Za-z0-9_:]*)\s*\(", str(sig or ""))
    return m.group(1).split("::")[-1] if m else ""


def select_scalar_source(row: dict[str, Any], benchmark: str) -> tuple[str, str, str]:
    """Return (source, function_name, source_label)."""
    audited = str(row.get("audited_nohelper_scalar") or "").strip()
    audited_ep = str(row.get("audited_nohelper_scalar_entrypoint") or "").strip()
    if audited:
        return audited, audited_ep or first_function_name(audited), "audited_nohelper_scalar"

    serial = str(row.get("serial_c_code") or "").strip()
    if serial:
        name = (
            function_name_from_signature(str(row.get("target_signature") or ""))
            or str(row.get("entrypoint_scalar") or "").strip()
            or first_function_name(serial)
        )
        return serial, name, "serial_c_code"

    sol = str(row.get("solution_scalar") or "").strip()
    if sol:
        return sol, str(row.get("entrypoint_scalar") or "").strip() or first_function_name(sol), "solution_scalar"

    response = str(row.get("response") or "").strip()
    if response:
        return response, first_function_name(response), "response"

    return "", "", "missing"


def display_signature(row: dict[str, Any], fallback_func: str, ast_header: str = "") -> str:
    target = str(row.get("target_signature") or "").strip()
    if target:
        return normalize_ws(target.rstrip(";"))
    prompt = str(row.get("prompt") or "")
    simd = str(row.get("entrypoint_simd") or "").strip()
    names = [simd, fallback_func]
    for name in names:
        if not name:
            continue
        # Prefer a real function-signature line.  A broad multi-line regex can
        # accidentally swallow examples such as ">>> foo(3)" from NL comments.
        m = re.search(
            rf"(?m)^[ \t]*([A-Za-z_~][\w:<>, \t\*&\[\]]*?\b{re.escape(name)}[ \t]*\([^;{{}}\n]*\))[ \t]*(?:\{{|;)?[ \t]*$",
            prompt,
        )
        if m:
            return normalize_ws(m.group(1))
    if ast_header:
        return normalize_ws(ast_header)
    return f"{fallback_func or 'target'}(...)"


def prompt_type_preamble(prompt: str) -> str:
    """Recover small user-defined type declarations carried only by the prompt."""
    text = str(prompt or "")
    chunks: list[str] = []
    for m in re.finditer(r"```(?:c|cpp|c\+\+)?\s*(.*?)```", text, re.S | re.I):
        block = m.group(1)
        if re.search(r"\b(?:typedef|struct|using)\b", block):
            chunks.append(block.strip())
    for m in re.finditer(r"typedef\s+struct\s*(?:[A-Za-z_][A-Za-z0-9_]*)?\s*\{.*?\}\s*[A-Za-z_][A-Za-z0-9_]*\s*;", text, re.S):
        decl = m.group(0).strip()
        if decl not in chunks:
            chunks.append(decl)
    return "\n\n".join(chunks)


def source_for_clang(code: str) -> str:
    src = str(code or "")
    # AST-only C++ compatibility shim for C-style packet parsers:
    #   T *p = (void *)(base + off);
    # is valid C but not C++.  Rewrite only the explicit initializer cast so
    # clang can build the AST; this does not alter any benchmark/problem file.
    src = re.sub(
        r"((?P<typ>(?:const\s+)?[A-Za-z_][A-Za-z0-9_:<>]*\s*\*)\s*[A-Za-z_][A-Za-z0-9_]*\s*=\s*)\(void\s*\*\)\s*\(",
        lambda m: f"{m.group(1)}({m.group('typ').strip()})(",
        src,
    )
    # Some contextualized external rows carry both the original half typedef
    # and the evaluation shim typedef.  Keep the wrapper's AST-only alias so
    # clang can parse the function body without treating duplicate typedefs as
    # a source-level failure.
    src = re.sub(r"(?m)^\s*typedef\s+(?:__fp16|float)\s+float16_t\s*;\s*$", "", src)
    src = re.sub(r"(?m)^\s*typedef\s+(?:__fp16|float)\s+FLOAT16_t\s*;\s*$", "", src)
    src = re.sub(r"(?m)^\s*#\s*define\s+fp16_to_native\s*\([^)]*\).*$", "", src)
    src = re.sub(r"(?m)^\s*#\s*define\s+native_to_fp16\s*\([^)]*\).*$", "", src)
    if "#include" in src[:300] or "typedef" in src[:500] or "using " in src[:500]:
        return STD_WRAPPER + "\n" + src
    return STD_WRAPPER + "\n" + src


def ssh_base(args: argparse.Namespace) -> list[str]:
    cmd = ["ssh", "-p", str(args.remote_port)]
    if args.remote_ssh_key:
        cmd += ["-i", args.remote_ssh_key]
    if args.remote_no_strict_hostkey:
        cmd += ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null"]
    cmd.append(f"{args.remote_user}@{args.remote_host}")
    return cmd


def scp_base(args: argparse.Namespace) -> list[str]:
    cmd = ["scp", "-P", str(args.remote_port)]
    if args.remote_ssh_key:
        cmd += ["-i", args.remote_ssh_key]
    if args.remote_no_strict_hostkey:
        cmd += ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null"]
    return cmd


def remote_tmp_root_abs(args: argparse.Namespace) -> str:
    root = str(args.remote_tmp_root or DEFAULT_REMOTE_TMP_ROOT).strip() or DEFAULT_REMOTE_TMP_ROOT
    if root == "~":
        return f"/home/{args.remote_user}"
    if root.startswith("~/"):
        return f"/home/{args.remote_user}/{root[2:]}"
    return root


def run(cmd: list[str], *, timeout: int = 300, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=check,
    )


def prepare_sources(rows: list[dict[str, Any]], problem_file: Path, out_dir: Path, limit: int = 0) -> list[dict[str, Any]]:
    src_dir = out_dir / "sources"
    src_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        if limit and len(manifest) >= limit:
            break
        task_id = stable_task_id(row, idx)
        bench = infer_benchmark(problem_file, row)
        source, func, label = select_scalar_source(row, bench)
        if not source.strip() or not func.strip():
            manifest.append(
                {
                    "task_id": task_id,
                    "row_index": idx,
                    "benchmark": bench,
                    "status": "missing_source_or_function",
                    "source_label": label,
                    "function": func,
                }
            )
            continue
        preamble_parts: list[str] = []
        candidate_prelude = str(row.get("candidate_prelude") or "").strip()
        if candidate_prelude:
            preamble_parts.append(candidate_prelude)
        prompt_preamble = "" if candidate_prelude else prompt_type_preamble(str(row.get("prompt") or ""))
        if prompt_preamble:
            preamble_parts.append(prompt_preamble)
        preamble = "\n\n".join(preamble_parts)
        if preamble and normalize_ws(preamble) not in normalize_ws(source):
            source = preamble + "\n\n" + source
        fname = f"{idx:04d}_{sanitize_filename(task_id)}.cpp"
        (src_dir / fname).write_text(source_for_clang(source), encoding="utf-8")
        manifest.append(
            {
                "task_id": task_id,
                "row_index": idx,
                "benchmark": bench,
                "status": "prepared",
                "source_label": label,
                "function": func,
                "source_file": fname,
                "display_signature": display_signature(row, str(row.get("entrypoint_simd") or func), ""),
            }
        )
    write_jsonl(out_dir / "manifest.jsonl", manifest)
    return manifest


def run_remote_clang_ast(args: argparse.Namespace, out_dir: Path) -> list[dict[str, Any]]:
    bundle = out_dir / "ast_inputs.tgz"
    with tarfile.open(bundle, "w:gz") as tf:
        tf.add(out_dir / "sources", arcname="sources")
        tf.add(out_dir / "manifest.jsonl", arcname="manifest.jsonl")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    remote_root = f"{remote_tmp_root_abs(args).rstrip('/')}/ast_semantic_bootstrap_{stamp}_{os.getpid()}"
    remote_q = shlex.quote(remote_root)
    run(ssh_base(args) + [f"mkdir -p {remote_q}"], timeout=60)
    run(scp_base(args) + [str(bundle), f"{args.remote_user}@{args.remote_host}:{remote_root}/inputs.tgz"], timeout=300)

    remote_py = r'''
import json, os, subprocess, tarfile
root = os.getcwd()
with tarfile.open("inputs.tgz", "r:gz") as tf:
    tf.extractall(root)
os.makedirs("ast", exist_ok=True)
meta = []
for line in open("manifest.jsonl", encoding="utf-8"):
    rec = json.loads(line)
    if rec.get("status") != "prepared":
        meta.append(rec)
        continue
    src = os.path.join("sources", rec["source_file"])
    func = rec.get("function") or ""
    ast_path = os.path.join("ast", rec["source_file"] + ".ast.json")
    cmd = [
        "clang++", "-std=c++17", "-fsyntax-only", "-Wno-everything",
        "-Xclang", "-ast-dump=json",
        "-Xclang", "-ast-dump-filter", "-Xclang", func,
        src,
    ]
    try:
        p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
    except subprocess.TimeoutExpired as e:
        rec.update({"ast_ok": False, "ast_status": "timeout", "clang_stderr_tail": str(e)[-2000:]})
        meta.append(rec)
        continue
    rec["clang_rc"] = p.returncode
    rec["clang_stderr_tail"] = (p.stderr or "")[-4000:]
    if p.returncode == 0 and (p.stdout or "").strip():
        open(ast_path, "w", encoding="utf-8").write(p.stdout)
        rec.update({"ast_ok": True, "ast_status": "ok", "ast_file": os.path.basename(ast_path)})
    else:
        rec.update({"ast_ok": False, "ast_status": "clang_failed"})
    meta.append(rec)
open("ast_meta.jsonl", "w", encoding="utf-8").write("\n".join(json.dumps(x, ensure_ascii=False) for x in meta) + "\n")
with tarfile.open("ast_outputs.tgz", "w:gz") as tf:
    tf.add("ast_meta.jsonl")
    if os.path.isdir("ast"):
        tf.add("ast")
'''
    run(ssh_base(args) + [f"cd {remote_q} && python3 - <<'PY'\n{remote_py}\nPY"], timeout=max(120, args.timeout))

    local_tgz = out_dir / "ast_outputs.tgz"
    with local_tgz.open("wb") as f:
        proc = subprocess.run(
            ssh_base(args) + [f"cat {remote_q}/ast_outputs.tgz"],
            stdout=f,
            stderr=subprocess.PIPE,
            timeout=max(120, args.timeout),
            check=True,
        )
        _ = proc
    with tarfile.open(local_tgz, "r:gz") as tf:
        tf.extractall(out_dir / "remote_ast")
    meta_path = out_dir / "remote_ast" / "ast_meta.jsonl"
    return load_jsonl(meta_path)


def iter_nodes(node: Any) -> Iterable[dict[str, Any]]:
    if isinstance(node, dict):
        yield node
        for child in node.get("inner") or []:
            yield from iter_nodes(child)
    elif isinstance(node, list):
        for item in node:
            yield from iter_nodes(item)


def find_function(ast: dict[str, Any], name: str) -> dict[str, Any] | None:
    funcs = [n for n in iter_nodes(ast) if n.get("kind") in {"FunctionDecl", "CXXMethodDecl"}]
    named = [fn for fn in funcs if str(fn.get("name") or "").split("::")[-1] == name]
    candidates = named or funcs

    def _has_body(fn: dict[str, Any]) -> bool:
        return any((c.get("kind") == "CompoundStmt") for c in (fn.get("inner") or []) if isinstance(c, dict))

    def _is_main_file(fn: dict[str, Any]) -> bool:
        loc = fn.get("loc") or {}
        file_name = str(loc.get("file") or "")
        if not file_name and isinstance(loc.get("includedFrom"), dict):
            file_name = str((loc.get("includedFrom") or {}).get("file") or "")
        return file_name.startswith("sources/")

    body = [fn for fn in candidates if _has_body(fn)]
    main_body = [fn for fn in body if _is_main_file(fn)]
    if main_body:
        return main_body[-1]
    if body:
        return body[-1]
    return candidates[0] if candidates else None


def load_ast_json(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        pos = 0
        objs: list[Any] = []
        while pos < len(text):
            while pos < len(text) and text[pos].isspace():
                pos += 1
            if pos >= len(text):
                break
            obj, end = decoder.raw_decode(text, pos)
            objs.append(obj)
            pos = end
        return {"kind": "TranslationUnitDecl", "inner": [x for x in objs if isinstance(x, dict)]}


def node_offsets(node: dict[str, Any]) -> tuple[int | None, int | None]:
    rng = node.get("range") or {}
    begin = rng.get("begin") or {}
    end = rng.get("end") or {}
    b = begin.get("offset")
    e = end.get("offset")
    tok_len = end.get("tokLen")
    if isinstance(e, int) and isinstance(tok_len, int) and tok_len > 0:
        e = e + tok_len - 1
    return (b if isinstance(b, int) else None, e if isinstance(e, int) else None)


def repair_fragment_start_offset(src: str, b: int) -> int:
    while (
        b > 0
        and (src[b - 1].isalnum() or src[b - 1] == "_")
        and (src[b].isalnum() or src[b] in {"_", "["})
    ):
        b -= 1
    return b


def snippet(node: dict[str, Any], src: str, max_chars: int = 400) -> str:
    b, e = node_offsets(node)
    if b is None or e is None or b < 0 or e < b or b >= len(src):
        return ""
    b = repair_fragment_start_offset(src, b)
    e = min(len(src) - 1, e)
    while e + 1 < len(src) and src[e + 1] in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_)]":
        e += 1
    text = normalize_ws(src[b : e + 1])
    if len(text) > max_chars:
        text = text[: max_chars - 3] + "..."
    return text


def function_header_from_source(src: str, func: str) -> str:
    clean = strip_comments(src)
    m = re.search(rf"([A-Za-z_~][\w:<>,\s\*&\[\]]*?\b{re.escape(func)}\s*\([^;{{}}]*\))\s*(?:const\s*)?\{{", clean, re.S)
    return normalize_ws(m.group(1)) if m else ""


def split_assignment(text: str) -> tuple[str, str, str]:
    for op in ("+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>="):
        if op in text:
            lhs, rhs = text.split(op, 1)
            return lhs.strip(), op, rhs.strip()
    m = re.search(r"(?<![=!<>])=(?!=)", text)
    if m:
        return text[: m.start()].strip(), "=", text[m.end() :].strip()
    return "", "", ""


def split_unary_update(text: str) -> tuple[str, str, str]:
    s = normalize_ws(text).rstrip(";")
    lhs_pat = r"[A-Za-z_][A-Za-z0-9_]*(?:\s*(?:->|\.)\s*[A-Za-z_][A-Za-z0-9_]*)?(?:\s*\[[^;\n{}]+\])*"
    m = re.fullmatch(rf"\+\+\s*({lhs_pat})", s)
    if m:
        return normalize_ws(m.group(1)), "+=", "1"
    m = re.fullmatch(rf"({lhs_pat})\s*\+\+", s)
    if m:
        return normalize_ws(m.group(1)), "+=", "1"
    m = re.fullmatch(rf"--\s*({lhs_pat})", s)
    if m:
        return normalize_ws(m.group(1)), "-=", "1"
    m = re.fullmatch(rf"({lhs_pat})\s*--", s)
    if m:
        return normalize_ws(m.group(1)), "-=", "1"
    return "", "", ""


def normalize_rw_snippet_key(text: str) -> str:
    key = normalize_ws(str(text or "")).rstrip(";").strip()
    return re.sub(r"\s+", "", key)


def normalize_rw_semantic_key(lhs: str, op: str, rhs: str) -> tuple[str, str, str]:
    return (
        re.sub(r"\s+", "", normalize_ws(lhs)),
        normalize_ws(op),
        re.sub(r"\s+", "", normalize_ws(rhs).rstrip(";")),
    )


def iter_simple_array_assignments(text: str) -> Iterable[tuple[int, str, str, str]]:
    """Yield simple array assignments from compact branch snippets.

    Clang JSON ranges for macro-heavy one-line branches can omit the then-arm
    assignment as a BinaryOperator in the filtered AST.  This fallback is kept
    deliberately narrow: only array-element assignments terminated by ';' are
    recovered from an already-extracted branch snippet.
    """
    pattern = re.compile(
        r"(?P<lhs>\b[A-Za-z_][A-Za-z0-9_]*\s*\[[^\];]+\])\s*"
        r"(?P<op>\+=|-=|\*=|/=|%=|&=|\|=|\^=|<<=|>>=|=)\s*"
        r"(?P<rhs>[^;]+)"
    )
    for m in pattern.finditer(str(text or "")):
        lhs = normalize_ws(m.group("lhs"))
        op = m.group("op")
        rhs = normalize_ws(m.group("rhs"))
        # Do not treat comparison tails or loop headers as assignments.
        if rhs.startswith("=") or " for " in lhs:
            continue
        yield m.start(), lhs, op, rhs


def iter_simple_statement_assignments(text: str) -> Iterable[tuple[int, str, str, str]]:
    """Yield simple source-level assignment statements missed by clang JSON.

    Clang represents some C++ overloaded operations, string/vector updates, and
    one-line bodies as operator/call nodes that do not always survive the narrow
    BinaryOperator extraction above.  This fallback only recovers semicolon
    terminated statements with a simple scalar/array/member lhs; loop-header
    matches are later filtered with the existing loop-initializer guard.
    """
    pattern = re.compile(
        r"(?P<lhs>(?:\*\s*[A-Za-z_][A-Za-z0-9_]*\s*(?:\+\+|--)|"
        r"\b[A-Za-z_][A-Za-z0-9_]*(?:\s*(?:->|\.)\s*[A-Za-z_][A-Za-z0-9_]*)?(?:\s*\[[^;\n{}]+\])*))\s*"
        r"(?P<op>\+=|-=|\*=|/=|%=|&=|\|=|\^=|<<=|>>=|=)\s*"
        r"(?P<rhs>(?:\{[^;\n{}]*\}|[^;{}])+);"
    )
    for m in pattern.finditer(str(text or "")):
        lhs = normalize_ws(m.group("lhs"))
        op = m.group("op")
        rhs = normalize_ws(m.group("rhs"))
        prefix = text[max(0, m.start() - 24) : m.start()]
        if re.search(r"\b(return|if|while|switch)\s*\([^)]*$", prefix):
            continue
        if rhs.startswith("="):
            continue
        comma_parts = [part.strip() for part in rhs.split(",") if part.strip()]
        if len(comma_parts) > 1 and any("=" in part for part in comma_parts[1:]):
            yield m.start(), lhs, op, comma_parts[0]
            for part in comma_parts[1:]:
                sub_lhs, sub_op, sub_rhs = split_assignment(part)
                if sub_lhs and sub_op and sub_rhs:
                    yield m.start() + m.group(0).find(part), normalize_ws(sub_lhs), sub_op, normalize_ws(sub_rhs)
            continue
        yield m.start(), lhs, op, rhs
    update_pattern = re.compile(
        r"(?P<expr>(?:\+\+|--)\s*\b[A-Za-z_][A-Za-z0-9_]*(?:\s*(?:->|\.)\s*[A-Za-z_][A-Za-z0-9_]*)?(?:\s*\[[^;\n{}]+\])*|"
        r"\b[A-Za-z_][A-Za-z0-9_]*(?:\s*(?:->|\.)\s*[A-Za-z_][A-Za-z0-9_]*)?(?:\s*\[[^;\n{}]+\])*\s*(?:\+\+|--))\s*;"
    )
    for m in update_pattern.finditer(str(text or "")):
        lhs, op, rhs = split_unary_update(m.group("expr"))
        if lhs and op and rhs:
            yield m.start(), lhs, op, rhs


def iter_source_vector_ctor_decls(text: str) -> Iterable[tuple[int, str, str, str]]:
    """Recover one-line std::vector constructor declarations missed by AST spans."""
    pattern = re.compile(
        r"(?m)^(?P<prefix>\s*)(?P<decl>std::vector\s*<.+>\s+"
        r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\((?P<args>[^;\n]*)\))\s*;"
    )
    for m in pattern.finditer(str(text or "")):
        decl = normalize_ws(m.group("decl"))
        name = str(m.group("name") or "").strip()
        args = normalize_ws(m.group("args"))
        if not name or "=" in decl:
            continue
        type_part = normalize_ws(decl[: decl.rfind(name)].strip())
        if not type_part or "std::vector" not in type_part:
            continue
        yield m.start("decl"), type_part, name, args


def branch_condition_abs_span(branch: dict[str, Any], src: str) -> tuple[int, int] | None:
    begin = branch.get("begin")
    end = branch.get("end")
    if not isinstance(begin, int) or not isinstance(end, int):
        return None
    if begin < 0 or begin >= len(src):
        return None
    raw = src[begin : min(len(src), end + 1)]
    m = re.search(r"\bif\s*\(", raw)
    if not m:
        return None
    open_pos = raw.find("(", m.start())
    close_pos = _find_matching_paren(raw, open_pos)
    if close_pos < 0:
        return None
    return begin + open_pos + 1, begin + close_pos


def offset_in_branch_condition(offset: Any, branches: list[dict[str, Any]], src: str) -> bool:
    if not isinstance(offset, int):
        return False
    for branch in branches:
        span = branch_condition_abs_span(branch, src)
        if not span:
            continue
        lo, hi = span
        if lo <= offset < hi:
            return True
    return False


def array_refs(text: str) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    s = str(text or "")
    i = 0
    while i < len(s):
        m = re.search(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\[", s[i:])
        if not m:
            break
        base = m.group(1)
        lbr = i + m.end() - 1
        depth = 0
        j = lbr
        while j < len(s):
            if s[j] == "[":
                depth += 1
            elif s[j] == "]":
                depth -= 1
                if depth == 0:
                    out.append({"base": base, "index": normalize_ws(s[lbr + 1 : j])})
                    i = j + 1
                    break
            j += 1
        else:
            break
    return out


def normalize_index(idx: str) -> str:
    idx = normalize_ws(idx)
    idx = re.sub(r"\b(size_t|int|long|uint32_t|uint64_t|std::size_t)\s*\(([^)]*)\)", r"\2", idx)
    idx = re.sub(r"\(([^()]+)\)", r"\1", idx)
    idx = re.sub(r"\s+", "", idx)
    return idx


def is_probably_same_index(a: str, b: str) -> bool:
    return normalize_index(a) == normalize_index(b)


def is_for_header_initializer(name: str, rhs: str, begin: Any, loops: list[dict[str, Any]]) -> bool:
    if not name or not isinstance(begin, int):
        return False
    rhs_norm = normalize_ws(rhs)
    for loop in loops:
        if not contains_offset(loop, begin):
            continue
        header = normalize_ws(str(loop.get("snippet") or "").split("{", 1)[0])
        m = re.match(
            r"for\s*\(\s*(?:(?:[A-Za-z_][\w:<>]*\s+)+)?([A-Za-z_]\w*)\s*=\s*([^;]+?)\s*;",
            header,
        )
        if m and m.group(1) == name and normalize_ws(m.group(2)) == rhs_norm:
            return True
    return False


def for_header_end_offset(loop: dict[str, Any]) -> int | None:
    begin = loop.get("begin")
    text = str(loop.get("snippet") or "")
    if not isinstance(begin, int):
        return None
    if not loop_header_line(loop).lstrip().startswith("for"):
        return None
    m = re.search(r"\bfor\s*\(", text)
    if not m:
        return None
    pos = m.end() - 1
    depth = 0
    while pos < len(text):
        ch = text[pos]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return begin + pos
        pos += 1
    return None


def is_for_header_control(begin: Any, loops: list[dict[str, Any]]) -> bool:
    if not isinstance(begin, int):
        return False
    for loop in loops:
        loop_begin = loop.get("begin")
        header_end = for_header_end_offset(loop)
        if isinstance(loop_begin, int) and isinstance(header_end, int) and loop_begin <= begin <= header_end:
            return True
    return False


def find_top_level_else_offset(text: str, absolute_begin: int | None) -> int | None:
    """Return the source offset of the top-level else belonging to this if.

    This is intentionally conservative.  It only recognizes the common
    `if (...) { ... } else ...` or `if (...) stmt; else ...` forms in the
    already-extracted source snippet, which is enough to avoid rendering both
    arms under the same `if` in the bootstrap text.
    """
    if not isinstance(absolute_begin, int):
        return None
    m = re.search(r"\bif\s*\(", text)
    if not m:
        return None
    pos = m.end() - 1
    depth = 0
    while pos < len(text):
        ch = text[pos]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                pos += 1
                break
        pos += 1
    while pos < len(text) and text[pos].isspace():
        pos += 1
    if pos >= len(text):
        return None
    if text[pos] == "{":
        brace_depth = 0
        while pos < len(text):
            ch = text[pos]
            if ch == "{":
                brace_depth += 1
            elif ch == "}":
                brace_depth -= 1
                if brace_depth == 0:
                    pos += 1
                    break
            pos += 1
    else:
        semi = text.find(";", pos)
        if semi < 0:
            return None
        pos = semi + 1
    while pos < len(text) and text[pos].isspace():
        pos += 1
    if re.match(r"\belse\b", text[pos:]):
        return absolute_begin + pos
    return None


def extract_ast_features(fn: dict[str, Any], src: str) -> dict[str, Any]:
    nodes = list(iter_nodes(fn))
    loops = []
    branches = []
    assignments = []
    returns = []
    controls = []
    calls = []
    array_subscripts = []
    members = []
    var_decls = []
    for n in nodes:
        kind = n.get("kind")
        if kind in {"ForStmt", "WhileStmt", "DoStmt", "CXXForRangeStmt"}:
            s = snippet(n, src, max_chars=700)
            if s:
                b, e = node_offsets(n)
                loops.append({"kind": kind, "snippet": s, "begin": b, "end": e})
        elif kind == "IfStmt":
            s = snippet(n, src, max_chars=700)
            if s:
                b, e = node_offsets(n)
                else_begin = None
                if isinstance(b, int) and isinstance(e, int) and 0 <= b < len(src):
                    fixed_b = repair_fragment_start_offset(src, b)
                    raw = src[fixed_b : min(len(src), e + 1)]
                    else_begin = find_top_level_else_offset(raw, fixed_b)
                    b = fixed_b
                branches.append({"kind": kind, "snippet": s, "begin": b, "end": e, "else_begin": else_begin})
        elif kind in {"BinaryOperator", "CompoundAssignOperator"}:
            op = str(n.get("opcode") or "")
            if kind == "CompoundAssignOperator" or op in {"=", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>="}:
                s = snippet(n, src, max_chars=4000)
                if s:
                    b, e = node_offsets(n)
                    assignments.append({"kind": kind, "opcode": op or kind, "snippet": s, "begin": b, "end": e})
        elif kind == "UnaryOperator":
            op = str(n.get("opcode") or "")
            if "++" in op or "--" in op:
                s = snippet(n, src, max_chars=4000)
                if s:
                    b, e = node_offsets(n)
                    assignments.append({"kind": kind, "opcode": op, "snippet": s, "begin": b, "end": e})
        elif kind == "ReturnStmt":
            s = snippet(n, src)
            if s:
                b, e = node_offsets(n)
                returns.append({"snippet": s, "begin": b, "end": e})
        elif kind in {"BreakStmt", "ContinueStmt"}:
            s = snippet(n, src)
            if s:
                b, e = node_offsets(n)
                controls.append({"kind": kind, "snippet": s, "begin": b, "end": e})
        elif kind in {"CallExpr", "CXXMemberCallExpr"}:
            s = snippet(n, src)
            if s:
                b, e = node_offsets(n)
                name = ""
                m = CALL_RE.search(s)
                if m:
                    name = m.group(1).split("::")[-1]
                calls.append({"name": name, "snippet": s, "begin": b, "end": e})
        elif kind == "ArraySubscriptExpr":
            s = snippet(n, src)
            if s:
                array_subscripts.append(s)
        elif kind == "MemberExpr":
            s = snippet(n, src)
            if s:
                members.append(s)
        elif kind in {"VarDecl", "ParmVarDecl"}:
            name = str(n.get("name") or "")
            typ = str((n.get("type") or {}).get("qualType") or "")
            s = snippet(n, src)
            if name or typ:
                b, e = node_offsets(n)
                var_decls.append({"kind": kind, "name": name, "type": typ, "snippet": s, "begin": b, "end": e})

    read_write = []
    for item in assignments:
        if item.get("kind") == "UnaryOperator" and offset_in_branch_condition(item.get("begin"), branches, src):
            continue
        lhs, op, rhs = split_assignment(item["snippet"])
        if not op and item.get("kind") == "UnaryOperator":
            lhs, op, rhs = split_unary_update(item["snippet"])
        if not op:
            continue
        writes = array_refs(lhs)
        reads = array_refs(rhs)
        if op != "=":
            reads.extend(array_refs(lhs))
        read_write.append({
            "lhs": lhs,
            "op": op,
            "rhs": rhs,
            "writes": writes,
            "reads": reads,
            "snippet": item["snippet"],
            "begin": item.get("begin"),
            "end": item.get("end"),
            "source": "assignment",
            "is_loop_initializer": is_for_header_control(item.get("begin"), loops),
        })
    for item in var_decls:
        if item.get("kind") != "VarDecl":
            continue
        s = str(item.get("snippet") or "").strip()
        name = str(item.get("name") or "").strip()
        if not s or not name:
            continue
        if "=" in s:
            m_init = re.search(rf"\b{re.escape(name)}\s*=\s*([^,;]+)", s)
            rhs = (m_init.group(1) if m_init else s.split("=", 1)[1]).strip().rstrip(";")
        else:
            m_ctor = re.search(rf"\b{re.escape(name)}\s*\(([^;]*)\)\s*$", s)
            if not m_ctor:
                typ = normalize_ws(str(item.get("type") or ""))
                if is_default_constructed_string_type(typ):
                    rhs = '""'
                elif is_default_constructed_vector_type(typ):
                    rhs = f"{typ}()"
                else:
                    continue
            else:
                typ = normalize_ws(str(item.get("type") or ""))
                args = normalize_ws(m_ctor.group(1))
                rhs = f"{typ}({args})" if typ else f"{name}({args})"
        if rhs.endswith("+") and "++" in s:
            rhs += "+"
        if rhs.endswith("-") and "--" in s:
            rhs += "-"
        if not rhs:
            continue
        reads = array_refs(rhs)
        read_write.append({
            "lhs": name,
            "op": "=",
            "rhs": rhs,
            "writes": [],
            "reads": reads,
            "snippet": f"{name} = {rhs}",
            "begin": item.get("begin"),
            "end": item.get("end"),
            "source": "var_decl",
            "decl_type": normalize_ws(str(item.get("type") or "")),
            "is_loop_initializer": is_for_header_control(item.get("begin"), loops),
        })

    seen_decl_semantics = {
        normalize_rw_semantic_key(str(item.get("lhs") or ""), str(item.get("op") or ""), str(item.get("rhs") or ""))
        for item in read_write
    }
    for rel_begin, typ, name, args in iter_source_vector_ctor_decls(src):
        rhs = f"{typ}({args})"
        semantic_key = normalize_rw_semantic_key(name, "=", rhs)
        if semantic_key in seen_decl_semantics:
            continue
        seen_decl_semantics.add(semantic_key)
        read_write.append({
            "lhs": name,
            "op": "=",
            "rhs": rhs,
            "writes": [],
            "reads": array_refs(rhs),
            "snippet": f"{name} = {rhs}",
            "begin": rel_begin,
            "end": rel_begin + len(f"{name} = {rhs}"),
            "source": "var_decl",
            "decl_type": normalize_ws(typ),
            "is_loop_initializer": is_for_header_control(rel_begin, loops),
        })

    uninitialized_scalar_decls: dict[str, dict[str, Any]] = {}
    for item in var_decls:
        if item.get("kind") != "VarDecl":
            continue
        name = str(item.get("name") or "").strip()
        typ = normalize_ws(str(item.get("type") or ""))
        s = str(item.get("snippet") or "").strip()
        if not name or not typ or not s:
            continue
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            continue
        if "=" in s or re.search(rf"\b{re.escape(name)}\s*\(", s):
            continue
        if "[" in typ or "]" in typ:
            continue
        if is_default_constructed_string_type(typ) or is_default_constructed_vector_type(typ):
            continue
        uninitialized_scalar_decls[name] = item

    typed_after_decl: set[str] = set()
    for item in sorted(read_write, key=lambda x: x.get("begin") if isinstance(x.get("begin"), int) else 10**18):
        if item.get("source") != "assignment" or str(item.get("op") or "") != "=":
            continue
        lhs = normalize_ws(str(item.get("lhs") or ""))
        if lhs in typed_after_decl or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", lhs):
            continue
        decl = uninitialized_scalar_decls.get(lhs)
        if not decl:
            continue
        begin = item.get("begin")
        decl_begin = decl.get("begin")
        if isinstance(begin, int) and isinstance(decl_begin, int) and begin > decl_begin:
            item["decl_type"] = normalize_ws(str(decl.get("type") or ""))
            item["source"] = "var_decl_assignment"
            typed_after_decl.add(lhs)

    seen_rw_snippets = {normalize_rw_snippet_key(str(item.get("snippet") or "")) for item in read_write}
    seen_rw_semantics = {
        normalize_rw_semantic_key(str(item.get("lhs") or ""), str(item.get("op") or ""), str(item.get("rhs") or ""))
        for item in read_write
    }
    for rel_begin, lhs, op, rhs in iter_simple_statement_assignments(src):
        snip = normalize_ws(f"{lhs} {op} {rhs}")
        key = normalize_rw_snippet_key(snip)
        semantic_key = normalize_rw_semantic_key(lhs, op, rhs)
        stream_pointer_store = bool(postinc_pointer_store_name(lhs))
        if key in seen_rw_snippets or semantic_key in seen_rw_semantics:
            if not stream_pointer_store:
                continue
            exact_duplicate = any(
                normalize_rw_snippet_key(str(existing.get("snippet") or "")) == key
                and isinstance(existing.get("begin"), int)
                and abs(int(existing.get("begin")) - rel_begin) <= 1
                for existing in read_write
            )
            if exact_duplicate:
                continue
        lhs_simple = re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", lhs)
        if lhs_simple:
            rhs_norm = normalize_ws(rhs)
            member_duplicate = any(
                str(existing.get("source") or "") != "source_statement_assignment"
                and normalize_ws(str(existing.get("rhs") or "")) == rhs_norm
                and re.search(rf"(?:\.|->)\s*{re.escape(lhs)}\s*$", normalize_ws(str(existing.get("lhs") or "")))
                for existing in read_write
            )
            if member_duplicate:
                continue
        if not stream_pointer_store:
            seen_rw_snippets.add(key)
            seen_rw_semantics.add(semantic_key)
        writes = array_refs(lhs)
        reads = array_refs(rhs)
        if op != "=":
            reads.extend(array_refs(lhs))
        read_write.append({
            "lhs": lhs,
            "op": op,
            "rhs": rhs,
            "writes": writes,
            "reads": reads,
            "snippet": snip,
            "begin": rel_begin,
            "end": rel_begin + len(snip),
            "source": "source_statement_assignment",
            "is_loop_initializer": is_for_header_control(rel_begin, loops),
        })

    for branch in branches:
        branch_text = str(branch.get("snippet") or "")
        branch_begin = branch.get("begin")
        if not isinstance(branch_begin, int):
            continue
        for rel_begin, lhs, op, rhs in iter_simple_array_assignments(branch_text):
            snip = normalize_ws(f"{lhs} {op} {rhs}")
            key = normalize_rw_snippet_key(snip)
            if key in seen_rw_snippets:
                continue
            seen_rw_snippets.add(key)
            writes = array_refs(lhs)
            reads = array_refs(rhs)
            if op != "=":
                reads.extend(array_refs(lhs))
            read_write.append({
                "lhs": lhs,
                "op": op,
                "rhs": rhs,
                "writes": writes,
                "reads": reads,
                "snippet": snip,
                "begin": branch_begin + rel_begin,
                "end": branch_begin + rel_begin + len(snip),
                "source": "branch_snippet_assignment",
                "is_loop_initializer": False,
            })

    annotate_read_write_postinc_subscripts(read_write)

    return_snippets = [str(x.get("snippet") or "") for x in returns]
    all_text = "\n".join(
        [x["snippet"] for x in loops]
        + [x["snippet"] for x in assignments]
        + return_snippets
        + [x["snippet"] for x in calls]
        + array_subscripts
        + members
    )
    type_text = " ".join(str(v.get("type") or "") for v in var_decls)
    semantic_text = all_text + "\n" + type_text
    semantic_lower = semantic_text.lower()
    all_lower = all_text.lower()

    same_array_deps = []
    for rw in read_write:
        for w in rw["writes"]:
            for r in rw["reads"]:
                if w["base"] == r["base"] and not is_probably_same_index(w["index"], r["index"]):
                    same_array_deps.append({"base": w["base"], "write_index": w["index"], "read_index": r["index"], "snippet": rw["snippet"]})

    features: set[str] = set()
    if any(rw.get("writes") for rw in read_write):
        features.add("array_write")
    if any(rw.get("reads") for rw in read_write):
        features.add("array_read")
    if loops:
        features.add("loop_nest")
    if len(loops) >= 2 or re.search(r"\bfor\s*\([^)]*\)\s*\{[^{}]*\bfor\s*\(", strip_comments(src), re.S):
        features.add("nested_loop")
    if same_array_deps:
        features.add("carried_dependency_or_raw")
    for dep in same_array_deps:
        wi = normalize_index(str(dep.get("write_index") or ""))
        ri = normalize_index(str(dep.get("read_index") or ""))
        if re.search(r"\+1|-1", wi + ri):
            features.add("adjacent_row_or_neighbor_copy")
            break
    if re.search(r"\[[^\]]*[+-]\s*1[^\]]*\]", all_text) and same_array_deps:
        features.add("neighbor_read_write")
    if any(re.search(r"\+\+|--", rw["lhs"]) or re.search(r"\+\+|--", rw["rhs"]) for rw in read_write):
        features.add("compact_or_streaming_index")
    pointer_inc_writes = [
        rw
        for rw in read_write
        if re.search(r"\*\s*[A-Za-z_][A-Za-z0-9_]*\s*(\+\+|--)", str(rw.get("lhs") or ""))
    ]
    if pointer_inc_writes:
        features.add("pointer_increment_store")
    for a, b in zip(pointer_inc_writes, pointer_inc_writes[1:]):
        if normalize_ws(a.get("rhs") or "") == normalize_ws(b.get("rhs") or ""):
            features.add("duplicate_stream_store")
            break

    assignment_text = " ".join(x["snippet"] for x in assignments)
    if re.search(r"\b(sum|acc|res|result|total|count|cnt|minv|maxv|dot|norm|mean|var)\w*\s*(\+|-|\*|/|&|\||\^)?=", assignment_text, re.I):
        features.add("reduction_or_accumulator")
    if any((not rw.get("writes")) and rw.get("op") not in {"", "="} and rw.get("reads") for rw in read_write):
        features.add("reduction_or_accumulator")
    scalar_state_vars: set[str] = set()
    for rw in read_write:
        lhs_name = normalize_ws(str(rw.get("lhs") or ""))
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", lhs_name) and rw.get("reads"):
            rhs = str(rw.get("rhs") or "")
            if rw.get("op") != "=" or re.search(rf"\b{re.escape(lhs_name)}\b", rhs):
                scalar_state_vars.add(lhs_name)
            if re.search(r"\b(max|min|std::max|std::min)\s*\(", rhs):
                scalar_state_vars.add(lhs_name)
    if scalar_state_vars:
        for rw in read_write:
            if rw.get("writes") and any(re.search(rf"\b{re.escape(v)}\b", str(rw.get("rhs") or "")) for v in scalar_state_vars):
                features.add("scan_or_prefix_monoid")
                break
    if any("return" in r and re.search(r"\b(sum|acc|total|count|cnt|minv|maxv|dot|norm|mean|var)\w*\b", r, re.I) for r in return_snippets):
        features.add("scalar_reduction_return")
    if same_array_deps and ("max" in all_lower or "min" in all_lower or re.search(r"\bprefix|rolling|scan", all_lower)):
        features.add("scan_or_prefix_monoid")
    if (
        "jsamparray" in semantic_lower
        or "jsampimage" in semantic_lower
        or (
            re.search(r"\b(input_buf|output_buf|inptr|outptr|rowptr)\b", semantic_lower)
            and (re.search(r"\[[^\]]+\]\s*\[[^\]]+\]", all_text) or "**" in type_text)
        )
    ):
        features.add("row_or_plane_pointer_layout")
    if re.search(r"\b(input_buf|input_data|src_data|src)\s*\[[^\]]+\]\s*\[[^\]]+\]", all_text):
        features.add("row_pointer_read")
    if re.search(r"\b(output_buf|output_data|dst_data|dst)\s*\[[^\]]+\]\s*\[[^\]]+\]", all_text):
        features.add("row_pointer_write")
    if re.search(r"\b(input_buf|input_data)\s*\[\s*[0-9A-Za-z_]+\s*\]", all_text) and re.search(
        r"\b(outptr|output_buf|output_data)\b", all_text
    ):
        features.add("plane_or_row_to_output")
    interleaved_index = re.search(
        r"(\*\s*(channels?|cn|scn|srccn|dstcn)\b)|(\b(channels?|cn|scn|srccn|dstcn)\s*\*)|(\*\s*[34]\b)",
        all_text,
        re.I,
    )
    color_text = re.search(r"\b(rgb|bgr|yuv|ycbcr|gray|alpha|channel|scn|srccn|dstcn)\b", semantic_lower)
    if color_text and interleaved_index:
        features.add("interleaved_channel_map")
    if members and re.search(r"\.[A-Za-z_]\w*", " ".join(members)):
        features.add("aos_struct_field")
    if re.search(r"\b(cols?|rows?|width|height|stride|step|pitch|dim[123]?|plane|slice|block)\b", all_lower) and re.search(r"\*", " ".join(array_subscripts)):
        features.add("strided_or_flattened_layout")
    if (
        re.search(r"\b(kernel|anchor|border|roi|ksize|window|radius)\b", semantic_lower)
        or re.search(r"\b(dx|dy|kx|ky|sx|sy|xx|yy)\b", semantic_lower)
    ) and re.search(r"\[[^\]]*[+-][^\]]*\]", all_text):
        features.add("stencil_or_window")
    if re.search(r"\b(roundf|ceilf|floorf|round|ceil|floor|nearbyint|rint)\b", all_lower):
        features.add("rounding_exact")
    if re.search(r"\b(float|double|int16_t|int32_t|int64_t|uint16_t|uint32_t|uint64_t)\s*\(", all_text) or "static_cast" in all_text or "reinterpret" in all_lower:
        features.add("numeric_conversion")
    if re.search(r"\b(expf|tanhf|sinf|cosf|atanf|erfcf|sigmoid|gelu|mish|swish|softmax)\b", all_lower):
        features.add("transcendental_or_polynomial")
    if re.search(r">>|<<|&|\^|\||~", all_text):
        features.add("bitwise_or_shift")
    if re.search(r">>|<<", all_text) and re.search(r"\*|\+", all_text) and re.search(r"\b(clamp|min|max|255|128|bias|round|scale|descale|fix)\b", all_lower):
        features.add("fixed_point_numeric_map")
    if re.search(r"\b(mean|variance|var|stddev|inv_std|sqrtf|rsqrt|gamma|beta|affine)\b", semantic_lower):
        features.add("two_pass_statistic")
    if re.search(r"\b(if|while|break|continue)\b", all_lower) and not features.intersection({"reduction_or_accumulator", "strided_or_flattened_layout"}):
        features.add("control_or_hybrid")
    if re.search(r"\bif\s*\(", all_text) and read_write and not same_array_deps:
        features.add("predicate_or_select_map")

    return {
        "features": sorted(features),
        "semantic_text": semantic_text,
        "loops": loops[:12],
        "branches": branches[:16],
        "assignments": assignments[:24],
        "returns": returns[:8],
        "controls": controls[:16],
        "calls": calls[:24],
        "array_subscripts": array_subscripts[:32],
        "members": members[:16],
        "var_decls": var_decls[:32],
        "read_write": sorted_by_begin(read_write)[:64],
        "same_array_dependencies": same_array_deps[:12],
    }


def route_from_features(features: set[str], source_text: str) -> dict[str, str]:
    lower = source_text.lower()
    if "stencil_or_window" in features:
        return {"style": "dataflow_pseudocode", "pattern": "stencil_or_window", "reason": "AST saw neighbor/window/border-indexed reads."}
    if "row_or_plane_pointer_layout" in features and (
        "duplicate_stream_store" in features or "adjacent_row_or_neighbor_copy" in features
    ):
        return {"style": "dataflow_pseudocode", "pattern": "row_pointer_expand_or_copy", "reason": "AST saw row pointer layout plus duplicate/adjacent-row writes."}
    if "row_or_plane_pointer_layout" in features and "fixed_point_numeric_map" in features:
        return {"style": "dataflow_pseudocode", "pattern": "row_pointer_fixed_point_map", "reason": "AST saw row pointer layout plus fixed-point arithmetic."}
    if "interleaved_channel_map" in features and "fixed_point_numeric_map" in features:
        return {"style": "dataflow_pseudocode", "pattern": "interleaved_fixed_point_channel_map", "reason": "AST saw channel-strided fixed-point conversion."}
    if "two_pass_statistic" in features:
        return {"style": "dataflow_pseudocode", "pattern": "two_pass_statistic", "reason": "AST saw mean/variance/affine statistic structure."}
    if "row_or_plane_pointer_layout" in features:
        return {"style": "dataflow_pseudocode", "pattern": "row_pointer_layout", "reason": "AST saw row/plane pointer indexed layout."}
    if "fixed_point_numeric_map" in features:
        return {"style": "dataflow_pseudocode", "pattern": "fixed_point_numeric_map", "reason": "AST saw fixed-point multiply/add/shift/clamp structure."}
    if "transcendental_or_polynomial" in features:
        return {"style": "dataflow_pseudocode", "pattern": "transcendental_or_polynomial", "reason": "AST saw transcendental/polynomial scalar kernel."}
    if "interleaved_channel_map" in features:
        return {"style": "dataflow_pseudocode", "pattern": "interleaved_channel_map", "reason": "AST saw channel-strided memory layout."}
    if "scan_or_prefix_monoid" in features or ("carried_dependency_or_raw" in features and "neighbor_read_write" in features):
        return {"style": "dataflow_pseudocode", "pattern": "scan_or_prefix_monoid", "reason": "AST saw read-after-write neighbor/carry dependency."}
    if "carried_dependency_or_raw" in features:
        return {"style": "dataflow_pseudocode", "pattern": "reverse_or_two_pointer_swap", "reason": "AST saw same-array cross-index dependency that is not a prefix neighbor."}
    if "reduction_or_accumulator" in features or "scalar_reduction_return" in features:
        return {"style": "dataflow_pseudocode", "pattern": "reduction_or_two_pass", "reason": "AST saw scalar accumulator/reduction."}
    if "predicate_or_select_map" in features:
        return {"style": "dataflow_pseudocode", "pattern": "predicate_select_store_map", "reason": "AST saw independent predicated/selective array store."}
    if "compact_or_streaming_index" in features:
        return {"style": "dataflow_pseudocode", "pattern": "compact_or_streaming_write", "reason": "AST saw output index advanced separately from loop index."}
    if "array_write" in features and ("strided_or_flattened_layout" in features or "nested_loop" in features):
        return {"style": "dataflow_pseudocode", "pattern": "store_map", "reason": "AST saw independent affine/indexed load-store layout."}
    if "numeric_conversion" in features:
        return {"style": "dataflow_pseudocode", "pattern": "numeric_conversion_lane_mapping", "reason": "AST saw numeric conversion/cast kernel."}
    if "bitwise_or_shift" in features:
        return {"style": "dataflow_pseudocode", "pattern": "bit_shift_or_rotate", "reason": "AST saw bitwise/shift map without stronger layout/fixed-point marker."}
    if re.search(r"\bstring|char|digit|prime|factor|base|palindrome|bracket\b", lower):
        return {"style": "pseudocode", "pattern": "scalar_or_irregular_control", "reason": "AST/text suggests irregular scalar/string control."}
    return {"style": "pseudocode", "pattern": "generic_code_like_pseudocode", "reason": "No strong vector dataflow pattern detected."}


def trim_statement(stmt: str, max_len: int = 160) -> str:
    stmt = normalize_ws(stmt)
    # Clang source ranges for for-loop init/increment expressions can include
    # the closing paren of the loop header.  Drop only unmatched trailing parens
    # so the emitted code-like pseudocode stays readable.
    while stmt.endswith(")") and stmt.count(")") > stmt.count("("):
        stmt = stmt[:-1].rstrip()
    if len(stmt) > max_len:
        stmt = stmt[: max_len - 3] + "..."
    return stmt


def _scan_matching_left(text: str, close_pos: int, open_ch: str, close_ch: str) -> int:
    depth = 0
    i = close_pos
    while i >= 0:
        ch = text[i]
        if ch == close_ch:
            depth += 1
        elif ch == open_ch:
            depth -= 1
            if depth == 0:
                return i
        i -= 1
    return close_pos


def _scan_matching_right(text: str, open_pos: int, open_ch: str, close_ch: str) -> int:
    depth = 0
    i = open_pos
    while i < len(text):
        ch = text[i]
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return open_pos


def _left_factor_start(text: str, end_exclusive: int) -> int:
    i = end_exclusive - 1
    while i >= 0 and text[i].isspace():
        i -= 1
    if i < 0:
        return 0
    if text[i] == ")":
        start = _scan_matching_left(text, i, "(", ")")
        j = start - 1
        while j >= 0 and text[j].isspace():
            j -= 1
        while j >= 0 and re.match(r"[A-Za-z0-9_:.]", text[j]):
            j -= 1
        return j + 1 if j + 1 < start else start
    if text[i] == "]":
        start = _scan_matching_left(text, i, "[", "]")
        j = start - 1
        while j >= 0 and text[j].isspace():
            j -= 1
        while j >= 0 and re.match(r"[A-Za-z0-9_:.]", text[j]):
            j -= 1
        return j + 1
    if text[i] in {"'", '"'}:
        quote = text[i]
        j = i - 1
        while j >= 0:
            if text[j] == quote and (j == 0 or text[j - 1] != "\\"):
                return j
            j -= 1
        return i
    while i >= 0 and re.match(r"[A-Za-z0-9_:.]", text[i]):
        i -= 1
    return i + 1


def _left_operand_start(text: str, op_pos: int) -> int:
    start = _left_factor_start(text, op_pos)
    while True:
        j = start - 1
        while j >= 0 and text[j].isspace():
            j -= 1
        if j < 0 or text[j] != "*":
            return _include_left_casts(text, start)
        start = _left_factor_start(text, j)


def _is_c_style_cast_text(text: str) -> bool:
    cast = normalize_ws(text)
    return bool(
        re.fullmatch(
            r"(?:const\s+|volatile\s+)*(?:signed\s+|unsigned\s+)?[A-Za-z_][A-Za-z0-9_:<>]*"
            r"(?:\s+[A-Za-z_][A-Za-z0-9_:<>]*)*(?:\s*[*&])?",
            cast,
        )
    )


def _include_left_casts(text: str, start: int) -> int:
    while True:
        j = start - 1
        while j >= 0 and text[j].isspace():
            j -= 1
        if j < 0 or text[j] != ")":
            return start
        cast_start = _scan_matching_left(text, j, "(", ")")
        if cast_start < 0 or not _is_c_style_cast_text(text[cast_start + 1 : j]):
            return start
        start = cast_start


def _right_operand_end(text: str, start: int) -> int:
    i = start
    while i < len(text) and text[i].isspace():
        i += 1
    if i >= len(text):
        return i
    if text[i] == "(":
        return _scan_matching_right(text, i, "(", ")") + 1
    if text[i] == "[":
        return _scan_matching_right(text, i, "[", "]") + 1
    if text[i] in {"'", '"'}:
        quote = text[i]
        j = i + 1
        while j < len(text):
            if text[j] == quote and text[j - 1] != "\\":
                return j + 1
            j += 1
        return i + 1
    j = i
    while j < len(text) and re.match(r"[A-Za-z0-9_:.]", text[j]):
        j += 1
    while j < len(text):
        k = j
        while k < len(text) and text[k].isspace():
            k += 1
        if k < len(text) and text[k] == "(":
            j = _scan_matching_right(text, k, "(", ")") + 1
            continue
        if k < len(text) and text[k] == "[":
            j = _scan_matching_right(text, k, "[", "]") + 1
            continue
        break
    return j


def _top_level_numeric_op(text: str) -> tuple[int, str] | None:
    depth_paren = depth_bracket = 0
    in_quote = ""
    i = 0
    while i < len(text):
        ch = text[i]
        if in_quote:
            if ch == in_quote and (i == 0 or text[i - 1] != "\\"):
                in_quote = ""
            i += 1
            continue
        if ch in {"'", '"'}:
            in_quote = ch
        elif ch == "(":
            depth_paren += 1
        elif ch == ")":
            depth_paren = max(0, depth_paren - 1)
        elif ch == "[":
            depth_bracket += 1
        elif ch == "]":
            depth_bracket = max(0, depth_bracket - 1)
        elif ch in {"/", "%"} and depth_paren == 0 and depth_bracket == 0:
            if ch == "/" and i + 1 < len(text) and text[i + 1] in {"/", "*"}:
                i += 1
            else:
                return i, ch
        i += 1
    return None


def _has_float_hint(expr: str, float_names: set[str] | None = None) -> bool:
    float_names = float_names or set()
    if any(re.search(rf"\b{re.escape(name)}\b", str(expr or "")) for name in float_names if name):
        return True
    return bool(
        re.search(r"\b(?:float|double|roundf?|ceilf?|floorf?|fabsf?)\b", expr)
        or re.search(r"(?<![A-Za-z0-9_])\d+\.\d*(?:[fF])?\b", expr)
    )


def _has_integer_cast(expr: str) -> bool:
    return bool(re.match(r"\s*\((?:signed\s+|unsigned\s+)?(?:char|short|int|long|int\d+_t|uint\d+_t|size_t)\b", expr))


FLOAT_CAST_CONVERSIONS = {
    "float": "float",
    "double": "double",
}


def _float_cast_conversion(cast_text: str) -> str:
    return FLOAT_CAST_CONVERSIONS.get(normalize_ws(cast_text).replace("std::", ""))


def strip_redundant_outer_parens(expr: str) -> str:
    s = str(expr or "").strip()
    while s.startswith("("):
        close = _scan_matching_right(s, 0, "(", ")")
        if close != len(s) - 1:
            break
        s = s[1:-1].strip()
    return s


def sanitize_float_casts(text: str) -> str:
    """Keep explicit floating C-style casts while binding their operand.

    This preserves the operand-level C semantics without introducing DSL helper
    names: (float)x * y becomes (float)(x) * y, while (float)(x * y)
    stays (float)(x * y).
    """
    s = str(text or "")
    guard = 0
    while guard < 32:
        guard += 1
        changed = False
        out: list[str] = []
        i = 0
        quote = ""
        while i < len(s):
            ch = s[i]
            if quote:
                out.append(ch)
                if ch == quote and (i == 0 or s[i - 1] != "\\"):
                    quote = ""
                i += 1
                continue
            if ch in {"'", '"'}:
                quote = ch
                out.append(ch)
                i += 1
                continue
            if ch != "(":
                out.append(ch)
                i += 1
                continue
            cast_close = _scan_matching_right(s, i, "(", ")")
            if cast_close <= i:
                out.append(ch)
                i += 1
                continue
            conv = _float_cast_conversion(s[i + 1 : cast_close])
            if not conv:
                out.append(ch)
                i += 1
                continue
            operand_start = cast_close + 1
            while operand_start < len(s) and s[operand_start].isspace():
                operand_start += 1
            operand_end = _right_operand_end(s, operand_start)
            operand = s[operand_start:operand_end].strip()
            if not operand:
                out.append(s[i : cast_close + 1])
                i = cast_close + 1
                continue
            cleaned_operand = sanitize_float_casts(strip_redundant_outer_parens(operand))
            replacement = f"({conv})({cleaned_operand})"
            out.append(replacement)
            if replacement != s[i:operand_end]:
                changed = True
            i = operand_end
        rendered = "".join(out)
        if not changed:
            return rendered
        s = rendered
    return s


def normalize_generated_operator_spacing(text: str) -> str:
    """Clean spacing around generated helper calls in pseudocode expressions."""
    s = str(text or "")
    callish = r"(?:math_kernel\[)"
    s = re.sub(r"\s*(&&|\|\|)\s*", r" \1 ", s)
    s = re.sub(rf"(?<=[A-Za-z0-9_)\]])\s*\+\s*(?={callish})", " + ", s)
    s = re.sub(rf"(?<=[A-Za-z0-9_)\]])\s*-\s*(?={callish})", " - ", s)
    return s


SIMPLE_CAST_MACROS = {
    "fp16_to_native": "float",
    "native_to_fp16": "__fp16",
}


def expand_simple_cast_macros(text: str) -> str:
    s = str(text or "")
    name_re = re.compile(r"(?<![A-Za-z0-9_:])(" + "|".join(map(re.escape, SIMPLE_CAST_MACROS)) + r")\s*\(")
    out: list[str] = []
    pos = 0
    while True:
        m = name_re.search(s, pos)
        if not m:
            out.append(s[pos:])
            break
        open_pos = m.end() - 1
        close_pos = _find_matching_paren(s, open_pos)
        if close_pos < 0:
            out.append(s[pos:])
            break
        cast_type = SIMPLE_CAST_MACROS[m.group(1)]
        arg = expand_simple_cast_macros(s[open_pos + 1 : close_pos])
        out.append(s[pos : m.start()])
        out.append(f"({cast_type})({strip_redundant_outer_parens(arg)})")
        pos = close_pos + 1
    return "".join(out)


def expand_simple_cast_macros_in_json(value: Any) -> Any:
    if isinstance(value, str):
        return expand_simple_cast_macros(value)
    if isinstance(value, list):
        return [expand_simple_cast_macros_in_json(x) for x in value]
    if isinstance(value, dict):
        return {k: expand_simple_cast_macros_in_json(v) for k, v in value.items()}
    return value


def sanitize_numeric_operators(
    text: str,
    *,
    division_mode: str = "auto",
    float_names: set[str] | None = None,
) -> str:
    """Preserve native C/C++ / and % syntax in bootstrap text.

    Earlier versions rewrote these operators into helper-like names.  Once
    types are shown in RSB, native C++ syntax is clearer for the model.
    """
    return normalize_generated_operator_spacing(str(text or ""))


MATH_KERNEL_CALLS = {
    "exp": "exp",
    "expf": "exp",
    "std::exp": "exp",
    "tanh": "tanh",
    "tanhf": "tanh",
    "std::tanh": "tanh",
    "sin": "sin",
    "sinf": "sin",
    "std::sin": "sin",
    "cos": "cos",
    "cosf": "cos",
    "std::cos": "cos",
    "log": "log",
    "logf": "log",
    "log1p": "log1p",
    "log1pf": "log1p",
    "std::log": "log",
    "std::log1p": "log1p",
    "pow": "pow",
    "powf": "pow",
    "std::pow": "pow",
    "atan": "atan",
    "atanf": "atan",
    "std::atan": "atan",
    "erfc": "erfc",
    "erfcf": "erfc",
    "std::erfc": "erfc",
}

SORT_SIDE_EFFECT_CALLS = {"sort", "stable_sort"}
SIDE_EFFECT_CALLS = {"memset", "memcpy", "memmove", "bzero", "resize", "clear", "push_back"} | SORT_SIDE_EFFECT_CALLS
INIT_SIDE_EFFECT_CALLS = {"memset", "bzero"}


def _find_matching_paren(text: str, open_pos: int) -> int:
    depth = 0
    i = open_pos
    quote = ""
    escaped = False
    while i < len(text):
        ch = text[i]
        if quote:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                quote = ""
            i += 1
            continue
        if ch in {"'", '"'}:
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def sanitize_math_kernel_calls(
    text: str,
    *,
    division_mode: str = "auto",
    numeric_ops: bool = True,
    float_names: set[str] | None = None,
) -> str:
    """Replace scalar libm call syntax with a non-callable math_kernel marker.

    The goal is to preserve the scalar semantics while avoiding bootstrap text
    that looks like C/SVE code the model can mechanically copy, e.g. expf(v)
    where v later becomes an SVE vector.
    """
    s = expand_simple_cast_macros(str(text or ""))
    name_re = re.compile(
        r"(?<![A-Za-z0-9_:])((?:std::)?(?:expf?|tanhf?|sinf?|cosf?|log1pf?|logf?|powf?|atanf?|erfcf?))\s*\("
    )
    out: list[str] = []
    pos = 0
    while True:
        m = name_re.search(s, pos)
        if not m:
            out.append(s[pos:])
            break
        open_pos = m.end() - 1
        close_pos = _find_matching_paren(s, open_pos)
        if close_pos < 0:
            out.append(s[pos:])
            break
        raw_name = m.group(1)
        kernel = MATH_KERNEL_CALLS.get(raw_name, MATH_KERNEL_CALLS.get(raw_name.replace("std::", ""), raw_name))
        arg = sanitize_math_kernel_calls(
            s[open_pos + 1 : close_pos],
            division_mode=division_mode,
            numeric_ops=numeric_ops,
            float_names=float_names,
        )
        out.append(s[pos : m.start()])
        out.append(f"math_kernel[{kernel}]({arg})")
        pos = close_pos + 1
    rendered = sanitize_float_casts("".join(out))
    return sanitize_numeric_operators(rendered, division_mode=division_mode, float_names=float_names) if numeric_ops else rendered


VECTOR_SCALAR_OP_CALLS = {
    "round": "round",
    "roundf": "roundf",
    "std::round": "std::round",
    "ceil": "ceil",
    "ceilf": "ceil",
    "std::ceil": "ceil",
    "floor": "floor",
    "floorf": "floor",
    "std::floor": "floor",
    "abs": "abs",
    "llabs": "abs",
    "labs": "abs",
    "fabs": "abs",
    "fabsf": "abs",
    "std::abs": "abs",
    "min": "min",
    "std::min": "min",
    "fmin": "min",
    "fminf": "min",
    "max": "max",
    "std::max": "max",
    "fmax": "max",
    "fmaxf": "max",
}


def sanitize_vector_scalar_ops(
    text: str,
    *,
    division_mode: str = "auto",
    float_names: set[str] | None = None,
) -> str:
    """Abstract scalar helper calls that would be misleading inside vector lanes."""
    float_names = float_names or set()
    s = sanitize_math_kernel_calls(text, division_mode=division_mode, numeric_ops=False, float_names=float_names)
    name_re = re.compile(
        r"(?<![A-Za-z0-9_:])((?:std::)?(?:roundf?|ceilf?|floorf?|abs|llabs|labs|fabsf?|min|max|fminf?|fmaxf?))\s*\("
    )
    out: list[str] = []
    pos = 0
    while True:
        m = name_re.search(s, pos)
        if not m:
            out.append(s[pos:])
            break
        open_pos = m.end() - 1
        close_pos = _find_matching_paren(s, open_pos)
        if close_pos < 0:
            out.append(s[pos:])
            break
        raw_name = m.group(1)
        op = VECTOR_SCALAR_OP_CALLS.get(raw_name, VECTOR_SCALAR_OP_CALLS.get(raw_name.replace("std::", ""), raw_name))
        arg_mode = "float" if raw_name.replace("std::", "") in {"round", "roundf", "ceil", "ceilf", "floor", "floorf", "fabs", "fabsf"} else division_mode
        arg = sanitize_vector_scalar_ops(
            s[open_pos + 1 : close_pos],
            division_mode=arg_mode,
            float_names=float_names,
        )
        out.append(s[pos : m.start()])
        out.append(f"{op}({arg})")
        pos = close_pos + 1
    return sanitize_numeric_operators(sanitize_float_casts("".join(out)), division_mode=division_mode, float_names=float_names)


def is_side_effect_call(item: dict[str, Any]) -> bool:
    name = str(item.get("name") or "").split("::")[-1]
    return name in SIDE_EFFECT_CALLS


def render_side_effect_call(item: dict[str, Any]) -> str:
    raw = trim_statement(sanitize_math_kernel_calls(str(item.get("snippet") or "")), 240)
    name = str(item.get("name") or "").split("::")[-1]
    if name in SORT_SIDE_EFFECT_CALLS:
        m = re.fullmatch(
            rf"(?:std::)?{re.escape(name)}\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\.begin\(\)\s*,\s*\1\.end\(\)\s*(?:,\s*(.+))?\)",
            raw,
        )
        if m:
            comparator = normalize_ws(str(m.group(2) or ""))
            order = "ascending"
            if comparator:
                if re.search(r"\bgreater\b|>\s*[^=]", comparator):
                    order = "descending"
                elif re.search(r"\bless\b|<\s*[^=]", comparator):
                    order = "ascending"
                else:
                    order = f"comparator={comparator}"
            return f"{name}({m.group(1)}, {order})"
    return raw


def side_effect_call_label(item: dict[str, Any]) -> str:
    name = str(item.get("name") or "").split("::")[-1]
    if name in SORT_SIDE_EFFECT_CALLS:
        return "scalar_step"
    if name == "resize":
        return "resize"
    return "init" if name in INIT_SIDE_EFFECT_CALLS else "update"


def render_control_statement(item: dict[str, Any]) -> str:
    kind = str(item.get("kind") or "")
    if kind == "BreakStmt":
        return "break"
    if kind == "ContinueStmt":
        return "continue"
    return trim_statement(str(item.get("snippet") or ""))


TYPE_CONTEXT = {
    "JSAMPLE": "unsigned char",
    "JSAMPROW": "JSAMPLE*",
    "JSAMPARRAY": "JSAMPROW*",
    "JSAMPIMAGE": "JSAMPARRAY*",
    "JDIMENSION": "unsigned int",
    "JCOEF": "short",
    "JCOEFPTR": "JCOEF*",
    "DCTELEM": "short",
    "JLONG": "long",
    "ISLOW_MULT_TYPE": "short",
    "float16_t": "__fp16",
}


def context_type_lines(text: str) -> list[str]:
    lines = []
    blob = str(text or "")
    for name, typ in TYPE_CONTEXT.items():
        if re.search(rf"\b{re.escape(name)}\b", blob):
            lines.append(f"{name} = {typ}")
    return lines


def variable_type_lines(feats: dict[str, Any]) -> list[str]:
    """Return a generic AST-derived symbol table for variables in the function."""
    by_name: dict[str, list[str]] = {}
    seen: set[tuple[str, str]] = set()
    for decl in feats.get("var_decls") or []:
        name = normalize_ws(str(decl.get("name") or ""))
        typ = normalize_ws(str(decl.get("type") or ""))
        if not name or not typ:
            continue
        if name.startswith("__") or name in {"std"}:
            continue
        key = (name, typ)
        if key in seen:
            continue
        seen.add(key)
        by_name.setdefault(name, []).append(typ)

    lines: list[str] = []
    for name, types in by_name.items():
        if len(types) == 1:
            lines.append(f"{name}: {types[0]}")
        else:
            lines.append(f"{name}: {' | '.join(types)}")
    return lines


def external_symbol_lines(signature: str, feats: dict[str, Any]) -> list[str]:
    """List unresolved macro/constant-like symbols used by the dataflow.

    These are not VarDecl nodes, so we must not fake variable types for them.
    The point is to make benchmark-provided constants visible instead of
    leaving names such as ONE_HALF or RGB_RED unexplained in the pseudocode.
    """
    blob_parts = [signature, str(feats.get("semantic_text") or "")]
    for decl in feats.get("var_decls") or []:
        blob_parts.append(str(decl.get("type") or ""))
    blob = "\n".join(blob_parts)

    skip = set(TYPE_CONTEXT)
    for decl in feats.get("var_decls") or []:
        name = normalize_ws(str(decl.get("name") or ""))
        if name:
            skip.add(name)
    skip.update(
        {
            "NULL",
            "TRUE",
            "FALSE",
            "NAN",
            "INFINITY",
            "EOF",
        }
    )

    symbols: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(r"\b[A-Z][A-Z0-9_]*\b", blob):
        name = match.group(0)
        if len(name) <= 1:
            continue
        if any(type_name.startswith(name) and type_name != name for type_name in TYPE_CONTEXT):
            continue
        if name in {"CONS"}:
            continue
        if name in skip or name.startswith("__"):
            continue
        if name not in seen:
            seen.add(name)
            symbols.append(name)
    return [f"{name}: context constant/macro symbol" for name in symbols[:32]]


def semantic_subpatterns(features: set[str]) -> list[str]:
    ordered = [
        ("duplicate_stream_store", "horizontal_duplicate_or_streaming_store"),
        ("adjacent_row_or_neighbor_copy", "adjacent_row_copy_or_neighbor_dependency"),
        ("plane_or_row_to_output", "plane_or_row_to_output_layout"),
        ("interleaved_channel_map", "interleaved_channel_stride"),
        ("fixed_point_numeric_map", "fixed_point_multiply_add_shift_clamp"),
        ("stencil_or_window", "neighbor_window_or_border_access"),
        ("two_pass_statistic", "two_pass_mean_variance_affine"),
        ("pointer_increment_store", "pointer_increment_output"),
        ("compact_or_streaming_index", "compact_or_streaming_index"),
        ("numeric_conversion", "numeric_conversion"),
        ("transcendental_or_polynomial", "math_approximation_kernel"),
    ]
    return [name for feat, name in ordered if feat in features]


def render_decl_type(typ: str) -> str:
    typ = normalize_ws(str(typ or ""))
    if not typ or typ in {"auto"}:
        return ""
    return typ


def typed_decl_name(name: str, typ: str) -> str:
    name = normalize_ws(str(name or ""))
    typ = render_decl_type(typ)
    if not name or not typ:
        return name
    if re.search(rf"\b{re.escape(name)}\b", typ):
        return name
    return f"{typ} {name}"


def loop_var_decl_type(loop: dict[str, Any], feats: dict[str, Any] | None, var: str) -> str:
    if not feats or not var:
        return ""
    loop_begin = loop.get("begin")
    if not isinstance(loop_begin, int):
        return ""
    header = loop_header_line(loop)
    header_end = loop_begin + len(header) + 8
    best: tuple[int, str] | None = None
    for decl in feats.get("var_decls") or []:
        if decl.get("kind") != "VarDecl":
            continue
        if str(decl.get("name") or "") != var:
            continue
        begin = decl.get("begin")
        if not isinstance(begin, int) or begin < loop_begin or begin > header_end:
            continue
        typ = render_decl_type(str(decl.get("type") or ""))
        if typ and (best is None or begin < best[0]):
            best = (begin, typ)
    return best[1] if best else ""


def render_assignment_line(
    item: dict[str, Any],
    *,
    vector_execution: bool = False,
    affine_spec: dict[str, str] | None = None,
    pointer_recurrences: dict[str, str] | None = None,
    scalar_exprs: dict[str, str] | None = None,
    float_names: set[str] | None = None,
) -> str:
    float_names = float_names or set()
    lhs_source = str(item.get("lhs_without_postinc_subscript") or item.get("lhs") or "")
    lhs_source = rewrite_affine_pointer_subscripts(lhs_source, affine_spec)
    lhs_source = rewrite_loop_body_pointer_subscripts(lhs_source, affine_spec, pointer_recurrences)
    lhs_source = replace_symbol_exprs(lhs_source, scalar_exprs)
    lhs_division_mode = "float" if expr_uses_float_symbol(lhs_source, float_names) else "auto"
    lhs = trim_statement(sanitize_vector_scalar_ops(lhs_source, float_names=float_names), 120).rstrip(";").strip()
    rhs_source = str(item.get("rhs_without_postinc_subscript") or item.get("rhs") or "")
    rhs_source = replace_postinc_pointer_reads(rhs_source)
    rhs_source = rewrite_affine_pointer_subscripts(rhs_source, affine_spec)
    rhs_source = rewrite_loop_body_pointer_subscripts(rhs_source, affine_spec, pointer_recurrences)
    rhs_source = replace_symbol_exprs(rhs_source, scalar_exprs)
    rhs = trim_statement(
        sanitize_vector_scalar_ops(rhs_source, division_mode=lhs_division_mode, float_names=float_names),
        1200,
    ).rstrip(";").strip()
    op = str(item.get("op") or "=")
    snippet_source = str(item.get("snippet") or "")
    snippet_source = replace_postinc_pointer_reads(snippet_source)
    snippet_source = replace_postinc_subscript_indices(snippet_source)
    snippet_source = rewrite_affine_pointer_subscripts(snippet_source, affine_spec)
    snippet_source = rewrite_loop_body_pointer_subscripts(snippet_source, affine_spec, pointer_recurrences)
    snippet_source = replace_symbol_exprs(snippet_source, scalar_exprs)
    snippet = trim_statement(
        sanitize_vector_scalar_ops(snippet_source, division_mode=lhs_division_mode, float_names=float_names),
        1200,
    ).rstrip(";").strip()
    if lhs and rhs:
        if item.get("source") in {"var_decl", "var_decl_assignment"} and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", lhs):
            lhs = typed_decl_name(lhs, str(item.get("decl_type") or ""))
        return f"{lhs} {op} {rhs}"
    return snippet


def loop_header_line(loop: dict[str, Any]) -> str:
    text = str(loop.get("snippet") or "")
    text = normalize_ws(text)
    if text.startswith("for") or text.startswith("while"):
        open_pos = text.find("(")
        if open_pos >= 0:
            close_pos = _find_matching_paren(text, open_pos)
            if close_pos >= 0:
                text = text[: close_pos + 1].strip()
    elif "{" in text:
        text = text.split("{", 1)[0].strip()
    return trim_statement(text, 220)


def branch_condition_line(branch: dict[str, Any]) -> str:
    cond = branch_condition_expr(branch)
    if cond:
        return "if " + trim_statement(sanitize_vector_scalar_ops(cond), 220)
    text = str(branch.get("snippet") or "")
    text = normalize_ws(text)
    return trim_statement(text.split("{", 1)[0], 220)


def branch_condition_expr(branch: dict[str, Any]) -> str:
    text = str(branch.get("snippet") or "")
    text = normalize_ws(text)
    m = re.search(r"\bif\s*\(", text)
    if not m:
        return ""
    open_pos = text.find("(", m.start())
    close_pos = _find_matching_paren(text, open_pos)
    if close_pos < 0:
        return ""
    cond = text[open_pos + 1 : close_pos]
    return trim_statement(cond, 220)


def render_refs(
    refs: list[dict[str, str]],
    *,
    affine_spec: dict[str, str] | None = None,
    pointer_recurrences: dict[str, str] | None = None,
) -> str:
    parts = []
    seen: set[str] = set()
    for ref in refs:
        base = str(ref.get("base") or "").strip()
        index = replace_postinc_scalar_expr(str(ref.get("index") or "").strip())
        index = affine_pointer_index(base, index, affine_spec)
        index = loop_body_pointer_index(base, index, affine_spec, pointer_recurrences)
        if base:
            rendered = f"{base}[{index}]"
            if rendered not in seen:
                seen.add(rendered)
                parts.append(rendered)
    return ", ".join(parts)


def ref_sort_key(ref: dict[str, str]) -> tuple[int, str, str]:
    """Prefer loop-invariant indexed reads before per-lane reads."""
    base = str(ref.get("base") or "").strip()
    index = normalize_ws(str(ref.get("index") or "").strip())
    const_index = bool(re.fullmatch(r"[-+]?(?:0x[0-9A-Fa-f]+|\d+)", index))
    return (0 if const_index else 1, base, index)


def render_grouped_refs(refs: list[dict[str, str]]) -> str:
    ordered: list[dict[str, str]] = []
    seen: set[str] = set()
    for ref in sorted(refs, key=ref_sort_key):
        base = str(ref.get("base") or "").strip()
        index = str(ref.get("index") or "").strip()
        key = f"{base}[{index}]"
        if base and key not in seen:
            seen.add(key)
            ordered.append({"base": base, "index": index})
    return render_refs(ordered)


def nested_index_read_refs(refs: list[dict[str, str]]) -> list[dict[str, str]]:
    nested: list[dict[str, str]] = []
    seen: set[str] = set()
    for ref in refs:
        index = str(ref.get("index") or "")
        for match in re.finditer(r"\b([A-Za-z_]\w*)\s*\[([^\[\]]+)\]", index):
            base = match.group(1)
            idx = normalize_ws(match.group(2))
            key = f"{base}[{idx}]"
            if key not in seen:
                seen.add(key)
                nested.append({"base": base, "index": idx})
    return nested


def read_refs_for_item(item: dict[str, Any]) -> list[dict[str, str]]:
    refs = list(item.get("reads") or [])
    for name in postinc_pointer_read_names(str(item.get("rhs") or "")):
        refs.append({"base": name, "index": "0"})
    refs.extend(nested_index_read_refs(list(item.get("reads") or [])))
    refs.extend(nested_index_read_refs(list(item.get("writes") or [])))
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for ref in refs:
        base = str(ref.get("base") or "").strip()
        index = str(ref.get("index") or "").strip()
        key = f"{base}[{index}]"
        if base and key not in seen:
            seen.add(key)
            out.append({"base": base, "index": index})
    return out


def contains_offset(container: dict[str, Any], offset: Any) -> bool:
    if not isinstance(offset, int):
        return False
    b = container.get("begin")
    e = container.get("end")
    return isinstance(b, int) and isinstance(e, int) and b <= offset <= e


def containing_count(containers: list[dict[str, Any]], offset: Any) -> int:
    return sum(1 for item in containers if contains_offset(item, offset))


def sorted_by_begin(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(items, key=lambda x: (x.get("begin") if isinstance(x.get("begin"), int) else 10**18))


def split_top_level_commas(text: str) -> list[str]:
    """Split a C/C++ comma list without splitting inside calls/subscripts."""
    s = str(text or "")
    parts: list[str] = []
    start = 0
    depth = 0
    quote = ""
    escaped = False
    for i, ch in enumerate(s):
        if quote:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                quote = ""
            continue
        if ch in {"'", '"'}:
            quote = ch
            continue
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        elif ch == "," and depth == 0:
            part = normalize_ws(s[start:i])
            if part:
                parts.append(part)
            start = i + 1
    tail = normalize_ws(s[start:])
    if tail:
        parts.append(tail)
    return parts


def primary_induction_step(var: str, inc_parts: list[str]) -> str | None:
    """Return the step expression for the loop induction variable only."""
    escaped = re.escape(var)
    for raw in inc_parts:
        inc = normalize_ws(raw)
        if re.fullmatch(rf"(?:\+\+{escaped}|{escaped}\+\+)", inc):
            return "1"
        if re.fullmatch(rf"(?:--{escaped}|{escaped}--)", inc):
            return "-1"
        m_step = re.fullmatch(rf"{escaped}\s*\+=\s*(.+)", inc)
        if m_step:
            return sanitize_vector_scalar_ops(normalize_ws(m_step.group(1)))
        m_step = re.fullmatch(rf"{escaped}\s*-=\s*(.+)", inc)
        if m_step:
            return "-" + sanitize_vector_scalar_ops(normalize_ws(m_step.group(1)))
        m_step = re.fullmatch(rf"{escaped}\s*=\s*{escaped}\s*\+\s*(.+)", inc)
        if m_step:
            return sanitize_vector_scalar_ops(normalize_ws(m_step.group(1)))
        m_step = re.fullmatch(rf"{escaped}\s*=\s*{escaped}\s*-(?!>)\s*(.+)", inc)
        if m_step:
            return "-" + sanitize_vector_scalar_ops(normalize_ws(m_step.group(1)))
    return None


def affine_recurrence_update(part: str, primary_var: str) -> tuple[str, str] | None:
    """Parse an extra loop-header recurrence such as src += scn.

    These are not vector-loop steps.  They are affine state recurrences whose
    value at lane i can be rewritten as base[(i-start)*stride + index].
    """
    inc = normalize_ws(part)
    escaped_primary = re.escape(primary_var)
    if re.fullmatch(rf"(?:\+\+{escaped_primary}|{escaped_primary}\+\+|--{escaped_primary}|{escaped_primary}--)", inc):
        return None
    if re.fullmatch(rf"{escaped_primary}\s*(?:[+\-]?=).+", inc):
        return None
    m = re.fullmatch(r"(?:\+\+([A-Za-z_][A-Za-z0-9_]*)|([A-Za-z_][A-Za-z0-9_]*)\+\+)", inc)
    if m:
        return (m.group(1) or m.group(2)), "1"
    m = re.fullmatch(r"(?:--([A-Za-z_][A-Za-z0-9_]*)|([A-Za-z_][A-Za-z0-9_]*)--)", inc)
    if m:
        return (m.group(1) or m.group(2)), "-1"
    m = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)\s*\+=\s*(.+)", inc)
    if m:
        return m.group(1), trim_statement(sanitize_vector_scalar_ops(normalize_ws(m.group(2))))
    m = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)\s*-=\s*(.+)", inc)
    if m:
        return m.group(1), "-" + trim_statement(sanitize_vector_scalar_ops(normalize_ws(m.group(2))))
    m = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*\1\s*\+\s*(.+)", inc)
    if m:
        return m.group(1), trim_statement(sanitize_vector_scalar_ops(normalize_ws(m.group(2))))
    m = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*\1\s*-(?!>)\s*(.+)", inc)
    if m:
        return m.group(1), "-" + trim_statement(sanitize_vector_scalar_ops(normalize_ws(m.group(2))))
    return None


def affine_header_recurrences(var: str, inc_parts: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in inc_parts:
        parsed = affine_recurrence_update(part, var)
        if not parsed:
            continue
        name, stride = parsed
        if name and name != var:
            out[name] = stride
    return out


def bounded_conjunctive_range_spec(loop: dict[str, Any]) -> dict[str, str] | None:
    """Canonicalize `i < bound && base + i < limit` style loop domains.

    This is a structural AST/header rewrite for scalar-control loops whose
    effective trip count is the intersection of a count bound and a pointer or
    base-expression bound.  It avoids leaving a raw C condition in the
    Bootstrap, without inventing a new loop kind.
    """
    text = loop_header_line(loop)
    m = re.match(r"for\s*\(\s*(.*?)\s*;\s*(.*?)\s*;\s*(.*?)\s*\)\s*$", text)
    if not m:
        return None
    init, cond, inc = [normalize_ws(x) for x in m.groups()]
    lhs, op, start = split_assignment(init)
    var = lhs.split()[-1] if lhs else ""
    if not var or op != "=":
        return None
    parts = [normalize_ws(x) for x in cond.split("&&") if normalize_ws(x)]
    if len(parts) != 2:
        return None
    inc_parts = split_top_level_commas(inc)
    step = primary_induction_step(var, inc_parts)
    if step is None or step.startswith("-"):
        return None

    count_bound = ""
    base_expr = ""
    limit_expr = ""
    for part in parts:
        part_s = sanitize_vector_scalar_ops(part)
        m_count = re.fullmatch(rf"{re.escape(var)}\s*<\s*(.+)", part_s)
        if m_count:
            count_bound = sanitize_vector_scalar_ops(normalize_ws(m_count.group(1)))
            continue
        m_ptr = re.fullmatch(rf"(.+?)\s*\+\s*{re.escape(var)}\s*<\s*(.+)", part_s)
        if not m_ptr:
            m_ptr = re.fullmatch(rf"{re.escape(var)}\s*\+\s*(.+?)\s*<\s*(.+)", part_s)
        if m_ptr:
            base_expr = sanitize_vector_scalar_ops(normalize_ws(m_ptr.group(1)))
            limit_expr = sanitize_vector_scalar_ops(normalize_ws(m_ptr.group(2)))
            continue
        return None
    if not count_bound or not base_expr or not limit_expr:
        return None
    return {
        "var": var,
        "start": sanitize_vector_scalar_ops(start),
        "end": f"min({count_bound}, {limit_expr} - {base_expr})",
        "step": sanitize_vector_scalar_ops(step),
    }


def render_loop_as_range(
    loop: dict[str, Any],
    feats: dict[str, Any] | None = None,
    *,
    cpp_for: bool = False,
) -> str:
    """Render a C/C++ for-loop header as compact code-like pseudocode."""
    text = loop_header_line(loop)
    if str(loop.get("kind") or "") == "DoStmt" or text.strip() == "do":
        m_do = re.search(r"\}\s*while\s*\((.*?)\)\s*;?", str(loop.get("snippet") or ""), re.S)
        if m_do:
            cond = sanitize_vector_scalar_ops(normalize_ws(m_do.group(1))).replace("&&", " and ").replace("||", " or ")
            return f"do while {cond}:"
        return "do:"
    m_while = re.match(r"while\s*\((.*)\)\s*$", text)
    if m_while:
        cond = sanitize_vector_scalar_ops(normalize_ws(m_while.group(1)))
        cond = cond.replace("&&", " and ").replace("||", " or ")
        cond = normalize_ws(cond)
        return f"while {cond}:"
    if cpp_for and text.startswith("for"):
        return text.rstrip(":") + ":"
    bounded = bounded_conjunctive_range_spec(loop)
    if bounded:
        bounded_var = typed_decl_name(bounded["var"], loop_var_decl_type(loop, feats, bounded["var"]))
        extras: list[str] = []
        if bounded["step"] not in {"1", "+1"}:
            extras.append(f"step={bounded['step']}")
        if extras:
            return f"for {bounded_var} in range({bounded['start']}, {bounded['end']}, {', '.join(extras)}):"
        return f"for {bounded_var} in range({bounded['start']}, {bounded['end']}):"
    m = re.match(
        r"for\s*\(\s*(?:(?:[A-Za-z_][\w:<>]*\s+)+)?([A-Za-z_]\w*)\s*=\s*([^;]+?)\s*;\s*"
        r"\1\s*([<>]=?)\s*([^;]+?)\s*;\s*(.*?)\s*\)\s*$",
        text,
    )
    if not m:
        return render_generic_for_loop(text)
    var, start, cmp_op, end, inc = [normalize_ws(x) for x in m.groups()]
    if re.search(r"&&|\|\|", end):
        return render_generic_for_loop(text)
    rendered_var = typed_decl_name(var, loop_var_decl_type(loop, feats, var))
    start = sanitize_vector_scalar_ops(start)
    end = sanitize_vector_scalar_ops(end)
    inc_parts = split_top_level_commas(inc)
    step = primary_induction_step(var, inc_parts) or "1"
    if cmp_op in {">", ">="} and not step.startswith("-"):
        step = "-" + step
    extras: list[str] = []
    if step not in {"1", "+1"}:
        extras.append(f"step={step}")
    if cmp_op in {"<=", ">="}:
        extras.append("inclusive=True")
    if extras:
        return f"for {rendered_var} in range({start}, {end}, {', '.join(extras)}):"
    if step in {"1", "+1"}:
        return f"for {rendered_var} in range({start}, {end}):"
    return f"for {rendered_var} in range({start}, {end}, step={step}):"


def render_generic_for_loop(text: str) -> str:
    """Render non-affine/range-for headers without leaving raw C syntax."""
    text = normalize_ws(text)
    m_range = re.match(
        r"for\s*\(\s*(?:const\s+)?(?:auto|[A-Za-z_][\w:<>]*(?:\s*[*&])?)\s*(?:&\s*)?([A-Za-z_]\w*)\s*:\s*([^)]+)\)\s*$",
        text,
    )
    if m_range:
        return f"for {m_range.group(1)} in {normalize_ws(m_range.group(2))}:"
    m = re.match(r"for\s*\(\s*(.*?)\s*;\s*(.*?)\s*;\s*(.*?)\s*\)\s*$", text)
    if not m:
        return text.rstrip(":") + ":"
    init, cond, inc = [normalize_ws(x) for x in m.groups()]
    lhs, op, start = split_assignment(init)
    var = re.sub(r"^[*&]+", "", lhs.split()[-1]) if lhs else ""
    if var and op == "=":
        start = sanitize_vector_scalar_ops(start)
        cond = sanitize_vector_scalar_ops(cond)
        inc_parts = split_top_level_commas(inc)
        step = primary_induction_step(var, inc_parts) or sanitize_vector_scalar_ops(normalize_ws(inc))
        return f"for {var} in loop(start={start}, condition={cond}, step={step}):"
    return f"for loop(init={sanitize_vector_scalar_ops(init)}, condition={sanitize_vector_scalar_ops(cond)}, step={sanitize_vector_scalar_ops(inc)}):"


def parse_for_loop_spec(loop: dict[str, Any]) -> dict[str, str] | None:
    text = loop_header_line(loop)
    m = re.match(
        r"for\s*\(\s*(?:(?:[A-Za-z_][\w:<>]*\s+)+)?([A-Za-z_]\w*)\s*=\s*([^;]+?)\s*;\s*"
        r"\1\s*([<>]=?)\s*([^;]+?)\s*;\s*(.*?)\s*\)\s*$",
        text,
    )
    if not m:
        return None
    var, start, cmp_op, end, inc = [normalize_ws(x) for x in m.groups()]
    # Multi-induction headers such as `for (i = 0, j = n - 1; i < j; ++i, --j)`
    # are not a single independent lane variable.  Handle them with the
    # dedicated converging-loop rule or keep them as scalar code.
    if len(split_top_level_commas(start)) > 1 or re.search(r"(^|,)\s*[A-Za-z_]\w*\s*=", start):
        return None
    start = sanitize_vector_scalar_ops(start)
    extra_condition = ""
    if "&&" in end:
        parts = [normalize_ws(x) for x in end.split("&&") if normalize_ws(x)]
        end = sanitize_vector_scalar_ops(parts[0])
        if len(parts) > 1:
            extra_condition = " && ".join(sanitize_vector_scalar_ops(x) for x in parts[1:])
    elif "||" in end:
        return None
    else:
        end = sanitize_vector_scalar_ops(end)
    inc_parts = split_top_level_commas(inc)
    step = primary_induction_step(var, inc_parts)
    if step is None:
        return None
    if cmp_op in {">", ">="} and not step.startswith("-"):
        step = "-" + step
    spec = {"var": var, "start": start, "end": end, "cmp": cmp_op, "step": step}
    recurrences = affine_header_recurrences(var, inc_parts)
    if recurrences:
        spec["affine_recurrences"] = json.dumps(recurrences, sort_keys=True)
    if extra_condition:
        spec["extra_condition"] = extra_condition
    return spec


def parse_generic_vector_loop_spec(loop: dict[str, Any]) -> dict[str, str] | None:
    """Parse non-affine but vector-candidate for headers.

    This is intentionally structural, not benchmark-specific.  It keeps the
    original loop condition as the active guard, while choosing a conservative
    vector iteration bound from the condition's upper expression.
    """
    text = loop_header_line(loop)
    m = re.match(r"for\s*\(\s*(.*?)\s*;\s*(.*?)\s*;\s*(.*?)\s*\)\s*$", text)
    if not m:
        return None
    init, cond, inc = [normalize_ws(x) for x in m.groups()]
    if len(split_top_level_commas(init)) > 1:
        return None
    lhs, op, start = split_assignment(init)
    var = lhs.split()[-1] if lhs else ""
    if not var or op != "=":
        return None
    start = sanitize_vector_scalar_ops(start)
    inc_parts = split_top_level_commas(inc)
    step = primary_induction_step(var, inc_parts)
    if step is None:
        return None
    if step.startswith("-"):
        return None
    cond_norm = sanitize_vector_scalar_ops(normalize_ws(cond))
    if "||" in cond_norm:
        return None
    end = ""
    cmp_op = "<"
    # Candidate-domain loops such as i*i <= n.
    m_sqrt = re.match(rf"{re.escape(var)}\s*\*\s*{re.escape(var)}\s*(<=|<)\s*(.+)$", cond_norm)
    if m_sqrt:
        cmp_op = m_sqrt.group(1)
        end = f"sqrt_bound({sanitize_vector_scalar_ops(normalize_ws(m_sqrt.group(2)))})"
    else:
        # Affine/index expressions such as i*2+1 < size or i < size-1-i.
        m_cmp = re.match(r"(.+?)\s*(<=|<)\s*(.+)$", cond_norm)
        if not m_cmp:
            return None
        lhs_expr, cmp_op, rhs_expr = [sanitize_vector_scalar_ops(normalize_ws(x)) for x in m_cmp.groups()]
        if not re.search(rf"\b{re.escape(var)}\b", lhs_expr):
            return None
        end = rhs_expr
        if re.search(rf"\b{re.escape(var)}\b", end):
            # For symmetric bounds such as i < n - 1 - i, keep the original
            # condition as the guard and use the non-self part only as a safe
            # outer bound.
            parts = re.split(rf"\s*[-+]\s*{re.escape(var)}\b", end, maxsplit=1)
            end = normalize_ws(parts[0]) if parts and parts[0].strip() else end
    if not end:
        return None
    spec = {"var": var, "start": start, "end": end, "cmp": cmp_op, "step": step, "active_condition": cond_norm}
    recurrences = affine_header_recurrences(var, inc_parts)
    if recurrences:
        spec["affine_recurrences"] = json.dumps(recurrences, sort_keys=True)
    return spec


def vector_loop_spec(loop: dict[str, Any]) -> dict[str, str] | None:
    spec = parse_for_loop_spec(loop)
    if spec and not re.search(rf"\b{re.escape(spec['var'])}\b", str(spec.get("end") or "")):
        return spec
    return parse_generic_vector_loop_spec(loop) or spec


def vector_lane_template_type(lane_type: str = "") -> str:
    typ = render_decl_type(lane_type)
    if not typ:
        return "int"
    typ = re.sub(r"\b(const|volatile)\b", "", typ).strip()
    typ = typ.replace("&", "").replace("*", "").strip()
    return normalize_ws(typ) or "int"


def split_template_args(text: str) -> list[str]:
    args: list[str] = []
    start = 0
    depth = 0
    for i, ch in enumerate(str(text or "")):
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth = max(0, depth - 1)
        elif ch == "," and depth == 0:
            args.append(text[start:i].strip())
            start = i + 1
    tail = str(text or "")[start:].strip()
    if tail:
        args.append(tail)
    return args


def type_is_container_like(typ: str) -> bool:
    return bool(re.search(r"\b(?:std::)?(?:vector|string|map|unordered_map|set|unordered_set|deque|list)\b", render_decl_type(typ)))


def canonical_vector_element_type(typ: str) -> str:
    text = render_decl_type(typ)
    if not text:
        return ""
    text = re.sub(r"\b(__restrict__|__restrict|restrict|const|volatile)\b", "", text)
    text = normalize_ws(text.replace(" &", "&").replace("&", " ").replace(" *", "*").replace("* ", "*"))
    text = text.strip()
    vector_match = re.search(r"\bstd::vector\s*<(.+)>\s*$", text)
    if vector_match:
        args = split_template_args(vector_match.group(1))
        if not args:
            return ""
        elem = normalize_ws(args[0])
        if type_is_container_like(elem):
            return ""
        return elem
    if re.search(r"\b(?:std::)?string\b", text):
        return "char"
    text = re.sub(r"\[[^\]]*\]", "", text).strip()
    while text.endswith("*"):
        text = text[:-1].strip()
    alias_map = {
        "JSAMPROW": "JSAMPLE",
        "JSAMPARRAY": "JSAMPROW",
        "DCTELEM": "DCTELEM",
        "float16_t": "float16_t",
        "FLOAT16_t": "FLOAT16_t",
    }
    text = alias_map.get(text, text)
    if text in {"auto", "void", "bool"}:
        return ""
    if re.search(r"\b(?:std::)?(?:vector|string|map|unordered_map|set|unordered_set|deque|list)\b", text):
        return ""
    return normalize_ws(text)


def ref_index_mentions_var(ref: dict[str, str], var: str) -> bool:
    idx = str(ref.get("index") or "")
    return bool(var and re.search(rf"\b{re.escape(var)}\b", idx))


def infer_vector_element_type(
    direct_rw: list[dict[str, Any]],
    feats: dict[str, Any] | None,
    spec: dict[str, str] | None,
) -> str:
    if not feats or not spec:
        return ""
    type_map = variable_type_map(feats)
    loop_var = str(spec.get("var") or "")
    scored: list[tuple[int, int, str]] = []
    order = 0
    for item in sorted_by_begin([x for x in direct_rw if not is_loop_initializer_assignment(x)]):
        ref_groups = [
            (list(item.get("writes") or []), 100),
            (read_refs_for_item(item), 50),
        ]
        for refs, base_score in ref_groups:
            for ref in refs:
                base = str(ref.get("base") or "")
                elem_type = canonical_vector_element_type(type_map.get(base, ""))
                if not elem_type:
                    continue
                score = base_score
                if ref_index_mentions_var(ref, loop_var):
                    score += 20
                if elem_type not in {"bool", "char"}:
                    score += 1
                scored.append((score, -order, elem_type))
                order += 1
    if not scored:
        return ""
    scored.sort(reverse=True)
    return scored[0][2]


def vector_step_multiplier(step: str) -> str:
    step = sanitize_vector_scalar_ops(step)
    if step in {"1", "+1"}:
        return ""
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*|\d+", step):
        return f" * {step}"
    return f" * ({step})"


def vector_range_header(
    spec: dict[str, str],
    base_name: str = "vec_base",
    lane_type: str = "",
    vector_type: str = "",
) -> str:
    start = sanitize_vector_scalar_ops(spec["start"])
    end = sanitize_vector_scalar_ops(spec["end"])
    step = sanitize_vector_scalar_ops(spec["step"])
    args = [start, end]
    if step not in {"1", "+1"}:
        args.append(f"step={step}")
    if spec.get("cmp") in {"<=", ">="}:
        args.append("inclusive=True")
    return f"for {base_name} in vector_range({', '.join(args)}):"


def vector_active_lanes_line(
    spec: dict[str, str],
    base_name: str = "vec_base",
    lane_type: str = "",
    vector_type: str = "",
) -> str:
    return active_predicate_line(spec)


def vector_lane_line(spec: dict[str, str], base_name: str = "vec_base", lane_type: str = "") -> str:
    var = spec["var"]
    step = sanitize_vector_scalar_ops(spec["step"])
    if step in {"1", "+1"}:
        return f"lane {var} = {base_name} + lane_id"
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*|\d+", step):
        return f"lane {var} = {base_name} + lane_id * {step}"
    return f"lane {var} = {base_name} + lane_id * ({step})"


def append_vector_loop_prelude(
    lines: list[str],
    indent: str,
    spec: dict[str, str],
    *,
    lane_type: str = "",
    vector_type: str = "",
    base_name: str = "vec_base",
) -> None:
    lines.append(f"{indent}{vector_range_header(spec, base_name=base_name, lane_type=lane_type, vector_type=vector_type)}")
    lines.append(f"{indent}  {vector_lane_line(spec, base_name=base_name, lane_type=lane_type)}")
    lines.append(f"{indent}  {active_guard_line(spec)}")


def append_vector_chunk_prelude(
    lines: list[str],
    indent: str,
    spec: dict[str, str],
    *,
    lane_type: str = "",
    vector_type: str = "",
    base_name: str = "vec_base",
) -> None:
    lines.append(f"{indent}{vector_range_header(spec, base_name=base_name, lane_type=lane_type, vector_type=vector_type)}")


def active_predicate_line(spec: dict[str, str]) -> str:
    if spec.get("active_condition"):
        return f"active = {spec['active_condition']}"
    pred = f"{spec['var']} {spec['cmp']} {spec['end']}"
    if spec.get("extra_condition"):
        pred += f" && {spec['extra_condition']}"
    return f"active = {pred}"


def active_guard_line(spec: dict[str, str]) -> str:
    if spec.get("active_condition"):
        return f"where active = {spec['active_condition']}:"
    pred = f"{spec['var']} {spec['cmp']} {spec['end']}"
    if spec.get("extra_condition"):
        pred += f" && {spec['extra_condition']}"
    return f"where active = {pred}:"


def spec_affine_recurrences(spec: dict[str, str] | None) -> dict[str, str]:
    if not spec:
        return {}
    raw = spec.get("affine_recurrences") or ""
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except Exception:
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {str(k): sanitize_vector_scalar_ops(str(v)) for k, v in parsed.items() if str(k)}


def lane_ordinal_expr(spec: dict[str, str]) -> str:
    var = sanitize_vector_scalar_ops(str(spec.get("var") or "i"))
    start = sanitize_vector_scalar_ops(str(spec.get("start") or "0"))
    step = sanitize_vector_scalar_ops(str(spec.get("step") or "1"))
    ordinal = var if start in {"0", "+0"} else f"({var} - ({start}))"
    if step not in {"1", "+1"}:
        ordinal = f"(({ordinal}) / ({step}))"
    return ordinal


def pointer_recurrence_offset_expr(spec: dict[str, str], stride: str) -> str:
    ordinal = lane_ordinal_expr(spec)
    stride = sanitize_vector_scalar_ops(str(stride or "1"))
    if stride in {"1", "+1"}:
        return ordinal
    if stride == "-1":
        return f"-({ordinal})"
    return f"{ordinal} * {stride}"


def combine_affine_index(offset: str, index: str) -> str:
    offset = normalize_ws(offset)
    index = normalize_ws(index)
    if not index or index in {"0", "+0"}:
        return offset
    if not offset or offset in {"0", "+0"}:
        return index
    if re.fullmatch(r"[-+]?(?:0x[0-9A-Fa-f]+|\d+|[A-Za-z_][A-Za-z0-9_]*)", index):
        return f"{offset} + {index}"
    return f"{offset} + ({index})"


POSTINC_POINTER_READ_RE = re.compile(
    r"\*\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:\+\+|--)"
)


def postinc_pointer_read_names(text: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for match in POSTINC_POINTER_READ_RE.finditer(str(text or "")):
        name = match.group(1)
        if name and name not in seen:
            seen.add(name)
            out.append(name)
    return out


def postinc_pointer_read_counts(text: str) -> Counter[str]:
    return Counter(match.group(1) for match in POSTINC_POINTER_READ_RE.finditer(str(text or "")) if match.group(1))


def replace_postinc_pointer_reads(text: str, indexes: dict[str, str] | None = None) -> str:
    indexes = indexes or {}

    def repl(match: re.Match[str]) -> str:
        name = match.group(1)
        index = indexes.get(name, "0")
        return f"{name}[{index}]"

    return POSTINC_POINTER_READ_RE.sub(repl, str(text or ""))


def postinc_subscript_update_names(text: str) -> list[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str]] = []
    for match in re.finditer(r"\[([^\[\]]*)\]", str(text or "")):
        for upd in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*(\+\+|--)", match.group(1)):
            item = (upd.group(1), "+= 1" if upd.group(2) == "++" else "-= 1")
            if item not in seen:
                seen.add(item)
                out.append(item)
    return out


def annotate_read_write_postinc_subscripts(read_write: list[dict[str, Any]]) -> None:
    """Attach AST-derived post-increment subscript metadata to read/write items.

    This keeps side-effectful subscripts such as out[index++] in the structured
    item representation.  The renderer consumes these fields rather than
    post-processing already-rendered bootstrap text.
    """
    for item in read_write:
        lhs = str(item.get("lhs") or "")
        rhs = str(item.get("rhs") or "")
        updates: list[dict[str, str]] = []
        for name, op_text in postinc_subscript_update_names(lhs + " " + rhs):
            op, value = op_text.split(maxsplit=1)
            updates.append({"name": name, "op": op, "rhs": value})
        if updates:
            item["postinc_subscript_updates"] = updates
        lhs_norm = replace_postinc_subscript_indices(lhs)
        rhs_norm = replace_postinc_subscript_indices(rhs)
        if lhs_norm != lhs:
            item["lhs_without_postinc_subscript"] = lhs_norm
        if rhs_norm != rhs:
            item["rhs_without_postinc_subscript"] = rhs_norm


def replace_postinc_subscript_indices(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        inner = re.sub(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*(?:\+\+|--)", r"\1", match.group(1))
        return "[" + inner + "]"

    return re.sub(r"\[([^\[\]]*)\]", repl, str(text or ""))


def replace_postinc_scalar_expr(text: str) -> str:
    return re.sub(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*(?:\+\+|--)", r"\1", str(text or ""))


def affine_pointer_index(base: str, index: str, spec: dict[str, str] | None) -> str:
    recurrences = spec_affine_recurrences(spec)
    if base not in recurrences or not spec:
        return sanitize_vector_scalar_ops(index)
    offset = pointer_recurrence_offset_expr(spec, recurrences[base])
    return sanitize_vector_scalar_ops(combine_affine_index(offset, index))


def loop_body_pointer_index(
    base: str,
    index: str,
    spec: dict[str, str] | None,
    pointer_recurrences: dict[str, str] | None,
) -> str:
    if not spec or not pointer_recurrences or base not in pointer_recurrences:
        return sanitize_vector_scalar_ops(index)
    offset = pointer_recurrence_offset_expr(spec, pointer_recurrences[base])
    return sanitize_vector_scalar_ops(combine_affine_index(offset, index))


def rewrite_affine_pointer_subscripts(text: str, spec: dict[str, str] | None) -> str:
    recurrences = spec_affine_recurrences(spec)
    if not recurrences:
        return str(text or "")
    s = str(text or "")
    out: list[str] = []
    i = 0
    while i < len(s):
        match = None
        for name in sorted(recurrences, key=len, reverse=True):
            m = re.match(rf"\b{re.escape(name)}\s*\[", s[i:])
            if m:
                match = (name, i + m.end() - 1)
                break
        if not match:
            out.append(s[i])
            i += 1
            continue
        name, lbr = match
        close = _scan_matching_right(s, lbr, "[", "]")
        if close <= lbr:
            out.append(s[i])
            i += 1
            continue
        index = s[lbr + 1 : close]
        rewritten_index = affine_pointer_index(name, index, spec)
        out.append(f"{name}[{rewritten_index}]")
        i = close + 1
    return "".join(out)


def rewrite_loop_body_pointer_subscripts(
    text: str,
    spec: dict[str, str] | None,
    pointer_recurrences: dict[str, str] | None,
) -> str:
    if not spec or not pointer_recurrences:
        return str(text or "")
    s = str(text or "")
    out: list[str] = []
    i = 0
    while i < len(s):
        match = None
        for name in sorted(pointer_recurrences, key=len, reverse=True):
            m = re.match(rf"\b{re.escape(name)}\s*\[", s[i:])
            if m:
                match = (name, i + m.end() - 1)
                break
        if not match:
            out.append(s[i])
            i += 1
            continue
        name, lbr = match
        close = _scan_matching_right(s, lbr, "[", "]")
        if close <= lbr:
            out.append(s[i])
            i += 1
            continue
        index = s[lbr + 1 : close]
        rewritten_index = loop_body_pointer_index(name, index, spec, pointer_recurrences)
        out.append(f"{name}[{rewritten_index}]")
        i = close + 1
    return "".join(out)


def replace_symbol_exprs(text: str, replacements: dict[str, str] | None) -> str:
    if not replacements:
        return str(text or "")
    out = str(text or "")
    for name in sorted(replacements, key=len, reverse=True):
        expr = replacements[name]
        if not name or not expr:
            continue
        out = re.sub(rf"\b{re.escape(name)}\b", f"({expr})", out)
    return out


def scalar_induction_lane_aliases(loop: dict[str, Any], direct_rw: list[dict[str, Any]], feats: dict[str, Any]) -> dict[str, str]:
    """Lift scalar ordinal counters inside vector loops into lane expressions."""
    spec = vector_loop_spec(loop)
    if not spec:
        return {}
    loop_begin = loop.get("begin")
    if not isinstance(loop_begin, int):
        return {}
    used_indices: set[str] = set()
    for item in direct_rw:
        for ref in (item.get("reads") or []) + (item.get("writes") or []):
            idx = normalize_ws(str(ref.get("index") or ""))
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", idx):
                used_indices.add(idx)
    out: dict[str, str] = {}
    for item in direct_rw:
        lhs = normalize_ws(str(item.get("lhs") or ""))
        if not lhs or lhs == spec["var"] or lhs not in used_indices:
            continue
        op = str(item.get("op") or "")
        rhs = normalize_ws(str(item.get("rhs") or ""))
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", lhs):
            continue
        if not ((op == "+=" and rhs in {"1", "1u", "1U"}) or (op == "=" and rhs in {f"{lhs} + 1", f"1 + {lhs}"})):
            continue
        init = "0"
        for prev in sorted_by_begin(feats.get("read_write") or []):
            begin = prev.get("begin")
            if not isinstance(begin, int) or begin >= loop_begin:
                break
            if normalize_ws(str(prev.get("lhs") or "")) == lhs and str(prev.get("op") or "") == "=":
                init = normalize_ws(str(prev.get("rhs") or "0"))
        step = normalize_ws(str(spec.get("step") or "1"))
        ordinal = f"({spec['var']} - ({spec['start']}))"
        if step not in {"1", "+1"}:
            ordinal = f"(({ordinal}) / ({step}))"
        out[lhs] = ordinal if init in {"0", "0u", "0U"} else f"({init}) + {ordinal}"
    return out


def item_is_scalar_induction_update(item: dict[str, Any], aliases: dict[str, str]) -> bool:
    lhs = normalize_ws(str(item.get("lhs") or ""))
    if lhs not in aliases:
        return False
    op = str(item.get("op") or "")
    rhs = normalize_ws(str(item.get("rhs") or ""))
    return (op == "+=" and rhs in {"1", "1u", "1U"}) or (op == "=" and rhs in {f"{lhs} + 1", f"1 + {lhs}"})


def render_branch_as_code(branch: dict[str, Any]) -> str:
    line = branch_condition_line(branch)
    if line.startswith("if "):
        return line.rstrip(":") + ":"
    return line.rstrip(":") + ":"


def is_loop_initializer_assignment(item: dict[str, Any]) -> bool:
    return bool(item.get("is_loop_initializer"))


def function_return_type(signature: str) -> str:
    head = str(signature or "").split("(", 1)[0].strip()
    if not head or " " not in head:
        return ""
    ret = head.rsplit(" ", 1)[0].strip()
    ret = re.sub(r"\b(static|inline|extern|constexpr)\b", "", ret).strip()
    return normalize_ws(ret)


def render_return_expr(text: str, *, float_names: set[str] | None = None, return_type: str = "") -> str:
    expr = trim_statement(sanitize_vector_scalar_ops(text or "", float_names=float_names))
    expr = re.sub(r"^\s*return\b\s*", "", expr)
    expr = expr.rstrip(";").strip()
    ret_type = render_decl_type(return_type)
    if expr.startswith("{") and ret_type.startswith("std::vector"):
        return f"{ret_type}{expr}"
    return expr


def render_vector_predicate_return_expr(
    text: str,
    spec: dict[str, str] | None,
    *,
    float_names: set[str] | None = None,
) -> str:
    expr = render_return_expr(text, float_names=float_names)
    if not spec:
        return expr
    var = spec.get("var") or ""
    if var and re.search(rf"\b{re.escape(var)}\b", expr):
        expr = re.sub(rf"\b{re.escape(var)}\b", f"first_matching_lane({var})", expr)
    return expr


def is_scalar_zero_or_empty_init(item: dict[str, Any]) -> bool:
    if item.get("writes") or item.get("reads"):
        return False
    if str(item.get("op") or "") != "=":
        return False
    lhs = normalize_ws(str(item.get("lhs") or ""))
    rhs = normalize_ws(str(item.get("rhs") or ""))
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", lhs):
        return False
    return rhs in {"0", "0u", "0U", "0L", "0UL", "0.0", "0.0f", "false", "NULL", "nullptr", '""'}


def is_simple_scalar_var_init(item: dict[str, Any]) -> bool:
    if item.get("source") != "var_decl":
        return False
    if item.get("writes") or item.get("reads"):
        return False
    if str(item.get("op") or "") != "=":
        return False
    lhs = normalize_ws(str(item.get("lhs") or ""))
    rhs = normalize_ws(str(item.get("rhs") or ""))
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", lhs):
        return False
    if re.search(r"\b[A-Za-z_][A-Za-z0-9_:]*\s*\(", rhs):
        return False
    if re.search(r"\[[^\]]+\]|->|\.|[?:*/%<>=!&|^]", rhs):
        return False
    if re.search(r"(?<!^)[+-]", rhs):
        return False
    return bool(re.fullmatch(r"(?:[-+]?(?:0x[0-9A-Fa-f]+|\d+(?:\.\d+)?)(?:[uUlLfF]*)|true|false|NULL|nullptr|\"\"|[A-Za-z_][A-Za-z0-9_]*)", rhs))


def is_std_vector_constructor_init(item: dict[str, Any]) -> bool:
    if item.get("source") != "var_decl":
        return False
    if str(item.get("op") or "") != "=":
        return False
    lhs = normalize_ws(str(item.get("lhs") or ""))
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", lhs):
        return False
    decl_type = render_decl_type(str(item.get("decl_type") or ""))
    if not re.search(r"\bstd::vector\s*<", decl_type):
        return False
    rhs = normalize_ws(str(item.get("rhs") or ""))
    if rhs in {"{}", "{ }"}:
        return True
    return bool(re.match(r"^std::vector\s*<", rhs))


def item_identity(item: dict[str, Any]) -> str:
    return f"{item.get('begin')}:{item.get('end')}:{item.get('source')}:{item.get('snippet')}"


def rw_context_key(item: dict[str, Any], loops: list[dict[str, Any]], branches: list[dict[str, Any]]) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    offset = item.get("begin")
    loop_key = tuple((x.get("begin"), x.get("end")) for x in loops if contains_offset(x, offset))
    branch_key = tuple((x.get("begin"), x.get("end"), x.get("else_begin")) for x in branches if contains_offset(x, offset))
    return loop_key, branch_key


def build_segment_read_map(
    rw: list[dict[str, Any]],
    loops: list[dict[str, Any]],
    branches: list[dict[str, Any]],
) -> dict[str, list[dict[str, str]]]:
    read_map: dict[str, list[dict[str, str]]] = {}
    segment: list[dict[str, Any]] = []
    current_key: tuple[tuple[Any, ...], tuple[Any, ...]] | None = None

    def flush() -> None:
        nonlocal segment, current_key
        if not segment:
            current_key = None
            return
        refs: list[dict[str, str]] = []
        for seg_item in segment:
            refs.extend(read_refs_for_item(seg_item))
        if refs:
            read_map[item_identity(segment[0])] = refs
        segment = []
        current_key = None

    for item in sorted_by_begin(rw):
        if is_loop_initializer_assignment(item):
            continue
        key = rw_context_key(item, loops, branches)
        if current_key is None:
            current_key = key
        elif key != current_key:
            flush()
            current_key = key
        segment.append(item)
        if item.get("writes"):
            flush()
    flush()
    return read_map


def append_read_write_index_lines(
    lines: list[str],
    item: dict[str, Any],
    indent: str,
    *,
    grouped_reads: list[dict[str, str]] | None = None,
    suppress_item_reads: bool = False,
) -> None:
    if is_loop_initializer_assignment(item):
        return
    rendered = render_assignment_line(item)
    if is_malformed_semantic_line(rendered):
        return
    writes = render_refs(item.get("writes") or [])
    reads = render_refs(read_refs_for_item(item))
    if grouped_reads:
        grouped = render_grouped_refs(grouped_reads)
        if grouped:
            lines.append(f"{indent}read: {grouped}")
    elif reads and not suppress_item_reads:
        lines.append(f"{indent}read: {reads}")
    if (
        is_std_vector_constructor_init(item)
        or is_simple_scalar_var_init(item)
        or is_scalar_zero_or_empty_init(item)
    ) and rendered:
        lines.append(f"{indent}init: {rendered}")
    elif item.get("op") and item.get("op") != "=" and rendered:
        lines.append(f"{indent}update: {rendered}")
    elif writes:
        lines.append(f"{indent}write: {rendered}")
    elif reads:
        lines.append(f"{indent}compute: {rendered}")
    elif rendered:
        # Keep scalar state that is part of the algorithm, but avoid labeling it
        # as dataflow.  Loop initializers are filtered above.
        lines.append(f"{indent}compute: {rendered}")


def render_unified_dataflow_bootstrap(signature: str, feats: dict[str, Any]) -> str:
    """Emit one uniform code-like read/write/index pseudocode form.

    This renderer deliberately avoids route names, SVE hints, and DSL lifts such
    as store_map/sum/exists.  It also avoids report-style block labels
    (variable_types/loops/dataflow).  The output is intended to look like
    compact algorithmic pseudocode:

        void foo(...):
          for i in range(0, n):
            read: a[i], b[i]
            write: dst[i] = a[i] + b[i]
    """
    lines = [f"{signature.rstrip(';')}:"]
    loops = feats.get("loops") or []

    deps = feats.get("same_array_dependencies") or []
    if deps:
        for dep in deps[:6]:
            base = str(dep.get("base") or "").strip()
            write_index = str(dep.get("write_index") or "").strip()
            read_index = str(dep.get("read_index") or "").strip()
            if base:
                lines.append(f"  dependency: {base}[{write_index}] reads {base}[{read_index}]")

    rw = feats.get("read_write") or []
    branches = sorted_by_begin(feats.get("branches") or [])
    segment_read_map = build_segment_read_map(rw, loops, branches) if rw else {}
    emitted_branches: set[str] = set()
    events: list[tuple[str, dict[str, Any]]] = []
    for loop in loops[:12]:
        events.append(("loop", loop))
    for call in (feats.get("calls") or [])[:24]:
        if is_side_effect_call(call):
            events.append(("call", call))
    if rw:
        for item in rw[:24]:
            events.append(("rw", item))
    elif feats.get("assignments"):
        for item in (feats.get("assignments") or [])[:12]:
            events.append(("assignment", item))
    for ret in (feats.get("returns") or [])[:4]:
        events.append(("return", ret))

    for kind, item in sorted(events, key=lambda x: (x[1].get("begin") if isinstance(x[1].get("begin"), int) else 10**18, {"loop": 0, "call": 1, "assignment": 1, "rw": 1, "return": 2}.get(x[0], 9))):
        if kind == "loop":
            line = render_loop_as_range(item)
            if line and not is_malformed_semantic_line(line):
                depth = containing_count([x for x in loops if x is not item], item.get("begin"))
                lines.append(f"{'  ' * (depth + 1)}{line}")
        elif kind == "rw":
            original_item_id = item_identity(item)
            offset = item.get("begin")
            containing_loops = [x for x in loops if contains_offset(x, offset)]
            loop_depth = len(containing_loops)
            indent = "  " * (loop_depth + 1)
            containing_branches = [b for b in branches if contains_offset(b, offset)]
            if len(containing_branches) == 1:
                branch = containing_branches[0]
                else_begin = branch.get("else_begin")
                in_else = isinstance(else_begin, int) and isinstance(offset, int) and offset >= else_begin
                branch_line = "else:" if in_else else render_branch_as_code(branch)
                branch_key = f"{branch.get('begin')}:{'else' if in_else else 'if'}:{branch_line}"
                branch_indent = "  " * (loop_depth + 1)
                if branch_line and branch_key not in emitted_branches and not is_malformed_semantic_line(branch_line):
                    lines.append(f"{branch_indent}{branch_line}")
                    emitted_branches.add(branch_key)
                indent = branch_indent + "  "
            append_read_write_index_lines(
                lines,
                item,
                indent,
                grouped_reads=segment_read_map.get(original_item_id),
                suppress_item_reads=True,
            )
        elif kind == "assignment":
            loop_depth = containing_count(loops, item.get("begin"))
            indent = "  " * (loop_depth + 1)
            line = trim_statement(item.get("snippet") or "")
            if line and not is_malformed_semantic_line(line):
                lines.append(f"{indent}compute: {line}")
        elif kind == "call":
            loop_depth = containing_count(loops, item.get("begin"))
            indent = "  " * (loop_depth + 1)
            line = render_side_effect_call(item)
            if line and not is_malformed_semantic_line(line):
                lines.append(f"{indent}{side_effect_call_label(item)}: {line}")
        elif kind == "return":
            loop_depth = containing_count(loops, item.get("begin"))
            indent = "  " * (loop_depth + 1)
            line = render_return_expr(item.get("snippet") or "")
            if line and not is_malformed_semantic_line(line):
                lines.append(f"{indent}return: {line}")

    return "\n".join(lines)


def loop_key(loop: dict[str, Any]) -> tuple[Any, Any, str]:
    return (loop.get("begin"), loop.get("end"), str(loop.get("snippet") or ""))


def immediate_loop_parent(loop: dict[str, Any], loops: list[dict[str, Any]]) -> dict[str, Any] | None:
    parents = [
        cand
        for cand in loops
        if cand is not loop and contains_offset(cand, loop.get("begin")) and contains_offset(cand, loop.get("end"))
    ]
    if not parents:
        return None
    return min(parents, key=lambda x: (int(x.get("end") or 0) - int(x.get("begin") or 0)))


def build_loop_children(loops: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[tuple[Any, Any, str], list[dict[str, Any]]]]:
    children: dict[tuple[Any, Any, str], list[dict[str, Any]]] = {loop_key(loop): [] for loop in loops}
    roots: list[dict[str, Any]] = []
    for loop in sorted_by_begin(loops):
        parent = immediate_loop_parent(loop, loops)
        if parent is None:
            roots.append(loop)
        else:
            children.setdefault(loop_key(parent), []).append(loop)
    for vals in children.values():
        vals.sort(key=lambda x: x.get("begin") if isinstance(x.get("begin"), int) else 10**18)
    return roots, children


def item_inside_any_loop(item: dict[str, Any], loops: list[dict[str, Any]]) -> bool:
    return any(contains_offset(loop, item.get("begin")) for loop in loops)


def item_directly_in_loop(item: dict[str, Any], loop: dict[str, Any], child_loops: list[dict[str, Any]]) -> bool:
    if not contains_offset(loop, item.get("begin")):
        return False
    return not item_inside_any_loop(item, child_loops)


def items_in_loop_subtree(items: list[dict[str, Any]], loop: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in items if contains_offset(loop, item.get("begin"))]


def direct_loop_local_scalar_names(items: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for item in items:
        if item.get("source") != "var_decl" or is_loop_initializer_assignment(item):
            continue
        if item.get("writes"):
            continue
        lhs = normalize_ws(str(item.get("lhs") or ""))
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", lhs):
            names.add(lhs)
    return names


def variable_type_map(feats: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for decl in feats.get("var_decls") or []:
        name = normalize_ws(str(decl.get("name") or ""))
        typ = normalize_ws(str(decl.get("type") or ""))
        if name and typ and name not in out:
            out[name] = typ
    return out


def is_float_type(typ: str) -> bool:
    text = normalize_ws(str(typ or "")).replace("const ", "").replace("volatile ", "")
    if re.search(r"\b(?:std::)?(?:vector|string|map|unordered_map|set|unordered_set|deque|list)\b", text):
        return False
    return bool(re.search(r"\b(?:float|double|__fp16|float16_t|FLOAT16_t)\b", text))


def floating_symbol_names(feats: dict[str, Any]) -> set[str]:
    return {
        normalize_ws(str(decl.get("name") or ""))
        for decl in feats.get("var_decls") or []
        if normalize_ws(str(decl.get("name") or "")) and is_float_type(str(decl.get("type") or ""))
    }


def expr_uses_float_symbol(expr: str, float_names: set[str] | None = None) -> bool:
    names = float_names or set()
    return any(re.search(rf"\b{re.escape(name)}\b", str(expr or "")) for name in names if name)


def is_dynamic_container_type(typ: str) -> bool:
    return bool(re.search(r"\b(?:std::)?(?:vector|string|map|unordered_map|set|unordered_set|deque|list)\b", typ))


def is_string_like_type(typ: str) -> bool:
    return bool(re.search(r"\b(?:std::)?(?:string|string_view)\b", typ))


def is_default_constructed_string_type(typ: str) -> bool:
    text = normalize_ws(str(typ or "")).replace("const ", "").replace("volatile ", "")
    if re.search(r"\b(?:std::)?(?:vector|map|unordered_map|set|unordered_set|deque|list)\b", text):
        return False
    if "string_view" in text:
        return False
    return bool(
        re.search(r"\b(?:std::)?string\b", text)
        or re.search(r"\bbasic_string\s*<\s*char\b", text)
    )


def is_default_constructed_vector_type(typ: str) -> bool:
    text = normalize_ws(str(typ or "")).replace("const ", "").replace("volatile ", "")
    return bool(re.search(r"\b(?:std::)?vector\s*<", text))


def is_dynamic_container_scalar(name: str, feats: dict[str, Any]) -> bool:
    return is_dynamic_container_type(variable_type_map(feats).get(normalize_ws(name), ""))


def is_string_like_scalar(name: str, feats: dict[str, Any]) -> bool:
    return is_string_like_type(variable_type_map(feats).get(normalize_ws(name), ""))


def refs_use_string_like_scalar(refs: list[dict[str, str]], feats: dict[str, Any]) -> bool:
    return any(is_string_like_scalar(str(ref.get("base") or ""), feats) for ref in refs)


def branch_condition_uses_string_like_scalar(branch: dict[str, Any], feats: dict[str, Any]) -> bool:
    return refs_use_string_like_scalar(array_refs(branch_condition_expr(branch)), feats)


def branch_has_return(branch: dict[str, Any], returns: list[dict[str, Any]]) -> bool:
    return any(contains_offset(branch, ret.get("begin")) for ret in returns)


def loop_has_direct_predicate_return(
    loop: dict[str, Any],
    direct_branches: list[dict[str, Any]],
    feats: dict[str, Any],
) -> bool:
    returns = feats.get("returns") or []
    if not vector_loop_spec(loop):
        return False
    return any(branch_has_return(branch, returns) for branch in direct_branches)


def item_inside_direct_branch(item: dict[str, Any], direct_branches: list[dict[str, Any]]) -> bool:
    return any(contains_offset(branch, item.get("begin")) for branch in direct_branches)


def item_is_scalar_selection_update(item: dict[str, Any], feats: dict[str, Any]) -> bool:
    if not item_has_scalar_lhs(item):
        return False
    if item.get("source") == "var_decl" or is_loop_initializer_assignment(item):
        return False
    if str(item.get("op") or "") != "=":
        return False
    lhs = normalize_ws(str(item.get("lhs") or ""))
    rhs = normalize_ws(str(item.get("rhs") or ""))
    if not lhs or not rhs or lhs == rhs:
        return False
    if is_dynamic_container_scalar(lhs, feats):
        return False
    if re.search(r"\b(?:std::)?(?:vector|string|map|unordered_map|set|unordered_set)\b", rhs):
        return False
    # Either select an array/member value, or select a lane/index variable.
    return bool(read_refs_for_item(item) or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", rhs))


def item_is_small_aggregate_selection_update(item: dict[str, Any], feats: dict[str, Any]) -> bool:
    if not item_has_scalar_lhs(item):
        return False
    if item.get("source") == "var_decl" or is_loop_initializer_assignment(item):
        return False
    if str(item.get("op") or "") != "=":
        return False
    lhs = normalize_ws(str(item.get("lhs") or ""))
    rhs = normalize_ws(str(item.get("rhs") or ""))
    if not is_dynamic_container_scalar(lhs, feats):
        return False
    if not (rhs.startswith("{") and rhs.endswith("}")):
        return False
    return bool(read_refs_for_item(item) or re.search(r"\b[A-Za-z_][A-Za-z0-9_]*\b", rhs))


def item_is_arg_index_update(item: dict[str, Any]) -> bool:
    lhs = normalize_ws(str(item.get("lhs") or ""))
    rhs = normalize_ws(str(item.get("rhs") or ""))
    if rhs in {"true", "false"}:
        return False
    if re.search(r"(?:^|_)(idx|index|pos|arg|max_i|min_i)(?:_|$)", lhs, re.I):
        return True
    return False


def item_is_boolean_state_assignment(item: dict[str, Any]) -> bool:
    return item_has_scalar_lhs(item) and str(item.get("op") or "") == "=" and normalize_ws(str(item.get("rhs") or "")) in {
        "true",
        "false",
    }


def self_dependent_scalar_state_names(items: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for item in items:
        if not item_has_scalar_lhs(item):
            continue
        if item.get("source") == "var_decl" or is_loop_initializer_assignment(item):
            continue
        lhs = normalize_ws(str(item.get("lhs") or ""))
        rhs = normalize_ws(str(item.get("rhs") or ""))
        op = str(item.get("op") or "")
        if not lhs:
            continue
        if op != "=" or re.search(rf"\b{re.escape(lhs)}\b", rhs):
            names.add(lhs)
    return names


def declared_local_scalar_names(items: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for item in items:
        if item.get("source") != "var_decl" or is_loop_initializer_assignment(item):
            continue
        if item.get("writes"):
            continue
        lhs = normalize_ws(str(item.get("lhs") or ""))
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", lhs):
            names.add(lhs)
    return names


def branch_condition_mentions_names(branches: list[dict[str, Any]], names: set[str]) -> bool:
    if not names:
        return False
    for branch in branches:
        cond = branch_condition_expr(branch)
        if any(re.search(rf"\b{re.escape(name)}\b", cond) for name in names):
            return True
    return False


def branch_conditions_mention_name(branches: list[dict[str, Any]], name: str) -> bool:
    if not name:
        return False
    return branch_condition_mentions_names(branches, {name})


def scalar_names_assigned_in_loop(items: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for item in items:
        if not item_has_scalar_lhs(item):
            continue
        if item.get("source") == "var_decl" or is_loop_initializer_assignment(item):
            continue
        lhs = normalize_ws(str(item.get("lhs") or ""))
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", lhs):
            names.add(lhs)
    return names


def loop_has_scalar_state_control_dependency(
    loop: dict[str, Any],
    active_rw: list[dict[str, Any]],
    direct_branches: list[dict[str, Any]],
    feats: dict[str, Any],
) -> bool:
    """Detect scalar state that controls later behavior inside the same loop.

    A scalar assigned in one iteration and used by a branch condition is a
    loop-carried control state, not an ordinary lane-local value.  Rendering
    that as `any_lane(cond)` changes the ordered scalar semantics.
    """
    if not direct_branches:
        return False
    assigned = scalar_names_assigned_in_loop(active_rw)
    if not assigned:
        return False
    declared_here = declared_local_scalar_names(active_rw)
    assigned -= declared_here
    if not assigned:
        return False
    for branch in direct_branches:
        cond = branch_condition_expr(branch)
        for name in assigned:
            if re.search(rf"\b{re.escape(name)}\b", cond):
                return True
            if re.search(rf"(?:\+\+|--)\s*{re.escape(name)}\b|\b{re.escape(name)}\s*(?:\+\+|--)", cond):
                return True
    return False


def loop_has_carried_scalar_output_dependency(active_rw: list[dict[str, Any]]) -> bool:
    """Return true when a carried scalar state feeds an array write.

    This is not a lane-local map: each output element depends on the scalar
    state after processing earlier iterations.  Unless a dedicated prefix-scan
    transform recognizes it, keep the loop in scalar order.
    """
    states = self_dependent_scalar_state_names(active_rw) - declared_local_scalar_names(active_rw)
    if not states:
        return False
    for item in active_rw:
        if not item.get("writes"):
            continue
        rhs = normalize_ws(str(item.get("rhs") or ""))
        if any(re.search(rf"\b{re.escape(name)}\b", rhs) for name in states):
            return True
    return False


def branch_conditions_have_side_effect_update(branches: list[dict[str, Any]]) -> bool:
    for branch in branches:
        cond = branch_condition_expr(branch)
        if re.search(r"(?:\+\+|--)\s*[A-Za-z_][A-Za-z0-9_]*|[A-Za-z_][A-Za-z0-9_]*\s*(?:\+\+|--)", cond):
            return True
        if re.search(r"[A-Za-z_][A-Za-z0-9_]*\s*(?:\+=|-=|\*=|/=|%=|&=|\|=|\^=|(?<![!<>=])=(?!=))", cond):
            return True
    return False


def is_two_pointer_swap_loop(loop: dict[str, Any], active_rw: list[dict[str, Any]]) -> bool:
    header = loop_header_line(loop)
    if not re.match(r"while\s*\(\s*([A-Za-z_]\w*)\s*<\s*([A-Za-z_]\w*)\s*\)\s*$", header):
        return False
    writes = [item for item in active_rw if item.get("writes")]
    if len(writes) < 2:
        return False
    bases = [ref.get("base") for item in writes for ref in item.get("writes") or []]
    if not bases or len(set(bases)) != 1:
        return False
    scalar_updates = [item for item in active_rw if item_has_scalar_lhs(item) and str(item.get("op") or "") in {"+=", "-="}]
    return any(str(x.get("op")) == "+=" for x in scalar_updates) and any(str(x.get("op")) == "-=" for x in scalar_updates)


def previous_index_for_loop_start(start: str) -> str:
    start = normalize_ws(start)
    if re.fullmatch(r"[-+]?\d+", start):
        return str(int(start) - 1)
    return f"({start}) - 1"


def find_prior_array_seed(
    loop: dict[str, Any],
    feats: dict[str, Any],
    base: str,
    seed_index: str,
) -> dict[str, Any] | None:
    loop_begin = loop.get("begin")
    if not isinstance(loop_begin, int):
        return None
    seed_norm = normalize_index(seed_index)
    candidates: list[dict[str, Any]] = []
    for item in feats.get("read_write") or []:
        begin = item.get("begin")
        if not isinstance(begin, int) or begin >= loop_begin:
            continue
        if str(item.get("op") or "") != "=":
            continue
        writes = item.get("writes") or []
        if len(writes) != 1:
            continue
        w = writes[0]
        if str(w.get("base") or "") != base:
            continue
        if normalize_index(str(w.get("index") or "")) != seed_norm:
            continue
        candidates.append(item)
    if not candidates:
        return None
    return max(candidates, key=lambda x: x.get("begin") if isinstance(x.get("begin"), int) else -1)


def parse_affine_neighbor_recurrence(
    loop: dict[str, Any],
    active_rw: list[dict[str, Any]],
    feats: dict[str, Any],
) -> dict[str, str] | None:
    spec = vector_loop_spec(loop)
    if not spec:
        return None
    var = spec["var"]
    for item in active_rw:
        writes = item.get("writes") or []
        if len(writes) != 1 or str(item.get("op") or "") != "=":
            continue
        w = writes[0]
        base = str(w.get("base") or "")
        idx = normalize_ws(str(w.get("index") or ""))
        rhs = normalize_ws(str(item.get("rhs") or ""))
        if idx != var:
            continue
        m = re.search(
            rf"\b{re.escape(base)}\s*\[\s*{re.escape(var)}\s*-\s*1\s*\]\s*([+-])\s*([-+]?\d+|[A-Za-z_][A-Za-z0-9_]*)",
            rhs,
        )
        if not m:
            continue
        sign, step = m.group(1), normalize_ws(m.group(2))
        delta = step if sign == "+" else f"-{step}"
        seed_index = previous_index_for_loop_start(spec.get("start", "1"))
        seed = find_prior_array_seed(loop, feats, base, seed_index)
        if not seed:
            continue
        seed_expr = normalize_ws(str(seed.get("rhs") or ""))
        if not seed_expr:
            continue
        return {
            "base": base,
            "var": var,
            "delta": delta,
            "seed_index": seed_index,
            "seed_expr": seed_expr,
            "rule_rhs": rhs,
        }
    return None


def loop_has_closed_form_neighbor_recurrence(loop: dict[str, Any], active_rw: list[dict[str, Any]], feats: dict[str, Any]) -> bool:
    return parse_affine_neighbor_recurrence(loop, active_rw, feats) is not None


def array_index_is_indirect(index: str) -> bool:
    return bool(re.search(r"\b[A-Za-z_][A-Za-z0-9_]*\s*\[", str(index or "")))


def item_has_indirect_write(item: dict[str, Any]) -> bool:
    return any(array_index_is_indirect(str(ref.get("index") or "")) for ref in item.get("writes") or [])


def item_has_array_write(item: dict[str, Any]) -> bool:
    return bool(item.get("writes"))


def item_write_indices_mention_loop_var(item: dict[str, Any], loop: dict[str, Any]) -> bool:
    spec = vector_loop_spec(loop)
    var = str((spec or {}).get("var") or "")
    if not var:
        return True
    writes = item.get("writes") or []
    if not writes:
        return True
    return any(re.search(rf"\b{re.escape(var)}\b", str(ref.get("index") or "")) for ref in writes)


def invariant_array_reduction_update_items(loop: dict[str, Any], active_rw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    spec = vector_loop_spec(loop)
    var = str((spec or {}).get("var") or "")
    if not var:
        return []
    out: list[dict[str, Any]] = []
    for item in active_rw:
        if not item_has_array_write(item):
            continue
        if str(item.get("op") or "") == "=":
            continue
        writes = item.get("writes") or []
        if not writes:
            continue
        if any(re.search(rf"\b{re.escape(var)}\b", str(ref.get("index") or "")) for ref in writes):
            continue
        out.append(item)
    return out


def loop_has_invariant_array_reduction_update(loop: dict[str, Any], active_rw: list[dict[str, Any]]) -> bool:
    return bool(invariant_array_reduction_update_items(loop, active_rw))


def loop_branch_has_direct_control(loop: dict[str, Any], branches: list[dict[str, Any]], feats: dict[str, Any]) -> bool:
    controls = feats.get("controls") or []
    for ctrl in controls:
        if str(ctrl.get("kind") or "") not in {"BreakStmt", "ContinueStmt"}:
            continue
        if not contains_offset(loop, ctrl.get("begin")):
            continue
        if any(contains_offset(branch, ctrl.get("begin")) for branch in branches):
            return True
    return False


def loop_has_side_effect_call(loop: dict[str, Any], feats: dict[str, Any]) -> bool:
    return any(is_side_effect_call(call) and contains_offset(loop, call.get("begin")) for call in feats.get("calls") or [])


def postinc_write_ref(item: dict[str, Any]) -> dict[str, Any] | None:
    writes = item.get("writes") or []
    if len(writes) != 1:
        return None
    idx = normalize_ws(str(writes[0].get("index") or ""))
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*\s*(?:\+\+|--)", idx):
        return writes[0]
    return None


def postinc_index_name(index: str) -> str:
    m = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:\+\+|--)\s*$", str(index or ""))
    return m.group(1) if m else "compact_count"


def postinc_write_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in items if postinc_write_ref(item)]


def postinc_pointer_store_name(lhs: str) -> str:
    m = re.fullmatch(r"\*\s*(?:\(\s*)?([A-Za-z_][A-Za-z0-9_]*)\s*(?:\+\+|--)\s*(?:\))?", normalize_ws(lhs))
    return m.group(1) if m else ""


def postinc_pointer_store_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in items if postinc_pointer_store_name(str(item.get("lhs") or ""))]


def pointer_memory_names(items: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for item in items:
        if is_loop_initializer_assignment(item):
            continue
        lhs = normalize_ws(str(item.get("lhs") or ""))
        ptr = postinc_pointer_store_name(lhs)
        if ptr:
            names.add(ptr)
        for ref in (item.get("reads") or []) + (item.get("writes") or []):
            base = str(ref.get("base") or "")
            if base:
                names.add(base)
    return names


def signed_stride_expr(op: str, rhs: str) -> str:
    rhs_norm = sanitize_vector_scalar_ops(normalize_ws(rhs))
    if op == "-=":
        if re.fullmatch(r"\d+(?:[uUlL]*)?", rhs_norm):
            return "-" + re.sub(r"[uUlL]+$", "", rhs_norm)
        return f"-({rhs_norm})"
    return rhs_norm


def combine_stride_exprs(strides: list[str]) -> str:
    cleaned = [normalize_ws(x) for x in strides if normalize_ws(x)]
    if not cleaned:
        return ""
    ints: list[int] = []
    all_int = True
    for stride in cleaned:
        if re.fullmatch(r"[-+]?\d+", stride):
            ints.append(int(stride))
        else:
            all_int = False
            break
    if all_int:
        return str(sum(ints))
    if len(cleaned) == 1:
        return cleaned[0]
    return " + ".join(f"({x})" if re.search(r"\s|[+\-*/]", x) else x for x in cleaned)


def loop_body_pointer_recurrences(direct_rw: list[dict[str, Any]]) -> dict[str, str]:
    memory_names = pointer_memory_names(direct_rw)
    updates: dict[str, list[str]] = {}
    for item in direct_rw:
        if is_loop_initializer_assignment(item):
            continue
        lhs = normalize_ws(str(item.get("lhs") or ""))
        if lhs not in memory_names:
            continue
        op = str(item.get("op") or "")
        if op not in {"+=", "-="}:
            continue
        rhs = normalize_ws(str(item.get("rhs") or ""))
        if not rhs:
            continue
        updates.setdefault(lhs, []).append(signed_stride_expr(op, rhs))
    return {name: stride for name, vals in updates.items() if (stride := combine_stride_exprs(vals))}


def stream_loop_iteration_count_expr(spec: dict[str, str]) -> str:
    """Return a forward ordinal trip count for a simple unit-stride loop."""
    start = sanitize_vector_scalar_ops(str(spec.get("start") or "0"))
    end = sanitize_vector_scalar_ops(str(spec.get("end") or ""))
    cmp_op = str(spec.get("cmp") or "<")
    step = sanitize_vector_scalar_ops(str(spec.get("step") or "1"))
    if not end:
        return ""
    if step in {"1", "+1"} and cmp_op in {"<", "<="}:
        if start in {"0", "+0"}:
            count = end
        else:
            count = f"({end}) - ({start})"
        if cmp_op == "<=":
            count = f"({count}) + 1"
        return sanitize_vector_scalar_ops(count)
    if step == "-1" and cmp_op in {">", ">="}:
        if end in {"0", "+0"}:
            count = start
        else:
            count = f"({start}) - ({end})"
        if cmp_op == ">=":
            count = f"({count}) + 1"
        return sanitize_vector_scalar_ops(count)
    return ""


def stream_index_expr(ordinal: str, stride: int, offset: int = 0) -> str:
    """Render pointer-stream index = ordinal * stride + offset."""
    ordinal = normalize_ws(ordinal)
    if stride == 0:
        base = "0"
    elif stride == 1:
        base = ordinal
    elif stride == -1:
        base = f"-({ordinal})"
    else:
        base = f"{stride} * ({ordinal})"
    if offset == 0:
        return sanitize_vector_scalar_ops(base)
    if base in {"0", "+0"}:
        return str(offset)
    if offset > 0:
        return sanitize_vector_scalar_ops(f"{base} + {offset}")
    return sanitize_vector_scalar_ops(f"{base} - {abs(offset)}")


def parse_integer_offset_expr(index: str) -> int | None:
    index = normalize_ws(str(index or "0"))
    if index in {"", "+0", "0"}:
        return 0
    if re.fullmatch(r"[-+]?\d+", index):
        return int(index)
    return None


def stream_index_with_subscript_offset(ordinal: str, stride: int, current_offset: int, subscript_index: str) -> str:
    parsed = parse_integer_offset_expr(subscript_index)
    if parsed is not None:
        return stream_index_expr(ordinal, stride, current_offset + parsed)
    base = stream_index_expr(ordinal, stride, current_offset)
    idx = sanitize_vector_scalar_ops(normalize_ws(subscript_index))
    if base in {"0", "+0"}:
        return idx
    return sanitize_vector_scalar_ops(f"{base} + ({idx})")


def pointer_stream_postinc_strides(direct_rw: list[dict[str, Any]]) -> dict[str, int]:
    """Count fixed per-iteration pointer motion implied by post-increment uses."""
    counts: Counter[str] = Counter()
    for item in direct_rw:
        if is_loop_initializer_assignment(item):
            continue
        for name, count in postinc_pointer_read_counts(str(item.get("rhs") or "")).items():
            counts[name] += count
        ptr = postinc_pointer_store_name(str(item.get("lhs") or ""))
        if ptr:
            counts[ptr] += 1
    return {name: int(count) for name, count in counts.items() if count}


def rewrite_stream_pointer_expr(
    text: str,
    *,
    pointer_strides: dict[str, int],
    pointer_offsets: dict[str, int],
    ordinal: str,
) -> str:
    """Rewrite post-increment pointer streams as lane-indexed subscripts.

    The scan is textual but driven by AST-extracted pointer names and preserves
    expression order, so `*p++` updates the offset seen by later `p[k]` in the
    same statement.
    """
    s = str(text or "")
    out: list[str] = []
    i = 0
    names = sorted(pointer_strides, key=len, reverse=True)
    while i < len(s):
        matched = False
        for name in names:
            m_post = re.match(rf"\*\s*{re.escape(name)}\s*(\+\+|--)", s[i:])
            if m_post:
                idx = stream_index_expr(ordinal, pointer_strides[name], pointer_offsets.get(name, 0))
                out.append(f"{name}[{idx}]")
                pointer_offsets[name] = pointer_offsets.get(name, 0) + (1 if m_post.group(1) == "++" else -1)
                i += m_post.end()
                matched = True
                break
            m_sub = re.match(rf"\b{re.escape(name)}\s*\[", s[i:])
            if m_sub:
                lbr = i + m_sub.end() - 1
                close = _scan_matching_right(s, lbr, "[", "]")
                if close > lbr:
                    idx = stream_index_with_subscript_offset(
                        ordinal,
                        pointer_strides[name],
                        pointer_offsets.get(name, 0),
                        s[lbr + 1 : close],
                    )
                    out.append(f"{name}[{idx}]")
                    i = close + 1
                    matched = True
                    break
        if not matched:
            out.append(s[i])
            i += 1
    return "".join(out)


def item_is_stream_pointer_update(item: dict[str, Any], pointer_strides: dict[str, int]) -> bool:
    lhs = normalize_ws(str(item.get("lhs") or ""))
    return bool(lhs in pointer_strides and str(item.get("op") or "") in {"+=", "-="})


def item_text_mentions_name(item: dict[str, Any], name: str) -> bool:
    if not name:
        return False
    parts = [str(item.get("lhs") or ""), str(item.get("rhs") or ""), str(item.get("snippet") or "")]
    for ref in (item.get("reads") or []) + (item.get("writes") or []):
        parts.append(str(ref.get("base") or ""))
        parts.append(str(ref.get("index") or ""))
    text = " ".join(parts)
    return bool(re.search(rf"\b{re.escape(name)}\b", text))


def loop_carried_scalar_reads_before_write(active_items: list[dict[str, Any]]) -> set[str]:
    """Find scalar temporaries used before local definition inside a loop body.

    In a pointer-stream vector kernel, scalar values must be lane-local
    temporaries produced before use in the same iteration.  If a scalar is read
    before its first write and later updated inside the loop, it represents
    carried state from a previous iteration and needs a separate recurrence or
    prefix/stencil transform.
    """
    assigned: set[str] = set()
    first_read_before_write: set[str] = set()
    assigned_names: set[str] = {
        normalize_ws(str(item.get("lhs") or ""))
        for item in active_items
        if item_has_scalar_lhs(item) and not is_loop_initializer_assignment(item)
    }
    assigned_names = {name for name in assigned_names if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name)}
    if not assigned_names:
        return set()
    for item in sorted_by_begin(active_items):
        rhs = str(item.get("rhs") or "")
        for name in assigned_names:
            if name not in assigned and re.search(rf"\b{re.escape(name)}\b", rhs):
                first_read_before_write.add(name)
        lhs = normalize_ws(str(item.get("lhs") or ""))
        if lhs in assigned_names:
            assigned.add(lhs)
    return first_read_before_write


def item_is_pointer_recurrence_update(item: dict[str, Any], pointer_recurrences: dict[str, str]) -> bool:
    lhs = normalize_ws(str(item.get("lhs") or ""))
    return bool(lhs and lhs in pointer_recurrences and str(item.get("op") or "") in {"+=", "-="})


def loop_scalar_toggle_exprs(direct_rw: list[dict[str, Any]], spec: dict[str, str] | None) -> dict[str, str]:
    if not spec:
        return {}
    out: dict[str, str] = {}
    for item in direct_rw:
        lhs = normalize_ws(str(item.get("lhs") or ""))
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", lhs):
            continue
        if str(item.get("op") or "") != "^=":
            continue
        rhs = sanitize_vector_scalar_ops(normalize_ws(str(item.get("rhs") or "")))
        if not rhs:
            continue
        ordinal = lane_ordinal_expr(spec)
        out[lhs] = f"select(({ordinal} & 1) != 0, {rhs}, 0)"
    return out


def item_is_scalar_toggle_update(item: dict[str, Any], scalar_exprs: dict[str, str]) -> bool:
    lhs = normalize_ws(str(item.get("lhs") or ""))
    return bool(lhs and lhs in scalar_exprs and str(item.get("op") or "") == "^=")


def streaming_store_slot_index(spec: dict[str, str], slots: int, slot: int) -> str:
    ordinal = lane_ordinal_expr(spec)
    if slots <= 1:
        return ordinal
    if slot == 0:
        return f"{slots} * ({ordinal})"
    return f"{slots} * ({ordinal}) + {slot}"


def streaming_pointer_store_slots(
    direct_rw: list[dict[str, Any]],
    spec: dict[str, str] | None,
) -> dict[str, tuple[str, str]]:
    if not spec:
        return {}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in sorted_by_begin(direct_rw):
        ptr = postinc_pointer_store_name(str(item.get("lhs") or ""))
        if ptr:
            grouped.setdefault(ptr, []).append(item)
    out: dict[str, tuple[str, str]] = {}
    for ptr, items in grouped.items():
        slots = len(items)
        for slot, item in enumerate(items):
            out[item_identity(item)] = (ptr, streaming_store_slot_index(spec, slots, slot))
    return out


def item_has_postinc_pointer_read(item: dict[str, Any]) -> bool:
    return bool(postinc_pointer_read_names(str(item.get("rhs") or "")))


def loop_has_sequential_pointer_stream(items: list[dict[str, Any]]) -> bool:
    active = [item for item in items if not is_loop_initializer_assignment(item)]
    return any(item_has_postinc_pointer_read(item) for item in active) and bool(postinc_pointer_store_items(active))


def postinc_pointer_read_update_items(items: list[dict[str, Any]]) -> set[str]:
    out: set[str] = set()
    for item in items:
        if is_loop_initializer_assignment(item):
            continue
        for name in postinc_pointer_read_names(str(item.get("rhs") or "")):
            out.add(name)
    return out


def item_is_separate_postinc_read_update(item: dict[str, Any], read_update_ptrs: set[str]) -> bool:
    lhs = normalize_ws(str(item.get("lhs") or ""))
    return bool(lhs and lhs in read_update_ptrs and str(item.get("op") or "") in {"+=", "-="})


SCALAR_ACCUMULATOR_RE = re.compile(
    r"^(?:sum|acc|res|result|total|count|cnt|num|ans|answer|dot|norm|mean|var|minv|maxv|min|max)(?:_|$|\d)",
    re.I,
)


def is_scalar_accumulator_name(name: str) -> bool:
    text = str(name or "")
    return bool(
        SCALAR_ACCUMULATOR_RE.search(text)
        or re.search(r"(sum|count|cnt|num|res|total|acc|dot|norm|mean|var|minv|maxv)", text, re.I)
    )


def item_has_scalar_reduction(item: dict[str, Any]) -> bool:
    lhs = normalize_ws(str(item.get("lhs") or ""))
    rhs = normalize_ws(str(item.get("rhs") or ""))
    if item.get("writes"):
        return False
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", lhs):
        return False
    if str(item.get("op") or "") != "=":
        return True
    if str(item.get("op") or "") == "*=":
        return True
    if not is_scalar_accumulator_name(lhs):
        return False
    return bool(lhs and re.search(rf"\b{re.escape(lhs)}\b", rhs))


def item_is_prefix_scan_update(item: dict[str, Any]) -> bool:
    if not item_has_scalar_reduction(item):
        return False
    if not read_refs_for_item(item):
        return False
    lhs = normalize_ws(str(item.get("lhs") or ""))
    rhs = normalize_ws(str(item.get("rhs") or ""))
    op = str(item.get("op") or "")
    if not lhs:
        return False
    if op in {"+=", "-=", "*=", "&=", "|=", "^="}:
        return True
    if op == "=" and re.search(rf"\b{re.escape(lhs)}\b", rhs):
        return True
    return False


def item_has_scalar_lhs(item: dict[str, Any]) -> bool:
    lhs = normalize_ws(str(item.get("lhs") or ""))
    return (not item.get("writes")) and bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", lhs))


def item_is_existing_scalar_state_update(item: dict[str, Any]) -> bool:
    if not item_has_scalar_lhs(item):
        return False
    if item.get("source") == "var_decl" or is_loop_initializer_assignment(item):
        return False
    if is_simple_scalar_var_init(item):
        return False
    lhs = normalize_ws(str(item.get("lhs") or ""))
    rhs = normalize_ws(str(item.get("rhs") or ""))
    op = str(item.get("op") or "")
    if op != "=":
        return True
    if lhs and re.search(rf"\b{re.escape(lhs)}\b", rhs):
        return True
    # Assigning a scalar state to a constant under a vector predicate is a
    # state reduction (e.g. any-zero flag), not a per-lane temporary compute.
    return rhs in {"0", "1", "true", "false", "-1"}


def loop_dependency_refs(loop: dict[str, Any], deps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dep for dep in deps if contains_offset(loop, dep.get("begin")) or contains_offset(loop, dep.get("snippet_begin"))]


def dependency_refs_for_loop(loop: dict[str, Any], feats: dict[str, Any]) -> list[dict[str, Any]]:
    deps: list[dict[str, Any]] = []
    for dep in feats.get("same_array_dependencies") or []:
        snip = str(dep.get("snippet") or "")
        found = False
        for item in feats.get("read_write") or []:
            if item.get("snippet") == snip and contains_offset(loop, item.get("begin")):
                found = True
                break
        if found:
            deps.append(dep)
    return deps


def loop_has_prefix_scan_predicate(
    loop: dict[str, Any],
    active_rw: list[dict[str, Any]],
    direct_branches: list[dict[str, Any]],
    feats: dict[str, Any],
) -> bool:
    if parse_conditional_prefix_scan_predicate(loop, active_rw, direct_branches, feats):
        return True
    if not direct_branches or not vector_loop_spec(loop):
        return False
    if loop_has_irregular_cpp_logic(loop):
        return False
    if not loop_has_direct_predicate_return(loop, direct_branches, feats):
        return False
    for item in active_rw:
        if not item_is_prefix_scan_update(item):
            continue
        lhs = normalize_ws(str(item.get("lhs") or ""))
        if branch_conditions_mention_name(direct_branches, lhs):
            return True
    return False


def direct_child_loops_for(loop: dict[str, Any], feats: dict[str, Any]) -> list[dict[str, Any]]:
    loops = feats.get("loops") or []
    out: list[dict[str, Any]] = []
    parent_key = loop_key(loop)
    for cand in loops:
        if cand is loop:
            continue
        if not contains_offset(loop, cand.get("begin")):
            continue
        parent = immediate_loop_parent(cand, loops)
        if parent is not None and loop_key(parent) == parent_key:
            out.append(cand)
    return sorted_by_begin(out)


def item_has_array_ref_index_mentioning_var(item: dict[str, Any], key: str, var: str) -> bool:
    if not var:
        return False
    return any(
        re.search(rf"\b{re.escape(var)}\b", str(ref.get("index") or ""))
        for ref in item.get(key) or []
    )


def loop_is_per_element_scalar_region(
    loop: dict[str, Any],
    subtree_rw: list[dict[str, Any]],
    feats: dict[str, Any],
) -> bool:
    """Detect an outer element map whose body is lane-local scalar control.

    This is deliberately structural: the outer induction must directly feed
    array input/output indices, while nested loops and scalar state are kept as
    the per-element body instead of being reclassified into separate vector
    dimensions.
    """
    spec = vector_loop_spec(loop)
    if not spec:
        return False
    var = str(spec.get("var") or "")
    if not var:
        return False
    child_loops = direct_child_loops_for(loop, feats)
    if not child_loops:
        return False
    if loop_has_irregular_cpp_logic(loop) or loop_has_side_effect_call(loop, feats):
        return False
    if loop_has_sequential_pointer_stream(subtree_rw):
        return False
    if dependency_refs_for_loop(loop, feats):
        return False
    controls = feats.get("controls") or []
    if any(
        str(ctrl.get("kind") or "") in {"BreakStmt", "ContinueStmt"}
        and contains_offset(loop, ctrl.get("begin"))
        for ctrl in controls
    ):
        return False
    if any(contains_offset(loop, ret.get("begin")) for ret in feats.get("returns") or []):
        return False

    direct_rw = [
        item
        for item in subtree_rw
        if item_directly_in_loop(item, loop, child_loops) and not is_loop_initializer_assignment(item)
    ]
    direct_reads_var = any(item_has_array_ref_index_mentioning_var(item, "reads", var) for item in direct_rw)
    direct_writes = [item for item in direct_rw if item.get("writes")]
    direct_writes_var = any(item_has_array_ref_index_mentioning_var(item, "writes", var) for item in direct_writes)
    if not direct_reads_var or not direct_writes_var:
        return False
    if any(str(item.get("op") or "") != "=" for item in direct_writes):
        return False
    if any(item_has_indirect_write(item) for item in direct_writes):
        return False

    array_writes = [item for item in subtree_rw if item_has_array_write(item)]
    if any(str(item.get("op") or "") != "=" for item in array_writes):
        return False
    if any(not item_write_indices_mention_loop_var(item, loop) for item in array_writes):
        return False

    scalar_states = self_dependent_scalar_state_names(subtree_rw)
    declared_scalars = declared_local_scalar_names(subtree_rw)
    if not scalar_states or not scalar_states.issubset(declared_scalars):
        return False
    if not any(contains_offset(loop, branch.get("begin")) for branch in feats.get("branches") or []):
        return False
    return True


def loop_is_outer_lane_local_scalar_region(
    loop: dict[str, Any],
    subtree_rw: list[dict[str, Any]],
    feats: dict[str, Any],
) -> bool:
    spec = vector_loop_spec(loop)
    if not spec:
        return False
    var = str(spec.get("var") or "")
    if not var:
        return False
    child_loops = direct_child_loops_for(loop, feats)
    if not child_loops:
        return False
    if loop_has_irregular_cpp_logic(loop) or loop_has_side_effect_call(loop, feats):
        return False
    if loop_has_sequential_pointer_stream(subtree_rw) or dependency_refs_for_loop(loop, feats):
        return False
    direct_rw = [
        item
        for item in subtree_rw
        if item_directly_in_loop(item, loop, child_loops) and not is_loop_initializer_assignment(item)
    ]
    direct_writes = [item for item in direct_rw if item.get("writes")]
    if not any(item_has_array_ref_index_mentioning_var(item, "writes", var) for item in direct_writes):
        return False
    if any(str(item.get("op") or "") != "=" or item_has_indirect_write(item) for item in direct_writes):
        return False
    if not any(re.search(rf"\b{re.escape(var)}\b", loop_header_line(child)) for child in child_loops):
        return False
    scalar_states = self_dependent_scalar_state_names(subtree_rw)
    if not scalar_states:
        return False
    return bool(declared_local_scalar_names(subtree_rw) & scalar_states)


def signed_delta_expr(op: str, rhs: str) -> str:
    rhs_norm = sanitize_vector_scalar_ops(normalize_ws(rhs))
    if op == "+=":
        return rhs_norm
    if op == "-=":
        if re.fullmatch(r"\d+(?:[uUlL]*)?", rhs_norm):
            return "-" + re.sub(r"[uUlL]+$", "", rhs_norm)
        return f"-({rhs_norm})"
    return rhs_norm


def select_zero_expr(condition: str, value: str) -> str:
    cond = sanitize_vector_scalar_ops(normalize_ws(condition))
    val = sanitize_vector_scalar_ops(normalize_ws(value))
    return f"select({cond}, {val}, 0)" if cond else val


def combine_delta_terms(terms: list[tuple[str, str]]) -> str:
    rendered = [select_zero_expr(cond, value) for cond, value in terms if value]
    if not rendered:
        return ""
    return " + ".join(rendered)


def replace_name_in_expr(expr: str, old: str, new: str) -> str:
    if not old:
        return expr
    return re.sub(rf"\b{re.escape(old)}\b", new, expr)


def parse_conditional_prefix_scan_predicate(
    loop: dict[str, Any],
    active_rw: list[dict[str, Any]],
    direct_branches: list[dict[str, Any]],
    feats: dict[str, Any],
) -> dict[str, Any] | None:
    spec = vector_loop_spec(loop)
    if not spec or not direct_branches:
        return None
    if any(item_has_array_write(item) for item in active_rw):
        return None
    returns = feats.get("returns") or []
    state_names = self_dependent_scalar_state_names(active_rw) - declared_local_scalar_names(active_rw)
    for state in sorted(state_names):
        threshold_branches = [
            branch
            for branch in direct_branches
            if branch_has_return(branch, returns)
            and re.search(rf"\b{re.escape(state)}\b", branch_condition_expr(branch))
        ]
        if not threshold_branches:
            continue
        update_items = [
            item
            for item in active_rw
            if item_has_scalar_lhs(item)
            and normalize_ws(str(item.get("lhs") or "")) == state
            and str(item.get("op") or "") in {"+=", "-="}
        ]
        if not update_items:
            continue
        terms: list[tuple[str, str]] = []
        for item in update_items:
            conds = [
                cond
                for cond in item_branch_conditions(item, direct_branches)
                if not re.search(rf"\b{re.escape(state)}\b", cond)
            ]
            cond = conds[-1] if conds else ""
            terms.append((cond, signed_delta_expr(str(item.get("op") or ""), str(item.get("rhs") or ""))))
        delta_expr = combine_delta_terms(terms)
        if not delta_expr:
            continue
        threshold = threshold_branches[0]
        ret = next((r for r in returns if contains_offset(threshold, r.get("begin"))), None)
        return {
            "state": state,
            "delta": delta_expr,
            "threshold_condition": sanitize_vector_scalar_ops(branch_condition_expr(threshold)),
            "return_expr": render_return_expr(ret.get("snippet") if ret else "return false"),
        }
    return None


def classify_RSB_region(loop: dict[str, Any], subtree_rw: list[dict[str, Any]], direct_branches: list[dict[str, Any]], feats: dict[str, Any]) -> str:
    header = loop_header_line(loop).lower()
    active_rw = [item for item in subtree_rw if not is_loop_initializer_assignment(item)]
    loop_controls = [c for c in feats.get("controls") or [] if contains_offset(loop, c.get("begin"))]
    has_child_loop = any(x is not loop and contains_offset(loop, x.get("begin")) for x in feats.get("loops") or [])
    has_side_effect_call = loop_has_side_effect_call(loop, feats)
    if header.startswith("while") and is_two_pointer_swap_loop(loop, active_rw):
        return "two_pointer_swap"
    if header.startswith("while") or str(loop.get("kind") or "") in {"WhileStmt", "DoStmt"}:
        return "pseudocode_control"
    if (
        has_child_loop
        and direct_branches
        and loop_is_outer_lane_local_scalar_region(loop, active_rw, feats)
    ):
        return "hybrid_scalar_region_per_element"
    if direct_branches and loop_branch_has_direct_control(loop, direct_branches, feats):
        return "pseudocode_loop"
    loop_carried_scalar_names = self_dependent_scalar_state_names(active_rw) - declared_local_scalar_names(active_rw)
    if direct_branches and branch_conditions_have_side_effect_update(direct_branches):
        return "pseudocode_loop"
    if loop_has_carried_scalar_output_dependency(active_rw):
        if loop_has_prefix_scan_predicate(loop, active_rw, direct_branches, feats):
            return "prefix_scan_predicate"
        return "pseudocode_loop"
    if (
        has_child_loop
        and vector_loop_spec(loop) is not None
        and loop_has_lane_local_scalar_child(loop, feats)
        and not loop_has_irregular_cpp_logic(loop)
        and not has_side_effect_call
        and not any(item_has_array_write(item) for item in active_rw)
        and any(item_has_scalar_reduction(item) for item in active_rw)
    ):
        return "hybrid_elementwise_reduction"
    if direct_branches and loop_has_scalar_state_control_dependency(loop, active_rw, direct_branches, feats):
        if loop_has_prefix_scan_predicate(loop, active_rw, direct_branches, feats):
            return "prefix_scan_predicate"
        return "pseudocode_loop"
    if direct_branches and branch_condition_mentions_names(direct_branches, loop_carried_scalar_names):
        if loop_has_prefix_scan_predicate(loop, active_rw, direct_branches, feats):
            return "prefix_scan_predicate"
        return "pseudocode_loop"
    if (
        direct_branches
        and any(item_is_existing_scalar_state_update(item) for item in active_rw)
        and any(item_has_scalar_lhs(item) and str(item.get("op") or "") != "=" for item in active_rw)
    ):
        return "pseudocode_loop"
    if (
        has_child_loop
        and any(contains_offset(loop, branch.get("begin")) for branch in feats.get("branches") or [])
        and any(item_is_existing_scalar_state_update(item) for item in active_rw)
    ):
        if loop_is_per_element_scalar_region(loop, active_rw, feats):
            return "hybrid_scalar_region_per_element"
        return "pseudocode_loop"
    if (
        loop_controls
        and direct_branches
        and vector_loop_spec(loop) is not None
        and any(item_is_existing_scalar_state_update(item) for item in active_rw)
    ):
        return "bounded_predicate_scan"
    if (
        direct_branches
        and not has_child_loop
        and loop_has_direct_predicate_return(loop, direct_branches, feats)
        and not loop_has_irregular_cpp_logic(loop)
        and not any(branch_condition_uses_string_like_scalar(branch, feats) for branch in direct_branches)
    ):
        return "predicate_return_scan"
    if (
        direct_branches
        and not has_child_loop
        and vector_loop_spec(loop) is not None
        and not loop_has_irregular_cpp_logic(loop)
        and not any(branch_condition_uses_string_like_scalar(branch, feats) for branch in direct_branches)
    ):
        selection_items = [
            item
            for item in active_rw
            if (item_is_scalar_selection_update(item, feats) or item_is_small_aggregate_selection_update(item, feats))
            and item_inside_direct_branch(item, direct_branches)
        ]
        if selection_items:
            if any(
                item_is_arg_index_update(item)
                or item_is_boolean_state_assignment(item)
                or item_is_small_aggregate_selection_update(item, feats)
                for item in selection_items
            ):
                return "arg_reduction"
            return "pseudocode_loop"
    if direct_branches and any(contains_offset(loop, r.get("begin")) for r in feats.get("returns") or []):
        return "pseudocode_loop"
    deps = dependency_refs_for_loop(loop, feats)
    if deps:
        if loop_has_closed_form_neighbor_recurrence(loop, active_rw, feats):
            return "affine_recurrence_closed_form"
        if any(re.search(r"\+1|-1", normalize_index(str(d.get("write_index") or "") + str(d.get("read_index") or ""))) for d in deps):
            return "prefix_or_neighbor_dependency"
        return "dependency_or_inplace_update"
    if (
        has_child_loop
        and vector_loop_spec(loop) is not None
        and not loop_has_irregular_cpp_logic(loop)
        and not has_side_effect_call
        and not any(item_has_array_write(item) for item in active_rw)
        and any(item_has_scalar_reduction(item) for item in active_rw)
    ):
        return "hybrid_elementwise_reduction"
    if (
        has_child_loop
        and vector_loop_spec(loop) is not None
        and not loop_has_irregular_cpp_logic(loop)
        and not has_side_effect_call
        and any(item_has_array_write(item) for item in active_rw)
        and declared_local_scalar_names(active_rw)
    ):
        return "hybrid_elementwise_store"
    if any(item_has_indirect_write(item) and str(item.get("op") or "") != "=" for item in active_rw):
        return "conflict_update"
    array_updates = [item for item in active_rw if item_has_array_write(item) and str(item.get("op") or "") != "="]
    if array_updates:
        if any(not item_write_indices_mention_loop_var(item, loop) for item in array_updates):
            return "nested_reduction_update"
        if len([x for x in feats.get("loops") or [] if contains_offset(loop, x.get("begin"))]) >= 2:
            return "nested_reduction_update"
        return "array_update"
    if (
        not has_child_loop
        and any(item_has_scalar_reduction(item) for item in active_rw)
        and any(item_has_array_write(item) for item in active_rw)
    ):
        return "prefix_or_rolling_output"
    if any(item_has_scalar_reduction(item) for item in active_rw):
        return "scalar_reduction"
    if direct_branches and any(item_has_array_write(item) for item in active_rw):
        return "predicate_store_map"
    if any(item_has_array_write(item) for item in active_rw):
        if any(item_has_indirect_write(item) or array_index_is_indirect(str(ref.get("index") or "")) for item in active_rw for ref in (item.get("reads") or [])):
            return "indexed_gather_scatter_map"
        return "store_map"
    return "pseudocode_loop"


VECTOR_EXECUTION_REGIONS = {
    "store_map",
    "predicate_store_map",
    "indexed_gather_scatter_map",
    "array_update",
    "nested_reduction_update",
    "scalar_reduction",
    "prefix_or_rolling_output",
    "prefix_scan_predicate",
    "bounded_predicate_scan",
    "predicate_return_scan",
    "arg_reduction",
    "affine_recurrence_closed_form",
    "two_pointer_swap",
    "hybrid_elementwise_reduction",
    "hybrid_elementwise_store",
    "hybrid_scalar_region_per_element",
}


def loop_has_irregular_cpp_logic(loop: dict[str, Any]) -> bool:
    text = str(loop.get("snippet") or "")
    lower = text.lower()
    return bool(
        re.search(
            r"\bstd::(?:to_string|string)\b|\.rbegin\s*\(|\.rend\s*\(|\.length\s*\(|\.substr\s*\(|"
            r"\.push_back\s*\(|\.erase\s*\(|\.insert\s*\(|\bfind\s*\([^;\n]*\.begin\s*\(|"
            r"\bmem(?:cpy|move)\s*\(",
            lower,
        )
    )


def loop_is_scheduler_loop(
    loop: dict[str, Any],
    child_loops: list[dict[str, Any]],
    subtree_rw: list[dict[str, Any]],
    feats: dict[str, Any],
) -> bool:
    """Return true when a loop defines a subdomain rather than data lanes.

    RSB should only emit vector_range for loops whose induction variable
    directly enumerates independent data elements.  Tiling, blocking, window,
    and other scheduling loops usually define bounds for inner loops; rendering
    those loop variables as lanes makes the bootstrap imply cross-region gather
    or scatter work that is not actually the vectorization target.
    """
    if not child_loops:
        return False
    spec = vector_loop_spec(loop)
    if not spec:
        return False
    var = str(spec.get("var") or "")
    if not var:
        return False

    descendant_loops = [
        item
        for item in feats.get("loops") or []
        if item is not loop and contains_offset(loop, item.get("begin"))
    ]
    descendant_headers = [loop_header_line(item) for item in descendant_loops]
    if any(re.search(rf"\b{re.escape(var)}\b", header) for header in descendant_headers):
        return True

    # Bound/extent temporaries are scalar expressions derived from the current
    # loop variable and later consumed by a nested loop header.  Exclude
    # temporaries loaded from arrays, which are lane-local data rather than
    # scheduling bounds.
    bound_names: set[str] = set()
    for item in subtree_rw:
        if is_loop_initializer_assignment(item):
            continue
        if item.get("writes") or read_refs_for_item(item):
            continue
        lhs = normalize_ws(str(item.get("lhs") or ""))
        rhs = normalize_ws(str(item.get("rhs") or ""))
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", lhs):
            continue
        if re.search(rf"\b{re.escape(var)}\b", rhs):
            bound_names.add(lhs)
    if not bound_names:
        return False
    return any(
        re.search(rf"\b{re.escape(name)}\b", header)
        for name in bound_names
        for header in descendant_headers
    )


def loop_is_base_pointer_scheduler_loop(
    loop: dict[str, Any],
    child_loops: list[dict[str, Any]],
    subtree_rw: list[dict[str, Any]],
    feats: dict[str, Any],
) -> bool:
    """Detect outer loops that only select a base pointer/slice for children.

    Shape:
        for q in range(...):
          ptr = data + q * size
          for i in range(...):
            ptr[i] = ...

    The vectorizable data lane is the inner `i` loop.  Rendering `q` as a lane
    makes the bootstrap imply cross-channel lane work instead of a scalar
    scheduling loop over slices.
    """
    if not child_loops:
        return False
    spec = vector_loop_spec(loop)
    if not spec:
        return False
    outer_var = str(spec.get("var") or "")
    if not outer_var:
        return False

    direct_rw = [
        item
        for item in subtree_rw
        if item_directly_in_loop(item, loop, child_loops) and not is_loop_initializer_assignment(item)
    ]
    pointer_like_aliases: set[str] = set()
    for item in direct_rw:
        if item.get("writes"):
            continue
        lhs = normalize_ws(str(item.get("lhs") or ""))
        rhs = normalize_ws(str(item.get("rhs") or ""))
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", lhs):
            continue
        # Keep the rule structural: a base/slice pointer expression combines
        # loop/slice state with address arithmetic or row/plane storage.
        # Pure scalar math derived from q should not turn the loop into a
        # scheduler, but row-pointer expressions such as input_data[row] do
        # carry array reads and must still be recognized.
        pointerish = bool(
            re.search(r"\b(?:row|col|channel|plane|slice|data|ptr|buf|base|src|dst|input|output|stride|step|size|width|height)\b", lhs + " " + rhs, re.I)
            or read_refs_for_item(item)
        )
        if not pointerish:
            continue
        if not (
            re.search(rf"\b{re.escape(outer_var)}\b", rhs)
            or read_refs_for_item(item)
            or re.search(r"\+|\*|<<|>>", rhs)
        ):
            continue
        pointer_like_aliases.add(lhs)
    if not pointer_like_aliases:
        return False

    descendant_loops = [
        item
        for item in feats.get("loops") or []
        if item is not loop and contains_offset(loop, item.get("begin"))
    ]
    for child in descendant_loops:
        child_spec = vector_loop_spec(child)
        child_rw = items_in_loop_subtree(subtree_rw, child)
        for item in child_rw:
            lhs = normalize_ws(str(item.get("lhs") or ""))
            if postinc_pointer_store_name(lhs) in pointer_like_aliases:
                return True
            if lhs in pointer_like_aliases and str(item.get("op") or "") in {"+=", "-="}:
                return True
        if not child_spec:
            continue
        child_var = str(child_spec.get("var") or "")
        if not child_var:
            continue
        for item in child_rw:
            for ref in (item.get("reads") or []) + (item.get("writes") or []):
                base = str(ref.get("base") or "")
                index = str(ref.get("index") or "")
                if base in pointer_like_aliases and re.search(rf"\b{re.escape(child_var)}\b", index):
                    return True
    return False


def should_render_vector_execution(
    loop: dict[str, Any],
    region_kind: str,
    child_loops: list[dict[str, Any]],
    subtree_rw: list[dict[str, Any]],
    feats: dict[str, Any],
) -> bool:
    if in_place_aggregate_container_loop(subtree_rw):
        return False
    if region_kind not in VECTOR_EXECUTION_REGIONS:
        return False
    if region_kind == "two_pointer_swap":
        return True
    if region_kind != "hybrid_scalar_region_per_element" and loop_is_scheduler_loop(loop, child_loops, subtree_rw, feats):
        return False
    if loop_is_base_pointer_scheduler_loop(loop, child_loops, subtree_rw, feats):
        return False
    if region_kind == "nested_reduction_update" and loop_has_invariant_array_reduction_update(
        loop,
        [item for item in subtree_rw if not is_loop_initializer_assignment(item)],
    ):
        return False
    if loop_has_sequential_pointer_stream(subtree_rw):
        return False
    if child_loops and region_kind not in {"hybrid_elementwise_reduction", "hybrid_elementwise_store", "hybrid_scalar_region_per_element"}:
        return False
    if loop_has_irregular_cpp_logic(loop):
        return False
    return vector_loop_spec(loop) is not None


def in_place_aggregate_container_loop(items: list[dict[str, Any]]) -> bool:
    for item in items:
        lhs = normalize_ws(str(item.get("lhs") or ""))
        rhs = normalize_ws(str(item.get("rhs") or ""))
        if item.get("writes") and rhs.startswith("{") and rhs.endswith("}"):
            return True
        if re.search(r"\]\s*\[", lhs) and str(item.get("op") or "") != "=":
            return True
    return False


def loop_is_lane_local_scalar_subloop(
    loop: dict[str, Any],
    subtree_rw: list[dict[str, Any]],
    feats: dict[str, Any],
) -> bool:
    """Detect per-lane scalar dynamic work nested under an outer vector loop.

    This is for shapes such as digit extraction or bounded per-element scalar
    state.  The outer loop can still expose vector_range/reduction structure,
    but this nested loop should not be rendered as a synchronous vector loop.
    """
    header = loop_header_line(loop).lower()
    loop_kind = str(loop.get("kind") or "")
    is_while_like = header.startswith("while") or loop_kind in {"WhileStmt", "DoStmt"}
    spec = vector_loop_spec(loop)
    if not is_while_like and not spec:
        return False
    if loop_has_side_effect_call(loop, feats):
        return False
    if any(item_has_array_write(item) for item in subtree_rw):
        return False
    controls = feats.get("controls") or []
    if any(str(c.get("kind") or "") in {"BreakStmt", "ContinueStmt"} and contains_offset(loop, c.get("begin")) for c in controls):
        return False
    local_state = declared_local_scalar_names(subtree_rw) | self_dependent_scalar_state_names(subtree_rw)
    if not local_state:
        return False
    header_text = loop_header_line(loop)
    if any(re.search(rf"\b{re.escape(name)}\b", header_text) for name in local_state):
        return True
    if spec and any(item_has_scalar_reduction(item) for item in subtree_rw):
        loop_var = str(spec.get("var") or "")
        header_names = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", header_text))
        header_names.discard(loop_var)
        header_names -= {"for", "int", "long", "size_t", "std", "uint32_t", "uint64_t", "std::size_t"}
        return bool(header_names)
    return False


def loop_has_lane_local_scalar_child(loop: dict[str, Any], feats: dict[str, Any]) -> bool:
    for child in feats.get("loops") or []:
        if child is loop:
            continue
        if child.get("begin") == loop.get("begin") and child.get("end") == loop.get("end"):
            continue
        if not contains_offset(loop, child.get("begin")):
            continue
        if loop_is_lane_local_scalar_subloop(child, items_in_loop_subtree(feats.get("read_write") or [], child), feats):
            return True
    return False


def RSB_item_label(
    item: dict[str, Any],
    region_kind: str,
    *,
    vector_execution: bool = False,
    reduction_vector_context: bool = False,
    local_scalar_names: set[str] | None = None,
    direct_local_scalar_names: set[str] | None = None,
) -> str:
    if is_loop_initializer_assignment(item):
        return ""
    local_scalar_names = local_scalar_names or set()
    direct_local_scalar_names = direct_local_scalar_names or set()
    lhs = normalize_ws(str(item.get("lhs") or ""))
    op = str(item.get("op") or "=")
    if item.get("source") == "var_decl":
        if (
            is_std_vector_constructor_init(item)
            or is_scalar_zero_or_empty_init(item)
            or is_simple_scalar_var_init(item)
        ):
            return "init"
        return "compute"
    if (
        region_kind in {"prefix_or_rolling_output", "prefix_scan_predicate"}
        and item_is_prefix_scan_update(item)
    ):
        return "scan"
    if (
        vector_execution
        and lhs in direct_local_scalar_names
        and item_has_scalar_lhs(item)
        and op != "="
    ):
        return "update"
    if (
        reduction_vector_context
        and item_has_scalar_lhs(item)
        and op != "="
        and (vector_execution or lhs not in local_scalar_names)
    ):
        return "reduce"
    if (
        vector_execution
        and region_kind == "arg_reduction"
        and item_has_scalar_lhs(item)
        and item.get("source") != "var_decl"
        and op == "="
    ):
        if item_is_boolean_state_assignment(item):
            return "state_reduce"
        return "arg_reduce" if item_is_arg_index_update(item) else "reduce"
    if vector_execution and item_is_existing_scalar_state_update(item) and not item_has_scalar_reduction(item):
        return "update" if region_kind == "bounded_predicate_scan" else "state_reduce"
    if (
        is_std_vector_constructor_init(item)
        or is_simple_scalar_var_init(item)
        or (item.get("source") == "var_decl" and is_scalar_zero_or_empty_init(item))
    ):
        return "init"
    if item.get("writes"):
        if region_kind == "conflict_update":
            return "conflict_update"
        if region_kind in {"dependency_or_inplace_update", "prefix_or_neighbor_dependency"} and op != "=":
            return "carried_update"
        if region_kind == "nested_reduction_update" and op != "=":
            return "reduce"
        if region_kind in {"predicate_store_map", "indexed_gather_scatter_map", "store_map"} and op == "=":
            return "write"
        if vector_execution and region_kind == "array_update" and op != "=":
            return "lane_update"
        if op != "=":
            return "update"
        return "write"
    if item_has_scalar_reduction(item):
        lhs = normalize_ws(str(item.get("lhs") or ""))
        if vector_execution and reduction_vector_context:
            return "reduce"
        if lhs in local_scalar_names:
            return "update"
        if reduction_vector_context:
            return "reduce"
        if region_kind in {"prefix_or_rolling_output", "prefix_scan_predicate"}:
            return "scan" if vector_execution else "update"
        if region_kind in {"pseudocode_control", "pseudocode_loop"}:
            return "update"
        return "reduce" if vector_execution else "update"
    if op != "=":
        return "update"
    if item_has_scalar_lhs(item) and item.get("source") != "var_decl" and not item.get("reads"):
        return "update"
    if item.get("reads"):
        return "compute"
    return "compute"


def render_scan_update_line(item: dict[str, Any], rendered: str) -> str:
    lhs = normalize_ws(str(item.get("lhs") or ""))
    rhs = normalize_ws(str(item.get("rhs") or ""))
    if not lhs or not rhs:
        return rendered
    op = str(item.get("op") or "=")
    if op in {"+=", "-=", "*=", "&=", "|=", "^="}:
        scan_op = op[:-1]
        return f"{lhs} = prefix_scan({scan_op}, carry={lhs}, value={sanitize_vector_scalar_ops(rhs)})"
    for op_name in ("max", "min"):
        pattern = rf"^(?:std::)?{op_name}\s*\((.*)\)$"
        m = re.match(pattern, rhs)
        if not m:
            continue
        args = [normalize_ws(x) for x in m.group(1).split(",")]
        if len(args) == 2 and lhs in args:
            other = args[1] if args[0] == lhs else args[0]
            return f"{lhs} = prefix_scan({op_name}, carry={lhs}, value={sanitize_vector_scalar_ops(other)})"
    for op_symbol in ["+", "-", "*", "&", "|", "^"]:
        patterns = [
            rf"^{re.escape(lhs)}\s*{re.escape(op_symbol)}\s*(.+)$",
            rf"^(.+)\s*{re.escape(op_symbol)}\s*{re.escape(lhs)}$",
        ]
        for pat in patterns:
            m = re.match(pat, rhs)
            if m:
                value = normalize_ws(m.group(1))
                return f"{lhs} = prefix_scan({op_symbol}, carry={lhs}, value={sanitize_vector_scalar_ops(value)})"
    return rendered


def render_RSB_item(
    lines: list[str],
    item: dict[str, Any],
    indent: str,
    region_kind: str,
    *,
    vector_execution: bool = False,
    reduction_vector_context: bool = False,
    local_scalar_names: set[str] | None = None,
    direct_local_scalar_names: set[str] | None = None,
    affine_spec: dict[str, str] | None = None,
    pointer_recurrences: dict[str, str] | None = None,
    scalar_exprs: dict[str, str] | None = None,
    streaming_slots: dict[str, tuple[str, str]] | None = None,
    float_names: set[str] | None = None,
) -> None:
    float_names = float_names or set()
    if is_loop_initializer_assignment(item):
        return
    streaming_slots = streaming_slots or {}
    stream_slot = streaming_slots.get(item_identity(item))
    if stream_slot and affine_spec:
        ptr, index = stream_slot
        rhs_source = str(item.get("rhs") or "")
        rhs_source = rewrite_affine_pointer_subscripts(rhs_source, affine_spec)
        rhs_source = rewrite_loop_body_pointer_subscripts(rhs_source, affine_spec, pointer_recurrences)
        rhs = trim_statement(sanitize_vector_scalar_ops(rhs_source, float_names=float_names), 1200).rstrip(";").strip()
        if not rhs or is_malformed_semantic_line(rhs):
            return
        reads = render_refs(read_refs_for_item(item), affine_spec=affine_spec, pointer_recurrences=pointer_recurrences)
        if reads:
            lines.append(f"{indent}read: {reads}")
        lines.append(f"{indent}write: {ptr}[{index}] = {rhs}")
        return
    scalar_stream_ptr = postinc_pointer_store_name(str(item.get("lhs") or ""))
    if scalar_stream_ptr:
        rhs_source = str(item.get("rhs") or "")
        rhs = trim_statement(
            sanitize_vector_scalar_ops(replace_postinc_pointer_reads(rhs_source), float_names=float_names),
            1200,
        ).rstrip(";").strip()
        if not rhs or is_malformed_semantic_line(rhs):
            return
        reads = render_refs(read_refs_for_item(item))
        if reads:
            lines.append(f"{indent}read: {reads}")
        lines.append(f"{indent}write_next: *{scalar_stream_ptr} = {rhs}")
        for name in postinc_pointer_read_names(rhs_source):
            lines.append(f"{indent}update: {name} += 1")
        return
    label = RSB_item_label(
        item,
        region_kind,
        vector_execution=vector_execution,
        reduction_vector_context=reduction_vector_context,
        local_scalar_names=local_scalar_names,
        direct_local_scalar_names=direct_local_scalar_names,
    )
    rendered = render_assignment_line(
        item,
        vector_execution=vector_execution,
        affine_spec=affine_spec,
        pointer_recurrences=pointer_recurrences,
        float_names=float_names,
    )
    if label == "scan":
        rendered = render_scan_update_line(item, rendered)
    if not label or is_malformed_semantic_line(rendered):
        return
    reads = render_refs(read_refs_for_item(item), affine_spec=affine_spec, pointer_recurrences=pointer_recurrences)
    if reads and label in {"write", "update", "lane_update", "carried_update", "conflict_update", "accumulate", "compute", "reduce", "arg_reduce", "scan", "state_reduce"}:
        lines.append(f"{indent}read: {reads}")
    lines.append(f"{indent}{label}: {rendered}")
    if not vector_execution:
        for name in postinc_pointer_read_names(str(item.get("rhs") or "")):
            lines.append(f"{indent}update: {name} += 1")
        for update in item.get("postinc_subscript_updates") or []:
            name = str(update.get("name") or "")
            op = str(update.get("op") or "+=")
            rhs = str(update.get("rhs") or "1")
            if name:
                lines.append(f"{indent}update: {name} {op} {rhs}")


def render_streaming_pointer_kernel_loop(
    lines: list[str],
    loop: dict[str, Any],
    direct_rw: list[dict[str, Any]],
    direct_branches: list[dict[str, Any]],
    child_loops: list[dict[str, Any]],
    direct_controls: list[dict[str, Any]],
    direct_calls: list[dict[str, Any]],
    direct_returns: list[dict[str, Any]],
    indent: str,
    *,
    feats: dict[str, Any],
    float_names: set[str] | None = None,
) -> bool:
    """Render fixed pointer-progress loops as indexed vector kernels.

    This is an AST-level canonicalization for loops where the source program
    uses pointer post-increment as a sequential notation for a fixed stream:

        tmp = *src++;
        *dst++ = f(tmp, src[-1], src[0]);
        *dst++ = g(tmp, src[0]);

    The renderer converts that to a forward lane ordinal over the loop trip
    count and indexes every pointer relative to the loop-entry pointer.  It is
    intentionally structural: no task ids or benchmark names are consulted.
    """
    if child_loops or direct_branches or direct_controls or direct_calls or direct_returns:
        return False
    spec = vector_loop_spec(loop)
    if not spec:
        return False
    loop_var = str(spec.get("var") or "")
    trip_count = stream_loop_iteration_count_expr(spec)
    if not trip_count:
        return False
    active_items = [item for item in direct_rw if not is_loop_initializer_assignment(item)]
    # This transform is for loops whose C induction variable is only a trip
    # counter around pointer streams.  If the induction variable participates in
    # address formation, the loop is selecting rows/tiles/elements directly and
    # should be handled by the ordinary indexed-loop renderer.
    if loop_var and any(item_text_mentions_name(item, loop_var) for item in active_items):
        return False
    if loop_carried_scalar_reads_before_write(active_items):
        return False
    pointer_strides = pointer_stream_postinc_strides(active_items)
    if not pointer_strides:
        return False
    if not postinc_pointer_store_items(active_items):
        return False
    if not any(item_has_postinc_pointer_read(item) for item in active_items):
        return False

    # Avoid turning arbitrary control-heavy pointer code into a vector kernel.
    # The accepted shape has fixed stream motion and only scalar temporaries or
    # stream stores in the loop body.
    for item in active_items:
        lhs = normalize_ws(str(item.get("lhs") or ""))
        if item_is_stream_pointer_update(item, pointer_strides):
            continue
        if postinc_pointer_store_name(lhs):
            continue
        if item.get("writes"):
            return False

    ordinal = "stream_i"
    vec_spec = {"var": ordinal, "start": "0", "end": trip_count, "cmp": "<", "step": "1"}
    append_vector_loop_prelude(
        lines,
        indent,
        vec_spec,
        vector_type=infer_vector_element_type(active_items, feats, vec_spec),
    )
    body_indent = indent + "    "
    pointer_offsets: dict[str, int] = {name: 0 for name in pointer_strides}
    float_names = float_names or set()

    for item in active_items:
        if item_is_stream_pointer_update(item, pointer_strides):
            continue
        lhs = normalize_ws(str(item.get("lhs") or ""))
        rhs_source = str(item.get("rhs") or "")
        rhs_offsets = dict(pointer_offsets)
        rhs_rewritten = rewrite_stream_pointer_expr(
            rhs_source,
            pointer_strides=pointer_strides,
            pointer_offsets=rhs_offsets,
            ordinal=ordinal,
        )
        store_ptr = postinc_pointer_store_name(lhs)
        if store_ptr:
            index = stream_index_expr(ordinal, pointer_strides.get(store_ptr, 1), pointer_offsets.get(store_ptr, 0))
            pointer_offsets[store_ptr] = pointer_offsets.get(store_ptr, 0) + 1
            pointer_offsets.update({k: v for k, v in rhs_offsets.items() if k != store_ptr})
            rhs = trim_statement(
                sanitize_vector_scalar_ops(rhs_rewritten, float_names=float_names),
                1200,
            ).rstrip(";").strip()
            if rhs and not is_malformed_semantic_line(rhs):
                lines.append(f"{body_indent}write: {store_ptr}[{index}] = {rhs}")
            continue

        pointer_offsets.update(rhs_offsets)
        rendered_lhs = sanitize_vector_scalar_ops(replace_postinc_scalar_expr(lhs), float_names=float_names)
        rendered_rhs = trim_statement(
            sanitize_vector_scalar_ops(rhs_rewritten, float_names=float_names),
            1200,
        ).rstrip(";").strip()
        if rendered_lhs and rendered_rhs and not is_malformed_semantic_line(rendered_rhs):
            lines.append(f"{body_indent}compute: {rendered_lhs} = {rendered_rhs}")
    return True


def render_two_pointer_swap_loop(
    lines: list[str],
    loop: dict[str, Any],
    direct_rw: list[dict[str, Any]],
    feats: dict[str, Any],
    indent: str,
) -> bool:
    header = loop_header_line(loop)
    m = re.match(r"while\s*\(\s*([A-Za-z_]\w*)\s*<\s*([A-Za-z_]\w*)\s*\)\s*$", header)
    if not m:
        return False
    left, right = m.group(1), m.group(2)
    write_refs = [ref for item in direct_rw for ref in item.get("writes") or []]
    if len(write_refs) < 2:
        return False
    base = str(write_refs[0].get("base") or "")
    if not base:
        return False
    temp_items = [item for item in direct_rw if item_has_scalar_lhs(item) and item.get("reads")]
    temp_name = normalize_ws(str(temp_items[0].get("lhs") or "temp")) if temp_items else "temp"
    half_span = sanitize_numeric_operators(f"(({right} - {left}) + 1) / 2")
    swap_spec = {"var": "offset", "start": "0", "end": half_span, "cmp": "<", "step": "1"}
    append_vector_loop_prelude(
        lines,
        indent,
        swap_spec,
        vector_type=infer_vector_element_type(direct_rw, feats, swap_spec),
    )
    body_indent = indent + "    "
    lines.append(f"{body_indent}lane left = {left} + offset")
    lines.append(f"{body_indent}lane right = {right} - offset")
    lines.append(f"{body_indent}where left < right:")
    nested_indent = body_indent + "  "
    lines.append(f"{nested_indent}read: {base}[left], {base}[right]")
    lines.append(f"{nested_indent}compute: {temp_name} = {base}[left]")
    lines.append(f"{nested_indent}write: {base}[left] = {base}[right]")
    lines.append(f"{nested_indent}write: {base}[right] = {temp_name}")
    return True


def loop_condition_expr(loop: dict[str, Any]) -> str:
    if str(loop.get("kind") or "") == "DoStmt":
        m_do = re.search(r"\}\s*while\s*\((.*?)\)\s*;?", str(loop.get("snippet") or ""), re.S)
        return normalize_ws(m_do.group(1)) if m_do else ""
    header = loop_header_line(loop)
    m_while = re.match(r"while\s*\((.*)\)\s*$", header)
    return normalize_ws(m_while.group(1)) if m_while else ""


def convergence_flag_for_neighbor_swap(loop: dict[str, Any], feats: dict[str, Any]) -> str | None:
    if str(loop.get("kind") or "") not in {"DoStmt", "WhileStmt"}:
        return None
    condition = loop_condition_expr(loop)
    if not condition:
        return None
    subtree_rw = items_in_loop_subtree(feats.get("read_write") or [], loop)
    true_names: set[str] = set()
    false_names: set[str] = set()
    for item in subtree_rw:
        if not item_has_scalar_lhs(item):
            continue
        lhs = normalize_ws(str(item.get("lhs") or ""))
        rhs = normalize_ws(str(item.get("rhs") or ""))
        if rhs == "true":
            true_names.add(lhs)
        elif rhs == "false":
            false_names.add(lhs)
    for name in sorted(true_names & false_names):
        if re.search(rf"\b{re.escape(name)}\b", condition):
            return name
    return None


def ref_text(ref: dict[str, str]) -> str:
    return f"{str(ref.get('base') or '').strip()}[{sanitize_vector_scalar_ops(str(ref.get('index') or '').strip())}]"


def loop_var_offset(index: str, var: str) -> int | None:
    text = normalize_ws(index)
    escaped = re.escape(var)
    if re.fullmatch(escaped, text):
        return 0
    m = re.fullmatch(rf"{escaped}\s*\+\s*(\d+)", text)
    if m:
        return int(m.group(1))
    m = re.fullmatch(rf"{escaped}\s*-\s*(\d+)", text)
    if m:
        return -int(m.group(1))
    return None


def parse_simple_compare_refs(condition: str) -> tuple[dict[str, str], str, dict[str, str]] | None:
    cond = normalize_ws(condition)
    m = re.fullmatch(r"(.+?)\s*(<=|>=|<|>)\s*(.+)", cond)
    if not m:
        return None
    lhs, op, rhs = m.groups()
    lhs_refs = array_refs(lhs)
    rhs_refs = array_refs(rhs)
    if len(lhs_refs) != 1 or len(rhs_refs) != 1:
        return None
    if str(lhs_refs[0].get("base") or "") != str(rhs_refs[0].get("base") or ""):
        return None
    return lhs_refs[0], op, rhs_refs[0]


def parse_neighbor_compare_swap_pass(
    child: dict[str, Any],
    feats: dict[str, Any],
    children: dict[tuple[Any, Any, str], list[dict[str, Any]]],
) -> dict[str, str] | None:
    spec = vector_loop_spec(child)
    if not spec:
        return None
    var = str(spec.get("var") or "")
    if not var:
        return None
    step = sanitize_vector_scalar_ops(str(spec.get("step") or "1"))
    if step not in {"2", "+2"}:
        return None
    direct_branches = [
        b
        for b in feats.get("branches") or []
        if item_directly_in_loop(b, child, children.get(loop_key(child), []))
    ]
    direct_rw = [
        item
        for item in feats.get("read_write") or []
        if item_directly_in_loop(item, child, children.get(loop_key(child), []))
    ]
    write_keys = {ref_text(ref) for item in direct_rw for ref in item.get("writes") or []}
    for branch in direct_branches:
        parsed = parse_simple_compare_refs(branch_condition_expr(branch))
        if not parsed:
            continue
        cond_lhs, op, cond_rhs = parsed
        refs = [cond_lhs, cond_rhs]
        offsets = [loop_var_offset(str(ref.get("index") or ""), var) for ref in refs]
        if any(offset is None for offset in offsets):
            continue
        if abs(int(offsets[0]) - int(offsets[1])) != 1:
            continue
        ordered = sorted(zip(offsets, refs), key=lambda x: int(x[0]))
        left_ref = ordered[0][1]
        right_ref = ordered[1][1]
        if ref_text(left_ref) not in write_keys or ref_text(right_ref) not in write_keys:
            continue
        lhs_name = "left" if ref_text(cond_lhs) == ref_text(left_ref) else "right"
        rhs_name = "left" if ref_text(cond_rhs) == ref_text(left_ref) else "right"
        start = sanitize_vector_scalar_ops(str(spec.get("start") or "0"))
        end = sanitize_vector_scalar_ops(str(spec.get("end") or ""))
        lane_i = f"{step} * pair_id" if start in {"0", "+0"} else f"{start} + {step} * pair_id"
        max_offset = max(int(offsets[0]), int(offsets[1]))
        if max_offset > 0:
            trip_count = f"{end} > {max_offset} ? (({end} - {max_offset}) / ({step})) : 0"
        else:
            trip_count = f"(({end}) / ({step}))"
        active = sanitize_vector_scalar_ops(str(spec.get("active_condition") or ""))
        if not active:
            active = sanitize_vector_scalar_ops(f"{var} {spec.get('cmp', '<')} {spec.get('end', end)}")
        return {
            "var": var,
            "trip_count": trip_count,
            "lane_i": lane_i,
            "active": active,
            "left": ref_text(left_ref),
            "right": ref_text(right_ref),
            "swapped": f"{lhs_name} {op} {rhs_name}",
        }
    return None


def render_iterative_neighbor_swap_loop(
    lines: list[str],
    loop: dict[str, Any],
    child_loops: list[dict[str, Any]],
    feats: dict[str, Any],
    children: dict[tuple[Any, Any, str], list[dict[str, Any]]],
    indent: str,
) -> bool:
    state = convergence_flag_for_neighbor_swap(loop, feats)
    if not state:
        return False
    deps = dependency_refs_for_loop(loop, feats)
    child_vars = {
        str((vector_loop_spec(child) or {}).get("var") or "")
        for child in child_loops
    }
    child_vars.discard("")
    if not any(
        (
            (read_off := loop_var_offset(str(dep.get("read_index") or ""), var)) is not None
            and (write_off := loop_var_offset(str(dep.get("write_index") or ""), var)) is not None
            and abs(read_off - write_off) == 1
        )
        for var in child_vars
        for dep in deps
    ):
        return False
    passes: list[dict[str, str]] = []
    for child in sorted_by_begin(child_loops):
        parsed = parse_neighbor_compare_swap_pass(child, feats, children)
        if parsed:
            passes.append(parsed)
    if not passes:
        return False
    lines.append(f"{indent}do:")
    body_indent = indent + "  "
    lines.append(f"{body_indent}set: {state} = true")
    for parsed in passes:
        pair_spec = {"var": "pair_id", "start": "0", "end": parsed["trip_count"], "cmp": "<", "step": "1"}
        append_vector_loop_prelude(lines, body_indent, pair_spec)
        loop_body_indent = body_indent + "    "
        lines.append(f"{loop_body_indent}lane {parsed['var']} = {parsed['lane_i']}")
        lines.append(f"{loop_body_indent}where {parsed['active']}:")
        pass_indent = loop_body_indent + "  "
        lines.append(f"{pass_indent}read: left = {parsed['left']}")
        lines.append(f"{pass_indent}read: right = {parsed['right']}")
        lines.append(f"{pass_indent}compute: swapped = {parsed['swapped']}")
        lines.append(f"{pass_indent}where swapped:")
        lines.append(f"{pass_indent}  write: {parsed['left']} = right")
        lines.append(f"{pass_indent}  write: {parsed['right']} = left")
        lines.append(f"{pass_indent}if any active pair swapped:")
        lines.append(f"{pass_indent}  set: {state} = false")
    lines.append(f"{indent}while !{state}")
    return True


def render_affine_recurrence_loop(
    lines: list[str],
    loop: dict[str, Any],
    direct_rw: list[dict[str, Any]],
    feats: dict[str, Any],
    indent: str,
) -> bool:
    spec = vector_loop_spec(loop)
    if not spec:
        return False
    rec = parse_affine_neighbor_recurrence(loop, direct_rw, feats)
    if not rec:
        return False
    base = rec["base"]
    var = rec["var"]
    delta = rec["delta"]
    seed_index = rec["seed_index"]
    seed_expr = rec["seed_expr"]
    offset_expr = var if seed_index == "0" else f"{var} - ({seed_index})"
    append_vector_loop_prelude(
        lines,
        indent,
        spec,
        lane_type=loop_var_decl_type(loop, feats, spec["var"]),
        vector_type=infer_vector_element_type(direct_rw, feats, spec),
    )
    lines.append(f"{indent}    write: {base}[{var}] = {seed_expr} + ({offset_expr}) * ({delta})")
    return True


def render_prefix_rolling_output_loop(
    lines: list[str],
    loop: dict[str, Any],
    direct_rw: list[dict[str, Any]],
    feats: dict[str, Any],
    indent: str,
    *,
    float_names: set[str] | None = None,
) -> bool:
    spec = vector_loop_spec(loop)
    if not spec:
        return False
    active_items = [item for item in sorted_by_begin(direct_rw) if not is_loop_initializer_assignment(item)]
    scan_items = [item for item in active_items if item_is_prefix_scan_update(item)]
    if len(scan_items) != 1:
        return False
    scan_item = scan_items[0]
    state = normalize_ws(str(scan_item.get("lhs") or ""))
    if not state:
        return False

    lane_type = loop_var_decl_type(loop, feats, spec["var"])
    vector_type = infer_vector_element_type(active_items, feats, spec)
    append_vector_chunk_prelude(
        lines,
        indent,
        spec,
        lane_type=lane_type,
        vector_type=vector_type,
    )
    body_indent = indent + "  "
    lines.append(f"{body_indent}scalar_region:")
    lines.append(f"{body_indent}  for each active lane in increasing lane_id:")
    lane_indent = body_indent + "    "
    lines.append(f"{lane_indent}{vector_lane_line(spec, lane_type=lane_type)}")
    for item in active_items:
        if item is scan_item:
            reads = render_refs(read_refs_for_item(item), affine_spec=spec)
            if reads:
                lines.append(f"{lane_indent}read: {reads}")
            lhs = normalize_ws(str(item.get("lhs") or ""))
            op = str(item.get("op") or "=")
            rhs = sanitize_numeric_operators(normalize_ws(str(item.get("rhs") or "")), float_names=float_names)
            rendered = f"{lhs} {op} {rhs}" if lhs and rhs else render_assignment_line(item, vector_execution=False, affine_spec=spec, float_names=float_names)
            if not is_malformed_semantic_line(rendered):
                lines.append(f"{lane_indent}update: {rendered}")
            continue
        label = RSB_item_label(item, "prefix_or_rolling_output", vector_execution=False)
        rendered = render_assignment_line(item, vector_execution=False, affine_spec=spec, float_names=float_names)
        if not label or is_malformed_semantic_line(rendered):
            continue
        reads = render_refs(read_refs_for_item(item), affine_spec=spec)
        if reads and label in {"write", "update", "compute"}:
            lines.append(f"{lane_indent}read: {reads}")
        lines.append(f"{lane_indent}{label}: {rendered}")
    return True


def postinc_fixed_position_expr(var: str, start: str, slots: int, slot: int) -> str:
    base = var if normalize_ws(start) == "0" else f"({var} - ({start}))"
    if slots == 1:
        return base if slot == 0 else f"{base} + {slot}"
    if slot == 0:
        return f"{slots} * {base}"
    return f"{slots} * {base} + {slot}"


def render_fixed_position_expansion_loop(
    lines: list[str],
    loop: dict[str, Any],
    direct_rw: list[dict[str, Any]],
    direct_branches: list[dict[str, Any]],
    indent: str,
    feats: dict[str, Any] | None = None,
) -> bool:
    spec = vector_loop_spec(loop)
    if not spec:
        return False
    active_items = [item for item in direct_rw if not is_loop_initializer_assignment(item)]
    post_items = postinc_write_items(active_items)
    if len(post_items) < 2:
        return False
    refs = [postinc_write_ref(item) for item in post_items]
    if any(ref is None for ref in refs):
        return False
    bases = [str(ref.get("base") or "") for ref in refs if ref]
    idxs = [postinc_index_name(str(ref.get("index") or "")) for ref in refs if ref]
    if not bases or len(set(bases)) != 1 or len(set(idxs)) != 1:
        return False
    var = spec["var"]
    append_vector_loop_prelude(
        lines,
        indent,
        spec,
        lane_type=loop_var_decl_type(loop, feats, spec["var"]),
        vector_type=infer_vector_element_type(active_items, feats, spec),
    )
    body_indent = indent + "    "
    emitted_branch_keys: set[str] = set()
    for slot, item in enumerate(post_items):
        item_indent = branch_indent_for_item(lines, item, direct_branches, body_indent, emitted_branch_keys, vector_execution=True)
        reads = render_refs(read_refs_for_item(item))
        if reads:
            lines.append(f"{item_indent}read: {reads}")
        rhs = trim_statement(sanitize_vector_scalar_ops(str(item.get("rhs") or "")), 1200).rstrip(";").strip()
        if not rhs:
            continue
        pos = postinc_fixed_position_expr(var, spec.get("start", "0"), len(post_items), slot)
        lines.append(f"{item_indent}write: {bases[0]}[{pos}] = {rhs}")
    return True


def render_compact_filter_loop(
    lines: list[str],
    loop: dict[str, Any],
    direct_rw: list[dict[str, Any]],
    direct_branches: list[dict[str, Any]],
    indent: str,
    feats: dict[str, Any] | None = None,
) -> bool:
    spec = vector_loop_spec(loop)
    if not spec:
        return False
    active_items = [item for item in direct_rw if not is_loop_initializer_assignment(item)]
    post_items = postinc_write_items(active_items)
    if len(post_items) != 1:
        return False
    item = post_items[0]
    ref = postinc_write_ref(item)
    if not ref:
        return False
    branches = [b for b in direct_branches if contains_offset(b, item.get("begin"))]
    if not branches:
        return False
    predicate = sanitize_vector_scalar_ops(branch_condition_expr(branches[-1]))
    if not predicate:
        return False
    base = str(ref.get("base") or "out")
    count_name = postinc_index_name(str(ref.get("index") or "compact_count++"))
    rhs = trim_statement(sanitize_vector_scalar_ops(str(item.get("rhs") or "")), 1200).rstrip(";").strip()
    if not rhs:
        return False
    append_vector_loop_prelude(
        lines,
        indent,
        spec,
        lane_type=loop_var_decl_type(loop, feats, spec["var"]),
        vector_type=infer_vector_element_type(active_items, feats, spec),
    )
    body_indent = indent + "    "
    reads = render_refs(read_refs_for_item(item))
    if reads:
        lines.append(f"{body_indent}read: {reads}")
    lines.append(f"{body_indent}compact_write: {base}, count={count_name}, value={rhs}, predicate={predicate}")
    return True


def pointer_end_bound_expr(
    loop: dict[str, Any],
    outptr: str,
    outend: str,
    feats: dict[str, Any],
) -> str:
    loop_begin = loop.get("begin")
    candidates = [
        item
        for item in feats.get("read_write") or []
        if normalize_ws(str(item.get("lhs") or "")) == outend
        and isinstance(item.get("begin"), int)
        and (not isinstance(loop_begin, int) or item.get("begin") < loop_begin)
    ]
    for item in sorted_by_begin(candidates):
        rhs = sanitize_vector_scalar_ops(normalize_ws(str(item.get("rhs") or "")))
        m = re.fullmatch(rf"{re.escape(outptr)}\s*\+\s*(.+)", rhs)
        if m:
            return normalize_ws(m.group(1))
        m = re.fullmatch(rf"(.+)\s*\+\s*{re.escape(outptr)}", rhs)
        if m:
            return normalize_ws(m.group(1))
    return f"{outend} - {outptr}"


def render_duplicate_pointer_expand_loop(
    lines: list[str],
    loop: dict[str, Any],
    direct_rw: list[dict[str, Any]],
    feats: dict[str, Any],
    indent: str,
) -> bool:
    header = loop_header_line(loop)
    m = re.match(r"while\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*<\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)\s*$", header)
    if not m:
        return False
    outptr, outend = m.group(1), m.group(2)
    active = [item for item in sorted_by_begin(direct_rw) if not is_loop_initializer_assignment(item)]
    stores = [
        item
        for item in active
        if postinc_pointer_store_name(str(item.get("lhs") or "")) == outptr
    ]
    if len(stores) < 2:
        return False
    source_item = next(
        (
            item
            for item in active
            if item_has_scalar_lhs(item)
            and len(postinc_pointer_read_names(str(item.get("rhs") or ""))) == 1
        ),
        None,
    )
    if not source_item:
        return False
    value_name = normalize_ws(str(source_item.get("lhs") or "value"))
    inptr = postinc_pointer_read_names(str(source_item.get("rhs") or ""))[0]
    if not value_name or any(normalize_ws(str(item.get("rhs") or "")) != value_name for item in stores[:2]):
        return False
    slots = len(stores)
    span = pointer_end_bound_expr(loop, outptr, outend, feats)
    trip_count = f"(({span}) / ({slots}))" if slots != 1 else span
    slot_spec = {"var": "x", "start": "0", "end": trip_count, "cmp": "<", "step": "1"}
    append_vector_loop_prelude(
        lines,
        indent,
        slot_spec,
        vector_type=infer_vector_element_type(active, feats, slot_spec),
    )
    body_indent = indent + "    "
    lines.append(f"{body_indent}read: {inptr}[x]")
    for slot, _ in enumerate(stores[:slots]):
        index = f"{slots} * x" if slot == 0 else f"{slots} * x + {slot}"
        lines.append(f"{body_indent}write: {outptr}[{index}] = {inptr}[x]")
    return True


def direct_postinc_pointer_read_assignment(item: dict[str, Any]) -> tuple[str, str] | None:
    lhs = normalize_ws(str(item.get("lhs") or ""))
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", lhs):
        return None
    names = postinc_pointer_read_names(str(item.get("rhs") or ""))
    if len(names) != 1:
        return None
    return lhs, names[0]


def render_expr_with_postinc_indexes(expr: str, indexes: dict[str, str]) -> str:
    expr = replace_postinc_pointer_reads(expr, indexes)
    return trim_statement(sanitize_vector_scalar_ops(expr), 1200).rstrip(";").strip()


def loop_child_condition(loop: dict[str, Any]) -> str:
    spec = vector_loop_spec(loop)
    if spec:
        cond = active_predicate_line(spec).removeprefix("active = ")
        return sanitize_vector_scalar_ops(normalize_ws(cond))
    text = loop_header_line(loop)
    m = re.match(r"for\s*\(\s*.*?;\s*(.*?);\s*.*?\)\s*$", text)
    return sanitize_vector_scalar_ops(normalize_ws(m.group(1))) if m else ""


def render_merged_chroma_pointer_loop(
    lines: list[str],
    loop: dict[str, Any],
    child_loops: list[dict[str, Any]],
    direct_rw: list[dict[str, Any]],
    feats: dict[str, Any],
    children: dict[tuple[Any, Any, str], list[dict[str, Any]]],
    indent: str,
) -> bool:
    spec = vector_loop_spec(loop)
    if not spec or normalize_ws(str(spec.get("step") or "")) not in {"2", "+2"} or len(child_loops) != 1:
        return False
    outer_var = str(spec.get("var") or "")
    child = child_loops[0]
    child_spec = vector_loop_spec(child)
    if not child_spec:
        return False
    child_var = str(child_spec.get("var") or "")
    child_cond = loop_child_condition(child)
    if not (outer_var and child_var and re.search(rf"\b{re.escape(outer_var)}\b", child_cond) and re.search(rf"\b{re.escape(child_var)}\b", child_cond)):
        return False

    outer_index = lane_ordinal_expr(spec)
    outer_post_items = [
        item
        for item in sorted_by_begin(direct_rw)
        if direct_postinc_pointer_read_assignment(item)
    ]
    if len(outer_post_items) < 2:
        return False

    child_direct_rw = [
        item
        for item in feats.get("read_write") or []
        if item_directly_in_loop(item, child, children.get(loop_key(child), []))
    ]
    child_post_sources: dict[str, tuple[str, str]] = {}
    child_post_order: list[str] = []
    for item in sorted_by_begin(child_direct_rw):
        parsed = direct_postinc_pointer_read_assignment(item)
        if not parsed:
            continue
        lhs, ptr = parsed
        if lhs not in child_post_order:
            child_post_order.append(lhs)
        child_post_sources[lhs] = (ptr, f"{outer_var} + {child_var}")
    write_items = [item for item in sorted_by_begin(child_direct_rw) if item.get("writes")]
    write_strides = loop_body_pointer_recurrences(child_direct_rw)
    if not child_post_sources or not write_items or not write_strides:
        return False

    append_vector_loop_prelude(
        lines,
        indent,
        spec,
        lane_type=loop_var_decl_type(loop, feats, spec["var"]),
        vector_type=infer_vector_element_type(direct_rw + child_direct_rw, feats, spec),
    )
    body_indent = indent + "    "
    outer_indexes: dict[str, str] = {}
    for item in outer_post_items:
        parsed = direct_postinc_pointer_read_assignment(item)
        if not parsed:
            continue
        lhs, ptr = parsed
        outer_indexes[ptr] = outer_index
        rhs = render_expr_with_postinc_indexes(str(item.get("rhs") or ""), outer_indexes)
        lines.append(f"{body_indent}read: {ptr}[{outer_index}]")
        lines.append(f"{body_indent}compute: {lhs} = {rhs}")
    for item in sorted_by_begin(direct_rw):
        if is_loop_initializer_assignment(item) or item in outer_post_items:
            continue
        if item.get("writes") or item_has_postinc_pointer_read(item):
            continue
        lhs = normalize_ws(str(item.get("lhs") or ""))
        if not lhs or lhs == outer_var:
            continue
        if str(item.get("op") or "") not in {"=", ""}:
            continue
        rendered = render_assignment_line(item, float_names=floating_symbol_names(feats))
        if rendered and not is_malformed_semantic_line(rendered):
            lines.append(f"{body_indent}compute: {rendered}")

    lines.append(f"{body_indent}for {child_var} in range({child_spec['start']}, {child_spec['end']}):")
    guard = child_cond
    if guard:
        lines.append(f"{body_indent}  where {guard}:")
        child_indent = body_indent + "    "
    else:
        child_indent = body_indent + "  "
    for lhs in child_post_order:
        ptr, idx = child_post_sources[lhs]
        lines.append(f"{child_indent}read: {ptr}[{idx}]")
        lines.append(f"{child_indent}compute: {lhs} = {ptr}[{idx}]")
    for item in write_items:
        writes = item.get("writes") or []
        if not writes:
            continue
        base = str(writes[0].get("base") or "")
        index = sanitize_vector_scalar_ops(str(writes[0].get("index") or "0"))
        if base in write_strides:
            offset = f"({outer_var} + {child_var}) * {write_strides[base]}"
            index = combine_affine_index(offset, index)
        rhs = trim_statement(sanitize_vector_scalar_ops(str(item.get("rhs") or "")), 1200).rstrip(";").strip()
        if not rhs or is_malformed_semantic_line(rhs):
            continue
        lines.append(f"{child_indent}write: {base}[{index}] = {rhs}")
    return True


def parse_scalar_bound_compare(cond: str, name: str) -> tuple[str, str] | None:
    cond_norm = normalize_ws(cond)
    escaped = re.escape(name)
    m = re.fullmatch(rf"{escaped}\s*(<=|>=|<|>)\s*(.+)", cond_norm)
    if m:
        return m.group(1), normalize_ws(m.group(2))
    m = re.fullmatch(rf"(.+)\s*(<=|>=|<|>)\s*{escaped}", cond_norm)
    if not m:
        return None
    bound = normalize_ws(m.group(1))
    op = m.group(2)
    flip = {"<": ">", ">": "<", "<=": ">=", ">=": "<="}
    return flip[op], bound


def item_branch_conditions(item: dict[str, Any], direct_branches: list[dict[str, Any]]) -> list[str]:
    return [
        branch_condition_expr(branch)
        for branch in direct_branches
        if contains_offset(branch, item.get("begin")) and branch_condition_expr(branch)
    ]


def write_target_key(item: dict[str, Any]) -> str:
    writes = item.get("writes") or []
    if writes:
        ref = writes[0]
        return f"{ref.get('base')}[{ref.get('index')}]"
    return normalize_ws(str(item.get("lhs") or ""))


def rhs_is_cast_of_name(rhs: str, name: str) -> bool:
    rhs_norm = normalize_ws(rhs)
    escaped = re.escape(name)
    return bool(
        re.fullmatch(rf"\([A-Za-z_][A-Za-z0-9_:]*\)\s*{escaped}", rhs_norm)
        or re.fullmatch(rf"static_cast\s*<\s*[^>]+\s*>\s*\(\s*{escaped}\s*\)", rhs_norm)
    )


def render_round_clamp_cast_loop(
    lines: list[str],
    loop: dict[str, Any],
    direct_rw: list[dict[str, Any]],
    direct_branches: list[dict[str, Any]],
    feats: dict[str, Any],
    indent: str,
) -> bool:
    spec = vector_loop_spec(loop)
    if not spec:
        return False
    active_rw = [item for item in direct_rw if not is_loop_initializer_assignment(item)]
    round_items = [
        item
        for item in active_rw
        if item_has_scalar_lhs(item)
        and str(item.get("op") or "") == "="
        and re.search(r"(?<![A-Za-z0-9_:])(?:std::)?roundf?\s*\(", str(item.get("rhs") or ""))
    ]
    if not round_items:
        return False
    write_items = [item for item in active_rw if item.get("writes") and str(item.get("op") or "") == "="]
    for round_item in round_items:
        value_name = normalize_ws(str(round_item.get("lhs") or ""))
        if not value_name:
            continue
        groups: dict[str, list[dict[str, Any]]] = {}
        for item in write_items:
            groups.setdefault(write_target_key(item), []).append(item)
        for _, group in groups.items():
            final_item = None
            lower_bound = ""
            upper_bound = ""
            for item in group:
                rhs_raw = str(item.get("rhs") or "")
                rhs = normalize_ws(rhs_raw)
                if rhs_is_cast_of_name(rhs, value_name):
                    final_item = item
                    continue
                for cond in item_branch_conditions(item, direct_branches):
                    parsed = parse_scalar_bound_compare(cond, value_name)
                    if not parsed:
                        continue
                    op, bound = parsed
                    if rhs == bound:
                        if op in {">", ">="}:
                            upper_bound = bound
                        elif op in {"<", "<="}:
                            lower_bound = bound
            if not (final_item and lower_bound and upper_bound):
                continue
            reads = render_refs(read_refs_for_item(round_item))
            round_expr = trim_statement(sanitize_vector_scalar_ops(str(round_item.get("rhs") or "")), 1200).rstrip(";").strip()
            cast_rhs = trim_statement(sanitize_vector_scalar_ops(str(final_item.get("rhs") or "")), 1200).rstrip(";").strip()
            target = normalize_ws(str(final_item.get("lhs") or ""))
            if not (round_expr and cast_rhs and target):
                continue
            append_vector_loop_prelude(
                lines,
                indent,
                spec,
                lane_type=loop_var_decl_type(loop, feats, spec["var"]),
                vector_type=infer_vector_element_type(active_rw, feats, spec),
            )
            body_indent = indent + "    "
            if reads:
                lines.append(f"{body_indent}read: {reads}")
            lines.append(f"{body_indent}compute: {value_name} = {round_expr}")
            lines.append(f"{body_indent}compute: {value_name} = clamp({value_name}, {lower_bound}, {upper_bound})")
            lines.append(f"{body_indent}write: {target} = {cast_rhs}")
            return True
    return False


def render_conditional_prefix_scan_predicate_loop(
    lines: list[str],
    loop: dict[str, Any],
    direct_rw: list[dict[str, Any]],
    direct_branches: list[dict[str, Any]],
    feats: dict[str, Any],
    indent: str,
) -> bool:
    spec = vector_loop_spec(loop)
    if not spec:
        return False
    active_rw = [item for item in direct_rw if not is_loop_initializer_assignment(item)]
    parsed = parse_conditional_prefix_scan_predicate(loop, active_rw, direct_branches, feats)
    if not parsed:
        return False
    state = str(parsed["state"])
    threshold = str(parsed["threshold_condition"])
    reads: list[dict[str, str]] = []
    for branch in direct_branches:
        if not re.search(rf"\b{re.escape(state)}\b", branch_condition_expr(branch)):
            reads.extend(array_refs(branch_condition_expr(branch)))
    reads.extend(array_refs(str(parsed["delta"])))
    read_line = render_refs(reads)
    lane_type = loop_var_decl_type(loop, feats, spec["var"])
    vector_type = infer_vector_element_type(active_rw, feats, spec)
    if not vector_type and re.search(r"\b[A-Za-z_][A-Za-z0-9_]*\.(?:length|size)\s*\(\s*\)", str(spec.get("end") or "")):
        vector_type = "char" if ".length" in str(spec.get("end") or "") else vector_type
    append_vector_chunk_prelude(
        lines,
        indent,
        spec,
        lane_type=lane_type,
        vector_type=vector_type,
    )
    body_indent = indent + "  "
    lines.append(f"{body_indent}scalar_region:")
    lines.append(f"{body_indent}  for each active lane in increasing lane_id:")
    lane_indent = body_indent + "    "
    lines.append(f"{lane_indent}{vector_lane_line(spec, lane_type=lane_type)}")
    if read_line:
        lines.append(f"{lane_indent}read: {read_line}")
    state_updates = [
        item
        for item in sorted_by_begin(active_rw)
        if item_has_scalar_lhs(item)
        and normalize_ws(str(item.get("lhs") or "")) == state
        and str(item.get("op") or "") in {"+=", "-="}
    ]
    if state_updates:
        for item in state_updates:
            op = str(item.get("op") or "")
            rhs = sanitize_numeric_operators(normalize_ws(str(item.get("rhs") or "")))
            rendered = f"{state} {op} {rhs}"
            conds = [
                cond
                for cond in item_branch_conditions(item, direct_branches)
                if not re.search(rf"\b{re.escape(state)}\b", cond)
            ]
            if conds:
                cond = sanitize_numeric_operators(normalize_ws(conds[-1]))
                lines.append(f"{lane_indent}if {cond}:")
                lines.append(f"{lane_indent}  update: {rendered}")
            else:
                lines.append(f"{lane_indent}update: {rendered}")
    else:
        lines.append(f"{lane_indent}compute: delta = {parsed['delta']}")
        lines.append(f"{lane_indent}update: {state} += delta")
    lines.append(f"{lane_indent}if {threshold}:")
    lines.append(f"{lane_indent}  return: {parsed['return_expr']}")
    return True


def branch_line_for_offset(
    branch: dict[str, Any],
    offset: Any,
    *,
    vector_execution: bool = False,
    predicate_return: bool = False,
) -> str:
    else_begin = branch.get("else_begin")
    if isinstance(else_begin, int) and isinstance(offset, int) and offset >= else_begin:
        return "else:"
    line = render_branch_as_code(branch)
    if vector_execution and line.startswith("if "):
        cond = sanitize_vector_scalar_ops(line.removeprefix("if ").rstrip(":"))
        if predicate_return:
            return f"if any_lane({cond}):"
        return f"where {cond}:"
    return line


def branch_indent_for_item(
    lines: list[str],
    item: dict[str, Any],
    branches: list[dict[str, Any]],
    base_indent: str,
    emitted_branch_keys: set[str],
    *,
    vector_execution: bool = False,
    predicate_return: bool = False,
    include_branch_reads: bool = False,
) -> str:
    containing_branches = sorted(
        [b for b in branches if contains_offset(b, item.get("begin"))],
        key=lambda b: (b.get("begin") if isinstance(b.get("begin"), int) else -1, -(b.get("end") if isinstance(b.get("end"), int) else -1)),
    )
    if not containing_branches:
        return base_indent
    for depth, branch in enumerate(containing_branches):
        branch_line = branch_line_for_offset(
            branch,
            item.get("begin"),
            vector_execution=vector_execution,
            predicate_return=predicate_return,
        )
        branch_key = f"{branch.get('begin')}:{'else' if branch_line == 'else:' else 'if'}:{branch_line}"
        branch_indent = base_indent + ("  " * depth)
        if branch_line and branch_key not in emitted_branch_keys and not is_malformed_semantic_line(branch_line):
            if include_branch_reads and vector_execution:
                refs = render_refs(array_refs(branch_condition_expr(branch)))
                if refs:
                    lines.append(f"{branch_indent}read: {refs}")
            lines.append(f"{branch_indent}{branch_line}")
            emitted_branch_keys.add(branch_key)
    return base_indent + ("  " * len(containing_branches))


def render_RSB_loop(
    lines: list[str],
    loop: dict[str, Any],
    *,
    feats: dict[str, Any],
    children: dict[tuple[Any, Any, str], list[dict[str, Any]]],
    depth: int,
    in_vector_context: bool = False,
    inherited_local_scalars: set[str] | None = None,
    float_names: set[str] | None = None,
    return_type: str = "",
) -> None:
    float_names = float_names or floating_symbol_names(feats)
    child_loops = children.get(loop_key(loop), [])
    loops = feats.get("loops") or []
    rw_all = feats.get("read_write") or []
    branches_all = feats.get("branches") or []
    returns_all = feats.get("returns") or []
    calls_all = feats.get("calls") or []
    direct_rw = [item for item in rw_all if item_directly_in_loop(item, loop, child_loops)]
    subtree_rw = items_in_loop_subtree(rw_all, loop)
    direct_branches = [b for b in branches_all if item_directly_in_loop(b, loop, child_loops)]
    direct_returns = [r for r in returns_all if item_directly_in_loop(r, loop, child_loops)]
    direct_calls = [c for c in calls_all if is_side_effect_call(c) and item_directly_in_loop(c, loop, child_loops)]
    direct_controls = [c for c in (feats.get("controls") or []) if item_directly_in_loop(c, loop, child_loops)]
    direct_local_scalar_names = direct_loop_local_scalar_names(direct_rw)
    local_scalar_names = (inherited_local_scalars or set()) | direct_local_scalar_names
    region_kind = classify_RSB_region(loop, subtree_rw, direct_branches, feats)
    indent = "  " * depth
    if render_iterative_neighbor_swap_loop(lines, loop, child_loops, feats, children, indent):
        return
    if region_kind == "two_pointer_swap" and render_two_pointer_swap_loop(lines, loop, direct_rw, feats, indent):
        return
    if region_kind == "affine_recurrence_closed_form" and render_affine_recurrence_loop(lines, loop, direct_rw, feats, indent):
        return
    if region_kind == "prefix_or_rolling_output" and render_prefix_rolling_output_loop(
        lines,
        loop,
        direct_rw,
        feats,
        indent,
        float_names=float_names,
    ):
        return
    if region_kind == "prefix_scan_predicate" and render_conditional_prefix_scan_predicate_loop(lines, loop, direct_rw, direct_branches, feats, indent):
        return
    if render_round_clamp_cast_loop(lines, loop, direct_rw, direct_branches, feats, indent):
        return
    if render_streaming_pointer_kernel_loop(
        lines,
        loop,
        direct_rw,
        direct_branches,
        child_loops,
        direct_controls,
        direct_calls,
        direct_returns,
        indent,
        feats=feats,
        float_names=float_names,
    ):
        return
    if render_merged_chroma_pointer_loop(lines, loop, child_loops, direct_rw, feats, children, indent):
        return
    if render_duplicate_pointer_expand_loop(lines, loop, direct_rw, feats, indent):
        return
    if render_fixed_position_expansion_loop(lines, loop, direct_rw, direct_branches, indent, feats=feats):
        return
    if render_compact_filter_loop(lines, loop, direct_rw, direct_branches, indent, feats=feats):
        return
    force_scalar_loop = bool(in_vector_context and loop_is_lane_local_scalar_subloop(loop, subtree_rw, feats))
    vector_execution = False if force_scalar_loop else should_render_vector_execution(loop, region_kind, child_loops, subtree_rw, feats)
    wrap_vector_body_as_scalar_region = vector_execution and region_kind == "hybrid_scalar_region_per_element"
    branch_vector_execution = vector_execution and not wrap_vector_body_as_scalar_region
    reduction_vector_context = vector_execution or in_vector_context
    spec = vector_loop_spec(loop) if vector_execution else None
    scalar_aliases = scalar_induction_lane_aliases(loop, direct_rw, feats) if vector_execution else {}
    affine_spec = spec if vector_execution and spec_affine_recurrences(spec) else None
    loop_pointer_recurrences = loop_body_pointer_recurrences(direct_rw) if vector_execution else {}
    loop_scalar_exprs = loop_scalar_toggle_exprs(direct_rw, spec) if vector_execution else {}
    streaming_slots = streaming_pointer_store_slots(direct_rw, spec) if vector_execution else {}
    direct_postinc_read_ptrs = postinc_pointer_read_update_items(direct_rw) if not vector_execution else set()
    item_affine_spec = spec if vector_execution else None
    if vector_execution:
        if spec:
            append_vector_loop_prelude(
                lines,
                indent,
                spec,
                lane_type=loop_var_decl_type(loop, feats, spec["var"]),
                vector_type=infer_vector_element_type(direct_rw, feats, spec),
            )
            for alias, expr in sorted(scalar_aliases.items()):
                lines.append(f"{indent}    lane {alias} = {expr}")
            for name, expr in sorted(loop_scalar_exprs.items()):
                lines.append(f"{indent}    compute: {name} = {expr}")
    else:
        loop_line = render_loop_as_range(loop, feats, cpp_for=True)
        if loop_line and not is_malformed_semantic_line(loop_line):
            lines.append(f"{indent}{loop_line}")
    body_indent = (indent + "    ") if vector_execution else ("  " * (depth + 1))
    if wrap_vector_body_as_scalar_region:
        lines.append(f"{body_indent}scalar_region:")
        body_indent += "  "

    events: list[tuple[str, dict[str, Any]]] = []
    for call in direct_calls:
        events.append(("call", call))
    for item in direct_rw:
        events.append(("rw", item))
    for child in child_loops:
        events.append(("loop", child))
    for ctrl in direct_controls:
        events.append(("control", ctrl))
    for ret in direct_returns:
        events.append(("return", ret))
    events.sort(key=lambda x: (x[1].get("begin") if isinstance(x[1].get("begin"), int) else 10**18, {"call": 0, "rw": 1, "loop": 2, "control": 3, "return": 4}.get(x[0], 9)))

    emitted_branch_keys: set[str] = set()
    for kind, item in events:
        if kind == "loop":
            item_indent = branch_indent_for_item(lines, item, direct_branches, body_indent, emitted_branch_keys, vector_execution=branch_vector_execution)
            child_subtree_rw = items_in_loop_subtree(rw_all, item)
            lane_local_scalar = bool(
                vector_execution
                and loop_is_lane_local_scalar_subloop(item, child_subtree_rw, feats)
            )
            child_depth = len(item_indent) // 2
            child_in_vector_context = reduction_vector_context
            if lane_local_scalar and not wrap_vector_body_as_scalar_region:
                lines.append(f"{item_indent}scalar_region:")
                child_depth += 1
            render_RSB_loop(
                lines,
                item,
                feats=feats,
                children=children,
                depth=child_depth,
                in_vector_context=child_in_vector_context,
                inherited_local_scalars=local_scalar_names,
                float_names=float_names,
                return_type=return_type,
            )
        elif kind == "call":
            item_indent = branch_indent_for_item(lines, item, direct_branches, body_indent, emitted_branch_keys, vector_execution=branch_vector_execution)
            lines.append(f"{item_indent}{side_effect_call_label(item)}: {render_side_effect_call(item)}")
        elif kind == "rw":
            if item_is_scalar_induction_update(item, scalar_aliases):
                continue
            if not vector_execution and item_is_separate_postinc_read_update(item, direct_postinc_read_ptrs):
                continue
            if vector_execution and item_is_pointer_recurrence_update(item, loop_pointer_recurrences):
                continue
            if vector_execution and item_is_scalar_toggle_update(item, loop_scalar_exprs):
                continue
            item_indent = branch_indent_for_item(
                lines,
                item,
                direct_branches,
                body_indent,
                emitted_branch_keys,
                vector_execution=branch_vector_execution,
                predicate_return=region_kind in {"predicate_return_scan", "prefix_scan_predicate", "bounded_predicate_scan"},
                include_branch_reads=region_kind in {"arg_reduction", "predicate_return_scan", "bounded_predicate_scan"},
            )
            render_RSB_item(
                lines,
                item,
                item_indent,
                region_kind,
                vector_execution=vector_execution,
                reduction_vector_context=reduction_vector_context,
                local_scalar_names=local_scalar_names,
                direct_local_scalar_names=direct_local_scalar_names,
                affine_spec=item_affine_spec,
                pointer_recurrences=loop_pointer_recurrences,
                scalar_exprs=loop_scalar_exprs,
                streaming_slots=streaming_slots,
                float_names=float_names,
            )
        elif kind == "control":
            item_indent = branch_indent_for_item(
                lines,
                item,
                direct_branches,
                body_indent,
                emitted_branch_keys,
                vector_execution=branch_vector_execution,
                predicate_return=region_kind in {"predicate_return_scan", "prefix_scan_predicate", "bounded_predicate_scan"},
            )
            rendered = render_control_statement(item)
            if rendered:
                prefix = "" if rendered in {"break", "continue"} else "control: "
                lines.append(f"{item_indent}{prefix}{rendered}")
        elif kind == "return":
            item_indent = branch_indent_for_item(
                lines,
                item,
                direct_branches,
                body_indent,
                emitted_branch_keys,
                vector_execution=branch_vector_execution,
                predicate_return=region_kind in {"predicate_return_scan", "prefix_scan_predicate"},
                include_branch_reads=region_kind in {"predicate_return_scan", "prefix_scan_predicate"},
            )
            ret = (
                render_vector_predicate_return_expr(item.get("snippet") or "", spec, float_names=float_names)
                if region_kind in {"predicate_return_scan", "prefix_scan_predicate"}
                else render_return_expr(item.get("snippet") or "", float_names=float_names, return_type=return_type)
            )
            if ret:
                lines.append(f"{item_indent}return: {ret}")


def append_missing_fallthrough_return(lines: list[str], feats: dict[str, Any], return_type: str, float_names: set[str]) -> None:
    if not return_type or return_type == "void":
        return
    returns = sorted_by_begin(feats.get("returns") or [])
    if not returns:
        return
    last = returns[-1]
    if any(contains_offset(branch, last.get("begin")) for branch in feats.get("branches") or []):
        return
    ret = render_return_expr(last.get("snippet") or "", float_names=float_names, return_type=return_type)
    if not ret:
        return
    wanted = f"return: {ret}"
    if lines and lines[-1].strip() == wanted:
        return
    lines.append(f"  {wanted}")


PREFIX_SCAN_OPERATOR_DEFINITION = """operator_definition:
  prefix_scan(op, carry, value):
    semantic contract:
      active lanes are processed in increasing lane_id order
      prev = carry
      for each active lane:
        prefix[lane] = op(prev, value[lane])
        prev = prefix[lane]
      inactive lanes are ignored
      return prefix
    implementation contract:
      exact prefix semantics are required
      do not replace prefix_scan with op(carry, value[lane])
      do not replace prefix_scan with a chunk reduction such as sum(value) or max(value)
      do not check predicates on value alone when the predicate is defined on prefix
      do not invent unavailable SVE prefix-scan intrinsics or use scalar reduction results as vectors
      if an exact vector prefix scan cannot be implemented safely with valid ACLE, preserve the same active-lane order with a scalar loop inside the current vector chunk
  last_active(prefix):
    return prefix[last active lane in the current vector chunk]
  any_lane(predicate):
    return true if predicate is true for any active lane"""


def operator_semantics_block(body: str) -> str:
    lines = str(body or "").splitlines()
    text = "\n".join(lines[1:]) if len(lines) > 1 else str(body or "")
    has_bitwise_and = bool("&=" in text or re.search(r"(?<=[A-Za-z0-9_\]\)])\s*&\s*(?=[A-Za-z0-9_\(])", text))
    if not (
        re.search(r"(?<!/)/(?!/)", text)
        or "%" in text
        or has_bitwise_and
        or re.search(r"\b(?:std::)?roundf?\s*\(", text)
    ):
        return ""
    return "\n".join(
        [
            "operator_semantics:",
            "  /: C++ division",
            "  %: C++ remainder",
            "  &: bitwise AND",
            "  roundf: C++ roundf",
        ]
    )


def render_RSB_bootstrap(signature: str, feats: dict[str, Any]) -> str:
    """Emit hierarchical RSB bootstrap.

    RSB keeps the stronger RSB intent, but routes each loop
    or stage independently.  It uses a conservative symbolic lift for common
    vectorizable regions and falls back to pseudocode-style labels for control
    or dependency-heavy regions.
    """
    lines = [f"{signature.rstrip(';')}:"]
    float_names = floating_symbol_names(feats)
    return_type = function_return_type(signature)
    loops = sorted_by_begin(feats.get("loops") or [])
    roots, children = build_loop_children(loops)
    root_branches = [b for b in feats.get("branches") or [] if not item_inside_any_loop(b, loops)]

    root_events: list[tuple[str, dict[str, Any]]] = []
    for call in (feats.get("calls") or [])[:24]:
        if is_side_effect_call(call) and not item_inside_any_loop(call, loops):
            root_events.append(("call", call))
    for item in feats.get("read_write") or []:
        if not item_inside_any_loop(item, loops):
            root_events.append(("rw", item))
    for loop in roots:
        root_events.append(("loop", loop))
    for ctrl in (feats.get("controls") or [])[:8]:
        if not item_inside_any_loop(ctrl, loops):
            root_events.append(("control", ctrl))
    for ret in (feats.get("returns") or [])[:4]:
        if not item_inside_any_loop(ret, loops):
            root_events.append(("return", ret))
    root_events.sort(key=lambda x: (x[1].get("begin") if isinstance(x[1].get("begin"), int) else 10**18, {"call": 0, "rw": 1, "loop": 2, "control": 3, "return": 4}.get(x[0], 9)))

    emitted_branch_keys: set[str] = set()
    for kind, item in root_events:
        if kind == "call":
            item_indent = branch_indent_for_item(lines, item, root_branches, "  ", emitted_branch_keys)
            lines.append(f"{item_indent}{side_effect_call_label(item)}: {render_side_effect_call(item)}")
        elif kind == "rw":
            item_indent = branch_indent_for_item(lines, item, root_branches, "  ", emitted_branch_keys)
            render_RSB_item(lines, item, item_indent, "pseudocode_top_level", float_names=float_names)
        elif kind == "loop":
            item_indent = branch_indent_for_item(lines, item, root_branches, "  ", emitted_branch_keys)
            render_RSB_loop(
                lines,
                item,
                feats=feats,
                children=children,
                depth=len(item_indent) // 2,
                float_names=float_names,
                return_type=return_type,
            )
        elif kind == "control":
            item_indent = branch_indent_for_item(lines, item, root_branches, "  ", emitted_branch_keys)
            rendered = render_control_statement(item)
            if rendered:
                lines.append(f"{item_indent}control: {rendered}")
        elif kind == "return":
            item_indent = branch_indent_for_item(lines, item, root_branches, "  ", emitted_branch_keys)
            ret = render_return_expr(item.get("snippet") or "", float_names=float_names, return_type=return_type)
            if ret:
                lines.append(f"{item_indent}return: {ret}")
    append_missing_fallthrough_return(lines, feats, return_type, float_names)
    body = "\n".join(lines)
    op_block = operator_semantics_block(body)
    if op_block:
        body = op_block + "\n\n" + body
    if "prefix_scan(" in body:
        return PREFIX_SCAN_OPERATOR_DEFINITION + "\n\n" + body
    return body


def is_malformed_semantic_line(line: str) -> bool:
    text = str(line or "").strip()
    if not text:
        return True
    if re.search(r"[\+\-\*/&|^]$", text):
        return True
    balance = []
    quote = ""
    escaped = False
    for ch in text:
        if quote:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                quote = ""
            balance.append(" ")
        elif ch in {"'", '"'}:
            quote = ch
            balance.append(" ")
        else:
            balance.append(ch)
    balance_text = "".join(balance)
    if balance_text.count("(") > balance_text.count(")") and not text.endswith("..."):
        return True
    return False


def render_bootstrap(signature: str, route: dict[str, str], feats: dict[str, Any]) -> str:
    lines = [f"{signature.rstrip(';')}:"]
    float_names = floating_symbol_names(feats)
    pattern = route.get("pattern") or "generic"
    features = set(feats.get("features") or [])
    # Keep this code-like.  Pattern is emitted as a label, not as SVE advice.
    lines.append(f"  pattern = {pattern}")
    subs = semantic_subpatterns(features)
    if subs:
        lines.append("  subpatterns = [" + ", ".join(subs[:8]) + "]")
    deps = feats.get("same_array_dependencies") or []
    if deps:
        lines.append("  dependency:")
        for dep in deps[:4]:
            dep_snip = trim_statement(str(dep.get("snippet") or ""), 180)
            lines.append(
                f"    {dep.get('base')}[{dep.get('write_index')}] depends_on {dep.get('base')}[{dep.get('read_index')}]"
            )
            if dep_snip:
                lines.append(f"    step: {dep_snip}")
    rw = feats.get("read_write") or []
    if rw:
        lines.append("  dataflow:")
        for item in rw[:24]:
            rendered = render_assignment_line(item, float_names=float_names)
            if is_malformed_semantic_line(rendered):
                continue
            lines.append("    " + rendered)
            writes = ", ".join(f"{x['base']}[{x['index']}]" for x in item.get("writes") or [])
            reads = ", ".join(f"{x['base']}[{x['index']}]" for x in item.get("reads") or [])
            if writes or reads:
                lines.append(f"      write_refs = [{writes or 'scalar'}]; read_refs = [{reads or 'scalar/constants'}]")
    elif feats.get("assignments"):
        lines.append("  body:")
        for item in (feats.get("assignments") or [])[:10]:
            lines.append("    " + trim_statement(item.get("snippet") or ""))

    if feats.get("returns"):
        lines.append("  returns:")
        for ret in feats.get("returns")[:4]:
            ret_text = ret.get("snippet") if isinstance(ret, dict) else ret
            lines.append("    " + trim_statement(ret_text or ""))
    calls = [c.get("name") for c in feats.get("calls") or [] if c.get("name")]
    if calls:
        uniq = []
        for c in calls:
            if c not in uniq:
                uniq.append(c)
        lines.append("  calls = [" + ", ".join(uniq[:12]) + "]")
    return "\n".join(lines)


def process_records(
    problem_file: Path,
    out_dir: Path,
    manifest: list[dict[str, Any]],
    ast_meta: list[dict[str, Any]],
    *,
    render_style: str = "legacy",
) -> list[dict[str, Any]]:
    rows_by_idx = {i: row for i, row in enumerate(load_jsonl(problem_file))}
    manifest_by_task = {m.get("task_id"): m for m in manifest}
    ast_dir = out_dir / "remote_ast" / "ast"
    records: list[dict[str, Any]] = []
    for meta in ast_meta:
        task_id = str(meta.get("task_id") or "")
        base = dict(manifest_by_task.get(task_id) or {})
        base.update(meta)
        row = rows_by_idx.get(int(base.get("row_index") or 0), {})
        source_file = base.get("source_file")
        source_text = ""
        if source_file:
            src_path = out_dir / "sources" / str(source_file)
            if src_path.exists():
                source_text = src_path.read_text(encoding="utf-8", errors="replace")
        signature = display_signature(row, str(row.get("entrypoint_simd") or base.get("function") or ""), "")
        ast_ok = bool(base.get("ast_ok"))
        result = {
            "task_id": task_id,
            "row_index": base.get("row_index"),
            "benchmark": base.get("benchmark"),
            "source_label": base.get("source_label"),
            "function": base.get("function"),
            "ast_ok": ast_ok,
            "ast_status": base.get("ast_status"),
            "clang_rc": base.get("clang_rc"),
            "clang_stderr_tail": base.get("clang_stderr_tail", ""),
        }
        if not ast_ok:
            result.update(
                {
                    "route_style": "pseudocode",
                    "route_pattern": "ast_unavailable",
                    "route_reason": "clang AST was unavailable; fall back to legacy pseudocode.",
                    "features": [],
                    "bootstrap_pseudocode": "",
                }
            )
            records.append(result)
            continue
        ast_path = ast_dir / str(base.get("ast_file") or "")
        try:
            ast = load_ast_json(ast_path)
            fn = find_function(ast, str(base.get("function") or ""))
            if not fn:
                raise ValueError("target function not found in AST")
            header = function_header_from_source(source_text, str(base.get("function") or ""))
            if not signature:
                signature = display_signature(row, str(base.get("function") or ""), header)
            feats = extract_ast_features(fn, source_text)
            feature_set = set(feats.get("features") or [])
            route = route_from_features(feature_set, str(feats.get("semantic_text") or source_text))
            if render_style == "unified_dataflow":
                route_style = "unified_dataflow_pseudocode"
                bootstrap = render_unified_dataflow_bootstrap(signature, feats)
            elif render_style == "RSB":
                route_style = "RSB_pseudocode"
                bootstrap = render_RSB_bootstrap(signature, feats)
            else:
                route_style = route.get("style")
                bootstrap = render_bootstrap(signature, route, feats)
            route_pattern = route.get("pattern")
            if render_style == "RSB" and "prefix_scan(" in bootstrap:
                route_pattern = "scan_or_prefix_monoid"
            result.update(
                {
                    "route_style": route_style,
                    "legacy_route_style": route.get("style"),
                    "renderer": render_style,
                    "route_pattern": route_pattern,
                    "route_reason": route.get("reason"),
                    "features": sorted(feature_set),
                    "bootstrap_pseudocode": bootstrap,
                    "ast_summary": {
                        "loop_count": len(feats.get("loops") or []),
                        "assignment_count": len(feats.get("assignments") or []),
                        "call_count": len(feats.get("calls") or []),
                        "same_array_dependency_count": len(feats.get("same_array_dependencies") or []),
                    },
                    "read_write": expand_simple_cast_macros_in_json(feats.get("read_write") or []),
                    "same_array_dependencies": expand_simple_cast_macros_in_json(feats.get("same_array_dependencies") or []),
                }
            )
        except Exception as exc:
            result.update(
                {
                    "route_style": "pseudocode",
                    "route_pattern": "ast_parse_failed",
                    "route_reason": f"AST parse failed: {exc}",
                    "features": [],
                    "bootstrap_pseudocode": "",
                }
            )
        records.append(result)
    return records


def write_summary(path: Path, records: list[dict[str, Any]]) -> None:
    style_counts = Counter(str(r.get("route_style") or "") for r in records)
    pattern_counts = Counter(str(r.get("route_pattern") or "") for r in records)
    ast_ok = sum(1 for r in records if r.get("ast_ok"))
    lines = [
        "# AST Semantic Bootstrap Summary",
        "",
        f"- total: {len(records)}",
        f"- ast_ok: {ast_ok}/{len(records)}",
        "",
        "## Route Styles",
        "",
    ]
    for key, val in style_counts.most_common():
        lines.append(f"- {key}: {val}")
    lines += ["", "## Patterns", ""]
    for key, val in pattern_counts.most_common():
        lines.append(f"- {key}: {val}")
    lines += ["", "## Samples", ""]
    for r in records[:20]:
        lines.append(f"### {r.get('task_id')}")
        lines.append("")
        lines.append(f"- route: {r.get('route_style')} / {r.get('route_pattern')}")
        lines.append(f"- features: {', '.join(r.get('features') or [])}")
        bp = str(r.get("bootstrap_pseudocode") or "").strip()
        if bp:
            lines.append("")
            lines.append("```text")
            lines.append(bp)
            lines.append("```")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem-file", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--remote-user", default=DEFAULT_REMOTE_USER)
    ap.add_argument("--remote-host", default=DEFAULT_REMOTE_HOST)
    ap.add_argument("--remote-port", type=int, default=22)
    ap.add_argument("--remote-ssh-key", default=DEFAULT_REMOTE_KEY)
    ap.add_argument("--remote-no-strict-hostkey", action="store_true", default=True)
    ap.add_argument("--remote-tmp-root", default=DEFAULT_REMOTE_TMP_ROOT)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument(
        "--render-style",
        choices=["legacy", "unified_dataflow", "RSB"],
        default="legacy",
        help="legacy keeps the previous routed/pattern renderer; unified_dataflow emits one read/write/index pseudocode form; RSB emits loop/stage-level routed dataflow.",
    )
    args = ap.parse_args()

    problem_file = Path(args.problem_file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(problem_file)
    manifest = prepare_sources(rows, problem_file, out_dir, limit=int(args.limit or 0))
    ast_meta = run_remote_clang_ast(args, out_dir)
    records = process_records(problem_file, out_dir, manifest, ast_meta, render_style=str(args.render_style))
    out_jsonl = out_dir / "ast_semantic_bootstrap.jsonl"
    write_jsonl(out_jsonl, records)
    write_summary(out_dir / "summary.md", records)
    print(f"[OK] wrote {out_jsonl}")
    print(f"[OK] wrote {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
