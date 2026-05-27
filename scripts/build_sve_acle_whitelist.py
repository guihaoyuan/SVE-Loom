#!/usr/bin/env python3
"""Build a compact Arm SVE ACLE whitelist from Clang's arm_sve.h.

The generated JSON is the schema consumed by the local NameFix / ShapeFix
repair passes:

  names:      valid intrinsic names
  intrinsics: name -> [{ret, args}, ...]
  meta:       source and extraction statistics

This script intentionally keeps the representation lightweight. It extracts
the public ACLE wrapper declarations from arm_sve.h, optionally removes blocks
guarded by target extensions that are not enabled by a baseline SVE compiler
configuration, and records overloads as multiple signatures under one name.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Iterable


DEFAULT_EXCLUDED_FEATURES = (
    "__ARM_FEATURE_SVE2",
    "__ARM_FEATURE_SVE_BF16",
    "__ARM_FEATURE_BF16",
    "__ARM_FEATURE_SVE_MATMUL",
    "__ARM_FEATURE_SME",
)


FUNC_DECL_RE = re.compile(
    r"^(?P<ret>[A-Za-z_][A-Za-z0-9_:<>\s\*&]*?)\s+"
    r"(?P<name>sv[A-Za-z0-9_]+)\s*"
    r"\((?P<args>[^;{}]*)\)\s*(?:;|\{)"
)

MACRO_RE = re.compile(r"^\s*#\s*define\s+(?P<name>sv[A-Za-z0-9_]+)\b")


def find_arm_sve_header(explicit: str = "") -> Path:
    if explicit:
        p = Path(explicit).expanduser()
        if not p.is_file():
            raise SystemExit(f"arm_sve.h not found: {p}")
        return p

    candidates = [
        "/usr/lib64/clang/15.0.7/include/arm_sve.h",
        "/usr/lib/clang/15.0.7/include/arm_sve.h",
        "/usr/lib/llvm-15/lib/clang/15.0.7/include/arm_sve.h",
    ]
    for c in candidates:
        p = Path(c)
        if p.is_file():
            return p

    for exe in ("clang-15", "clang"):
        if shutil.which(exe):
            try:
                rd = subprocess.check_output([exe, "-print-resource-dir"], text=True).strip()
            except Exception:
                continue
            p = Path(rd) / "include" / "arm_sve.h"
            if p.is_file():
                return p

    raise SystemExit("Could not find arm_sve.h. Pass --header explicitly.")


def _line_is_excluded_if(line: str, excluded_features: Iterable[str]) -> bool:
    stripped = line.strip()
    if not stripped.startswith("#if") and not stripped.startswith("#elif"):
        return False
    return any(feat in stripped for feat in excluded_features)


def filter_extension_blocks(lines: list[str], excluded_features: Iterable[str]) -> tuple[list[str], dict]:
    """Remove preprocessor blocks guarded by excluded Arm feature macros.

    The parser is intentionally conservative: once an excluded #if is entered,
    all nested lines are skipped until the matching #endif. This matches the
    intended use here, where we want a baseline SVE API surface rather than SVE2
    or optional extension APIs.
    """
    out: list[str] = []
    stack: list[bool] = []
    removed = 0
    blocks = 0

    for line in lines:
        is_if = line.lstrip().startswith("#if")
        is_endif = line.lstrip().startswith("#endif")

        if is_if:
            parent_skip = any(stack)
            this_skip = parent_skip or _line_is_excluded_if(line, excluded_features)
            if this_skip and not parent_skip:
                blocks += 1
            stack.append(this_skip)
            if this_skip:
                removed += 1
                continue
            out.append(line)
            continue

        if is_endif:
            skip = stack.pop() if stack else False
            if skip:
                removed += 1
                continue
            out.append(line)
            continue

        if any(stack):
            removed += 1
            continue
        out.append(line)

    return out, {
        "lines_total": len(lines),
        "lines_removed": removed,
        "excluded_if_blocks": blocks,
    }


def normalize_type(s: str) -> str:
    s = re.sub(r"\b(?:const|volatile|restrict|__restrict__|__restrict)\b", lambda m: m.group(0), s.strip())
    s = re.sub(r"\s+", "", s)
    return s


def split_args(arg_str: str) -> list[str]:
    arg_str = arg_str.strip()
    if not arg_str or arg_str == "void":
        return []

    args: list[str] = []
    cur: list[str] = []
    depth = 0
    for ch in arg_str:
        if ch == "," and depth == 0:
            args.append("".join(cur).strip())
            cur = []
            continue
        cur.append(ch)
        if ch in "([{<":
            depth += 1
        elif ch in ")]}>":
            depth = max(0, depth - 1)
    if cur:
        args.append("".join(cur).strip())
    return args


def strip_param_name(arg: str) -> str:
    arg = arg.strip()
    if not arg:
        return ""
    # Remove default parameter names from forms like "svint8_t op" while
    # preserving pointer stars attached to the type.
    m = re.match(r"^(?P<type>.+?)(?:\s+[A-Za-z_][A-Za-z0-9_]*)$", arg)
    if m and "*" not in m.group("type").split()[-1]:
        return m.group("type").strip()
    return arg


def extract_whitelist(text: str) -> tuple[dict[str, list[dict]], set[str], int]:
    intrinsics: dict[str, list[dict]] = {}
    names: set[str] = set()
    macro_only = 0

    pending_attr = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue

        mm = MACRO_RE.match(line)
        if mm:
            name = mm.group("name")
            names.add(name)
            if name not in intrinsics:
                macro_only += 1
            continue

        if "__clang_arm_builtin_alias" in line or line.startswith("__ai ") or line.startswith("__aio "):
            pending_attr = True
            line = re.sub(r"^__ai[o]?\s+", "", line).strip()
            if "__clang_arm_builtin_alias" in line:
                continue

        if pending_attr or line.startswith("__ai ") or line.startswith("__aio "):
            line = re.sub(r"^__ai[o]?\s+", "", line).strip()
            m = FUNC_DECL_RE.match(line)
            if not m:
                pending_attr = False
                continue
            ret = normalize_type(m.group("ret"))
            name = m.group("name")
            args = [normalize_type(strip_param_name(a)) for a in split_args(m.group("args"))]
            sig = {"ret": ret, "args": args}
            names.add(name)
            bucket = intrinsics.setdefault(name, [])
            if sig not in bucket:
                bucket.append(sig)
            pending_attr = False

    return intrinsics, names, macro_only


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--header", default="", help="Path to Clang arm_sve.h; auto-detected if omitted.")
    ap.add_argument("--output", required=True, help="Output whitelist JSON path.")
    ap.add_argument(
        "--no-filter-extensions",
        action="store_true",
        help="Do not remove optional SVE2/BF16/MATMUL/SME feature blocks.",
    )
    ap.add_argument(
        "--removed-names-output",
        default="",
        help="Optional text file listing names removed by extension filtering.",
    )
    args = ap.parse_args()

    header = find_arm_sve_header(args.header)
    raw_lines = header.read_text(encoding="utf-8", errors="replace").splitlines()
    raw_intrinsics, raw_names, _ = extract_whitelist("\n".join(raw_lines))

    filter_stats = {"lines_total": len(raw_lines), "lines_removed": 0, "excluded_if_blocks": 0}
    lines = raw_lines
    if not args.no_filter_extensions:
        lines, filter_stats = filter_extension_blocks(raw_lines, DEFAULT_EXCLUDED_FEATURES)

    intrinsics, names, macro_seen = extract_whitelist("\n".join(lines))
    removed_names = sorted(raw_names - names)
    for name in names:
        intrinsics.setdefault(name, [])

    obj = {
        "names": sorted(names),
        "intrinsics": {k: intrinsics[k] for k in sorted(intrinsics)},
        "meta": {
            "header": str(header),
            "count": len(names),
            "count_with_sigs": len(intrinsics),
            "count_macro_names": macro_seen,
            "filtered_extensions": not args.no_filter_extensions,
            "excluded_features": [] if args.no_filter_extensions else list(DEFAULT_EXCLUDED_FEATURES),
            "filter_stats": filter_stats,
            "removed_count": len(removed_names),
            "note": "best-effort extractor from Clang arm_sve.h; records names plus return/argument signatures for local NameFix/ShapeFix.",
        },
    }

    out = Path(args.output).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if args.removed_names_output:
        rpath = Path(args.removed_names_output).expanduser()
        rpath.parent.mkdir(parents=True, exist_ok=True)
        rpath.write_text("\n".join(removed_names) + ("\n" if removed_names else ""), encoding="utf-8")

    print(f"wrote {out}")
    print(f"names={len(names)} signatures={len(intrinsics)} removed={len(removed_names)}")


if __name__ == "__main__":
    main()
