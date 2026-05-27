#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import csv
import difflib
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from tree_sitter import Language, Node, Parser
import tree_sitter_cpp


DEFAULT_TRAIN = Path(
    "/home/user/simdbench_full/results/generated/"
    "exactcap5_vecintrin_imagecap5_p6vxttargeted30_plus_taskexactgap30/"
    "direct_codegen_train.exactcap5_vecintrin_imagecap5_p6vxttargeted30_plus_taskexactgap30.jsonl"
)
DEFAULT_DEV = Path(
    "/home/user/simdbench_full/results/generated/"
    "exactcap5_vecintrin_imagecap5_p6vxttargeted30_plus_taskexactgap30/"
    "direct_codegen_dev.exactcap5_vecintrin_imagecap5_p6vxttargeted30_plus_taskexactgap30.jsonl"
)
DEFAULT_SIMDBENCH = Path("/home/user/simdbench_full/data/simdbench_sve.jsonl")
DEFAULT_ARM_LOOPS = Path(
    "/home/user/selective_repo/results/generated/contextualized_benchmarks/"
    "arm_simd_loops/arm_simd_loops_contextualized.problem.jsonl"
)
DEFAULT_OUT = Path("/home/user/selective_repo/results/nonoverlap_audit_full_core_vs_simdbench_armloops")

CPP_LANGUAGE = Language(tree_sitter_cpp.language())
PARSER = Parser(CPP_LANGUAGE)

CONTROL_NAMES = {"if", "for", "while", "switch", "return", "sizeof", "alignof", "catch"}
GENERIC_FUNCTION_NAMES = {
    "add", "compare", "search", "solve", "solution", "test", "func", "f", "run", "process",
    "compute", "count", "sum", "max", "min", "digits",
}
TYPE_WORDS = {
    "void", "bool", "char", "short", "int", "long", "float", "double", "size_t",
    "uint8_t", "uint16_t", "uint32_t", "uint64_t", "int8_t", "int16_t", "int32_t", "int64_t",
    "std::vector", "std::string", "string", "vector",
}
MATH_CALLS = {"fabs", "fabsf", "sqrt", "sqrtf", "sin", "sinf", "cos", "cosf", "tanh", "tanhf", "exp", "expf", "floor", "floorf", "ceil", "ceilf", "round", "roundf"}
OP_TOKENS = {"+", "-", "*", "/", "%", "<<", ">>", "&", "|", "^", "&&", "||", "==", "!=", "<", ">", "<=", ">=", "=", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>=", "!", "~", "?", ":"}


@dataclass
class Record:
    dataset: str
    split: str
    row_index: int
    task_id: str
    source_type: str
    source_name: str
    template_family_id: str
    prompt: str
    scalar_code: str
    response_code: str
    signature: str
    raw: dict[str, Any]

    @property
    def uid(self) -> str:
        return f"{self.dataset}:{self.split}:{self.task_id or self.row_index}"


@dataclass
class CodeUnit:
    record: Record
    kind: str
    code: str

    @property
    def uid(self) -> str:
        return f"{self.record.uid}:{self.kind}"


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def h64(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "little")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def strip_comments_and_strings(code: str, *, keep_string_token: bool = True) -> str:
    code = re.sub(r"/\*.*?\*/", " ", code, flags=re.S)
    code = re.sub(r"//.*", " ", code)
    if keep_string_token:
        code = re.sub(r'"(?:\\.|[^"\\])*"', " STR ", code)
        code = re.sub(r"'(?:\\.|[^'\\])*'", " CHR ", code)
    return code


def strip_includes(code: str) -> str:
    return "\n".join(line for line in (code or "").splitlines() if not line.lstrip().startswith("#include"))


def tokenize(text: str) -> list[str]:
    text = strip_comments_and_strings(text)
    return re.findall(r"[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*|\d+\.\d+|\d+|==|!=|<=|>=|<<=|>>=|<<|>>|\+\+|--|&&|\|\||->|[{}()\[\];,.*&+\-/%<>!=|^?:~]", text)


def canonical_exact(tokens: list[str]) -> str:
    return " ".join(t.lower() for t in tokens)


def normalized_tokens(tokens: Iterable[str], *, drop_names: bool, const_to_num: bool) -> list[str]:
    out = []
    for t in tokens:
        if const_to_num and re.fullmatch(r"\d+\.\d+|\d+", t):
            out.append("NUM")
        elif drop_names and re.fullmatch(r"[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*", t) and t not in TYPE_WORDS and t not in CONTROL_NAMES:
            out.append("ID")
        else:
            out.append(t.lower())
    return out


def shingles(tokens: list[str], n: int) -> set[str]:
    if len(tokens) < n:
        return set(tokens)
    return {" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / (len(a) + len(b) - len(a & b))


def containment(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def simhash(sh: set[str]) -> int:
    weights = [0] * 64
    for s in sh:
        x = h64(s)
        for i in range(64):
            weights[i] += 1 if (x >> i) & 1 else -1
    out = 0
    for i, w in enumerate(weights):
        if w > 0:
            out |= 1 << i
    return out


MINHASH_SEEDS = [h64(f"seed_{i}") for i in range(64)]


def minhash(sh: set[str]) -> tuple[int, ...]:
    if not sh:
        return tuple([0] * len(MINHASH_SEEDS))
    vals = [2**64 - 1] * len(MINHASH_SEEDS)
    for s in sh:
        x = h64(s)
        for i, seed in enumerate(MINHASH_SEEDS):
            y = ((x ^ seed) * 11400714819323198485) & ((1 << 64) - 1)
            if y < vals[i]:
                vals[i] = y
    return tuple(vals)


def minhash_similarity(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    if not a or not b:
        return 0.0
    return sum(1 for x, y in zip(a, b) if x == y) / min(len(a), len(b))


def hamming64(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def edit_similarity(a_tokens: list[str], b_tokens: list[str]) -> float:
    # Token-level edit-like ratio; enough for high-similarity review candidates.
    return difflib.SequenceMatcher(a=a_tokens, b=b_tokens, autojunk=False).ratio()


def extract_signature(row: dict[str, Any]) -> str:
    for k in ["public_entry_signature", "target_signature", "harness_extern_decl"]:
        if row.get(k):
            return re.sub(r"\s+", " ", str(row[k]).rstrip(";")).strip()
    for k in ["response", "generated_sve_code_v22", "serial_c_code", "solution_scalar"]:
        code = str(row.get(k) or "")
        m = re.search(r"(?:^|\n)\s*([A-Za-z_][\w:<>,\s\*&]+?\s+[A-Za-z_]\w*\s*\([^;{}]*\))\s*\{", code, re.S)
        if m:
            return re.sub(r"\s+", " ", m.group(1)).strip()
    return ""


def load_train_dev(train: Path, dev: Path) -> list[Record]:
    out = []
    for split, path in [("train", train), ("dev", dev)]:
        for i, row in enumerate(read_jsonl(path), 1):
            prompt = str(row.get("prompt") or row.get("nl_description_ds_r1") or "")
            scalar = str(row.get("serial_c_code") or "")
            response = str(row.get("response") or row.get("generated_sve_code_v22") or "")
            sig = extract_signature(row)
            out.append(Record("core", split, i, str(row.get("task_id") or f"{split}_{i}"), str(row.get("source_type") or ""), str(row.get("source_name") or ""), str(row.get("template_family_id") or ""), prompt, scalar, response, sig, row))
    return out


def load_simdbench(path: Path) -> list[Record]:
    out = []
    for i, row in enumerate(read_jsonl(path), 1):
        if row.get("intrinsic") and str(row.get("intrinsic")).upper() != "SVE":
            continue
        scalar = str(row.get("solution_scalar") or "")
        out.append(Record("simdbench_sve", "benchmark", i, str(row.get("task_id") or f"simdbench_{i}"), "benchmark", "simdbench_sve", str(row.get("type") or ""), str(row.get("prompt") or ""), scalar, "", extract_signature({"serial_c_code": scalar}), row))
    return out


def load_arm_loops(path: Path) -> list[Record]:
    out = []
    for i, row in enumerate(read_jsonl(path), 1):
        scalar = str(row.get("serial_c_code") or "")
        out.append(Record("arm_simd_loops", "benchmark", i, str(row.get("task_id") or f"arm_loop_{i}"), str(row.get("source_type") or "benchmark"), str(row.get("source_name") or "arm_simd_loops"), str(row.get("benchmark_category") or ""), str(row.get("prompt") or row.get("nl_description_ds_r1") or ""), scalar, "", str(row.get("target_signature") or extract_signature({"serial_c_code": scalar})), row))
    return out


def code_units(records: list[Record], *, include_response: bool) -> list[CodeUnit]:
    out = []
    for r in records:
        if r.scalar_code.strip():
            out.append(CodeUnit(r, "scalar", r.scalar_code))
        if include_response and r.response_code.strip():
            out.append(CodeUnit(r, "response", r.response_code))
    return out


def node_text(node: Node, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")


def parse_cpp(code: str):
    return PARSER.parse(code.encode("utf-8", errors="ignore"))


def iter_nodes(node: Node):
    yield node
    for ch in node.children:
        yield from iter_nodes(ch)


def function_defs(root: Node) -> list[Node]:
    return [n for n in iter_nodes(root) if n.type == "function_definition"]


def canonical_ast(node: Node, src: bytes, *, typed: bool) -> list[str]:
    parts: list[str] = []
    def rec(n: Node) -> None:
        if not n.is_named:
            txt = node_text(n, src)
            if txt in OP_TOKENS or txt in {"[", "]", "(", ")", "{", "}", ",", ";"}:
                parts.append(txt)
            return
        t = n.type
        if t in {"identifier", "field_identifier", "namespace_identifier"}:
            parts.append("ID")
            return
        if t in {"number_literal", "char_literal"}:
            parts.append("NUM")
            return
        if t in {"string_literal", "raw_string_literal"}:
            parts.append("STR")
            return
        if "type" in t or t in {"primitive_type", "sized_type_specifier"}:
            parts.append(("TYPE:" + node_text(n, src).strip()) if typed else "TYPE")
            return
        parts.append(t)
        for ch in n.children:
            rec(ch)
        parts.append("/" + t)
    rec(node)
    return parts


def canon_expr(node: Node | None, src: bytes, *, keep_names: bool = False) -> str:
    if node is None:
        return ""
    if not node.is_named:
        txt = node_text(node, src)
        return txt if txt in OP_TOKENS else ""
    t = node.type
    if t in {"identifier", "field_identifier", "namespace_identifier"}:
        return node_text(node, src).strip() if keep_names else "ID"
    if t in {"number_literal", "char_literal"}:
        return "NUM"
    if t in {"string_literal", "raw_string_literal"}:
        return "STR"
    if "type" in t or t in {"primitive_type", "sized_type_specifier"}:
        return "TYPE"
    parts = [t]
    for ch in node.children:
        s = canon_expr(ch, src, keep_names=keep_names)
        if s:
            parts.append(s)
    return "(" + " ".join(parts) + ")"


def child_field(node: Node, name: str) -> Node | None:
    try:
        return node.child_by_field_name(name)
    except Exception:
        return None


def ast_features_for_code(code: str) -> dict[str, Any]:
    src = strip_includes(code).encode("utf-8", errors="ignore")
    tree = PARSER.parse(src)
    root = tree.root_node
    funcs = function_defs(root)
    target_nodes = funcs if funcs else [root]
    typed_parts: list[str] = []
    untyped_parts: list[str] = []
    loops: list[str] = []
    arrays: list[str] = []
    assignments: list[str] = []
    exprs: list[str] = []
    calls: list[str] = []
    returns: list[str] = []
    conditions: list[str] = []
    writes: list[str] = []
    reads: list[str] = []
    types: list[str] = []
    max_depth = 0

    def walk(n: Node, loop_depth: int = 0, in_lhs: bool = False) -> None:
        nonlocal max_depth
        if n.type in {"for_statement", "while_statement", "do_statement"}:
            max_depth = max(max_depth, loop_depth + 1)
            cond = child_field(n, "condition")
            upd = child_field(n, "update")
            init = child_field(n, "initializer")
            loops.append(f"{n.type}:init={canon_expr(init, src)}:cond={canon_expr(cond, src)}:upd={canon_expr(upd, src)}")
            for ch in n.children:
                walk(ch, loop_depth + 1, in_lhs=False)
            return
        if n.type == "if_statement":
            conditions.append(canon_expr(child_field(n, "condition"), src))
        if n.type == "assignment_expression":
            left = child_field(n, "left")
            right = child_field(n, "right")
            op = "="
            for ch in n.children:
                if not ch.is_named and node_text(ch, src) in {"=", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>="}:
                    op = node_text(ch, src)
                    break
            lshape = canon_expr(left, src)
            rshape = canon_expr(right, src)
            assignments.append(f"{lshape}{op}{rshape}")
            writes.append(lshape)
            if right is not None:
                for sub in iter_nodes(right):
                    if sub.type == "subscript_expression":
                        reads.append(canon_expr(sub, src))
        if n.type == "subscript_expression":
            shape = canon_expr(n, src)
            arrays.append(shape)
        if n.type == "call_expression":
            fn = child_field(n, "function")
            fname = canon_expr(fn, src, keep_names=True)
            calls.append(fname)
        if n.type == "return_statement":
            returns.append(canon_expr(n, src))
        if n.type in {"binary_expression", "unary_expression", "call_expression", "subscript_expression", "conditional_expression"}:
            exprs.append(canon_expr(n, src))
        if "type" in n.type or n.type in {"primitive_type", "sized_type_specifier"}:
            txt = node_text(n, src).strip()
            if txt:
                types.append(re.sub(r"\s+", " ", txt))
        for ch in n.children:
            walk(ch, loop_depth, in_lhs=False)

    for fn in target_nodes:
        typed_parts.extend(canonical_ast(fn, src, typed=True))
        untyped_parts.extend(canonical_ast(fn, src, typed=False))
        walk(fn)

    def h(items: list[str]) -> str:
        return sha("\n".join(items))

    call_class = []
    for c in calls:
        if "sv" in c:
            call_class.append(re.sub(r"_[suf](8|16|32|64).*", "_T", c))
        elif any(m in c for m in MATH_CALLS):
            call_class.append("math:" + c)
        else:
            call_class.append("call")
    return {
        "parse_error": root.has_error,
        "function_count": len(funcs),
        "typed_ast_hash": h(typed_parts),
        "untyped_ast_hash": h(untyped_parts),
        "loop_nest_hash": h(sorted(loops)),
        "array_access_hash": h(sorted(arrays)),
        "assignment_tree_hash": h(sorted(assignments)),
        "expression_tree_hash": h(sorted(exprs)),
        "call_graph_hash": h(sorted(call_class)),
        "composite_hash": h([
            "loops:" + "|".join(sorted(loops)),
            "arrays:" + "|".join(sorted(arrays)),
            "assign:" + "|".join(sorted(assignments)),
            "calls:" + "|".join(sorted(call_class)),
            "returns:" + "|".join(sorted(returns)),
        ]),
        "loop_nest": sorted(loops)[:40],
        "array_access": sorted(set(arrays))[:40],
        "assignment_tree": sorted(set(assignments))[:40],
        "expression_tree": sorted(set(exprs))[:40],
        "call_graph": sorted(set(call_class))[:40],
        "returns": sorted(set(returns))[:20],
        "conditions": sorted(set(conditions))[:30],
        "writes": sorted(set(writes))[:30],
        "reads": sorted(set(reads))[:30],
        "types": sorted(set(types))[:40],
        "max_loop_depth": max_depth,
    }


def signature_shape(sig: str) -> dict[str, Any]:
    ret = ""
    params = ""
    if "(" in sig and ")" in sig:
        ret = sig[:sig.find("(")].strip()
        params = sig[sig.find("(") + 1:sig.rfind(")")]
    else:
        params = sig
    parts = [p.strip() for p in params.split(",") if p.strip() and p.strip() != "void"]
    shaped = []
    for p in parts:
        toks = tokenize(p)
        types = [t for t in toks if t in TYPE_WORDS or t in {"const"}]
        ptr = sum(1 for t in toks if t in {"*", "&"})
        shaped.append({"types": types, "ptr_level": ptr})
    return {"return": " ".join([t for t in tokenize(ret) if t in TYPE_WORDS]), "params": shaped, "arity": len(shaped)}


def layout_class(code: str, astf: dict[str, Any]) -> list[str]:
    c = code
    out = []
    if "svld2" in c or "svst2" in c or re.search(r"\b2\s*\*", c): out.append("interleaved_or_pair")
    if "svld3" in c or "svst3" in c or "svld4" in c or "svst4" in c: out.append("tuple_channel")
    if "gather" in c: out.append("gather")
    if "scatter" in c: out.append("scatter")
    if "." in c or "->" in c: out.append("aos_or_struct")
    if any("subscript_expression" in a and ("*" in a or "+" in a) for a in astf.get("array_access", [])): out.append("affine_index")
    if not out: out.append("contiguous_or_scalar")
    return sorted(set(out))


def semantic_pattern(code: str, sig: str, astf: dict[str, Any]) -> dict[str, Any]:
    c = strip_comments_and_strings(code)
    calls = astf.get("call_graph", [])
    ops = sorted(set(re.findall(r"\+=|-=|\*=|/=|%=|<<|>>|==|!=|<=|>=|&&|\|\||[+\-*/%&|^<>]", c)))
    red = []
    if re.search(r"\+=|svaddv|svmaxv|svminv", c): red.append("sum_or_reduce")
    if re.search(r"\*=|product", c, re.I): red.append("product")
    if re.search(r"\bwhile\s*\(", c): red.append("while_control")
    if "svcntp" in c or "svptest" in c: red.append("predicate_test")
    data_flow = []
    for key in ["svcvt", "svunpk", "svreinterpret", "svld1sb", "svld1ub", "svld1sh", "svld1uh", "svst1", "svaddv", "svlsr", "svasr"]:
        if key in c: data_flow.append(key)
    return {
        "signature_shape": signature_shape(sig),
        "loop_domain": {
            "max_depth": astf.get("max_loop_depth"),
            "loop_count": len(astf.get("loop_nest", [])),
            "while_count": c.count("while"),
            "for_count": c.count("for"),
        },
        "write_target_shape": astf.get("writes", []),
        "read_index_shape": astf.get("reads", []),
        "reduction_operator": sorted(set(red)),
        "predicate_condition": astf.get("conditions", []),
        "data_type_flow": sorted(set(data_flow + astf.get("types", [])))[:60],
        "memory_layout_class": layout_class(c, astf),
        "ops": ops,
        "math_calls": sorted(m for m in MATH_CALLS if re.search(rf"\b{re.escape(m)}\b", c)),
    }


def semantic_hash(pat: dict[str, Any], *, strict: bool) -> str:
    if strict:
        obj = pat
    else:
        obj = {
            "arity": pat["signature_shape"]["arity"],
            "loop_domain": pat["loop_domain"],
            "write_target_shape": pat["write_target_shape"],
            "read_index_shape": pat["read_index_shape"],
            "reduction_operator": pat["reduction_operator"],
            "memory_layout_class": pat["memory_layout_class"],
            "ops": pat["ops"],
        }
    return sha(json.dumps(obj, sort_keys=True, ensure_ascii=False))


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8") as f:
        f.write("\t".join(keys) + "\n")
        for r in rows:
            f.write("\t".join(str(r.get(k, "")).replace("\n", "\\n") for k in keys) + "\n")


def rec_meta(unit: CodeUnit) -> dict[str, Any]:
    r = unit.record
    return {
        "dataset": r.dataset,
        "split": r.split,
        "task_id": r.task_id,
        "kind": unit.kind,
        "source_type": r.source_type,
        "source_name": r.source_name,
        "family": r.template_family_id,
    }


def pair_row(level: str, bench: CodeUnit | Record, core: CodeUnit | Record, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    br = bench.record if isinstance(bench, CodeUnit) else bench
    cr = core.record if isinstance(core, CodeUnit) else core
    out = {
        "level": level,
        "benchmark": br.dataset,
        "bench_id": br.task_id,
        "bench_kind": bench.kind if isinstance(bench, CodeUnit) else "record",
        "core_split": cr.split,
        "core_id": cr.task_id,
        "core_kind": core.kind if isinstance(core, CodeUnit) else "record",
        "core_source_type": cr.source_type,
        "core_source_name": cr.source_name,
        "core_family": cr.template_family_id,
    }
    if extra:
        out.update(extra)
    return out


def channel_tokens_prompt(r: Record) -> list[str]:
    return tokenize(r.prompt)


def channel_tokens_code(unit: CodeUnit) -> list[str]:
    return tokenize(strip_includes(unit.code))


def text_token_audit(core: list[Record], benches: list[Record], out: Path) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    core_units = code_units(core, include_response=True)
    bench_units = code_units(benches, include_response=False)
    channels: list[tuple[str, list[Any], list[Any], Any]] = [
        ("prompt", core, benches, channel_tokens_prompt),
        ("scalar_code", [u for u in core_units if u.kind == "scalar"], bench_units, channel_tokens_code),
        ("response_code", [u for u in core_units if u.kind == "response"], bench_units, channel_tokens_code),
    ]
    summary: dict[str, Any] = {}
    all_matches: dict[str, list[dict[str, Any]]] = {}
    for name, core_items, bench_items, tok_fn in channels:
        # Prepare exact and similarity representations.
        core_data = {}
        exact_index: dict[str, list[Any]] = collections.defaultdict(list)
        inv: dict[str, set[str]] = collections.defaultdict(set)
        uid_to_item = {}
        for item in core_items:
            toks = tok_fn(item)
            exact = canonical_exact(toks)
            sh = shingles([t.lower() for t in toks], 5)
            uid = item.uid
            uid_to_item[uid] = item
            mh = minhash(sh)
            core_data[uid] = {
                "tokens": toks,
                "exact": exact,
                "shingles": sh,
                "simhash": simhash(sh),
                "minhash": mh,
            }
            if exact:
                exact_index[sha(exact)].append(item)
            for s in sh:
                inv[s].add(uid)
        exact_matches: list[dict[str, Any]] = []
        ngram_matches: list[dict[str, Any]] = []
        minhash_matches: list[dict[str, Any]] = []
        simhash_matches: list[dict[str, Any]] = []
        edit_matches: list[dict[str, Any]] = []
        for b in bench_items:
            btoks = tok_fn(b)
            bexact = canonical_exact(btoks)
            bsh = shingles([t.lower() for t in btoks], 5)
            bmh = minhash(bsh)
            bsim = simhash(bsh)
            for c in exact_index.get(sha(bexact), []):
                exact_matches.append(pair_row(f"{name}:normalized_exact", b, c, {"hash": sha(bexact)}))
            if len(bsh) < 8:
                continue
            cand = collections.Counter()
            for s in bsh:
                for uid in inv.get(s, ()):
                    cand[uid] += 1
            for uid, shared in cand.items():
                cd = core_data[uid]
                csh = cd["shingles"]
                if len(csh) < 8:
                    continue
                jac = jaccard(bsh, csh)
                cont = containment(bsh, csh)
                mh_sim = minhash_similarity(bmh, cd["minhash"])
                ham = hamming64(bsim, cd["simhash"])
                # Report each metric independently. Thresholds are intentionally strict.
                if jac >= 0.80 or cont >= 0.95:
                    ngram_matches.append(pair_row(f"{name}:token_ngram_overlap", b, uid_to_item[uid], {"jaccard": round(jac, 6), "containment": round(cont, 6), "shared_shingles": shared}))
                    if min(len(btoks), len(cd["tokens"])) <= 3000:
                        ed = edit_similarity([t.lower() for t in btoks], [t.lower() for t in cd["tokens"]])
                        if ed >= 0.88:
                            edit_matches.append(pair_row(f"{name}:edit_similarity", b, uid_to_item[uid], {"edit_ratio": round(ed, 6), "jaccard": round(jac, 6), "containment": round(cont, 6)}))
                if mh_sim >= 0.84:
                    minhash_matches.append(pair_row(f"{name}:minhash_similarity", b, uid_to_item[uid], {"minhash_similarity": round(mh_sim, 6), "jaccard": round(jac, 6)}))
                if ham <= 4 and max(len(bsh), len(csh)) >= 20:
                    simhash_matches.append(pair_row(f"{name}:simhash_similarity", b, uid_to_item[uid], {"hamming": ham, "jaccard": round(jac, 6)}))
        for suffix, rows in [
            ("exact", exact_matches),
            ("ngram", ngram_matches),
            ("minhash", minhash_matches),
            ("simhash", simhash_matches),
            ("edit", edit_matches),
        ]:
            rows.sort(key=lambda r: (r.get("benchmark", ""), r.get("bench_id", ""), r.get("core_id", "")))
            write_tsv(out / f"{name}_{suffix}_matches.tsv", rows)
            all_matches[f"{name}_{suffix}"] = rows
        summary[name] = {k: len(v) for k, v in all_matches.items() if k.startswith(name + "_")}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def build_ast_records(units: list[CodeUnit]) -> dict[str, dict[str, Any]]:
    out = {}
    for u in units:
        astf = ast_features_for_code(u.code)
        sem = semantic_pattern(u.code, u.record.signature, astf)
        out[u.uid] = {"unit": u, "ast": astf, "semantic": sem}
    return out


def exact_hash_matches(core_data: dict[str, dict[str, Any]], bench_data: dict[str, dict[str, Any]], key: str, level: str) -> list[dict[str, Any]]:
    idx: dict[str, list[CodeUnit]] = collections.defaultdict(list)
    for d in core_data.values():
        idx[d["ast"][key]].append(d["unit"])
    rows = []
    for d in bench_data.values():
        h = d["ast"][key]
        for cu in idx.get(h, []):
            rows.append(pair_row(level, d["unit"], cu, {"hash": h}))
    return rows


def ast_structural_audit(core: list[Record], benches: list[Record], out: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    out.mkdir(parents=True, exist_ok=True)
    core_units = code_units(core, include_response=True)
    bench_units = code_units(benches, include_response=False)
    core_data = build_ast_records(core_units)
    bench_data = build_ast_records(bench_units)
    match_specs = [
        ("typed_ast_hash", "typed_ast_exact"),
        ("untyped_ast_hash", "untyped_ast_exact"),
        ("loop_nest_hash", "loop_nest_exact"),
        ("array_access_hash", "array_access_exact"),
        ("assignment_tree_hash", "assignment_tree_exact"),
        ("expression_tree_hash", "expression_tree_exact"),
        ("call_graph_hash", "call_graph_exact"),
        ("composite_hash", "composite_ast_exact"),
    ]
    summary: dict[str, Any] = {}
    for key, level in match_specs:
        rows = exact_hash_matches(core_data, bench_data, key, level)
        # Empty hashes are not meaningful for narrow sub-signatures.
        rows = [r for r in rows if r.get("hash") != sha("")]
        write_tsv(out / f"{level}.tsv", rows)
        summary[level] = len(rows)
    # Write feature dumps for reproducibility.
    with (out / "core_ast_features.jsonl").open("w", encoding="utf-8") as f:
        for uid, d in core_data.items():
            f.write(json.dumps({**rec_meta(d["unit"]), "uid": uid, "ast": d["ast"]}, ensure_ascii=False) + "\n")
    with (out / "benchmark_ast_features.jsonl").open("w", encoding="utf-8") as f:
        for uid, d in bench_data.items():
            f.write(json.dumps({**rec_meta(d["unit"]), "uid": uid, "ast": d["ast"]}, ensure_ascii=False) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary, core_data, bench_data


def semantic_pattern_audit(core_data: dict[str, dict[str, Any]], bench_data: dict[str, dict[str, Any]], out: Path) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    levels = [
        ("strict", True),
        ("abstract", False),
    ]
    summary = {}
    for name, strict in levels:
        idx: dict[str, list[CodeUnit]] = collections.defaultdict(list)
        for d in core_data.values():
            idx[semantic_hash(d["semantic"], strict=strict)].append(d["unit"])
        rows = []
        for d in bench_data.values():
            h = semantic_hash(d["semantic"], strict=strict)
            for cu in idx.get(h, []):
                rows.append(pair_row(f"semantic_{name}_exact", d["unit"], cu, {"hash": h}))
        write_tsv(out / f"semantic_{name}_exact.tsv", rows)
        summary[f"semantic_{name}_exact"] = len(rows)
    # High-risk semantic = abstract semantic match plus same typed/untyped AST or composite hash.
    high = []
    abstract_idx: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for d in core_data.values():
        abstract_idx[semantic_hash(d["semantic"], strict=False)].append(d)
    for bd in bench_data.values():
        h = semantic_hash(bd["semantic"], strict=False)
        for cd in abstract_idx.get(h, []):
            same_bits = []
            for k in ["typed_ast_hash", "untyped_ast_hash", "composite_hash", "array_access_hash", "assignment_tree_hash"]:
                if bd["ast"].get(k) == cd["ast"].get(k):
                    same_bits.append(k)
            if same_bits:
                high.append(pair_row("semantic_pattern_high_risk", bd["unit"], cd["unit"], {"semantic_hash": h, "same_structural_components": ",".join(same_bits)}))
    write_tsv(out / "semantic_pattern_high_risk.tsv", high)
    summary["semantic_pattern_high_risk"] = len(high)
    with (out / "core_semantic_patterns.jsonl").open("w", encoding="utf-8") as f:
        for uid, d in core_data.items():
            f.write(json.dumps({**rec_meta(d["unit"]), "uid": uid, "semantic": d["semantic"], "strict_hash": semantic_hash(d["semantic"], strict=True), "abstract_hash": semantic_hash(d["semantic"], strict=False)}, ensure_ascii=False) + "\n")
    with (out / "benchmark_semantic_patterns.jsonl").open("w", encoding="utf-8") as f:
        for uid, d in bench_data.items():
            f.write(json.dumps({**rec_meta(d["unit"]), "uid": uid, "semantic": d["semantic"], "strict_hash": semantic_hash(d["semantic"], strict=True), "abstract_hash": semantic_hash(d["semantic"], strict=False)}, ensure_ascii=False) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def identifier_audit(core: list[Record], benches: list[Record], out: Path) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    def fnames(code: str) -> set[str]:
        tree = parse_cpp(strip_includes(code))
        src = strip_includes(code).encode("utf-8", errors="ignore")
        names = set()
        for fn in function_defs(tree.root_node):
            decl = child_field(fn, "declarator")
            for n in iter_nodes(decl) if decl else []:
                if n.type == "identifier":
                    nm = node_text(n, src)
                    if nm not in CONTROL_NAMES:
                        names.add(nm)
                        break
        return names
    bench_fn: dict[str, list[Record]] = collections.defaultdict(list)
    for b in benches:
        for n in fnames(b.scalar_code + "\n" + b.signature):
            bench_fn[n].append(b)
    rows = []
    for c in core:
        blob = "\n".join([c.task_id, c.prompt, c.scalar_code, c.response_code, c.signature])
        for n, bs in bench_fn.items():
            if re.search(rf"\b{re.escape(n)}\b", blob):
                for b in bs:
                    rows.append(pair_row("identifier_or_literal", b, c, {
                        "literal": n,
                        "generic_name": n in GENERIC_FUNCTION_NAMES,
                        "signature_shape_same": signature_shape(b.signature) == signature_shape(c.signature),
                    }))
    write_tsv(out / "identifier_or_literal_matches.tsv", rows)
    summary = {"identifier_or_literal_matches": len(rows), "non_generic_or_same_signature": sum(1 for r in rows if r.get("generic_name") != True or r.get("signature_shape_same"))}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    ap.add_argument("--dev", type=Path, default=DEFAULT_DEV)
    ap.add_argument("--simdbench", type=Path, default=DEFAULT_SIMDBENCH)
    ap.add_argument("--arm-loops", type=Path, default=DEFAULT_ARM_LOOPS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    core = load_train_dev(args.train, args.dev)
    benches = load_simdbench(args.simdbench) + load_arm_loops(args.arm_loops)
    args.out.mkdir(parents=True, exist_ok=True)
    text_summary = text_token_audit(core, benches, args.out / "text_token")
    ast_summary, core_ast, bench_ast = ast_structural_audit(core, benches, args.out / "ast_structural")
    sem_summary = semantic_pattern_audit(core_ast, bench_ast, args.out / "semantic_pattern")
    ident_summary = identifier_audit(core, benches, args.out / "identifier")
    summary = {
        "core_rows": len(core),
        "core_train_rows": sum(1 for r in core if r.split == "train"),
        "core_dev_rows": sum(1 for r in core if r.split == "dev"),
        "benchmark_rows": len(benches),
        "benchmark_counts": dict(collections.Counter(r.dataset for r in benches)),
        "text_token": text_summary,
        "ast_structural": ast_summary,
        "semantic_pattern": sem_summary,
        "identifier": ident_summary,
        "paths": {
            "text_token": str(args.out / "text_token"),
            "ast_structural": str(args.out / "ast_structural"),
            "semantic_pattern": str(args.out / "semantic_pattern"),
            "identifier": str(args.out / "identifier"),
        },
    }
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md = [
        "# Full Non-overlap Audit",
        "",
        f"- Core rows: {summary['core_rows']} ({summary['core_train_rows']} train, {summary['core_dev_rows']} dev)",
        f"- Benchmark rows: {summary['benchmark_rows']} {summary['benchmark_counts']}",
        "",
        "## Text / Token Audit",
        "```json",
        json.dumps(text_summary, indent=2, ensure_ascii=False),
        "```",
        "",
        "## AST Structural Audit",
        "```json",
        json.dumps(ast_summary, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Semantic Pattern Audit",
        "```json",
        json.dumps(sem_summary, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Identifier / Literal Audit",
        "```json",
        json.dumps(ident_summary, indent=2, ensure_ascii=False),
        "```",
    ]
    (args.out / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
