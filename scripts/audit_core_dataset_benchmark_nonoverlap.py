#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


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
DEFAULT_OUT = Path("/home/user/selective_repo/results/nonoverlap_audit_core_vs_simdbench_armloops")


KEYWORDS = {
    "alignas", "alignof", "and", "asm", "auto", "bool", "break", "case", "catch", "char", "class",
    "const", "constexpr", "continue", "default", "delete", "do", "double", "else", "enum", "extern",
    "false", "float", "for", "if", "inline", "int", "long", "namespace", "new", "noexcept", "nullptr",
    "operator", "or", "private", "protected", "public", "register", "reinterpret_cast", "return",
    "short", "signed", "sizeof", "static", "struct", "switch", "template", "this", "throw", "true",
    "try", "typedef", "typename", "uint8_t", "uint16_t", "uint32_t", "uint64_t", "int8_t", "int16_t",
    "int32_t", "int64_t", "size_t", "std", "unsigned", "using", "void", "volatile", "while",
}
CONTROL_NAMES = {"if", "for", "while", "switch", "return", "sizeof", "alignof"}
TYPE_TOKENS = {
    "bool", "char", "uint8_t", "int8_t", "uint16_t", "int16_t", "uint32_t", "int32_t",
    "uint64_t", "int64_t", "float", "double", "size_t", "void", "short", "long", "int",
    "unsigned", "signed",
}
MATH_CALLS = {"fabs", "fabsf", "sqrt", "sqrtf", "sin", "sinf", "cos", "cosf", "tanh", "tanhf", "exp", "expf", "floor", "floorf", "ceil", "ceilf", "round", "roundf"}
GENERIC_FUNCTION_NAMES = {
    "add", "compare", "search", "solve", "solution", "test", "func", "f", "run", "process",
    "compute", "count", "sum", "max", "min", "digits",
}


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
    code: str
    scalar_code: str
    response_code: str
    signature: str
    raw: dict[str, Any]

    @property
    def uid(self) -> str:
        return f"{self.dataset}:{self.split}:{self.task_id or self.row_index}"


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def compact_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def strip_comments_and_strings(code: str) -> str:
    code = re.sub(r"/\*.*?\*/", " ", code, flags=re.S)
    code = re.sub(r"//.*", " ", code)
    code = re.sub(r'"(?:\\.|[^"\\])*"', " STR ", code)
    code = re.sub(r"'(?:\\.|[^'\\])*'", " CHR ", code)
    return code


def strip_includes(code: str) -> str:
    return "\n".join(line for line in code.splitlines() if not line.lstrip().startswith("#include"))


def tokenize(text: str) -> list[str]:
    text = strip_comments_and_strings(text)
    return re.findall(r"[A-Za-z_]\w*|\d+\.\d+|\d+|==|!=|<=|>=|<<|>>|\+\+|--|&&|\|\||->|[{}()\[\];,.*&+\-/%<>!=|^?:]", text)


def normalize_tokens(tokens: Iterable[str], *, keep_calls: bool = False) -> list[str]:
    toks = list(tokens)
    out: list[str] = []
    for i, tok in enumerate(toks):
        if re.fullmatch(r"\d+\.\d+|\d+", tok):
            out.append("NUM")
        elif re.fullmatch(r"[A-Za-z_]\w*", tok):
            next_tok = toks[i + 1] if i + 1 < len(toks) else ""
            if tok in KEYWORDS:
                out.append(tok)
            elif keep_calls and next_tok == "(" and tok not in CONTROL_NAMES:
                out.append("CALL")
            else:
                out.append("ID")
        else:
            out.append(tok)
    return out


def normalized_text(text: str) -> str:
    return " ".join(normalize_tokens(tokenize(text), keep_calls=False))


def normalized_code(code: str) -> str:
    return " ".join(normalize_tokens(tokenize(strip_includes(code)), keep_calls=True))


def canonical_exact_text(text: str, *, code: bool = False) -> str:
    """Canonical form for exact duplicate checks.

    This intentionally does *not* replace identifiers or constants. Exact overlap
    should mean literal-equivalent text/code after comments/includes/whitespace,
    not merely the same abstract template.
    """
    if code:
        text = strip_includes(text)
    text = strip_comments_and_strings(text)
    return " ".join(tokenize(text)).lower()


def similarity_token_text(text: str, *, code: bool = False) -> str:
    """Token stream for near-duplicate similarity.

    Keep identifiers and numeric constants. Structural normalization is reported
    separately, so this layer should not turn same-shaped but different examples
    into artificial near duplicates.
    """
    if code:
        text = strip_includes(text)
    text = strip_comments_and_strings(text)
    return " ".join(tok.lower() for tok in tokenize(text))


def shingles(tokens: list[str], n: int = 5) -> set[str]:
    if len(tokens) < n:
        return set(tokens)
    return {" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def containment(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def function_names(code: str) -> set[str]:
    names: set[str] = set()
    clean = strip_comments_and_strings(code)
    # Top-level-ish function definitions. This is intentionally conservative; a later brace-depth check
    # is unnecessary for identifier overlap because false positives are filtered by CONTROL_NAMES.
    pat = re.compile(r"(?:^|\n)\s*(?:[A-Za-z_]\w*::)?(?:[\w:<>,\s\*&]+?)\s+([A-Za-z_]\w*)\s*\([^;{}]*\)\s*\{", re.S)
    for m in pat.finditer(clean):
        name = m.group(1)
        if name not in CONTROL_NAMES:
            names.add(name)
    return names


def extract_signature(row: dict[str, Any]) -> str:
    for k in ["public_entry_signature", "target_signature", "harness_extern_decl"]:
        if row.get(k):
            return compact_ws(str(row[k]).rstrip(";"))
    for k in ["response", "generated_sve_code_v22", "serial_c_code", "solution_scalar"]:
        code = str(row.get(k) or "")
        m = re.search(r"(?:^|\n)\s*([A-Za-z_][\w:<>,\s\*&]+?\s+[A-Za-z_]\w*\s*\([^;{}]*\))\s*\{", code, re.S)
        if m:
            return compact_ws(m.group(1))
    return ""


def signature_type_shape(sig: str) -> str:
    if not sig:
        return ""
    inside = sig[sig.find("(")+1:sig.rfind(")")] if "(" in sig and ")" in sig else sig
    params = []
    for part in inside.split(","):
        part = part.strip()
        if not part or part == "void":
            continue
        toks = tokenize(part)
        types = []
        ptrs = 0
        for t in toks:
            if t in TYPE_TOKENS:
                types.append(t)
            elif t == "*":
                ptrs += 1
            elif t == "&":
                ptrs += 1
            elif t == "const":
                types.append("const")
        params.append(" ".join(types + ["ptr" if ptrs else "scalar"]))
    return "|".join(params)


def array_access_shapes(code: str) -> list[str]:
    clean = strip_comments_and_strings(code)
    shapes = []
    for m in re.finditer(r"([A-Za-z_]\w*)\s*\[([^\[\]]+)\]", clean):
        expr = m.group(2)
        toks = normalize_tokens(tokenize(expr), keep_calls=False)
        shapes.append(" ".join(toks))
    return sorted(collections.Counter(shapes).elements())


def assignment_ops(code: str) -> list[str]:
    toks = tokenize(strip_comments_and_strings(code))
    ops = [t for t in toks if t in ["=", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>="]]
    return sorted(collections.Counter(ops).elements())


def calls(code: str) -> list[str]:
    toks = tokenize(strip_comments_and_strings(code))
    out = []
    for i, t in enumerate(toks[:-1]):
        if toks[i + 1] == "(" and re.fullmatch(r"[A-Za-z_]\w*", t) and t not in CONTROL_NAMES and t not in KEYWORDS:
            if t.startswith("sv"):
                out.append("sve:" + re.sub(r"_[suf](?:8|16|32|64).*$", "_T", t))
            elif t in MATH_CALLS:
                out.append("math:" + t.rstrip("f"))
            else:
                out.append("call")
    return sorted(collections.Counter(out).elements())


def max_loop_depth(code: str) -> int:
    clean = strip_comments_and_strings(code)
    depth = 0
    maxd = 0
    pending_loop = False
    stack: list[bool] = []
    i = 0
    while i < len(clean):
        m = re.match(r"\b(for|while)\b", clean[i:])
        if m:
            pending_loop = True
            i += len(m.group(0))
            continue
        ch = clean[i]
        if ch == "{":
            stack.append(pending_loop)
            if pending_loop:
                depth += 1
                maxd = max(maxd, depth)
            pending_loop = False
        elif ch == "}":
            if stack:
                was_loop = stack.pop()
                if was_loop:
                    depth -= 1
            pending_loop = False
        elif ch == ";":
            pending_loop = False
        i += 1
    return maxd


def semantic_signature(code: str, sig: str) -> dict[str, Any]:
    clean = strip_includes(strip_comments_and_strings(code))
    toks = tokenize(clean)
    tokset = set(toks)
    arr_shapes = array_access_shapes(code)
    operators = sorted(t for t in set(toks) if t in {"+", "-", "*", "/", "%", "<<", ">>", "&", "|", "^", "==", "!=", "<", ">", "<=", ">="})
    features = {
        "sig_shape": signature_type_shape(sig),
        "loop_depth": max_loop_depth(code),
        "for_count": len(re.findall(r"\bfor\s*\(", clean)),
        "while_count": len(re.findall(r"\bwhile\s*\(", clean)),
        "if_count": len(re.findall(r"\bif\s*\(", clean)),
        "has_return_scalar": bool(re.search(r"\breturn\b", clean)),
        "has_reduction_update": any(op in clean for op in ["+=", "*=", "&=", "|=", "^="]) or "svaddv" in clean or "svmaxv" in clean or "svminv" in clean,
        "has_array_write": bool(re.search(r"[A-Za-z_]\w*\s*\[[^\]]+\]\s*(?:=|\+=|-=|\*=|/=|%=|&=|\|=|\^=)", clean)),
        "has_field_access": "." in tokset or "->" in tokset,
        "has_interleaved_index": bool(re.search(r"\b2\s*\*\s*[A-Za-z_]\w*|[A-Za-z_]\w*\s*\*\s*2\b", clean)) or "svld2" in clean or "svst2" in clean,
        "has_gather_scatter": "gather" in clean or "scatter" in clean,
        "has_math_call": any(c in clean for c in MATH_CALLS),
        "calls": calls(code)[:30],
        "ops": operators,
        "assign_ops": assignment_ops(code),
        "array_shapes": sorted(set(arr_shapes))[:20],
    }
    return features


def sem_hash(sig: dict[str, Any]) -> str:
    return sha(json.dumps(sig, sort_keys=True, ensure_ascii=False))


def load_train_dev(train: Path, dev: Path) -> list[Record]:
    out: list[Record] = []
    for split, path in [("train", train), ("dev", dev)]:
        for i, row in enumerate(read_jsonl(path), 1):
            prompt = str(row.get("prompt") or row.get("nl_description_ds_r1") or "")
            scalar = str(row.get("serial_c_code") or "")
            response = str(row.get("response") or row.get("generated_sve_code_v22") or "")
            code = "\n".join(x for x in [scalar, response] if x)
            sig = extract_signature(row)
            out.append(Record("core", split, i, str(row.get("task_id") or f"{split}_{i}"), str(row.get("source_type") or ""), str(row.get("source_name") or ""), str(row.get("template_family_id") or ""), prompt, code, scalar, response, sig, row))
    return out


def load_simdbench(path: Path) -> list[Record]:
    out: list[Record] = []
    for i, row in enumerate(read_jsonl(path), 1):
        task_id = str(row.get("task_id") or f"simdbench_{i}")
        if row.get("intrinsic") and str(row.get("intrinsic")).upper() != "SVE":
            continue
        prompt = str(row.get("prompt") or "")
        scalar = str(row.get("solution_scalar") or "")
        sig = extract_signature({"serial_c_code": scalar})
        out.append(Record("simdbench_sve", "benchmark", i, task_id, "benchmark", "simdbench_sve", str(row.get("type") or ""), prompt, scalar, scalar, "", sig, row))
    return out


def load_arm_loops(path: Path) -> list[Record]:
    out: list[Record] = []
    for i, row in enumerate(read_jsonl(path), 1):
        prompt = str(row.get("prompt") or row.get("nl_description_ds_r1") or "")
        scalar = str(row.get("serial_c_code") or "")
        sig = str(row.get("target_signature") or extract_signature({"serial_c_code": scalar}))
        out.append(Record("arm_simd_loops", "benchmark", i, str(row.get("task_id") or f"arm_loop_{i}"), str(row.get("source_type") or "benchmark"), str(row.get("source_name") or "arm_simd_loops"), str(row.get("benchmark_category") or ""), prompt, scalar, scalar, "", sig, row))
    return out


def build_indexes(records: list[Record]) -> dict[str, Any]:
    prompt_norm: dict[str, list[Record]] = collections.defaultdict(list)
    code_norm: dict[str, list[Record]] = collections.defaultdict(list)
    struct_norm: dict[str, list[Record]] = collections.defaultdict(list)
    fn_index: dict[str, list[Record]] = collections.defaultdict(list)
    sem_index: dict[str, list[Record]] = collections.defaultdict(list)
    prompt_shingles: dict[str, set[str]] = {}
    code_shingles: dict[str, set[str]] = {}
    sem: dict[str, dict[str, Any]] = {}
    for r in records:
        pn = canonical_exact_text(r.prompt, code=False)
        cn = canonical_exact_text(r.code, code=True)
        sn = normalized_code(r.code)
        if pn:
            prompt_norm[sha(pn)].append(r)
            prompt_shingles[r.uid] = shingles(similarity_token_text(r.prompt, code=False).split(), 5)
        if cn:
            code_norm[sha(cn)].append(r)
            code_shingles[r.uid] = shingles(similarity_token_text(r.code, code=True).split(), 5)
        if sn:
            struct_norm[sha(sn)].append(r)
        for fn in function_names("\n".join([r.signature, r.code])):
            fn_index[fn].append(r)
        ss = semantic_signature(r.code, r.signature)
        sem[r.uid] = ss
        sem_index[sem_hash(ss)].append(r)
    return {
        "prompt_norm": prompt_norm,
        "code_norm": code_norm,
        "struct_norm": struct_norm,
        "fn_index": fn_index,
        "sem_index": sem_index,
        "prompt_shingles": prompt_shingles,
        "code_shingles": code_shingles,
        "sem": sem,
    }


def inverted_index(shingle_map: dict[str, set[str]]) -> dict[str, set[str]]:
    inv: dict[str, set[str]] = collections.defaultdict(set)
    for uid, ss in shingle_map.items():
        for s in ss:
            inv[s].add(uid)
    return inv


def top_similar(
    bench: list[Record],
    core: list[Record],
    core_shingles: dict[str, set[str]],
    bench_shingles: dict[str, set[str]],
    *,
    threshold: float,
    max_candidates: int = 2000,
    min_shingles: int = 20,
) -> list[dict[str, Any]]:
    inv = inverted_index(core_shingles)
    core_by_uid = {r.uid: r for r in core}
    out: list[dict[str, Any]] = []
    for b in bench:
        bs = bench_shingles.get(b.uid, set())
        if len(bs) < min_shingles:
            continue
        cand_counter: collections.Counter[str] = collections.Counter()
        for sh in bs:
            for uid in inv.get(sh, ()):
                cand_counter[uid] += 1
        # Keep candidates with most shared shingles first; this is exact enough for audit.
        for uid, _ in cand_counter.most_common(max_candidates):
            cs = core_shingles.get(uid, set())
            if len(cs) < min_shingles:
                continue
            score = jaccard(bs, cs)
            cont = containment(bs, cs)
            if score >= threshold or cont >= max(threshold, 0.92):
                c = core_by_uid[uid]
                out.append({
                    "benchmark": b.dataset,
                    "bench_id": b.task_id,
                    "core_split": c.split,
                    "core_id": c.task_id,
                    "core_source_type": c.source_type,
                    "core_source_name": c.source_name,
                    "jaccard": round(score, 6),
                    "containment": round(cont, 6),
                })
    out.sort(key=lambda r: (r["benchmark"], -r["jaccard"], -r["containment"], r["bench_id"]))
    return out


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


def pair_rows(level: str, b: Record, c: Record, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    row = {
        "level": level,
        "benchmark": b.dataset,
        "bench_id": b.task_id,
        "bench_signature": b.signature,
        "core_split": c.split,
        "core_id": c.task_id,
        "core_source_type": c.source_type,
        "core_source_name": c.source_name,
        "core_family": c.template_family_id,
    }
    if extra:
        row.update(extra)
    return row


def run_audit(core: list[Record], benches: list[Record], out: Path) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    core_idx = build_indexes(core)
    bench_idx = build_indexes(benches)

    identifier_matches: list[dict[str, Any]] = []
    for fn, b_records in bench_idx["fn_index"].items():
        if fn in core_idx["fn_index"]:
            for b in b_records:
                for c in core_idx["fn_index"][fn]:
                    identifier_matches.append(pair_rows("function_name_exact", b, c, {
                        "function_name": fn,
                        "generic_name": fn in GENERIC_FUNCTION_NAMES,
                        "signature_type_shape_same": signature_type_shape(b.signature) == signature_type_shape(c.signature),
                    }))

    exact_prompt: list[dict[str, Any]] = []
    for h, b_records in bench_idx["prompt_norm"].items():
        for c in core_idx["prompt_norm"].get(h, []):
            for b in b_records:
                exact_prompt.append(pair_rows("prompt_normalized_exact", b, c, {"hash": h}))

    exact_code: list[dict[str, Any]] = []
    for h, b_records in bench_idx["code_norm"].items():
        for c in core_idx["code_norm"].get(h, []):
            for b in b_records:
                exact_code.append(pair_rows("code_normalized_exact", b, c, {"hash": h}))

    structural_exact: list[dict[str, Any]] = []
    for h, b_records in bench_idx["struct_norm"].items():
        for c in core_idx["struct_norm"].get(h, []):
            for b in b_records:
                structural_exact.append(pair_rows("normalized_structural_code_exact", b, c, {"hash": h}))

    semantic_exact_all: list[dict[str, Any]] = []
    for h, b_records in bench_idx["sem_index"].items():
        for c in core_idx["sem_index"].get(h, []):
            for b in b_records:
                semantic_exact_all.append(pair_rows("semantic_signature_exact", b, c, {"hash": h}))

    prompt_sim = top_similar(
        benches,
        core,
        core_idx["prompt_shingles"],
        bench_idx["prompt_shingles"],
        threshold=0.86,
        min_shingles=25,
    )
    code_sim = top_similar(
        benches,
        core,
        core_idx["code_shingles"],
        bench_idx["code_shingles"],
        threshold=0.80,
        min_shingles=20,
    )

    # High-risk semantic matches: same semantic signature plus non-trivial code n-gram overlap.
    core_code = core_idx["code_shingles"]
    bench_code = bench_idx["code_shingles"]
    core_by_uid = {r.uid: r for r in core}
    bench_by_uid = {r.uid: r for r in benches}
    semantic_high_risk: list[dict[str, Any]] = []
    for row in semantic_exact_all:
        b_uid = f"{row['benchmark']}:benchmark:{row['bench_id']}"
        # Some benchmark row ids contain task_id only; fallback search.
        b = next((x for x in benches if x.dataset == row["benchmark"] and x.task_id == row["bench_id"]), None)
        c = next((x for x in core if x.split == row["core_split"] and x.task_id == row["core_id"]), None)
        if not b or not c:
            continue
        cj = jaccard(bench_code.get(b.uid, set()), core_code.get(c.uid, set()))
        cc = containment(bench_code.get(b.uid, set()), core_code.get(c.uid, set()))
        if cj >= 0.45 or cc >= 0.75:
            semantic_high_risk.append({**row, "code_jaccard": round(cj, 6), "code_containment": round(cc, 6)})

    # Text scan for benchmark task IDs and exact function names inside core prompt/response.
    b_task_terms = {b.task_id for b in benches if b.task_id}
    b_fn_terms = set()
    for fn, rs in bench_idx["fn_index"].items():
        if len(fn) >= 4:
            b_fn_terms.add(fn)
    literal_mentions: list[dict[str, Any]] = []
    for c in core:
        blob = "\n".join([c.task_id, c.prompt, c.code])
        for term in b_task_terms:
            if term and term in blob:
                # Ignore source-looking SimdBench words in historical docs? Keep as evidence for manual review.
                b = next(x for x in benches if x.task_id == term)
                literal_mentions.append(pair_rows("benchmark_task_id_literal_mention", b, c, {"literal": term}))
        for fn in b_fn_terms:
            if re.search(rf"\b{re.escape(fn)}\b", blob):
                # If it is the same function-name match, identifier audit already catches it; this catches prompt mentions too.
                b = next((x for x in benches if fn in function_names(x.code + "\n" + x.signature)), None)
                if b:
                    literal_mentions.append(pair_rows("benchmark_function_literal_mention", b, c, {
                        "literal": fn,
                        "generic_name": fn in GENERIC_FUNCTION_NAMES,
                    }))

    files = {
        "identifier_matches": out / "identifier_matches.tsv",
        "literal_mentions": out / "literal_mentions.tsv",
        "exact_prompt_matches": out / "exact_prompt_matches.tsv",
        "exact_code_matches": out / "exact_code_matches.tsv",
        "structural_exact_matches": out / "structural_exact_matches.tsv",
        "semantic_exact_matches": out / "semantic_exact_matches.tsv",
        "semantic_high_risk_matches": out / "semantic_high_risk_matches.tsv",
        "prompt_similarity_matches": out / "prompt_similarity_matches.tsv",
        "code_similarity_matches": out / "code_similarity_matches.tsv",
    }
    write_tsv(files["identifier_matches"], identifier_matches)
    write_tsv(files["literal_mentions"], literal_mentions)
    write_tsv(files["exact_prompt_matches"], exact_prompt)
    write_tsv(files["exact_code_matches"], exact_code)
    write_tsv(files["structural_exact_matches"], structural_exact)
    write_tsv(files["semantic_exact_matches"], semantic_exact_all[:50000])
    write_tsv(files["semantic_high_risk_matches"], semantic_high_risk)
    write_tsv(files["prompt_similarity_matches"], prompt_sim)
    write_tsv(files["code_similarity_matches"], code_sim)

    # Signature dumps for reproducibility.
    with (out / "core_signatures.jsonl").open("w", encoding="utf-8") as f:
        for r in core:
            f.write(json.dumps({
                "uid": r.uid,
                "task_id": r.task_id,
                "split": r.split,
                "source_type": r.source_type,
                "source_name": r.source_name,
                "signature": r.signature,
                "function_names": sorted(function_names(r.signature + "\n" + r.code)),
                "prompt_hash": sha(canonical_exact_text(r.prompt)) if r.prompt else "",
                "code_hash": sha(canonical_exact_text(r.code, code=True)) if r.code else "",
                "structural_hash": sha(normalized_code(r.code)) if r.code else "",
                "semantic_hash": sem_hash(core_idx["sem"][r.uid]),
                "semantic_signature": core_idx["sem"][r.uid],
            }, ensure_ascii=False) + "\n")
    with (out / "benchmark_signatures.jsonl").open("w", encoding="utf-8") as f:
        for r in benches:
            f.write(json.dumps({
                "uid": r.uid,
                "task_id": r.task_id,
                "dataset": r.dataset,
                "signature": r.signature,
                "function_names": sorted(function_names(r.signature + "\n" + r.code)),
                "prompt_hash": sha(canonical_exact_text(r.prompt)) if r.prompt else "",
                "code_hash": sha(canonical_exact_text(r.code, code=True)) if r.code else "",
                "structural_hash": sha(normalized_code(r.code)) if r.code else "",
                "semantic_hash": sem_hash(bench_idx["sem"][r.uid]),
                "semantic_signature": bench_idx["sem"][r.uid],
            }, ensure_ascii=False) + "\n")

    high_risk = []
    for rows, name in [
        ([r for r in identifier_matches if not r.get("generic_name") or r.get("signature_type_shape_same")], "identifier"),
        ([r for r in literal_mentions if not r.get("generic_name")], "literal_mention"),
        (exact_prompt, "exact_prompt"),
        (exact_code, "exact_code"),
        (structural_exact, "structural_exact"),
        (semantic_high_risk, "semantic_high_risk"),
    ]:
        for r in rows:
            high_risk.append({"risk_source": name, **r})
    for r in prompt_sim:
        if r["jaccard"] >= 0.92 or r["containment"] >= 0.96:
            high_risk.append({"risk_source": "prompt_similarity", "level": "prompt_similarity", **r})
    for r in code_sim:
        if r["jaccard"] >= 0.85 or r["containment"] >= 0.95:
            high_risk.append({"risk_source": "code_similarity", "level": "code_similarity", **r})
    write_tsv(out / "high_risk_candidates.tsv", high_risk)

    by_benchmark = collections.defaultdict(lambda: collections.Counter())
    for b in benches:
        by_benchmark[b.dataset]["rows"] += 1
    for label, rows in [
        ("identifier_matches", identifier_matches),
        ("literal_mentions", literal_mentions),
        ("exact_prompt_matches", exact_prompt),
        ("exact_code_matches", exact_code),
        ("structural_exact_matches", structural_exact),
        ("semantic_exact_matches", semantic_exact_all),
        ("semantic_high_risk_matches", semantic_high_risk),
        ("prompt_similarity_matches", prompt_sim),
        ("code_similarity_matches", code_sim),
        ("high_risk_candidates", high_risk),
    ]:
        for r in rows:
            by_benchmark[r.get("benchmark", "unknown")][label] += 1

    summary = {
        "core_rows": len(core),
        "core_train_rows": sum(1 for r in core if r.split == "train"),
        "core_dev_rows": sum(1 for r in core if r.split == "dev"),
        "benchmark_rows": len(benches),
        "benchmark_counts": dict(collections.Counter(r.dataset for r in benches)),
        "audit_counts": {
            "identifier_matches": len(identifier_matches),
            "literal_mentions": len(literal_mentions),
            "exact_prompt_matches": len(exact_prompt),
            "exact_code_matches": len(exact_code),
            "structural_exact_matches": len(structural_exact),
            "semantic_exact_matches": len(semantic_exact_all),
            "semantic_high_risk_matches": len(semantic_high_risk),
            "prompt_similarity_matches": len(prompt_sim),
            "code_similarity_matches": len(code_sim),
            "high_risk_candidates": len(high_risk),
        },
        "by_benchmark": {k: dict(v) for k, v in by_benchmark.items()},
        "thresholds": {
            "prompt_similarity_jaccard": 0.86,
            "code_similarity_jaccard": 0.80,
            "semantic_high_risk_code_jaccard": 0.45,
            "semantic_high_risk_code_containment": 0.75,
        },
        "files": {k: str(v) for k, v in files.items()} | {
            "high_risk_candidates": str(out / "high_risk_candidates.tsv"),
            "core_signatures": str(out / "core_signatures.jsonl"),
            "benchmark_signatures": str(out / "benchmark_signatures.jsonl"),
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    md = [
        "# Core Dataset vs Benchmark Non-overlap Audit",
        "",
        "## Inputs",
        f"- Core train rows: {summary['core_train_rows']}",
        f"- Core dev rows: {summary['core_dev_rows']}",
        f"- Benchmark rows: {summary['benchmark_rows']} ({summary['benchmark_counts']})",
        "",
        "## Audit Counts",
        "",
        "| Level | Count |",
        "|---|---:|",
    ]
    for k, v in summary["audit_counts"].items():
        md.append(f"| {k} | {v} |")
    md += [
        "",
        "## Interpretation",
        "",
        "- `exact_prompt_matches`, `exact_code_matches`, and `structural_exact_matches` are the strongest direct-overlap signals.",
        "- `semantic_exact_matches` is intentionally broad: it means examples share an abstract access/control signature, not necessarily that they are duplicates.",
        "- `semantic_high_risk_matches` additionally requires meaningful code-token overlap and is the review queue for near-duplicate concerns.",
        "- `identifier_matches` and `literal_mentions` are metadata/text leakage checks and should be manually reviewed if nonzero.",
        "",
        "## Output Files",
    ]
    for k, v in summary["files"].items():
        md.append(f"- `{k}`: `{v}`")
    (out / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
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
    summary = run_audit(core, benches, args.out)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
