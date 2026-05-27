#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple


SHARED_DIAG_DIR = Path("/home/user/selective_repo/latest_pairs_DeesSeekR1-32B/scripts")
if str(SHARED_DIAG_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIAG_DIR))

try:
    from compile_diagnostics_pass1_v22 import extract_compile_diagnostics
except Exception as exc:  # pragma: no cover - runtime environment guard
    raise SystemExit(f"failed to import compile_diagnostics_pass1_v22: {exc}")


ERROR_LOC_RE = re.compile(
    r"^(?P<path>[^:\n]+):(?P<line>\d+):(?P<col>\d+):\s+"
    r"(?P<kind>fatal error|error|warning|note):\s+(?P<msg>.+)$"
)


def clean_items(items: Any, limit: int = 12) -> List[str]:
    out: List[str] = []
    seen = set()
    if not isinstance(items, list):
        return out
    for item in items:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
        if len(out) >= limit:
            break
    return out


def display_diagnostic_key(key: str) -> str:
    return {
        "non_constant_immediate_args": "compile_time_immediate_args",
    }.get(str(key), str(key))


def normalize_diagnostic_keys_for_output(diagnostics: Dict[str, List[str]]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for key, values in (diagnostics or {}).items():
        out_key = display_diagnostic_key(str(key))
        out.setdefault(out_key, [])
        out[out_key].extend(clean_items(values, 16))
    return {k: clean_items(v, 16) for k, v in out.items() if clean_items(v, 16)}


def extract_diagnostics_with_locations(text: str) -> Dict[str, List[str]]:
    diagnostics: Dict[str, List[str]] = dict(extract_compile_diagnostics(text) or {})
    error_locations: List[str] = []
    error_messages: List[str] = []
    for raw_line in str(text or "").splitlines():
        match = ERROR_LOC_RE.match(raw_line.strip())
        if not match:
            continue
        if str(match.group("kind") or "").lower() not in {"error", "fatal error"}:
            continue
        loc = f"{match.group('line')}:{match.group('col')}"
        msg = str(match.group("msg") or "").strip()
        error_locations.append(loc)
        error_messages.append(f"{loc} {msg}")
    if error_locations:
        diagnostics["error_locations"] = clean_items(error_locations, 12)
    if error_messages:
        diagnostics["error_messages"] = clean_items(error_messages, 12)
    return {str(k): clean_items(v, 16) for k, v in diagnostics.items() if clean_items(v, 16)}


def classify_diagnostics(diagnostics: Dict[str, List[str]]) -> List[str]:
    classes: List[str] = []
    if diagnostics.get("syntax_signals") or diagnostics.get("expected_tokens"):
        classes.append("syntax_closure")
    if diagnostics.get("unsupported_symbols"):
        classes.append("unsupported_or_fake_intrinsic")
    if diagnostics.get("missing_helper_symbols") or diagnostics.get("missing_index_symbols"):
        classes.append("missing_infile_helper")
    if diagnostics.get("undeclared_identifiers"):
        classes.append("undeclared_identifier_or_missing_include")
    if diagnostics.get("ptr_type_mismatch_calls"):
        classes.append("ptr_type_mismatch")
    if diagnostics.get("arg_type_mismatch_calls") or diagnostics.get("no_matching_calls"):
        classes.append("arg_type_mismatch")
    if diagnostics.get("ambiguous_calls"):
        classes.append("ambiguous_call")
    if diagnostics.get("non_constant_immediate_args") or diagnostics.get("compile_time_immediate_args"):
        classes.append("compile_time_immediate_required")
    if diagnostics.get("missing_include_headers"):
        classes.append("missing_standard_include")
    if diagnostics.get("missing_members"):
        classes.append("missing_member_or_wrong_std_version")
    if diagnostics.get("unknown_type_names"):
        classes.append("unknown_type_name")
    if diagnostics.get("redefined_symbols"):
        classes.append("redefinition")
    if diagnostics.get("type_mismatch_messages"):
        classes.append("expression_type_mismatch")
    if diagnostics.get("sizeless_type_messages"):
        classes.append("sizeless_sve_misuse")
    if diagnostics.get("const_assignment_messages"):
        classes.append("const_assignment")
    if diagnostics.get("invalid_cast_messages"):
        classes.append("invalid_cast")
    if diagnostics.get("invalid_address_messages"):
        classes.append("invalid_addressing")
    if diagnostics.get("invalid_cpp_construct_messages"):
        classes.append("invalid_cpp_construct")
    if diagnostics.get("remote_evaluator_artifacts"):
        classes.append("remote_evaluator_artifact")
    return classes or ["unclassified_structured"]


def iter_input_files(paths: List[str]) -> Iterator[Path]:
    for raw in paths:
        path = Path(raw).expanduser()
        if not path.exists():
            continue
        if path.is_file():
            yield path
            continue
        for child in path.rglob("*"):
            if child.name == "_remote_feedback_info.json":
                yield child
            elif child.suffix == ".json" and child.name.startswith("remote_result_in_round"):
                yield child
            elif child.suffix == ".jsonl" and "meta" in child.name:
                yield child


def load_json_file(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None


def iter_jsonl(path: Path) -> Iterator[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield lineno, obj


def is_compile_failure_result(result: Dict[str, Any]) -> bool:
    try:
        compile_ok = int(result.get("compile_ok", 0) or 0)
    except Exception:
        compile_ok = 0
    reason = str(result.get("reason") or "").lower()
    return compile_ok == 0 or "compile" in reason or bool(str(result.get("compile_log_tail") or "").strip())


def iter_result_dicts_from_remote_info(path: Path, obj: Dict[str, Any]) -> Iterator[Tuple[str, Dict[str, Any]]]:
    for key in ["final", "best", "last"]:
        value = obj.get(key)
        if isinstance(value, dict):
            yield key, value
    for idx, item in enumerate(obj.get("history") or []):
        if not isinstance(item, dict):
            continue
        result = item.get("result")
        if isinstance(result, dict):
            yield f"history[{idx}].result", result
    gate = obj.get("remote_cpp17_compile_gate")
    if isinstance(gate, dict):
        final = gate.get("final")
        if isinstance(final, dict):
            yield "remote_cpp17_compile_gate.final", final
        for idx, item in enumerate(gate.get("history") or []):
            if isinstance(item, dict):
                result = item.get("result") if isinstance(item.get("result"), dict) else item
                yield f"remote_cpp17_compile_gate.history[{idx}]", result


def summarize_result(
    *,
    source_file: Path,
    source_kind: str,
    source_label: str,
    result: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if not is_compile_failure_result(result):
        return None
    compile_tail = str(result.get("compile_log_tail") or "").strip()
    existing_diag = result.get("remote_compile_diagnostics")
    diagnostics = existing_diag if isinstance(existing_diag, dict) and existing_diag else {}
    if not diagnostics and compile_tail:
        diagnostics = extract_diagnostics_with_locations(compile_tail)
    diagnostics = {str(k): clean_items(v, 16) for k, v in diagnostics.items() if clean_items(v, 16)}
    classes = classify_diagnostics(diagnostics) if diagnostics else []
    diagnostics = normalize_diagnostic_keys_for_output(diagnostics)
    return {
        "source_file": str(source_file),
        "source_kind": source_kind,
        "source_label": source_label,
        "task_id": result.get("task_id"),
        "compile_ok": result.get("compile_ok"),
        "run_ok": result.get("run_ok"),
        "reason": result.get("reason"),
        "has_compile_log_tail": bool(compile_tail),
        "structured_ok": bool(diagnostics),
        "fallback_raw_needed": bool(compile_tail and not diagnostics),
        "classes": classes,
        "diagnostics": diagnostics,
    }


def iter_compile_feedback_records(path: Path) -> Iterator[Dict[str, Any]]:
    if path.suffix == ".jsonl":
        for lineno, rec in iter_jsonl(path):
            if "remote_compile_diagnostics" in rec or "remote_compile_ok" in rec:
                result = {
                    "task_id": rec.get("task_id"),
                    "compile_ok": rec.get("remote_compile_ok"),
                    "run_ok": rec.get("remote_run_ok"),
                    "reason": rec.get("remote_reason"),
                    "remote_compile_diagnostics": rec.get("remote_compile_diagnostics"),
                    "compile_log_tail": rec.get("compile_log_tail", ""),
                }
                summary = summarize_result(
                    source_file=path,
                    source_kind="meta_jsonl",
                    source_label=f"line:{lineno}",
                    result=result,
                )
                if summary:
                    yield summary
        return

    obj = load_json_file(path)
    if isinstance(obj, dict) and path.name == "_remote_feedback_info.json":
        for label, result in iter_result_dicts_from_remote_info(path, obj):
            summary = summarize_result(
                source_file=path,
                source_kind="remote_feedback_info",
                source_label=label,
                result=result,
            )
            if summary:
                yield summary
    elif isinstance(obj, dict):
        summary = summarize_result(
            source_file=path,
            source_kind="remote_result_json",
            source_label=path.name,
            result=obj,
        )
        if summary:
            yield summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit historical compile feedback and re-bucket it into structured diagnostics."
    )
    parser.add_argument("inputs", nargs="+", help="meta jsonl, _remote_feedback_info.json, remote_result_in json, or dirs")
    parser.add_argument("--out_dir", required=True, help="Output directory for JSON/TSV summaries")
    parser.add_argument("--limit", type=int, default=0, help="Optional max records to write")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    seen = set()
    for path in iter_input_files(args.inputs):
        for rec in iter_compile_feedback_records(path):
            key = (
                rec.get("source_file"),
                rec.get("source_label"),
                rec.get("task_id"),
                rec.get("reason"),
                json.dumps(rec.get("diagnostics", {}), sort_keys=True, ensure_ascii=False),
            )
            if key in seen:
                continue
            seen.add(key)
            records.append(rec)
            if args.limit and len(records) >= args.limit:
                break
        if args.limit and len(records) >= args.limit:
            break

    class_counts = Counter()
    diag_key_counts = Counter()
    reason_counts = Counter()
    task_counts = Counter()
    for rec in records:
        reason_counts[str(rec.get("reason") or "")] += 1
        task_counts[str(rec.get("task_id") or "")] += 1
        for cls in rec.get("classes") or ["fallback_raw_or_empty"]:
            class_counts[cls] += 1
        for key in (rec.get("diagnostics") or {}).keys():
            diag_key_counts[str(key)] += 1

    summary = {
        "total_compile_feedback_records": len(records),
        "structured_ok": sum(1 for rec in records if rec.get("structured_ok")),
        "fallback_raw_needed": sum(1 for rec in records if rec.get("fallback_raw_needed")),
        "has_compile_log_tail": sum(1 for rec in records if rec.get("has_compile_log_tail")),
        "class_counts": dict(class_counts.most_common()),
        "diagnostic_key_counts": dict(diag_key_counts.most_common()),
        "reason_counts": dict(reason_counts.most_common()),
        "top_tasks": dict(task_counts.most_common(30)),
    }

    (out_dir / "structured_compile_feedback_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (out_dir / "structured_compile_feedback_records.jsonl").open("w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
    with (out_dir / "structured_compile_feedback_records.tsv").open("w", encoding="utf-8") as handle:
        handle.write("task_id\treason\tstructured_ok\tfallback_raw_needed\tclasses\tdiagnostic_keys\tsource\n")
        for rec in records:
            handle.write(
                "\t".join(
                    [
                        str(rec.get("task_id") or ""),
                        str(rec.get("reason") or ""),
                        "1" if rec.get("structured_ok") else "0",
                        "1" if rec.get("fallback_raw_needed") else "0",
                        ";".join(rec.get("classes") or []),
                        ";".join((rec.get("diagnostics") or {}).keys()),
                        f"{rec.get('source_file')}::{rec.get('source_label')}",
                    ]
                )
                + "\n"
            )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
