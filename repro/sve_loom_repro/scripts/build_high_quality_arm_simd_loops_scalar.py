#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path
from textwrap import dedent
from typing import Any


DEFAULT_PROBLEM = Path(
    "/home/user/selective_repo/results/generated/contextualized_benchmarks/"
    "arm_simd_loops/arm_simd_loops_contextualized.problem.with_solution_scalar.jsonl"
)
DEFAULT_OUT_DIR = Path(
    "/home/user/simdbench_full/results/generated/high_quality_arm_simd_loops"
)


QUALITY_OVERRIDES: dict[str, str] = {
    "arm_simd_loops.loop_001": r"""
float arm_simd_loop_001(float * __restrict a, float * __restrict b, int n) {
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    int i = 0;
    for (; i + 3 < n; i += 4) {
        s0 += a[i] * b[i];
        s1 += a[i + 1] * b[i + 1];
        s2 += a[i + 2] * b[i + 2];
        s3 += a[i + 3] * b[i + 3];
    }
    float res = (s0 + s1) + (s2 + s3);
    for (; i < n; ++i) {
        res += a[i] * b[i];
    }
    return res;
}
""",
    "arm_simd_loops.loop_002": r"""
uint32_t arm_simd_loop_002(uint32_t * __restrict a, uint32_t * __restrict b, int n) {
    uint32_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    int i = 0;
    for (; i + 3 < n; i += 4) {
        s0 += a[i] * b[i];
        s1 += a[i + 1] * b[i + 1];
        s2 += a[i + 2] * b[i + 2];
        s3 += a[i + 3] * b[i + 3];
    }
    uint32_t res = (s0 + s1) + (s2 + s3);
    for (; i < n; ++i) {
        res += a[i] * b[i];
    }
    return res;
}
""",
    "arm_simd_loops.loop_003": r"""
double arm_simd_loop_003(double * __restrict a, double * __restrict b, int n) {
    double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
    int i = 0;
    for (; i + 3 < n; i += 4) {
        s0 += a[i] * b[i];
        s1 += a[i + 1] * b[i + 1];
        s2 += a[i + 2] * b[i + 2];
        s3 += a[i + 3] * b[i + 3];
    }
    double res = (s0 + s1) + (s2 + s3);
    for (; i < n; ++i) {
        res += a[i] * b[i];
    }
    return res;
}
""",
    "arm_simd_loops.loop_004": r"""
uint64_t arm_simd_loop_004(uint64_t * __restrict a, uint64_t * __restrict b, int n) {
    uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    int i = 0;
    for (; i + 3 < n; i += 4) {
        s0 += a[i] * b[i];
        s1 += a[i + 1] * b[i + 1];
        s2 += a[i + 2] * b[i + 2];
        s3 += a[i + 3] * b[i + 3];
    }
    uint64_t res = (s0 + s1) + (s2 + s3);
    for (; i < n; ++i) {
        res += a[i] * b[i];
    }
    return res;
}
""",
    "arm_simd_loops.loop_005": r"""
#include <string.h>
uint32_t arm_simd_loop_005(uint8_t * p, uint8_t * lmt) {
    uint32_t res = 0;
    while (p < lmt) {
        const void *z = memchr(p, 0, (size_t)(lmt - p));
        uint32_t len = z ? (uint32_t)((const uint8_t *)z - p) : (uint32_t)(lmt - p);
        p += (size_t)len + 1u;
        res += 1;
        res ^= (len % 0xffffu) << 16;
    }
    return res;
}
""",
    "arm_simd_loops.loop_006": r"""
#include <string.h>
uint32_t arm_simd_loop_006(uint8_t * p, uint8_t * lmt) {
    uint32_t res = 0;
    while (p < lmt) {
        const void *z = memchr(p, 0, (size_t)(lmt - p));
        uint32_t len = z ? (uint32_t)((const uint8_t *)z - p) : (uint32_t)(lmt - p);
        p += (size_t)len + 1u;
        res += 1;
        res ^= (len % 0xffffu) << 16;
    }
    return res;
}
""",
    "arm_simd_loops.loop_008": r"""
double arm_simd_loop_008(double * __restrict a, int n) {
    double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
    int i = 0;
    for (; i + 3 < n; i += 4) {
        s0 += a[i];
        s1 += a[i + 1];
        s2 += a[i + 2];
        s3 += a[i + 3];
    }
    double res = (s0 + s1) + (s2 + s3);
    for (; i < n; ++i) {
        res += a[i];
    }
    return res;
}
""",
    "arm_simd_loops.loop_010": r"""
int arm_simd_loop_010(float * __restrict a, uint64_t n) {
    bool any = false;
    bool all = true;
    for (uint64_t i = 0; i < n; ++i) {
        const bool neg = a[i] < 0.0f;
        any |= neg;
        all &= neg;
    }
    return all ? 1 : (any ? 2 : 3);
}
""",
    "arm_simd_loops.loop_023": r"""
double arm_simd_loop_023(double * __restrict a, double * __restrict b, uint32_t * __restrict indexes, int n) {
    double s0 = 0.0, s1 = 0.0;
    int i = 0;
    for (; i + 1 < n; i += 2) {
        s0 += a[indexes[i]] * b[i];
        s1 += a[indexes[i + 1]] * b[i + 1];
    }
    double res = s0 + s1;
    for (; i < n; ++i) {
        res += a[indexes[i]] * b[i];
    }
    return res;
}
""",
    "arm_simd_loops.loop_024": r"""
uint32_t arm_simd_loop_024(uint8_t * __restrict a, uint8_t * __restrict b, int64_t n) {
    uint32_t sum = 0;
    for (int64_t i = 0; i < n; ++i) {
        const uint32_t av = a[i];
        const uint32_t bv = b[i];
        sum += av > bv ? av - bv : bv - av;
    }
    return sum;
}
""",
    "arm_simd_loops.loop_027": r"""
#include <math.h>
void arm_simd_loop_027(float * __restrict input, float * __restrict output, int64_t size) {
    for (int64_t i = 0; i < size; ++i) {
        output[i] = sqrtf(input[i]);
    }
}
""",
    "arm_simd_loops.loop_028": r"""
void arm_simd_loop_028(double * __restrict input1, double * __restrict input2, double * __restrict output, int64_t size) {
    for (int64_t i = 0; i < size; ++i) {
        output[i] = input1[i] / input2[i];
    }
}
""",
    "arm_simd_loops.loop_031": r"""
#include <string.h>
void arm_simd_loop_031(uint8_t * a, uint8_t * b) {
    static const size_t count[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 31, 64, 80, 96, 127, 128, 200, 255, 512};
    uint8_t *src = a;
    uint8_t *to = b;
    for (int j = 0; j < 10; ++j) {
        for (int c = 0; c < 20; ++c) {
            const size_t n = count[c];
            memcpy(to, src, n);
            src += n;
            to += n;
        }
    }
}
""",
    "arm_simd_loops.loop_040": r"""
int32_t arm_simd_loop_040(int32_t count) {
    const int32_t value = count / 2;
    int32_t result = 0;
    for (int32_t i = 0; i < count; ++i) {
        const int32_t lower = i;
        const int32_t upper = 2 * i;
        const int32_t clamped = value < lower ? lower : (value > upper ? upper : value);
        result += clamped;
    }
    return result;
}
""",
    "arm_simd_loops.loop_032": r"""
double arm_simd_loop_032(double * __restrict a, double * __restrict b, int n) {
    double res = 0.0;
    int lw = 0;
    for (int j = 4; j < n; j += 5, ++lw) {
        res -= a[lw] * b[j];
    }
    return res;
}
""",
    "arm_simd_loops.loop_033": r"""
double arm_simd_loop_033(double * __restrict a, double * __restrict b, int64_t n) {
    double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
    int64_t i = 0;
    for (; i + 3 < n; i += 4) {
        s0 += a[i] * b[i];
        s1 += a[i + 1] * b[i + 1];
        s2 += a[i + 2] * b[i + 2];
        s3 += a[i + 3] * b[i + 3];
    }
    double res = (s0 + s1) + (s2 + s3);
    for (; i < n; ++i) {
        res += a[i] * b[i];
    }
    return res;
}
""",
    "arm_simd_loops.loop_035": r"""
void arm_simd_loop_035(float * __restrict a, float * __restrict b, float * __restrict c, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}
""",
    "arm_simd_loops.loop_037": r"""
void arm_simd_loop_037(cfloat32_t * __restrict a0, cfloat32_t * __restrict b0, cfloat32_t * __restrict c0, uint64_t size) {
    for (uint64_t i = 0; i < size; ++i) {
        const float ar = a0[i].re, ai = a0[i].im;
        const float br = b0[i].re, bi = b0[i].im;
        c0[i].re = ar * br - ai * bi;
        c0[i].im = ar * bi + ai * br;
    }
}
""",
    "arm_simd_loops.loop_038": r"""
void arm_simd_loop_038(__fp16 * __restrict a, __fp16 * __restrict b, __fp16 * __restrict c, int dim) {
    for (int row = 0; row < dim - 1; ++row) {
        const int base = row * dim;
        const int next = base + dim;
        for (int col = 0; col < dim - 1; ++col) {
            const float s0 = (float)a[base + col];
            const float s1 = (float)a[base + col + 1];
            const float s2 = (float)a[next + col];
            const float s3 = (float)a[next + col + 1];
            const float ac = (float)b[base + col];
            c[base + col] = (__fp16)(ac + 0.25f * (s0 + s1 + s2 + s3));
        }
    }
}
""",
    "arm_simd_loops.loop_101": r"""
void arm_simd_loop_101(uint8_t * __restrict a, uint8_t * __restrict b, int n) {
    for (int i = 0; i < n - 1; ++i) {
        const uint16_t s1 = b[i];
        const uint16_t s2 = b[i + 1];
        a[2 * i] = (uint8_t)((3 * s1 + s2 + 2) >> 2);
        a[2 * i + 1] = (uint8_t)((3 * s2 + s1 + 2) >> 2);
    }
}
""",
    "arm_simd_loops.loop_105": r"""
float arm_simd_loop_105(float * __restrict a, float * b, int n) {
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    int i = 0;
    for (; i + 3 < n; i += 4) {
        s0 += a[i];
        s1 += a[i + 1];
        s2 += a[i + 2];
        s3 += a[i + 3];
    }
    float sum = (s0 + s1) + (s2 + s3);
    for (; i < n; ++i) {
        sum += a[i];
    }
    return sum;
}
""",
    "arm_simd_loops.loop_108": r"""
void arm_simd_loop_108(uint32_t * __restrict rgba, uint8_t * __restrict y, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        const uint32_t px = rgba[i];
        uint32_t v = (px >> 24) >> 2;
        const uint32_t g = (px >> 16) & 0xffu;
        v += g >> 1;
        v += g >> 3;
        v += ((px >> 8) & 0xffu) >> 3;
        y[i] = (uint8_t)v;
    }
}
""",
    "arm_simd_loops.loop_109": r"""
void arm_simd_loop_109(cuint32_t * __restrict a0, cuint32_t * __restrict b0, cuint32_t * __restrict c0, uint64_t size) {
    for (uint64_t i = 0; i < size; ++i) {
        c0[i].re = a0[i].re - b0[i].im;
        c0[i].im = a0[i].im + b0[i].re;
    }
}
""",
    "arm_simd_loops.loop_112": r"""
void arm_simd_loop_112(cuint32_t * __restrict a0, cuint32_t * __restrict b0, cuint32_t * __restrict c0, uint64_t size) {
    for (uint64_t i = 0; i < size; ++i) {
        const uint32_t ar = a0[i].re, ai = a0[i].im;
        const uint32_t br = b0[i].re, bi = b0[i].im;
        c0[i].re = ar * br - ai * bi;
        c0[i].im = ar * bi + ai * br;
    }
}
""",
    "arm_simd_loops.loop_113": r"""
void arm_simd_loop_113(uint32_t * __restrict a0, uint32_t * __restrict b0, uint32_t * __restrict c0, uint64_t size) {
    for (uint64_t i = 0; i < size; i += 2) {
        c0[i] = a0[i] + a0[i + 1];
        c0[i + 1] = b0[i] + b0[i + 1];
    }
}
""",
    "arm_simd_loops.loop_120": r"""
#include <algorithm>
void arm_simd_loop_120(uint32_t n, int32_t * data) {
    std::sort(data, data + n);
}
""",
    "arm_simd_loops.loop_121": r"""
#include <algorithm>
void arm_simd_loop_121(uint32_t n, int32_t * data, int32_t * temp) {
    (void)temp;
    std::sort(data, data + n);
}
""",
    "arm_simd_loops.loop_122": r"""
#include <algorithm>
void arm_simd_loop_122(uint32_t n, int32_t * data) {
    std::sort(data, data + n);
}
""",
    "arm_simd_loops.loop_123": r"""
#include <algorithm>
void arm_simd_loop_123(uint32_t n, int32_t * data, int32_t * temp, uint32_t * block_sizes) {
    (void)temp;
    (void)block_sizes;
    std::sort(data, data + n);
}
""",
    "arm_simd_loops.loop_124": r"""
#include <algorithm>
void arm_simd_loop_124(uint32_t n, int32_t * data, int32_t * temp, uint32_t * hist, uint32_t * prfx) {
    (void)temp;
    (void)hist;
    (void)prfx;
    std::sort(data, data + n);
}
""",
    "arm_simd_loops.loop_126": r"""
uint32_t arm_simd_loop_126(uint32_t * __restrict a, uint32_t * __restrict b, int n) {
    uint32_t res = 0;
    for (int i = 0; i < n; ++i) {
        res += a[i] * b[i];
        res += res & 1u;
    }
    return res;
}
""",
    "arm_simd_loops.loop_128": r"""
void arm_simd_loop_128(uint32_t * __restrict a, uint32_t * __restrict b, uint32_t * __restrict c, int n) {
    for (int i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}
""",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def normalize_code(code: str) -> str:
    return dedent(code).strip() + "\n"


def strip_top_includes(code: str) -> str:
    lines = []
    for line in code.splitlines():
        if line.strip().startswith("#include"):
            continue
        lines.append(line)
    return "\n".join(lines).strip() + "\n"


def compile_and_run(row: dict[str, Any], serial_code: str, compiler: str) -> tuple[str, str]:
    # The local x86 g++ toolchain in this workspace does not parse __fp16
    # function signatures.  The ARM benchmark itself is evaluated on the
    # remote AArch64/SVE toolchain, so keep these rows in the artifact and mark
    # only the local smoke check as skipped.
    if "__fp16" in str(row.get("target_signature") or "") and "g++" in Path(compiler).name:
        return "local_skip_unsupported_fp16", "local g++ does not support __fp16 signatures"
    prelude = str(row.get("candidate_prelude") or "")
    harness = str(row.get("test_harness_code") or "")
    source = "\n".join(
        part
        for part in [
            "#include <algorithm>\n#include <cmath>\n#include <cstdint>\n#include <cstring>\n#include <cstdlib>",
            prelude,
            serial_code,
            harness,
        ]
        if part.strip()
    )
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "case.cpp"
        exe = Path(tmp) / "case.out"
        src.write_text(source, encoding="utf-8")
        res = subprocess.run(
            [compiler, "-std=c++17", "-O3", str(src), "-o", str(exe)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60,
        )
        if res.returncode != 0:
            return "compile_fail", res.stderr[-4000:]
        run = subprocess.run(
            [str(exe)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )
        if run.returncode != 0:
            return "run_fail", (run.stdout + "\n" + run.stderr)[-4000:]
    return "ok", ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default=str(DEFAULT_PROBLEM))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--compiler", default="/usr/bin/g++")
    args = ap.parse_args()

    problem = Path(args.problem)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    artifact_path = out_dir / "arm_simd_loops54.new_serial_code.v1.jsonl"
    overlay_path = out_dir / "arm_simd_loops54.problem.high_quality_serial_v1.jsonl"
    report_path = out_dir / "arm_simd_loops54.new_serial_code.v1.correctness_report.json"

    rows = read_jsonl(problem)
    artifact_rows: list[dict[str, Any]] = []
    overlay_rows: list[dict[str, Any]] = []
    report_rows: list[dict[str, Any]] = []

    for row in rows:
        tid = str(row["task_id"])
        if tid in QUALITY_OVERRIDES:
            serial_code = normalize_code(QUALITY_OVERRIDES[tid])
            source = "manual_high_quality_override_v1"
        else:
            serial_code = strip_top_includes(str(row.get("solution_scalar") or row.get("serial_c_code") or ""))
            source = "original_serial_passthrough_v1"

        status, detail = compile_and_run(row, serial_code, args.compiler)
        artifact_rows.append(
            {
                "task_id": tid,
                "target_signature": row.get("target_signature"),
                "candidate_prelude": row.get("candidate_prelude"),
                "serial_code": serial_code,
                "source": source,
                "correctness_status": status,
            }
        )
        new_row = dict(row)
        new_row["solution_scalar"] = serial_code
        new_row["serial_c_code"] = serial_code
        new_row["high_quality_serial_source"] = source
        new_row["high_quality_serial_version"] = "arm_simd_loops54_v1"
        overlay_rows.append(new_row)
        report_rows.append(
            {
                "task_id": tid,
                "source": source,
                "status": status,
                "detail_tail": detail,
            }
        )
        print(f"{tid} {source} {status}", flush=True)

    with artifact_path.open("w", encoding="utf-8") as f:
        for row in artifact_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with overlay_path.open("w", encoding="utf-8") as f:
        for row in overlay_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "problem": str(problem),
        "artifact": str(artifact_path),
        "overlay": str(overlay_path),
        "rows": len(rows),
        "manual_overrides": sum(1 for r in artifact_rows if r["source"].startswith("manual")),
        "passthrough": sum(1 for r in artifact_rows if r["source"].startswith("original")),
        "ok": sum(1 for r in report_rows if r["status"] == "ok"),
        "local_skip_unsupported_fp16": sum(
            1 for r in report_rows if r["status"] == "local_skip_unsupported_fp16"
        ),
        "compile_fail": sum(1 for r in report_rows if r["status"] == "compile_fail"),
        "run_fail": sum(1 for r in report_rows if r["status"] == "run_fail"),
    }
    report_path.write_text(
        json.dumps({"summary": summary, "rows": report_rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    locally_checked_or_skipped = summary["ok"] + summary["local_skip_unsupported_fp16"]
    if locally_checked_or_skipped != len(rows):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
