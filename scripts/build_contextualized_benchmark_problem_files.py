#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path("/home/user/selective_repo")
DEFAULT_OUT_DIR = REPO_ROOT / "results/generated/contextualized_benchmarks"
DEFAULT_ARM_ROWS = (
    REPO_ROOT
    / "results/generated/external_nohelper_eval_bundle/code2nl_semantic_audited/by_benchmark/arm_simd_loops/arm_simd_loops_generated_serial_c_seed_rows_v0.semantic_audited.jsonl"
)
DEFAULT_VEC_PROBLEM = (
    Path("/home/user/simdbench_full/results/generated/vecintrinbench_neon2nl_problem_pseudocode")
    / "vecintrinbench50_neon2nl.nohelper_parent_corrected.problem.jsonl"
)
DEFAULT_VEC_PARENT_ROWS = (
    REPO_ROOT
    / "results/vecintrinbench50_nohelper_serial_parent/vecintrinbench50_nohelper_serial_parent_rows.jsonl"
)

SVE_INCLUDE = "#include <arm_sve.h>"
ARM_C_HEADER = "\n".join(
    [
        "#include <stdint.h>",
        "#include <stddef.h>",
        "#include <stdbool.h>",
        "#include <string.h>",
        "#include <math.h>",
        "#include <stdio.h>",
        "#include <stdlib.h>",
    ]
)
VEC_CXX_HEADER = "\n".join(
    [
        "#include <algorithm>",
        "#include <cmath>",
        "#include <cstddef>",
        "#include <cstdint>",
        "#include <cstdlib>",
        "#include <cstring>",
        "#include <limits>",
        "#include <vector>",
    ]
)

BAD_PROMPT_API_RE = re.compile(r"\b(?:svexp|svtanh|svatan|sverf|svld1ub_gather_[A-Za-z0-9_]+)\b")
FUNC_DEF_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\([^;{}]*\)\s*\{", re.DOTALL)
CONTROL_NAMES = {"if", "for", "while", "switch", "return", "sizeof"}

ARM_SEQUENTIAL = {
    "arm_simd_loops.loop_009",  # linked-list pointer chasing
    "arm_simd_loops.loop_019",  # indexed object scatter
    "arm_simd_loops.loop_102",  # histogram scatter/update conflicts
    "arm_simd_loops.loop_104",  # histogram scatter/update conflicts
    "arm_simd_loops.loop_120",  # insertion sort
    "arm_simd_loops.loop_121",  # quicksort normalized as ascending sort
    "arm_simd_loops.loop_122",  # odd-even sort
    "arm_simd_loops.loop_123",  # bitonic sort normalized as ascending sort
    "arm_simd_loops.loop_124",  # radix sort normalized as ascending sort
}

ARM_HYBRID = {
    "arm_simd_loops.loop_005",  # bounded strlen over records
    "arm_simd_loops.loop_006",  # bounded strlen over records
    "arm_simd_loops.loop_022",  # variable-length TCP records + checksum loop
    "arm_simd_loops.loop_026",  # table-driven UTF-16 scan with exits
    "arm_simd_loops.loop_034",  # short string compare with variable advances
    "arm_simd_loops.loop_036",  # sparse indexed Gauss step
    "arm_simd_loops.loop_103",  # whitespace scanner
    "arm_simd_loops.loop_106",  # outer element loop with scalar bit packing
    "arm_simd_loops.loop_114",  # outer scalar lags, inner vectorizable dot
    "arm_simd_loops.loop_127",  # early-exit dot product
}

VEC_CONTEXT_TASKS = {
    "VecIntrinBench_atan",
    "VecIntrinBench_convsamp",
    "VecIntrinBench_cvtBGRtoHSV",
    "VecIntrinBench_eltwise",
    "VecIntrinBench_fdct_ifast",
    "VecIntrinBench_fdct_islow",
    "VecIntrinBench_h2v1_downsample",
    "VecIntrinBench_h2v1_fancy_upsample",
    "VecIntrinBench_h2v1_merged_upsample",
    "VecIntrinBench_h2v1_upsample",
    "VecIntrinBench_h2v2_downsample",
    "VecIntrinBench_h2v2_fancy_upsample",
    "VecIntrinBench_h2v2_merged_upsample",
    "VecIntrinBench_h2v2_upsample",
    "VecIntrinBench_idct_ifast",
    "VecIntrinBench_idct_islow",
    "VecIntrinBench_morph",
    "VecIntrinBench_quantize",
    "VecIntrinBench_rgb_gray_convert",
    "VecIntrinBench_rgb_ycc_convert",
    "VecIntrinBench_ycc_rgb_convert",
}

VEC_HYBRID_TASKS = {
    "VecIntrinBench_convsamp",
    "VecIntrinBench_fdct_ifast",
    "VecIntrinBench_fdct_islow",
    "VecIntrinBench_gelu",
    "VecIntrinBench_h2v1_downsample",
    "VecIntrinBench_h2v1_fancy_upsample",
    "VecIntrinBench_h2v1_merged_upsample",
    "VecIntrinBench_h2v1_upsample",
    "VecIntrinBench_h2v2_downsample",
    "VecIntrinBench_h2v2_fancy_upsample",
    "VecIntrinBench_h2v2_merged_upsample",
    "VecIntrinBench_h2v2_upsample",
    "VecIntrinBench_idct_ifast",
    "VecIntrinBench_idct_islow",
    "VecIntrinBench_instancenorm",
    "VecIntrinBench_mish",
    "VecIntrinBench_morph",
    "VecIntrinBench_packing",
    "VecIntrinBench_pooling",
    "VecIntrinBench_selu",
    "VecIntrinBench_sigmoid",
    "VecIntrinBench_softmax",
    "VecIntrinBench_swish",
    "VecIntrinBench_tanh",
}

VEC_ACTIVATION_MATH_TASKS = {
    "VecIntrinBench_gelu",
    "VecIntrinBench_mish",
    "VecIntrinBench_selu",
    "VecIntrinBench_sigmoid",
    "VecIntrinBench_softmax",
    "VecIntrinBench_swish",
    "VecIntrinBench_tanh",
}

VEC_JPEG_DOWNSAMPLE_TASKS = {
    "VecIntrinBench_h2v1_downsample",
    "VecIntrinBench_h2v2_downsample",
}

VEC_JPEG_MERGED_UPSAMPLE_TASKS = {
    "VecIntrinBench_h2v1_merged_upsample",
    "VecIntrinBench_h2v2_merged_upsample",
}

VEC_JPEG_HELPER_TASKS = VEC_JPEG_DOWNSAMPLE_TASKS | VEC_JPEG_MERGED_UPSAMPLE_TASKS

SVE_MATH_HELPER_NAMES = {
    "sve_exp_approx_f32",
    "sve_tanh_approx_f32",
    "sve_sigmoid_approx_f32",
    "sve_log1p_exp_hybrid_f32",
    "sve_erfc_hybrid_f32",
    "sve_gelu_erfc_hybrid_f32",
    "sve_mish_hybrid_f32",
}

SVE_MATH_HELPER_DECLS = """
/* Harness-provided SVE math helpers. Call them when needed; do not redefine them. */
svfloat32_t sve_exp_approx_f32(svbool_t pg, svfloat32_t x);
svfloat32_t sve_tanh_approx_f32(svbool_t pg, svfloat32_t x);
svfloat32_t sve_sigmoid_approx_f32(svbool_t pg, svfloat32_t x);
svfloat32_t sve_log1p_exp_hybrid_f32(svbool_t pg, svfloat32_t x);
svfloat32_t sve_erfc_hybrid_f32(svbool_t pg, svfloat32_t x);
svfloat32_t sve_gelu_erfc_hybrid_f32(svbool_t pg, svfloat32_t x);
svfloat32_t sve_mish_hybrid_f32(svbool_t pg, svfloat32_t x);
""".strip()

SVE_MATH_HELPER_IMPL = r"""
#include <arm_sve.h>
#include <cmath>
#include <cstddef>
#include <vector>

#ifndef SVE_LOOM_MATH_HELPERS_F32
#define SVE_LOOM_MATH_HELPERS_F32

static inline svfloat32_t sve_exp_approx_f32(svbool_t pg, svfloat32_t x)
{
    const svfloat32_t one = svdup_f32(1.0f);
    x = svmin_f32_x(pg, x, svdup_f32(88.3762626647949f));
    x = svmax_f32_x(pg, x, svdup_f32(-88.3762626647949f));

    svfloat32_t fx = svmla_f32_x(pg, svdup_f32(0.5f), x, svdup_f32(1.44269504088896341f));
    svint32_t emm0 = svcvt_s32_f32_x(pg, fx);
    svfloat32_t tmp = svcvt_f32_s32_x(pg, emm0);
    svbool_t mask = svcmpgt_f32(pg, tmp, fx);
    fx = svsel_f32(mask, svsub_f32_x(pg, tmp, one), tmp);

    tmp = svmul_f32_x(pg, fx, svdup_f32(0.693359375f));
    svfloat32_t z = svmul_f32_x(pg, fx, svdup_f32(-2.12194440e-4f));
    x = svsub_f32_x(pg, svsub_f32_x(pg, x, tmp), z);
    z = svmul_f32_x(pg, x, x);

    svfloat32_t y = svdup_f32(1.9875691500e-4f);
    y = svmla_f32_x(pg, svdup_f32(1.3981999507e-3f), y, x);
    y = svmla_f32_x(pg, svdup_f32(8.3334519073e-3f), y, x);
    y = svmla_f32_x(pg, svdup_f32(4.1665795894e-2f), y, x);
    y = svmla_f32_x(pg, svdup_f32(1.6666665459e-1f), y, x);
    y = svmla_f32_x(pg, svdup_f32(5.0000001201e-1f), y, x);
    y = svmla_f32_x(pg, x, y, z);
    y = svadd_f32_x(pg, y, one);

    svint32_t mm = svcvt_s32_f32_x(pg, fx);
    mm = svadd_s32_x(pg, mm, svdup_s32(0x7f));
    mm = svlsl_n_s32_x(pg, mm, 23);
    return svmul_f32_x(pg, y, svreinterpret_f32_s32(mm));
}

static inline svfloat32_t sve_tanh_approx_f32(svbool_t pg, svfloat32_t x)
{
    svfloat32_t ax = svabs_f32_x(pg, x);
    svbool_t non_tiny = svcmpge_f32(pg, ax, svdup_f32(1e-4f));
    ax = svmin_f32_x(pg, ax, svdup_f32(9.0f));

    svfloat32_t z = svmul_f32_x(pg, ax, ax);
    svfloat32_t y = svdup_f32(-2.76076847742355e-16f);
    y = svmla_f32_x(pg, svdup_f32(2.00018790482477e-13f), y, z);
    y = svmla_f32_x(pg, svdup_f32(-8.60467152213735e-11f), y, z);
    y = svmla_f32_x(pg, svdup_f32(5.12229709037114e-8f), y, z);
    y = svmla_f32_x(pg, svdup_f32(1.48572235717979e-5f), y, z);
    y = svmla_f32_x(pg, svdup_f32(6.37261928875436e-4f), y, z);
    y = svmla_f32_x(pg, svdup_f32(4.89352455891786e-3f), y, z);
    y = svmul_f32_x(pg, y, ax);

    svfloat32_t w = svdup_f32(1.19825839466702e-6f);
    w = svmla_f32_x(pg, svdup_f32(1.18534705686654e-4f), w, z);
    w = svmla_f32_x(pg, svdup_f32(2.26843463243900e-3f), w, z);
    w = svmla_f32_x(pg, svdup_f32(4.89352518554385e-3f), w, z);
    y = svdiv_f32_x(pg, y, w);

    svbool_t neg = svcmplt_f32(pg, x, svdup_f32(0.0f));
    y = svsel_f32(neg, svneg_f32_x(pg, y), y);
    return svsel_f32(non_tiny, y, x);
}

static inline svfloat32_t sve_sigmoid_approx_f32(svbool_t pg, svfloat32_t x)
{
    const svfloat32_t one = svdup_f32(1.0f);
    svfloat32_t e = sve_exp_approx_f32(pg, svneg_f32_x(pg, x));
    return svdiv_f32_x(pg, one, svadd_f32_x(pg, one, e));
}

static inline svfloat32_t sve_unary_f32_hybrid(
    svbool_t pg, svfloat32_t x, float (*fn)(float))
{
    const size_t vl = svcntw();
    std::vector<float> in(vl, 0.0f);
    std::vector<float> out(vl, 0.0f);
    svst1_f32(pg, in.data(), x);
    for (size_t i = 0; i < vl; ++i)
        out[i] = fn(in[i]);
    return svld1_f32(pg, out.data());
}

static inline float sve_log1p_exp_scalar(float x)
{
    if (x > 20.0f)
        return x;
    if (x < -20.0f)
        return std::exp(x);
    return std::log1p(std::exp(x));
}

static inline float sve_erfc_scalar(float x)
{
    return std::erfc(x);
}

static inline float sve_gelu_erfc_scalar(float x)
{
    return 0.5f * x * std::erfc(-0.7071067811865475f * x);
}

static inline float sve_mish_scalar(float x)
{
    float sp = sve_log1p_exp_scalar(x);
    return x * std::tanh(sp);
}

static inline svfloat32_t sve_log1p_exp_hybrid_f32(svbool_t pg, svfloat32_t x)
{
    return sve_unary_f32_hybrid(pg, x, sve_log1p_exp_scalar);
}

static inline svfloat32_t sve_erfc_hybrid_f32(svbool_t pg, svfloat32_t x)
{
    return sve_unary_f32_hybrid(pg, x, sve_erfc_scalar);
}

static inline svfloat32_t sve_gelu_erfc_hybrid_f32(svbool_t pg, svfloat32_t x)
{
    return sve_unary_f32_hybrid(pg, x, sve_gelu_erfc_scalar);
}

static inline svfloat32_t sve_mish_hybrid_f32(svbool_t pg, svfloat32_t x)
{
    return sve_unary_f32_hybrid(pg, x, sve_mish_scalar);
}

#endif
""".strip()

SVE_MORPH_HELPER_NAMES = {
    "sve_morph_const_row_u8",
    "sve_morph_row_u8",
    "sve_morph_kernel_3x3_u8",
    "sve_morph_erode_kernel_3x3_u8",
    "sve_morph_dilate_kernel_3x3_u8",
}

SVE_MORPH_HELPER_DECLS = """
/* Harness-provided SVE morphology helpers. Call them when needed; do not redefine them. */
int sve_morph_erode_kernel_3x3_u8(int width, int height, const uint8_t* srcBase,
                                  ptrdiff_t srcStride, uint8_t* dstBase,
                                  ptrdiff_t dstStride, const uint8_t* kernel,
                                  ptrdiff_t kernelStep, int anchorX, int anchorY,
                                  int borderType, uint8_t borderValue);
int sve_morph_dilate_kernel_3x3_u8(int width, int height, const uint8_t* srcBase,
                                   ptrdiff_t srcStride, uint8_t* dstBase,
                                   ptrdiff_t dstStride, const uint8_t* kernel,
                                   ptrdiff_t kernelStep, int anchorX, int anchorY,
                                   int borderType, uint8_t borderValue);
""".strip()

SVE_MORPH_HELPER_IMPL = r"""
#include <arm_sve.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#ifndef SVE_LOOM_MORPH_HELPERS_U8
#define SVE_LOOM_MORPH_HELPERS_U8

static inline const uint8_t* sve_morph_const_row_u8(const uint8_t* base, ptrdiff_t stride, size_t row)
{
    const char* raw = reinterpret_cast<const char*>(base);
    return reinterpret_cast<const uint8_t*>(raw + ptrdiff_t(row) * stride);
}

static inline uint8_t* sve_morph_row_u8(uint8_t* base, ptrdiff_t stride, size_t row)
{
    char* raw = reinterpret_cast<char*>(base);
    return reinterpret_cast<uint8_t*>(raw + ptrdiff_t(row) * stride);
}

static inline int sve_morph_kernel_3x3_u8(int width, int height, const uint8_t* srcBase,
                                          ptrdiff_t srcStride, uint8_t* dstBase,
                                          ptrdiff_t dstStride, const uint8_t* kernel,
                                          ptrdiff_t kernelStep, int anchorX, int anchorY,
                                          int borderType, uint8_t borderValue, bool erode)
{
    std::vector<std::pair<int, int>> kernel_positions;
    for (int ky = 0; ky < 3; ++ky) {
        for (int kx = 0; kx < 3; ++kx) {
            if (kernel[ptrdiff_t(ky) * kernelStep + kx] != 0)
                kernel_positions.push_back({ky - anchorY, kx - anchorX});
        }
    }

    if (kernel_positions.empty()) {
        for (int y = 0; y < height; ++y) {
            uint8_t* drow = sve_morph_row_u8(dstBase, dstStride, size_t(y));
            std::fill(drow, drow + width, borderValue);
        }
        return 0;
    }

    const size_t vl_bytes = svcntb();
    std::vector<uint8_t> temp(vl_bytes);

    for (int y = 0; y < height; ++y) {
        uint8_t* drow = sve_morph_row_u8(dstBase, dstStride, size_t(y));
        int x = 0;
        while (x < width) {
            svbool_t pg = svwhilelt_b8(x, width);
            const int active = std::min<int>(int(vl_bytes), width - x);
            svuint8_t result = svdup_u8(erode ? 255u : 0u);
            bool first = true;

            for (const auto& pos : kernel_positions) {
                int sy = y + pos.first;
                int sx_base = x + pos.second;
                svuint8_t data;

                if (sy < 0 || sy >= height) {
                    if (borderType == 0) {
                        data = svdup_u8(borderValue);
                    } else {
                        sy = std::max(0, std::min(sy, height - 1));
                        const uint8_t* sptr = sve_morph_const_row_u8(srcBase, srcStride, size_t(sy));
                        for (int lane = 0; lane < active; ++lane) {
                            int sx = sx_base + lane;
                            sx = std::max(0, std::min(sx, width - 1));
                            temp[size_t(lane)] = sptr[sx];
                        }
                        data = svld1_u8(pg, temp.data());
                    }
                } else {
                    const uint8_t* sptr = sve_morph_const_row_u8(srcBase, srcStride, size_t(sy));
                    if (sx_base >= 0 && sx_base + active <= width) {
                        data = svld1_u8(pg, sptr + sx_base);
                    } else {
                        for (int lane = 0; lane < active; ++lane) {
                            int sx = sx_base + lane;
                            if (sx < 0 || sx >= width) {
                                if (borderType == 0) {
                                    temp[size_t(lane)] = borderValue;
                                } else {
                                    sx = std::max(0, std::min(sx, width - 1));
                                    temp[size_t(lane)] = sptr[sx];
                                }
                            } else {
                                temp[size_t(lane)] = sptr[sx];
                            }
                        }
                        data = svld1_u8(pg, temp.data());
                    }
                }

                if (first) {
                    result = data;
                    first = false;
                } else if (erode) {
                    result = svmin_u8_x(pg, result, data);
                } else {
                    result = svmax_u8_x(pg, result, data);
                }
            }

            svst1_u8(pg, drow + x, result);
            x += int(vl_bytes);
        }
    }
    return 0;
}

static inline int sve_morph_erode_kernel_3x3_u8(int width, int height, const uint8_t* srcBase,
                                                ptrdiff_t srcStride, uint8_t* dstBase,
                                                ptrdiff_t dstStride, const uint8_t* kernel,
                                                ptrdiff_t kernelStep, int anchorX, int anchorY,
                                                int borderType, uint8_t borderValue)
{
    return sve_morph_kernel_3x3_u8(width, height, srcBase, srcStride, dstBase, dstStride,
                                   kernel, kernelStep, anchorX, anchorY, borderType, borderValue, true);
}

static inline int sve_morph_dilate_kernel_3x3_u8(int width, int height, const uint8_t* srcBase,
                                                 ptrdiff_t srcStride, uint8_t* dstBase,
                                                 ptrdiff_t dstStride, const uint8_t* kernel,
                                                 ptrdiff_t kernelStep, int anchorX, int anchorY,
                                                 int borderType, uint8_t borderValue)
{
    return sve_morph_kernel_3x3_u8(width, height, srcBase, srcStride, dstBase, dstStride,
                                   kernel, kernelStep, anchorX, anchorY, borderType, borderValue, false);
}

#endif
""".strip()

SVE_JPEG_HELPER_NAMES = {
    "jpeg_expand_right_edge_u8",
    "jpeg_clamp_u8_i32",
}

SVE_JPEG_HELPER_DECLS = """
/* Harness-provided JPEG helpers. Call them when needed; do not redefine them. */
void jpeg_expand_right_edge_u8(unsigned char** image_data, int num_rows,
                               unsigned int input_cols, unsigned int output_cols);
unsigned char jpeg_clamp_u8_i32(int value);
""".strip()

SVE_JPEG_HELPER_IMPL = r"""
#include <algorithm>
#include <cstddef>
#include <cstdint>

#ifndef SVE_LOOM_JPEG_HELPERS_U8
#define SVE_LOOM_JPEG_HELPERS_U8

static inline void jpeg_expand_right_edge_u8(unsigned char** image_data, int num_rows,
                                             unsigned int input_cols, unsigned int output_cols)
{
    if (!image_data || num_rows <= 0 || output_cols <= input_cols)
        return;
    const int extra = int(output_cols - input_cols);
    for (int row = 0; row < num_rows; ++row) {
        unsigned char* ptr = image_data[row] + input_cols;
        const unsigned char pixval = ptr[-1];
        for (int c = 0; c < extra; ++c)
            ptr[c] = pixval;
    }
}

static inline unsigned char jpeg_clamp_u8_i32(int value)
{
    return static_cast<unsigned char>(value < 0 ? 0 : (value > 255 ? 255 : value));
}

#endif
""".strip()

REQUIRED_FORBIDDEN_RESPONSE_TOKENS = {
    "#define",
    "typedef",
    "using",
    "enum",
    "static inline helper",
    "extra function definition",
    "extra entry point",
    "global variable definition",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


BARE_RESTRICT_RE = re.compile(r"(?<!_)\brestrict\b(?!_)")
POINTER_DECL_VOID_CAST_RE = re.compile(
    r"(?P<prefix>\b(?P<type>(?:const\s+|volatile\s+)*(?:struct\s+[A-Za-z_][A-Za-z0-9_]*|[A-Za-z_][A-Za-z0-9_]*))\s*\*\s*[A-Za-z_][A-Za-z0-9_]*\s*=\s*)\(void\s*\*\)"
)
CXX17_RESTRICT_CODE_FIELDS = {
    "serial_c_code",
    "test_harness_code",
    "test_correctness",
    "solution_scalar",
}
ARM_CODE_FIELDS = set(CXX17_RESTRICT_CODE_FIELDS)
ARM_LOOP038_FP16_PRELUDE = """typedef float FLOAT16_t;
#define fp16_to_native(x) ((float)(x))
#define native_to_fp16(x) ((__fp16)(x))"""


def normalize_cxx17_pointer_void_casts(text: str) -> str:
    """C++17 does not implicitly convert void* to typed object pointers."""

    def repl(match: re.Match[str]) -> str:
        pointee = re.sub(r"\s+", " ", match.group("type")).strip()
        return f"{match.group('prefix')}({pointee} *)"

    return POINTER_DECL_VOID_CAST_RE.sub(repl, str(text or ""))


def normalize_cxx17_restrict_keywords(row: dict[str, Any]) -> dict[str, Any]:
    """C++17 accepts compiler restrict spellings, not the C-only bare keyword."""
    out = dict(row)
    for key in CXX17_RESTRICT_CODE_FIELDS:
        value = out.get(key)
        if isinstance(value, str) and "restrict" in value:
            out[key] = BARE_RESTRICT_RE.sub("__restrict__", value)
        value = out.get(key)
        if isinstance(value, str) and "(void *)" in value:
            out[key] = normalize_cxx17_pointer_void_casts(value)
    return out


def normalize_arm_loop038_fp16_context(text: str) -> str:
    """Use raw __fp16 for the loop_038 API to avoid float16_t typedef conflicts."""
    source = str(text or "")
    if not source.strip():
        return source
    source = re.sub(r"(?m)^\s*typedef\s+(?:__fp16|float)\s+float16_t\s*;\s*\n?", "", source)
    source = re.sub(r"(?m)^\s*typedef\s+float\s+FLOAT16_t\s*;\s*\n?", "", source)
    source = re.sub(r"(?m)^\s*#define\s+fp16_to_native\(x\).*\n?", "", source)
    source = re.sub(r"(?m)^\s*#define\s+native_to_fp16\(x\).*\n?", "", source)
    source = re.sub(r"\bfloat16_t\b", "__fp16", source)
    return ARM_LOOP038_FP16_PRELUDE + "\n\n" + source.lstrip()


def normalize_arm_source_row(task_id: str, row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    if task_id == "arm_simd_loops.loop_038":
        for key in ARM_CODE_FIELDS:
            value = out.get(key)
            if isinstance(value, str) and value.strip():
                out[key] = normalize_arm_loop038_fp16_context(value)
        description = out.get("nl_description_ds_r1")
        if isinstance(description, str) and description.strip():
            out["nl_description_ds_r1"] = re.sub(r"\bfloat16_t\b", "__fp16", description)
    return out


def normalize_signature(signature: str) -> str:
    s = str(signature or "").strip()
    s = re.sub(r"^\s*extern\s+", "", s)
    s = re.sub(r";\s*$", "", s)
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s*([(),*])\s*", r"\1", s)
    s = s.replace("*", " * ")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s+,", ",", s)
    s = re.sub(r"\(\s+", "(", s)
    s = re.sub(r"\s+\)", ")", s)
    return s.strip()


def strip_markdown_fences(text: str) -> str:
    s = str(text or "").strip()
    match = re.search(r"```(?:c|cc|cpp|c\+\+)?\s*([\s\S]*?)```", s, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return s.replace("```", "").strip()


def extract_entrypoint_from_signature(signature: str) -> str:
    names = re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", str(signature or ""))
    for name in reversed(names):
        if name not in CONTROL_NAMES:
            return name
    return ""


def extract_signature_from_code(code: str, entrypoint: str) -> str:
    body = strip_markdown_fences(code)
    if not entrypoint:
        raise ValueError("missing entrypoint")
    pattern = re.compile(
        rf"^[ \t]*([^\n;{{}}]*\b{re.escape(entrypoint)}\s*\([^;{{}}]*\))\s*\{{",
        flags=re.MULTILINE,
    )
    match = pattern.search(body)
    if not match:
        raise ValueError(f"cannot extract signature for {entrypoint}")
    sig = " ".join(match.group(1).split())
    sig = re.sub(r"\s+\*", " *", sig)
    return sig.strip()


def split_prelude_items(text: str) -> list[str]:
    items: list[str] = []
    i = 0
    lines = str(text or "").strip().splitlines()
    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()
        if not stripped:
            i += 1
            continue
        if stripped.startswith("#include"):
            items.append(stripped)
            i += 1
            continue
        if stripped.startswith("#define "):
            block = [line]
            i += 1
            while i < len(lines):
                if not block[-1].rstrip().endswith("\\"):
                    break
                block.append(lines[i].rstrip())
                i += 1
            items.append("\n".join(block).strip())
            continue
        if "{" in stripped and not stripped.endswith(";"):
            block = [line]
            balance = line.count("{") - line.count("}")
            i += 1
            while i < len(lines) and balance > 0:
                block.append(lines[i].rstrip())
                balance += lines[i].count("{") - lines[i].count("}")
                i += 1
            items.append("\n".join(block).strip())
            continue
        if stripped.startswith("typedef ") and stripped.endswith(";"):
            items.append(stripped)
            i += 1
            continue
        if stripped.startswith("using ") and stripped.endswith(";"):
            items.append(stripped)
            i += 1
            continue
        if stripped.endswith(";"):
            items.append(stripped)
            i += 1
            continue
        block = [line]
        i += 1
        while i < len(lines) and lines[i].strip():
            block.append(lines[i].rstrip())
            i += 1
        items.append("\n".join(block).strip())
    return [item for item in items if item.strip()]


def prelude_item_key(item: str) -> tuple[str, str]:
    text = "\n".join(line.rstrip() for line in str(item or "").strip().splitlines()).strip()
    normalized = re.sub(r"\s+", " ", text).strip()
    macro = re.match(r"^\s*#\s*define\s+([A-Za-z_][A-Za-z0-9_]*)", text)
    if macro:
        return ("macro", macro.group(1))
    include = re.match(r"^\s*#\s*include\s+(.+?)\s*$", text)
    if include:
        return ("include", include.group(1).strip())
    using = re.match(r"^\s*using\s+([A-Za-z_][A-Za-z0-9_]*)\s*=", text)
    if using:
        return ("using", using.group(1))
    if re.match(r"^\s*typedef\b", text):
        function_pointer = re.search(r"\(\s*\*\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)", text)
        if function_pointer:
            return ("typedef", function_pointer.group(1))
        alias = re.search(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*;\s*$", text)
        if alias:
            return ("typedef", alias.group(1))
    return ("text", normalized)


def merge_prelude_items(items: list[str]) -> str:
    merged: list[str] = []
    seen: set[tuple[str, str]] = set()
    for item in items:
        text = str(item or "").strip()
        if not text:
            continue
        key = prelude_item_key(text)
        if key in seen:
            continue
        seen.add(key)
        merged.append(text)
    return "\n".join(merged).strip()


def extract_problem_prelude_from_code(code: str, signature: str) -> str:
    body = strip_markdown_fences(code)
    idx = body.find(signature)
    if idx < 0:
        entrypoint = extract_entrypoint_from_signature(signature)
        match = re.search(rf"\b{re.escape(entrypoint)}\s*\(", body) if entrypoint else None
        idx = match.start() if match else -1
    if idx < 0:
        return ""
    pre = body[:idx]
    items: list[str] = []
    for item in split_prelude_items(pre):
        stripped = item.strip()
        if stripped.startswith("#include"):
            continue
        if stripped.startswith("#define "):
            items.append(stripped)
            continue
        if stripped.startswith("typedef "):
            items.append(stripped)
            continue
        if stripped.startswith("using "):
            items.append(stripped)
            continue
    return merge_prelude_items(items)


def extract_comment_body(text: str) -> str:
    s = strip_markdown_fences(text)
    match = re.search(r"/\*([\s\S]*?)\*/", s)
    if match:
        s = match.group(1)
    s = re.sub(r"^\s*\*\s?", "", s, flags=re.MULTILINE)
    s = re.sub(
        r"\n\s*The requirement is to implement the function with SVE[\s\S]*?$",
        "",
        s,
        flags=re.IGNORECASE,
    )
    s = re.sub(r"\n\s*[A-Za-z_][\w\s\*]+[A-Za-z_]\w*\s*\([^;{}]*\)\s*\{\s*\}\s*$", "", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def append_task_notes(task_id: str, description: str) -> str:
    notes: list[str] = []
    if task_id in VEC_ACTIVATION_MATH_TASKS:
        notes.append(
            "Transcendental math note: implement the stated formula within the benchmark tolerance. "
            "The harness provides SVE math helper functions for exp/tanh/sigmoid/log1p-exp/erfc-style kernels; call them when needed and do not invent unavailable vector transcendental intrinsics."
        )
    if task_id == "VecIntrinBench_morph":
        notes.append(
            "Morph helper note: the harness provides SVE 3x3 uint8 erode and dilate kernel helpers; after validating the arguments and normalizing negative anchors to 1, call the matching helper instead of redefining morphology helper functions."
        )
    if task_id in VEC_JPEG_DOWNSAMPLE_TASKS:
        notes.append(
            "JPEG helper note: the harness provides jpeg_expand_right_edge_u8(...). Call it before the vectorized downsampling loop to reproduce the right-edge padding side effect; do not flatten row pointers."
        )
    if task_id in VEC_JPEG_MERGED_UPSAMPLE_TASKS:
        notes.append(
            "JPEG helper note: the harness provides jpeg_clamp_u8_i32(...). Use it for scalar tail or hybrid edge handling, and preserve JSAMPIMAGE/JSAMPARRAY row and plane pointer semantics."
        )
    if any(token in description for token in ["JSAMPARRAY", "JSAMPIMAGE", "output_buf", "input_buf"]):
        notes.append(
            "Pointer-layout note: preserve the row/plane pointer semantics exactly; do not flatten JSAMPARRAY or JSAMPIMAGE into a single byte pointer."
        )
    if "table" in description.lower() or "quant" in task_id.lower():
        notes.append(
            "Context table note: constants, typedefs, tables, and macros supplied by the prelude or harness are already initialized for this call; use them but do not redefine or rebuild them."
        )
    if not notes:
        return description.strip()
    return description.rstrip() + "\n\n" + "\n".join(notes)


def response_contract() -> dict[str, Any]:
    return {
        "allowed_output": "exact_target_function_definition_only",
        "forbidden_in_response": sorted(REQUIRED_FORBIDDEN_RESPONSE_TOKENS),
        "signature_must_match_prompt": True,
        "candidate_prelude_provided_by_harness": True,
    }


def render_prompt(
    *,
    signature: str,
    description: str,
    prelude: str,
    category: str,
    vectorization_strategy: str,
) -> str:
    del category, vectorization_strategy
    clean_description = description.strip()
    if SVE_REQUIREMENT_SENTENCE not in clean_description:
        clean_description = clean_description.rstrip() + "\n\n" + SVE_REQUIREMENT_SENTENCE

    parts = [SVE_INCLUDE]
    if prelude.strip():
        parts.append(prelude.strip())
    parts.append("/*\n" + clean_description.strip() + "\n*/")
    parts.append(signature.strip() + " {\n}")
    return "\n\n".join(parts).rstrip() + "\n"


def needs_sve_math_helpers(task_id: str, description: str) -> bool:
    if task_id in VEC_ACTIVATION_MATH_TASKS:
        return True
    text = f"{task_id}\n{description}".lower()
    return any(token in text for token in ["exp(", "expf", "tanh(", "tanhf", "erfc(", "erfcf", "log1p"])


def needs_sve_morph_helpers(task_id: str) -> bool:
    return task_id == "VecIntrinBench_morph"


def needs_sve_jpeg_helpers(task_id: str) -> bool:
    return task_id in VEC_JPEG_HELPER_TASKS


def merge_prelude(*chunks: str) -> str:
    items: list[str] = []
    for chunk in chunks:
        items.extend(split_prelude_items(chunk))
    return merge_prelude_items(items)


SVE_REQUIREMENT_SENTENCE = (
    "The requirement is to implement the function with SVE "
    "(Arm C Language Extensions (ACLE) for the Arm Scalable Vector Extension (SVE)) "
    "intrinsics for parallelism."
)


ARM_PROMPT_OVERRIDES = {
    "arm_simd_loops.loop_110": """#include <arm_sve.h>

typedef struct cint8_t { int8_t re, im; } cint8_t;
typedef struct cint32_t { int32_t re, im; } cint32_t;

/*
The inputs to this function are:
- a0: pointer to an AoS array of cint8_t complex numbers. Each element is stored in memory as adjacent fields {re, im}; the array memory order is re0, im0, re1, im1, ...
- b0: pointer to an AoS array of cint8_t complex numbers with the same layout.
- c0: pointer to an AoS output array of cint32_t complex numbers with memory order re0, im0, re1, im1, ...
- size: the number of output complex elements to compute.

Behavior:
for (i = 0; i < size; i++) {
    x0 = a0[2*i], y0 = b0[2*i]
    x1 = a0[2*i+1], y1 = b0[2*i+1]
    c0[i].re = int32_t((x0.re*y0.re - x0.im*y0.im) + (x1.re*y1.re - x1.im*y1.im))
    c0[i].im = int32_t((x0.im*y0.re + x0.re*y0.im) + (x1.im*y1.re + x1.re*y1.im))
}

AoS/interleaved layout note:
- Do not treat c0[i].re and c0[i].im as two separate contiguous arrays.
- When vectorizing, deinterleave real/imag fields from the AoS inputs before arithmetic, then interleave real/imag fields before storing back to the AoS output.
- If using SVE tuple load/store intrinsics, use valid ACLE tuple construction/access for the element type; do not invent .val[] fields or store a single vector into a struct array incorrectly.

The requirement is to implement the function with SVE intrinsics for parallelism.
*/

void arm_simd_loop_110(cint8_t * a0, cint8_t * b0, cint32_t * c0, uint64_t size) {
}
""",
    "arm_simd_loops.loop_111": """#include <arm_sve.h>

/*
The inputs to this function are:
- `input1`: pointer to an array of double-precision floating-point numbers (read-only).
- `input2`: pointer to an array of double-precision floating-point numbers (read-only).
- `output`: pointer to an array of double-precision floating-point numbers (writable).
- `exponent`: pointer to an array of 64-bit signed integers (writable).
- `size`: the number of elements to process (non-negative).

The function processes elements from index 0 to `size-1`. For each index `i`:
1. It computes `output[i] = frexp(input1[i], (int *)&exponent[i])`, storing the fraction in `output[i]` and the exponent through the exponent pointer.
2. It computes `output[i] = ldexp(output[i], 1)`.
3. It computes `output[i] = output[i] * input2[i]`.
4. It decrements `exponent[i]` by 1.

No return value. Precondition: all pointers must be valid, and `output` and `exponent` arrays must be writable with capacity for at least `size` elements.

The requirement is to implement the function with SVE intrinsics for parallelism.
*/

void arm_simd_loop_111(double * input1, double * input2, double * output, int64_t * exponent, int64_t size) {
}
""",
    "arm_simd_loops.loop_112": """#include <arm_sve.h>

typedef struct cuint32_t { uint32_t re, im; } cuint32_t;

/*
The inputs to this function are:
- a0: pointer to an AoS array of cuint32_t complex numbers. Each element is stored as adjacent fields {re, im}; memory order is re0, im0, re1, im1, ...
- b0: pointer to an AoS array of cuint32_t complex numbers with the same layout.
- c0: pointer to an AoS output array of cuint32_t complex numbers with the same interleaved real/imag layout.
- size: the number of complex elements to process.

Behavior:
for (i = 0; i < size; i++) {
    c0[i].re = (a0[i].re * b0[i].re) - (a0[i].im * b0[i].im);
    c0[i].im = (a0[i].re * b0[i].im) + (a0[i].im * b0[i].re);
}

AoS/interleaved layout note:
- `c0[i].re` and `c0[i].im` are fields of one struct array element, not separate contiguous arrays.
- Vector code should load/deinterleave real and imaginary fields, compute both result components, then interleave/store them back as AoS complex elements.
- If using SVE tuple intrinsics, use valid ACLE tuple types and accessors for interleaved load/store.

The requirement is to implement the function with SVE intrinsics for parallelism.
*/

void arm_simd_loop_112(cuint32_t * a0, cuint32_t * b0, cuint32_t * c0, uint64_t size) {
}
""",
    "arm_simd_loops.loop_113": """#include <arm_sve.h>

/*
The inputs to this function are:
- a0: pointer to an input array of uint32_t.
- b0: pointer to an input array of uint32_t.
- c0: pointer to an output array of uint32_t.
- size: number of elements in each array; it may be odd, and i + 1 is valid for each executed iteration.

Behavior:
for (i = 0; i < size; i += 2) {
    c0[i] = a0[i] + a0[i + 1];
    c0[i + 1] = b0[i] + b0[i + 1];
}

Interleaved pair layout note:
- Each loop iteration consumes one adjacent pair from a0 and b0.
- The output is interleaved by pair: even output positions come from a0 pairs, odd output positions come from b0 pairs.
- Do not collapse this into a single ordinary contiguous store map. Preserve the even/odd output layout, using indexed or interleaved stores if vectorized.

The requirement is to implement the function with SVE intrinsics for parallelism.
*/

void arm_simd_loop_113(uint32_t * a0, uint32_t * b0, uint32_t * c0, uint64_t size) {
}
""",
}


def render_arm_simdbench_style_prompt(
    *,
    signature: str,
    description: str,
    prelude: str,
) -> str:
    clean_description = description.strip()
    if SVE_REQUIREMENT_SENTENCE not in clean_description:
        clean_description = clean_description.rstrip() + "\n\n" + SVE_REQUIREMENT_SENTENCE

    parts = [SVE_INCLUDE]
    if prelude.strip():
        parts.append(prelude.strip())
    parts.append("/*\n" + clean_description + "\n*/")
    parts.append(signature.strip() + " {\n}")
    return "\n\n".join(parts).rstrip() + "\n"


def wrap_nonbenchmark_harness(harness: str) -> str:
    body = strip_markdown_fences(harness)
    if '"correctness"' in body:
        return body
    match = re.search(r"\bint\s+main\s*\(\s*(?:void\s*)?\)", body)
    if not match:
        return body
    rewritten = body[: match.start()] + "static int nonbenchmark_correctness_main(void)" + body[match.end() :]
    wrapper = (
        'int main(void) {\n'
        "    int rc = nonbenchmark_correctness_main();\n"
        '    printf("{ \\"correctness\\": %s }\\n", rc == 0 ? "1" : "0");\n'
        "    return 0;\n"
        "}\n"
    )
    return rewritten.rstrip() + "\n\n" + wrapper


def signature_to_extern_decl(signature: str) -> str:
    return "extern " + str(signature or "").strip().rstrip(";") + ";"


def primary_category(vectorization_strategy: str, context_required: bool, not_suitable: bool = False) -> str:
    if not_suitable:
        return "not_suitable_for_single_function"
    if vectorization_strategy == "sequential":
        return "sequential_holdout"
    if context_required:
        return "requires_context_prelude"
    if vectorization_strategy == "hybrid":
        return "hybrid_single_function"
    return "pure_sve_single_function"


def classify_arm(row: dict[str, Any], prelude: str) -> tuple[str, str, list[str]]:
    task_id = str(row.get("task_id") or row.get("sample_id") or "")
    reasons: list[str] = []
    if task_id in ARM_SEQUENTIAL:
        strategy = "sequential"
        reasons.append("algorithm has pointer chasing, sorting, indexed scatter, or update-conflict histogram behavior")
    elif task_id in ARM_HYBRID:
        strategy = "hybrid"
        reasons.append("algorithm has scalar control flow with vectorizable inner work")
    else:
        strategy = "pure_sve"
        reasons.append("single target function with direct vectorizable loop structure")
    context_required = bool(prelude.strip())
    if context_required:
        reasons.append("target signature/body depends on benchmark typedefs or macros supplied by candidate_prelude")
    return primary_category(strategy, context_required), strategy, reasons


def classify_vec(row: dict[str, Any], prelude: str) -> tuple[str, str, list[str]]:
    task_id = str(row.get("task_id") or "")
    signature = str(row.get("target_signature") or row.get("harness_extern_decl") or "")
    reasons: list[str] = []
    context_required = (
        task_id in VEC_CONTEXT_TASKS
        or bool(re.search(r"^\s*(?:#define|typedef|using|enum)\b", prelude, flags=re.MULTILINE))
        or "std::" in signature
        or any(token in signature for token in ["JSAMP", "DCT", "JCOEF", "JDIMENSION"])
    )
    if task_id in VEC_HYBRID_TASKS:
        strategy = "hybrid"
        reasons.append("requires scalar outer/control flow, reductions, table-driven math, or scalar math fallback")
    else:
        strategy = "pure_sve"
        reasons.append("single target function with direct elementwise or blockwise vectorizable work")
    if context_required:
        reasons.append("target depends on benchmark typedefs, macros, C++ types, or JPEG-style pointer aliases")
    return primary_category(strategy, context_required), strategy, reasons


def build_arm(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    input_path = Path(args.arm_rows).expanduser().resolve()
    rows = read_jsonl(input_path)
    problem_rows: list[dict[str, Any]] = []
    remote_rows: list[dict[str, Any]] = []
    classified_rows: list[dict[str, Any]] = []

    for row in rows:
        task_id = str(row.get("task_id") or row.get("sample_id") or "")
        row = normalize_arm_source_row(task_id, row)
        loop_no = task_id.rsplit("_", 1)[-1] if "_" in task_id else task_id.rsplit(".", 1)[-1]
        entrypoint = f"arm_simd_loop_{loop_no}"
        signature = extract_signature_from_code(str(row.get("serial_c_code") or ""), entrypoint)
        harness_extern_decl = signature_to_extern_decl(signature)
        prelude = extract_problem_prelude_from_code(str(row.get("serial_c_code") or ""), signature)
        category, strategy, reasons = classify_arm(row, prelude)
        prompt_override = ARM_PROMPT_OVERRIDES.get(task_id)
        if prompt_override:
            prompt = prompt_override.rstrip() + "\n"
            description = extract_comment_body(prompt)
        else:
            description = extract_comment_body(str(row.get("nl_description_ds_r1") or ""))
            prompt = render_arm_simdbench_style_prompt(
                signature=signature,
                description=description,
                prelude=prelude,
            )
        base_meta = {
            "benchmark_category": category,
            "vectorization_strategy": strategy,
            "context_required": bool(prelude.strip()),
            "candidate_prelude_present": bool(prelude.strip()),
            "classification_reasons": reasons,
            "target_signature": signature,
            "harness_extern_decl": harness_extern_decl,
            "harness_extern_entrypoint": entrypoint,
            "candidate_prelude": prelude,
            "response_contract": response_contract(),
            "benchmark_boundary": "benchmark_holdout_eval_only_do_not_train",
        }
        classified = dict(row)
        classified.update(base_meta)
        classified["contextualized_prompt"] = prompt
        classified_rows.append(classified)

        common = {
            "task_id": task_id,
            "sample_id": row.get("sample_id") or task_id,
            "source_name": row.get("source_name"),
            "source_type": row.get("source_type"),
            "schema_version": row.get("schema_version"),
            "prompt": prompt,
            "prompt_field": "contextualized_prompt",
            "nl_description_ds_r1": description,
            "target_signature": signature,
            "harness_extern_decl": harness_extern_decl,
            "harness_extern_entrypoint": entrypoint,
            "candidate_prelude": prelude,
            "response_contract": response_contract(),
            "benchmark_category": category,
            "vectorization_strategy": strategy,
            "context_required": bool(prelude.strip()),
            "classification_reasons": reasons,
            "serial_c_code": row.get("serial_c_code") or "",
            "test_harness_code": row.get("test_harness_code") or "",
            "intrinsic": "SVE",
            "task": "",
            "source_jsonl": str(input_path),
        }
        problem_rows.append(common)
        remote = dict(common)
        remote.update(
            {
                "entrypoint": entrypoint,
                "entry_point": entrypoint,
                "entrypoint_simd": entrypoint,
                "entrypoint_scalar": entrypoint,
                "solution_scalar": "",
                "test_correctness": harness_extern_decl
                + "\n\n"
                + wrap_nonbenchmark_harness(str(row.get("test_harness_code") or "")),
                "compile_modes": ["c11"],
            }
        )
        remote_rows.append(remote)

    manifest = manifest_for("arm_simd_loops", input_path, classified_rows)
    return problem_rows, remote_rows, manifest, classified_rows


def build_vec(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    input_path = Path(args.vec_problem).expanduser().resolve()
    parent_path = Path(args.vec_parent_rows).expanduser().resolve()
    rows = read_jsonl(input_path)
    parents = {str(r.get("task_id") or ""): r for r in read_jsonl(parent_path)}
    problem_rows: list[dict[str, Any]] = []
    remote_rows: list[dict[str, Any]] = []
    classified_rows: list[dict[str, Any]] = []

    for row in rows:
        task_id = str(row.get("task_id") or "")
        extern_decl = str(row.get("harness_extern_decl") or "").strip()
        if extern_decl:
            signature = re.sub(r"^\s*extern\s+", "", extern_decl)
            signature = re.sub(r";\s*$", "", signature).strip()
        else:
            signature = str(row.get("target_signature") or row.get("signature") or "").strip()
        if not signature:
            raise ValueError(f"{task_id}: missing target signature")
        base_prelude = str(row.get("candidate_prelude") or "").strip()
        description = append_task_notes(task_id, extract_comment_body(str(row.get("nl_description_ds_r1") or row.get("prompt") or "")))
        math_helpers_provided = needs_sve_math_helpers(task_id, description)
        morph_helpers_provided = needs_sve_morph_helpers(task_id)
        jpeg_helpers_provided = needs_sve_jpeg_helpers(task_id)
        prompt_prelude = merge_prelude(
            base_prelude,
            SVE_MATH_HELPER_DECLS if math_helpers_provided else "",
            SVE_MORPH_HELPER_DECLS if morph_helpers_provided else "",
            SVE_JPEG_HELPER_DECLS if jpeg_helpers_provided else "",
        )
        compile_prelude = merge_prelude(
            base_prelude,
            SVE_MATH_HELPER_IMPL if math_helpers_provided else "",
            SVE_MORPH_HELPER_IMPL if morph_helpers_provided else "",
            SVE_JPEG_HELPER_IMPL if jpeg_helpers_provided else "",
        )
        category, strategy, reasons = classify_vec({**row, "target_signature": signature}, compile_prelude)
        prompt = render_prompt(
            signature=signature,
            description=description,
            prelude=prompt_prelude,
            category=category,
            vectorization_strategy=strategy,
        )
        parent_sig = str(parents.get(task_id, {}).get("signature") or "").strip()
        base_meta = {
            "benchmark_category": category,
            "vectorization_strategy": strategy,
            "context_required": category == "requires_context_prelude",
            "candidate_prelude_present": bool(compile_prelude.strip()),
            "prompt_prelude": prompt_prelude,
            "classification_reasons": reasons,
            "target_signature": signature,
            "parent_signature": parent_sig,
            "candidate_prelude": compile_prelude,
            "sve_math_helpers_provided": math_helpers_provided,
            "sve_morph_helpers_provided": morph_helpers_provided,
            "sve_jpeg_helpers_provided": jpeg_helpers_provided,
            "response_contract": response_contract(),
            "benchmark_boundary": "benchmark_holdout_eval_only_do_not_train",
        }
        classified = dict(row)
        classified.update(base_meta)
        classified["contextualized_prompt"] = prompt
        classified_rows.append(classified)

        common = dict(row)
        common.update(base_meta)
        common.update(
            {
                "prompt": prompt,
                "prompt_field": "contextualized_prompt",
                "nl_description_ds_r1": description,
                "intrinsic": "SVE",
                "source_jsonl": str(input_path),
            }
        )
        problem_rows.append(common)
        remote = dict(common)
        remote["compile_modes"] = ["cxx17"]
        remote_rows.append(remote)

    manifest = manifest_for("vecintrinbench50", input_path, classified_rows)
    manifest["parent_rows"] = str(parent_path)
    return problem_rows, remote_rows, manifest, classified_rows


def manifest_for(name: str, input_path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    category_counts = Counter(str(r.get("benchmark_category") or "unknown") for r in rows)
    strategy_counts = Counter(str(r.get("vectorization_strategy") or "unknown") for r in rows)
    context_count = sum(1 for r in rows if r.get("context_required"))
    return {
        "builder": Path(__file__).name,
        "version": "contextualized_benchmark_problem_manifest_v0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": name,
        "input": str(input_path),
        "rows": len(rows),
        "category_distribution": dict(category_counts),
        "vectorization_strategy_distribution": dict(strategy_counts),
        "context_required_rows": context_count,
        "benchmark_boundary": {
            "training_data_allowed": False,
            "admission_status_required": "quarantine_or_eval_only",
            "note": "Evaluation-only benchmark holdout. Do not merge into training, preference, or repair data.",
        },
    }


def has_function_definition(text: str) -> bool:
    cleaned = re.sub(r"^\s*#.*$", "", str(text or ""), flags=re.MULTILINE)
    cleaned = re.sub(r"/\*[\s\S]*?\*/", "", cleaned)
    cleaned = re.sub(r"//.*", "", cleaned)
    for match in FUNC_DEF_RE.finditer(cleaned):
        if match.group(1) not in CONTROL_NAMES:
            return True
    return False


def function_definition_names(text: str) -> set[str]:
    cleaned = re.sub(r"^\s*#.*$", "", str(text or ""), flags=re.MULTILINE)
    cleaned = re.sub(r"/\*[\s\S]*?\*/", "", cleaned)
    cleaned = re.sub(r"//.*", "", cleaned)
    names: set[str] = set()
    for match in FUNC_DEF_RE.finditer(cleaned):
        name = match.group(1)
        if name not in CONTROL_NAMES:
            names.add(name)
    return names


def local_syntax_smoke(row: dict[str, Any], benchmark: str) -> tuple[bool, str]:
    signature = str(row.get("target_signature") or "").strip()
    prelude = str(row.get("candidate_prelude") or "").strip()
    solution_scalar = str(row.get("solution_scalar") or "").strip()
    test = str(row.get("test_correctness") or row.get("test_harness_code") or "").strip()
    if not signature or not test:
        return False, "missing signature or test harness"
    if benchmark == "vecintrinbench50" and (row.get("sve_math_helpers_provided") or row.get("sve_morph_helpers_provided")):
        return True, "skipped local host syntax smoke for SVE helper prelude"
    source = "\n\n".join([prelude, solution_scalar, signature + " {\n}", test])
    if benchmark == "arm_simd_loops":
        # Host-only audit smoke runs on x86 GCC, which may not know the target
        # __fp16 spelling. Keep generated remote artifacts unchanged.
        source = re.sub(r"\btypedef\s+__fp16\s+float16_t\s*;", "typedef float float16_t;", source)
        source = re.sub(r"\b__fp16\b", "float", source)
        source = ARM_C_HEADER + "\n\n" + source
        compiler = "gcc"
        args = [compiler, "-x", "c", "-std=gnu11", "-fsyntax-only", "-Wall", "-Wextra"]
        suffix = ".c"
    else:
        source = VEC_CXX_HEADER + "\n\n" + source
        compiler = "g++"
        args = [compiler, "-x", "c++", "-std=c++17", "-fsyntax-only", "-Wall", "-Wextra"]
        suffix = ".cpp"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=suffix, delete=False) as handle:
        handle.write(source)
        tmp_path = Path(handle.name)
    try:
        proc = subprocess.run(args + [str(tmp_path)], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
    if proc.returncode == 0:
        return True, ""
    return False, (proc.stderr or proc.stdout or "").strip()[:2000]


def audit_rows(rows: list[dict[str, Any]], benchmark: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    stats = Counter()
    for row in rows:
        task_id = str(row.get("task_id") or "")
        prompt = str(row.get("prompt") or "")
        signature = str(row.get("target_signature") or "")
        prelude = str(row.get("candidate_prelude") or "")
        contract = row.get("response_contract") or {}
        stats["rows"] += 1

        if not signature or signature not in prompt:
            findings.append({"task_id": task_id, "severity": "error", "kind": "signature_missing_from_prompt"})
        if not prompt.rstrip().endswith(signature.strip() + " {\n}"):
            findings.append({"task_id": task_id, "severity": "error", "kind": "prompt_not_terminated_by_empty_target_signature"})
        if BAD_PROMPT_API_RE.search(prompt):
            findings.append({"task_id": task_id, "severity": "error", "kind": "bad_api_name_leaked_in_prompt"})
        prelude_functions = function_definition_names(prelude)
        allowed_function_names: set[str] = set()
        if benchmark == "vecintrinbench50" and row.get("sve_math_helpers_provided"):
            allowed_function_names |= SVE_MATH_HELPER_NAMES | {
                "sve_unary_f32_hybrid",
                "sve_log1p_exp_scalar",
                "sve_erfc_scalar",
                "sve_gelu_erfc_scalar",
                "sve_mish_scalar",
            }
        if benchmark == "vecintrinbench50" and row.get("sve_morph_helpers_provided"):
            allowed_function_names |= SVE_MORPH_HELPER_NAMES
        if benchmark == "vecintrinbench50" and row.get("sve_jpeg_helpers_provided"):
            allowed_function_names |= SVE_JPEG_HELPER_NAMES
        allowed_prelude_functions = (
            benchmark == "vecintrinbench50"
            and prelude_functions
            and prelude_functions <= allowed_function_names
        )
        if prelude_functions and not allowed_prelude_functions:
            findings.append({"task_id": task_id, "severity": "error", "kind": "candidate_prelude_contains_function_definition"})
        if not contract.get("signature_must_match_prompt"):
            findings.append({"task_id": task_id, "severity": "error", "kind": "missing_response_contract"})
        missing_forbidden_tokens = REQUIRED_FORBIDDEN_RESPONSE_TOKENS - set(contract.get("forbidden_in_response") or [])
        if missing_forbidden_tokens:
            findings.append(
                {
                    "task_id": task_id,
                    "severity": "error",
                    "kind": "response_contract_missing_forbidden_tokens",
                    "missing": sorted(missing_forbidden_tokens),
                }
            )
        extern_decl = str(row.get("harness_extern_decl") or "")
        if extern_decl and normalize_signature(extern_decl) != normalize_signature(signature):
            findings.append({"task_id": task_id, "severity": "error", "kind": "harness_extern_signature_mismatch"})
        if benchmark == "vecintrinbench50":
            parent_sig = str(row.get("parent_signature") or "")
            if parent_sig and normalize_signature(parent_sig) != normalize_signature(signature):
                findings.append(
                    {
                        "task_id": task_id,
                        "severity": "error",
                        "kind": "parent_signature_mismatch",
                        "parent_signature": parent_sig,
                        "target_signature": signature,
                    }
                )
        ok, msg = local_syntax_smoke(row, benchmark)
        if ok:
            stats["local_syntax_smoke_pass"] += 1
        else:
            stats["local_syntax_smoke_fail"] += 1
            findings.append({"task_id": task_id, "severity": "error", "kind": "local_syntax_smoke_fail", "message": msg})

    summary = {
        "rows": stats["rows"],
        "issues": len(findings),
        "local_syntax_smoke_pass": stats["local_syntax_smoke_pass"],
        "local_syntax_smoke_fail": stats["local_syntax_smoke_fail"],
        "finding_distribution": dict(Counter(f["kind"] for f in findings)),
    }
    return findings, summary


def write_outputs(
    *,
    out_dir: Path,
    benchmark: str,
    problem_rows: list[dict[str, Any]],
    remote_rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    classified_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    bench_dir = out_dir / benchmark
    rows_path = bench_dir / f"{benchmark}_contextualized_rows.jsonl"
    problem_path = bench_dir / f"{benchmark}_contextualized.problem.jsonl"
    remote_path = bench_dir / f"{benchmark}_contextualized.remote_eval.problem.jsonl"
    remote_cxx17_path = bench_dir / f"{benchmark}_contextualized.remote_eval.cxx17.problem.jsonl"
    manifest_path = bench_dir / f"{benchmark}_contextualized.manifest.json"
    findings_path = bench_dir / f"{benchmark}_contextualized_static_audit_findings.jsonl"
    audit_path = bench_dir / f"{benchmark}_contextualized_static_audit_report.json"

    findings, audit_summary = audit_rows(remote_rows, benchmark)
    manifest = dict(manifest)
    manifest["outputs"] = {
        "classified_rows": str(rows_path),
        "problem_jsonl": str(problem_path),
        "remote_eval_problem_jsonl": str(remote_path),
        "manifest_json": str(manifest_path),
        "static_audit_findings_jsonl": str(findings_path),
        "static_audit_report_json": str(audit_path),
    }
    remote_cxx17_rows: list[dict[str, Any]] = []
    if benchmark == "arm_simd_loops":
        remote_cxx17_rows = [
            normalize_cxx17_restrict_keywords(dict(row, compile_modes=["cxx17"]))
            for row in remote_rows
        ]
        manifest["outputs"]["remote_eval_cxx17_problem_jsonl"] = str(remote_cxx17_path)
    manifest["static_audit_summary"] = audit_summary
    manifest["rows_written"] = {
        "classified_rows": len(classified_rows),
        "problem_jsonl": len(problem_rows),
        "remote_eval_problem_jsonl": len(remote_rows),
    }
    if remote_cxx17_rows:
        manifest["rows_written"]["remote_eval_cxx17_problem_jsonl"] = len(remote_cxx17_rows)

    write_jsonl(rows_path, classified_rows)
    write_jsonl(problem_path, problem_rows)
    write_jsonl(remote_path, remote_rows)
    if remote_cxx17_rows:
        write_jsonl(remote_cxx17_path, remote_cxx17_rows)
    write_jsonl(findings_path, findings)
    write_json(audit_path, audit_summary)
    write_json(manifest_path, manifest)
    return {
        "benchmark": benchmark,
        "rows": len(problem_rows),
        "category_distribution": manifest["category_distribution"],
        "vectorization_strategy_distribution": manifest["vectorization_strategy_distribution"],
        "context_required_rows": manifest["context_required_rows"],
        "static_audit_summary": audit_summary,
        "outputs": manifest["outputs"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build context-aware benchmark problem files for Arm SIMD Loops and VecIntrinBench.")
    parser.add_argument("--arm-rows", default=str(DEFAULT_ARM_ROWS))
    parser.add_argument("--vec-problem", default=str(DEFAULT_VEC_PROBLEM))
    parser.add_argument("--vec-parent-rows", default=str(DEFAULT_VEC_PARENT_ROWS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.output_dir).expanduser().resolve()
    arm_problem, arm_remote, arm_manifest, arm_rows = build_arm(args)
    vec_problem, vec_remote, vec_manifest, vec_rows = build_vec(args)

    summaries = [
        write_outputs(
            out_dir=out_dir,
            benchmark="arm_simd_loops",
            problem_rows=arm_problem,
            remote_rows=arm_remote,
            manifest=arm_manifest,
            classified_rows=arm_rows,
        ),
        write_outputs(
            out_dir=out_dir,
            benchmark="vecintrinbench50",
            problem_rows=vec_problem,
            remote_rows=vec_remote,
            manifest=vec_manifest,
            classified_rows=vec_rows,
        ),
    ]
    summary = {
        "builder": Path(__file__).name,
        "version": "contextualized_benchmark_problem_summary_v0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(out_dir),
        "benchmarks": summaries,
        "benchmark_boundary": {
            "training_data_allowed": False,
            "note": "Generated files are evaluation-only benchmark holdout artifacts.",
        },
    }
    write_json(out_dir / "contextualized_benchmark_problem_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
