#!/usr/bin/env python3
"""Build eval-only stricter correctness problem files.

The generated files are intended for post-hoc final evaluation only. They must
not be used inside the repair loop.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable


ROOT = Path("/home/user")
SIMDBENCH_IN = ROOT / "simdbench_full/data/simdbench_sve.jsonl"
SIMDBENCH_OUT = ROOT / "simdbench_full/data/simdbench_sve_hidden_strict.jsonl"

ARM_IN = (
    ROOT
    / "selective_repo/results/generated/contextualized_benchmarks/"
    / "arm_simd_loops/arm_simd_loops_contextualized.remote_eval.cxx17.problem.jsonl"
)
ARM_OUT = (
    ROOT
    / "selective_repo/results/generated/contextualized_benchmarks/"
    / "arm_simd_loops/arm_simd_loops_contextualized.remote_eval.cxx17_hidden_strict.problem.jsonl"
)


SIMDBENCH_HIDDEN_PREAMBLE = r"""

// Hidden-style stricter final evaluation configuration.
// This file is for post-hoc evaluation only; it is not used during repair.
#undef ITERATIONS
#define ITERATIONS 5000

static inline size_t hidden_eval_arg_1d(int iter) {
    static const size_t cases[] = {
        1, 2, 3, 4, 7, 8,
        15, 16, 17, 31, 32, 33,
        63, 64, 65, 127, 128, 129,
        255, 256, 257, 511, 512, 513,
        1023, 1024, 2048
    };
    const size_t n_cases = sizeof(cases) / sizeof(cases[0]);
    if (iter < (int)n_cases) return cases[iter];
    uint32_t x = (uint32_t)iter * 1103515245u + 12345u + 0x5EEDBEEFu;
    return (size_t)(1u + (x % 2048u));
}

static inline size_t hidden_eval_arg_2d(int iter) {
    static const size_t cases[] = {
        1, 2, 3, 4, 7, 8, 9,
        15, 16, 17, 31, 32, 33,
        47, 63, 64
    };
    const size_t n_cases = sizeof(cases) / sizeof(cases[0]);
    if (iter < (int)n_cases) return cases[iter];
    uint32_t x = (uint32_t)iter * 1664525u + 1013904223u + 0x5EEDBEEFu;
    return (size_t)(1u + (x % 64u));
}

static inline size_t hidden_eval_arg_3d(int iter) {
    static const size_t cases[] = {1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 16};
    const size_t n_cases = sizeof(cases) / sizeof(cases[0]);
    if (iter < (int)n_cases) return cases[iter];
    uint32_t x = (uint32_t)iter * 22695477u + 1u + 0x5EEDBEEFu;
    return (size_t)(1u + (x % 16u));
}

static inline size_t hidden_eval_arg_4d(int iter) {
    static const size_t cases[] = {1, 2, 3, 4, 5, 6, 7, 8};
    const size_t n_cases = sizeof(cases) / sizeof(cases[0]);
    if (iter < (int)n_cases) return cases[iter];
    uint32_t x = (uint32_t)iter * 214013u + 2531011u + 0x5EEDBEEFu;
    return (size_t)(1u + (x % 8u));
}

static inline size_t hidden_eval_arg_5d(int iter) {
    static const size_t cases[] = {1, 2, 3, 4, 5, 6};
    const size_t n_cases = sizeof(cases) / sizeof(cases[0]);
    if (iter < (int)n_cases) return cases[iter];
    uint32_t x = (uint32_t)iter * 747796405u + 2891336453u + 0x5EEDBEEFu;
    return (size_t)(1u + (x % 6u));
}

#undef Small_Arg_1D
#undef Small_Arg_2D
#undef Small_Arg_3D
#undef Small_Arg_4D
#undef Small_Arg_5D
#define Small_Arg_1D hidden_eval_arg_1d(i)
#define Small_Arg_2D hidden_eval_arg_2d(i)
#define Small_Arg_3D hidden_eval_arg_3d(i)
#define Small_Arg_4D hidden_eval_arg_4d(i)
#define Small_Arg_5D hidden_eval_arg_5d(i)

"""


ARM_HIDDEN_CASES = "0, 1, 2, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65"


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def make_simdbench_hidden(row: dict) -> dict:
    row = dict(row)
    tc = str(row.get("test_correctness") or "")
    if not tc:
        return row

    # Make the randomized harness reproducible. Most tasks declare one rng
    # outside the randomized loop; one task declares it inside. Both forms are
    # deterministic under this replacement.
    tc = re.sub(r"\bRandom\s+rng\s*;", "Random rng(0x5EEDBEEFu);", tc)

    # Keep existing fixed HumanEval checks, then strengthen the randomized pass.
    if "hidden_eval_arg_1d" not in tc:
        tc = SIMDBENCH_HIDDEN_PREAMBLE + tc
    tc = re.sub(r"correctness_check\s*\(\s*ITERATIONS\s*\)", "correctness_check(ITERATIONS)", tc)
    row["test_correctness"] = tc
    row["hidden_eval_config"] = {
        "name": "simdbench_hidden_strict",
        "repair_oracle": False,
        "iterations": 5000,
        "fixed_seed": "0x5EEDBEEF",
        "size_macros": "expanded deterministic Small_Arg_*D cases plus seeded pseudo-random tail",
        "notes": [
            "post-hoc final evaluation only",
            "does not expose concrete inputs or outputs to the model",
            "does not inject NaN/Inf globally because many tasks do not specify such inputs",
        ],
    }
    return row


def make_arm_hidden(row: dict) -> dict:
    row = dict(row)
    tc = str(row.get("test_correctness") or "")
    if not tc:
        return row

    tc = re.sub(
        r"const\s+int\s+n_cases\[\]\s*=\s*\{[^}]*\};",
        f"const int n_cases[] = {{{ARM_HIDDEN_CASES}}};\n"
        "    const int n_cases_count = (int)(sizeof(n_cases) / sizeof(n_cases[0]));",
        tc,
    )
    tc = re.sub(
        r"case_idx\s*<\s*5",
        "case_idx < n_cases_count",
        tc,
    )
    row["test_correctness"] = tc
    row["hidden_eval_config"] = {
        "name": "arm_simd_loops_hidden_strict",
        "repair_oracle": False,
        "case_values": [int(x.strip()) for x in ARM_HIDDEN_CASES.split(",")],
        "notes": [
            "post-hoc final evaluation only",
            "expands deterministic boundary/tail-size cases",
            "does not enter the repair loop",
        ],
    }
    return row


def main() -> None:
    simd_n = write_jsonl(SIMDBENCH_OUT, (make_simdbench_hidden(r) for r in iter_jsonl(SIMDBENCH_IN)))
    arm_n = write_jsonl(ARM_OUT, (make_arm_hidden(r) for r in iter_jsonl(ARM_IN)))
    print(f"[hidden_eval_build] simdbench rows={simd_n} out={SIMDBENCH_OUT}")
    print(f"[hidden_eval_build] arm_simd_loops rows={arm_n} out={ARM_OUT}")


if __name__ == "__main__":
    main()
