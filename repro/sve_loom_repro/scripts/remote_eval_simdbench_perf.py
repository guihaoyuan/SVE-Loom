#!/usr/bin/env python3
from __future__ import annotations

import argparse

from simdbench.data import SIMD_BENCH
from simdbench.evaluation import evaluate_performance


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-file", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--optimization", default="-O0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--n-reputation", type=int, default=1)
    args = parser.parse_args()

    print("[remote_perf] sample_file=", args.sample_file, flush=True)
    print("[remote_perf] output_path=", args.output_path, flush=True)
    print(
        "[remote_perf] optimization=",
        args.optimization,
        "nrep=",
        args.n_reputation,
        "workers=",
        args.workers,
        flush=True,
    )
    evaluate_performance(
        sample_file=args.sample_file,
        intrinsic="SVE",
        k=[1],
        n_workers=args.workers,
        timeout=args.timeout,
        problem_file=SIMD_BENCH,
        optimization=args.optimization,
        n_reputation=args.n_reputation,
        output_path=args.output_path,
    )
    print("[remote_perf] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
