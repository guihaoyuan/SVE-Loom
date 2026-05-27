import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-L', '--src', required=True, help="Path to the src directory of built google benchmark \
                    (where exists libbenchmark.a and libbenchmark_main.a)")
    parser.add_argument('-I', '--include', required=True, help="Path to the include directory of built google benchmark \
                    (where exists benchmark/benchmark.h)")
    args = parser.parse_args()
    src = args.src
    include = args.include
    assert(os.path.exists(src))
    assert(os.path.exists(include))
    assert(os.path.exists(os.path.join(src, "libbenchmark.a")))
    assert(os.path.exists(os.path.join(src, "libbenchmark_main.a")))
    assert(os.path.exists(os.path.join(include, "benchmark/benchmark.h")))
    print((f"-L{os.path.abspath(src)} -isystem {os.path.abspath(include)} -lbenchmark -lpthread"))