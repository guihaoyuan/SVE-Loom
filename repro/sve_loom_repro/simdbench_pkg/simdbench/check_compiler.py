import json
import os
import argparse
import subprocess
from simdbench.global_var import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--intrinsic', required=True, help="simd instrinsics (SSE, AVX, SVE, Neon, RVV) " \
    "that are supported by this machine") 
    args = parser.parse_args()
    intrinsic = args.intrinsic
    assert(intrinsic in intrin_list)
    compiler = cpp_compilers[intrinsic]
    
    if not os.path.exists(compiler):
        print(f"Compiler {compiler} not found. Please check the path in config_compiler.json.")
        exit(1)

    example_code = f"{get_header(intrinsic)}\nint main(){{ return 0; }}\n"
    example_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "example.cpp")
    example_output = os.path.join(os.path.dirname(os.path.abspath(__file__)), "example.out")
    with open(example_file, "w") as f:
        f.write(example_code)
    compile_cmd = f"{compiler} {example_file} {intrin_flags[intrinsic]} -o {example_output}"

    try:
        result = subprocess.run(compile_cmd, shell=True, check=True, capture_output=True)
        print(f"{compiler}: supports { intrin2fullname[intrinsic] }.")
    except subprocess.CalledProcessError as e:
        print(f"{compiler}: does not support { intrin2fullname[intrinsic] }.")
        print(f"Error message: {e.stderr.decode()}")
    finally:
        if os.path.exists(example_file):
            os.remove(example_file)
        if os.path.exists(example_output):
            os.remove(example_output)