import os

# arch_list = ["x86", "arm", "riscv"]
# intrin_list = ["SSE", "AVX", "SVE", "Neon", "RVV"]
arch_list = ["arm"]
intrin_list = ["SVE"]
cpp_compilers = {
    "scalar": "/usr/local/bin/g++",
    # "SSE": "/usr/bin/clang++",
    # "AVX": "/usr/bin/clang++",
    "SVE": "/home/llvm-19.1.7/arm/bin/clang++",
    # "Neon": "/usr/local/bin/g++",
    # "RVV": "/home/llvm-19.1.7/riscv64/bin/clang++"
}
emulators = {
    "scalar": None,
    "SSE": None,
    "AVX": None,
    "SVE": None,
    "Neon": None,
    "RVV": None
}
intrin_flags = {
    "scalar": "",
    "SSE": " -msse -msse2 ",
    "AVX": " -mavx -mavx2 ",
    "SVE": " -march=armv8-a+sve ",
    "Neon": " -march=armv8-a+simd ",
    "RVV": " -march=rv64gcv -mabi=lp64d -static "
} # for compilation

# simd intrinsics to arch keywords
intrin2arch = {
    "SSE": "x86",
    "AVX": "x86",
    "SVE": "arm",
    "Neon": "arm",
    "RVV": "riscv"
}

# simd intrinsics to arch suffix (used in function signature)
intrin2suffix = {
    "SSE": "sse",
    "AVX": "avx",
    "SVE": "sve",
    "Neon": "neon",
    "RVV": "rvv"
}

# simd intrinsics to intrinsic name (used in prompt)
intrin2fullname = {
    "SSE": "SSE/SSE2 (Streaming SIMD Extensions)",
    "AVX": "AVX/AVX2 (Advanced Vector Extensions)",
    "SVE": "SVE (Arm C Language Extensions (ACLE) for the Arm Scalable Vector Extension (SVE))",
    "Neon": "Neon (Advanced SIMD)",
    "RVV": "RVV (RISC-V Vector Extension)"
}

def get_header(intrinsic:str=None) -> str:
    # Returns the header file for general C code.
    if(intrinsic == "scalar"): return ""
    if intrinsic is None:
        header_file = "header.cpp"
    else:
        assert(intrinsic in intrin_list)
        header_file = f"header_{intrinsic.lower()}.c"

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "header", header_file), "r") as f:
        header = f.read()
    return header + "\n"

header = get_header()
# header_sse = get_header("SSE")
# header_avx = get_header("AVX")
header_sve = get_header("SVE")
# header_neon = get_header("Neon")
# header_rvv = get_header("RVV")
