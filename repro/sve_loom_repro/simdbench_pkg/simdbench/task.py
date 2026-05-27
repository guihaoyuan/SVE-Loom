import os
import sys
import random
import argparse
import json
from collections import OrderedDict
# from simdbench.global_var import *

from global_var import *



"""
This script generates the prompt for SimdBench.
data path: the path to the data directory (a group of task directories).
for each task directory, it at least should contain:
    - signature.cpp: the function signature with description
    - solution_scalar.cpp: the scalar implementation of the function
    - testcode.cpp: the test code for the function
"""

def clean_signature(signature:str, intrinsic:str="AVX"):
    res = signature.replace("[arch]", intrin2arch[intrinsic])
    res = res.replace("[suffix]", intrin2suffix[intrinsic])
    res = res.replace("[simd]", intrin2fullname[intrinsic])
    return res.strip()

def sys_prompt(intrinsic:str="AVX", task:str="generation", CoT:bool=False):
    '''
    prompt for {"role": "system"},
    intrinsic in ["SSE", "AVX", "SVE", "Neon", "RVV"]
    task in ["generation", "translation"]
    '''
    assert(intrinsic in intrin_list+["scalar"])
    if(intrinsic == "scalar"):
        return f"You are a code generator. Return the inline assembly code implementation for armv8 arch within the markdown code block. Note that you should use c/c++ code with assembly language, not C/C++ code with any intrinisics. Do not include any explanations, comments, or text outside the code block."
    # Your task is to generate the C/C++ code for this function using {intrin2fullname[intrinsic]} intrinisics to enable parallelism.
    prompt_generation = f"""You are a code generator. You will be given a function signature and a description of the function. 
Your task is to generate the assembly code with gcc inline assembly for this function using {intrin2fullname[intrinsic]} instruction set to enable parallelism. Note that you
should use c/c++ code with inline assembly language, not C/C++ code with any intrinisics.
Return the SIMD code implementation within the markdown code block. Do not include any explanations, comments, or text outside the code block.\n"""
    prompt_inline_ass = f"""You are an expert in low-level C/C++ SIMD programming.
Generate a C or C++ function implementation **using GCC-style inline assembly only**.
⚠️ Do not use any compiler intrinsics (such as `_mm_*`, `sv*`, or `vaddq_*`).
⚠️ Do not include any header files like `<arm_neon.h>` or `<arm_sve.h>`.
Use **only inline assembly (`asm` or `__asm__`)** to implement the function body.
Target architecture: {intrin2fullname[intrinsic]}
Return **only** the code implementation within a fenced code block. 
No explanations, no comments, no Markdown text outside the code block.
"""
    prompt_translation = f"""You are a code translator. You will be given a C/C++ code snippet of a scalar implementation, and a function signature for the vectorized implementation with a description. 
Your task is to translate the scalar code (source code) into assembly code with {intrin2fullname} instruction set.
Return the SIMD code implementation within the markdown code block. Do not include any explanations, comments, or text outside the code block.\n"""
    # {random.choice(['parallel', 'SIMD', 'vectorized'])} code (target code) with {intrin2fullname[intrinsic]} intrinsics.
    prompt = ""
    if(task == "generation"): prompt = prompt_inline_ass
    elif(task == "translation"): prompt = prompt_translation
    else:
        raise ValueError(f"Invalid task: {task}")
    
    if CoT:
        prompt += """Note that you should use the CoT method to generate the code. 
You should first generate the code in a step-by-step manner, and then combine the steps into a final code.\n"""
    #if(intrinsic == "RVV"):
        #prompt += "Note that RVV intrinsics have the common prefix `__riscv_`.\n"
    return prompt

# Code generation prompt: NL to code
def gen_prompt_codegen(signature:str, intrinsic:str="AVX"):
    assert(intrinsic in intrin_list)
    prompt = f"""{clean_signature(signature, intrinsic)}"""
    return prompt

# Code translation prompt: scalar code to SIMD code
def gen_prompt_codetrans(signature:str, scalar_code:str, intrinsic:str="AVX"):
    assert(intrinsic in intrin_list)
    prompt = f"""Source code:
```cpp
{scalar_code}
```
Target code:
```cpp
{clean_signature(signature, intrinsic)}
```
    """
    return prompt   

# task: generation or translation
def gen_prompt(read_dir:str, intrinsic:str="AVX", task:str = "generation", CoT:bool=False)-> dict:
    assert(intrinsic in intrin_list)
    assert(os.path.exists(read_dir))

    res = {}
    assert(os.path.exists(os.path.join(read_dir, "info.json")))
    with open(os.path.join(read_dir, "info.json"), "r") as file:
        info = json.load(file)
        res["type"] = info["type"]
        res["subtype"] = info["subtype"]
        res["entrypoint_scalar"] = info["entrypoint_scalar"]
        res["entrypoint_simd"] = info["entrypoint_simd"]
        assert( res["entrypoint_scalar"] != res["entrypoint_simd"] )

    assert(os.path.exists(os.path.join(read_dir, "signature.cpp")))
    with open(os.path.join(read_dir, "signature.cpp"), "r") as file:
        signature = file.read().strip()
        signature = clean_signature(signature, intrinsic)
        signature = get_header(intrinsic) + '\n' + signature
        assert( res["entrypoint_simd"] in signature )
    
    assert(os.path.exists(os.path.join(read_dir, "solution_scalar.cpp")))
    with open(os.path.join(read_dir, "solution_scalar.cpp"), "r") as file:
        scalar_code = file.read().strip()
        assert( res["entrypoint_simd"] not in scalar_code )
        assert( res["entrypoint_scalar"] in scalar_code )

    assert(os.path.exists(os.path.join(read_dir, "test_correctness.cpp")))
    with open(os.path.join(read_dir, "test_correctness.cpp"), "r") as file:
        test_correctness = file.read().strip()
        assert( res["entrypoint_simd"] in test_correctness )
        assert( res["entrypoint_scalar"] in test_correctness.replace(res["entrypoint_simd"], "") )

    assert(os.path.exists(os.path.join(read_dir, "test_performance.cpp")))
    with open(os.path.join(read_dir, "test_performance.cpp"), "r") as file:
        test_performance = file.read().strip()
        assert( res["entrypoint_simd"] in test_performance )
        assert( res["entrypoint_scalar"] in test_performance.replace(res["entrypoint_simd"], "") )
    
    if task == "generation":
        prompt = gen_prompt_codegen(signature, intrinsic)
    elif task == "translation":
        prompt = gen_prompt_codetrans(signature, scalar_code, intrinsic)
    else:
        raise ValueError(f"Unknown task: {task}. Please use 'generation' or 'translation'.")
    
    res["prompt"] = prompt
    res["solution_scalar"] = scalar_code
    res["test_correctness"] = test_correctness
    res["test_performance"] = test_performance
    res["task"] = task
    res["intrinsic"] = intrinsic
    return res


# deal with a signle task directory
# safe check the task directory
def gen_a_task(task_dir:str, intrinsic:str="AVX", task:str = "generation", CoT:bool=False)->OrderedDict:
    if(intrinsic not in intrin_list):
        print(f"Skipped: {intrinsic} is not supported.")
        return None
    if(os.path.exists(task_dir)== False):
        print(f"Skipped: {task_dir} does not exist.")
        return None
    if(os.path.isdir(task_dir) == False):
        print(f"Skipped: {task_dir} is not a directory.")
        return None
    if(task not in ["generation", "translation"]):
        print(f"Skipped: {task} is not a valid task.")
        return None
    
    necessary_files = ["signature.cpp", "solution_scalar.cpp", "test_correctness.cpp", "test_performance.cpp", "info.json"]
    for file in necessary_files:
        if(os.path.exists(os.path.join(task_dir, file)) == False):
            print(f"Skipped: {file} does not exist in {task_dir}.")
            return None
    
    with open(os.path.join(task_dir, "info.json"), "r") as file:
        info = json.load(file)
        necessary_keys = ["type", "subtype", "entrypoint_scalar", "entrypoint_simd"]
        for key in necessary_keys:
            if(key not in info):
                print(f"Skipped: {key} is not found in {task_dir}/info.json.")
                return None

    res = gen_prompt(task_dir, intrinsic, task, CoT)
    res["task_id"] = os.path.basename(task_dir) + "_" + intrinsic
    data = OrderedDict([
        ("task_id", res["task_id"]),
        ("prompt", res["prompt"]),
        ("solution_scalar", res["solution_scalar"]),
        ("test_correctness", res["test_correctness"]),
        ("test_performance", res["test_performance"]),
        ("intrinsic", res["intrinsic"]),
        ("task", res["task"]),
        ("type", res["type"]),
        ("subtype", res["subtype"]),
        ("entrypoint_scalar", res["entrypoint_scalar"]),
        ("entrypoint_simd", res["entrypoint_simd"]),
    ])
    return data

# deal with a group of task directories (data path)
def gen_tasks(data_path:str, task:str = "generation", CoT:bool=False)->list[OrderedDict]:
    assert(os.path.exists(data_path))
    assert(os.path.isdir(data_path))
    assert(task in ["generation", "translation"])

    res = []
    for task_id in os.listdir(data_path):
        for intrinsic in intrin_list:
            data = gen_a_task(os.path.join(data_path, task_id), intrinsic, task, CoT)
            if(data != None):
                res.append(data)
            else:
                break
    def get_idx(task_id:str):
        try:
            return int(task_id.split('_')[1])  # Extract the number: SimdBench_N_XXX
        except (IndexError, ValueError):
            return float('inf')  # Put invalid entries at the end
    res = sorted(
        res, 
        key=lambda x: get_idx(x["task_id"])
    )
    return res

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--data_path', required=True, help="data path")
    parser.add_argument('-t', '--task', default="generation", help="task: generation or translation")
    #parser.add_argument('--cot', action="store_true", help="whether CoT (Chain of Thought) is used")
    args = parser.parse_args()
    assert(args.task in ["generation", "translation"])
    assert(os.path.exists(args.data_path))
    assert(os.path.isdir(args.data_path))
    
    #simdbench_jsonl = gen_tasks(args.data_path, args.task, args.cot)
    simdbench_jsonl = gen_tasks(args.data_path, args.task, False)
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data", "simdbench.jsonl")
    with open(output_path, "w") as file:
        for data in simdbench_jsonl:
            file.write(json.dumps(data) + "\n")
    
    print(f"Simdbench generated in {output_path}.\n")
    #for intrinsic in intrin_list:
        #print(f"System prompt:\n{sys_prompt(intrinsic, args.task)}")
    