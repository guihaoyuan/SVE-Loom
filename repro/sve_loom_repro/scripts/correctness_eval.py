import tqdm
import os
import numpy as np
from contextlib import redirect_stdout
import io
import fire
import sys
from tabulate import tabulate
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from simdbench.data import SIMD_BENCH,simdbench_scalar
from simdbench.evaluation import evaluate_functional_correctness

simdbench_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")
#correctness_output = os.path.join(simdbench_path , "results/correctness-eval")

# model_lst = ["claude-3-5-haiku-snapshot", "claude-3-5-sonnet-snapshot", "codestral-latest", 
#              "deepseek-chat", "deepseek-reasoner", "gemini-2.0-flash", "gemini-2.5-flash-preview-05-20", 
#              "gpt-3.5-turbo", "gpt-4o-snapshot", "gpt-4o-mini-snapshot", 
#              "grok-3", "grok-3-beta", "mistral-large-latest", "qwen-coder-plus", 
#              "qwen-coder-turbo", "qwen-max-snapshot", "qwen-plus-snapshot", "qwen-turbo-snapshot"]
model_lst = ["qwen-plus", "qwen-max", "qwen3-coder-plus"]
# model_lst = ["claude-3-5-haiku-snapshot"] # for test
code_lst = ["scalar", "Neon"]

pass_at_k_results = []
headers = ["Model", "Scalar(1)", "Scalar(5)","Neon(1)", "Neon(5)"]

def entry_point(correctness_output:str):
    #check model output
    print("checking model output...")
    for model in model_lst:
        for intrinsic in code_lst:
            path = f"simdbench/{model}-inline-ass-{intrinsic}.jsonl"
            model_output = os.path.join(simdbench_path, path)
            assert( os.path.exists(model_output) )

    for model in tqdm.tqdm(model_lst, desc="Evaluating models", leave=False):
        model_res = [model]
        for intrinsic in tqdm.tqdm(code_lst, desc=f"Processing {model}", leave=False):
            path = f"simdbench/{model}-inline-ass-{intrinsic}.jsonl"
            model_output = os.path.join(simdbench_path, path)
            model_output = os.path.abspath(model_output)
                    
            # entry point
            sample_file = model_output
            k = "1,5"
            k = list(map(int, k.split(",")))
            # n_workers = min(64, os.cpu_count() + 4)
            n_workers=16
            timeout = 15.0
            problem_file = SIMD_BENCH
            if(intrinsic=="scalar" and problem_file == SIMD_BENCH): problem_file = simdbench_scalar
            output_path = correctness_output
            
            with redirect_stdout(io.StringIO()):
                pass_at_k = evaluate_functional_correctness(sample_file, intrinsic, k, n_workers, timeout, problem_file, output_path)

            print(f"pass: {pass_at_k}")
            pass_at_1 = np.around(pass_at_k['pass@1'] * 100, 2)
            pass_at_5 = np.around(pass_at_k['pass@5'] * 100, 2)
            model_res += [pass_at_1, pass_at_5]
        
        pass_at_k_results.append(model_res)

    print(tabulate(pass_at_k_results, headers=headers, tablefmt="grid"))

def main():
    fire.Fire(entry_point)

sys.exit(main())