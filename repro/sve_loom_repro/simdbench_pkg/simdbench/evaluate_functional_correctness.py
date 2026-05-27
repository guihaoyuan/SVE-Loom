import fire
import sys
import os
from simdbench.data import SIMD_BENCH,simdbench_scalar
from simdbench.evaluation import evaluate_functional_correctness

def entry_point(
    sample_file: str,
    intrinsic:str,
    k: str = "1,5,10",
    n_workers: int = None,
    timeout: float = 15.0,
    problem_file: str = SIMD_BENCH,
    output_path:str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"),
):
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """
    k = list(map(int, k.split(",")))
    if(intrinsic=="scalar" and problem_file == SIMD_BENCH): problem_file = simdbench_scalar
    results = evaluate_functional_correctness(sample_file, intrinsic, k, n_workers, timeout, problem_file, output_path)
    print(results)

def main():
    fire.Fire(entry_point)

sys.exit(main())