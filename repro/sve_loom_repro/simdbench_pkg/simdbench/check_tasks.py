import tqdm
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from simdbench.execution import exec_cpp
from simdbench.data import SIMD_BENCH, simdbench_scalar, read_problems
from simdbench.global_var import get_header

tasks = read_problems(simdbench_scalar, "scalar")

# correctness
def process_task_correctness(task_id, problem):
    intrinsic = problem['intrinsic']
    completion = problem['solution_scalar'].replace(problem['entrypoint_scalar'], problem['entrypoint_simd'])
    if(task_id == 'SimdBench_35'):
        completion = completion.replace("""typedef struct {
    int16_t min_num;
    int16_t max_num;
} MinMaxPair;""", "")

    correctness_cpp = (
        get_header() + '\n' + \
        get_header(intrinsic) + '\n' + problem['solution_scalar'] + '\n' + \
        completion + '\n' + \
        problem['test_correctness']
    )
    correctness_res, correctness_stdout = exec_cpp(correctness_cpp, intrinsic, timeout=10.0)
    assert correctness_res == 1, f"correctness check false {task_id}: {correctness_stdout}\n\n{correctness_cpp}"
    try:
        data = json.loads(correctness_stdout)
    except:
        print(f"Invalid stdout when correctness check {task_id}: {correctness_stdout[:30]}")
    assert data['correctness'] == 1, f"correctness check false {task_id}: correctness is 0 ({correctness_stdout}).\n\n{correctness_cpp}"


# performance
def process_task_performance(task_id, problem):
    intrinsic = problem['intrinsic']
    completion = problem['solution_scalar'].replace(problem['entrypoint_scalar'], problem['entrypoint_simd'])
    if(task_id == 'SimdBench_35'):
        completion = completion.replace("""typedef struct {
    int16_t min_num;
    int16_t max_num;
} MinMaxPair;""", "")
    
    performance_cpp = (
        "#include <benchmark/benchmark.h>\n" + get_header() + '\n' + \
        get_header(intrinsic) + '\n' + problem['solution_scalar'] + '\n' + \
        completion + '\n' + \
        problem['test_performance']
    )

    performance_res, performance_stdout = exec_cpp(performance_cpp, intrinsic, timeout=30.0, googlebench=True)
    assert performance_res == 1, f"correctness check false {task_id}: {performance_stdout[0:30]}\n\n{performance_cpp}"

print("check correctness test cases:")
with ThreadPoolExecutor() as executor:
    futures = {executor.submit(process_task_correctness, task_id, problem): task_id for task_id, problem in tasks.items()}
    for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
        future.result()
print("correctness test cases all passed!")
print()

print("check performance test cases:")
with ThreadPoolExecutor() as executor:
    futures = {executor.submit(process_task_performance, task_id, problem): task_id for task_id, problem in tasks.items()}
    for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
        future.result()
print("performance test cases all passed!")