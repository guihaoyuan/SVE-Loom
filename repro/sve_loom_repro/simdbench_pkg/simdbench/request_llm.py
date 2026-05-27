import json
import os
import tqdm
from typing import Dict
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type
# from simdbench.task import sys_prompt
from task import sys_prompt
from data import read_problems, simdbench_scalar, SIMD_BENCH
from global_var import intrin_list
# from request_local_llm import request_llm_once_local
qwen_key = os.environ.get("DASHSCOPE_API_KEY", "")
qwen_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# client = OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY", ""), base_url="https://api.deepseek.com")
client = OpenAI(api_key=qwen_key, base_url=qwen_base_url)
iteration = 5 # number per problem
# model = "deepseek-reasoner"
import re
inline_pattern = re.compile(r'(?<![A-Za-z0-9_])_{0,2}asm_{0,2}\b', re.MULTILINE)

models = ["qwen3-coder-plus", "qwen3-coder-flash","qwen3-flash"]
# model = models[0]
# task_name = f"{model}-inline-ass"
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def request_llm_once(task: Dict, model: str, prompt: str|None =None):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_prompt(task['intrinsic'], task['task'])},
            {"role": "user", "content": task['prompt']},
        ],
        temperature = 0.2,
        stream=False
    )
    return response.choices[0].message.content

# return simdbench.json
def get_data(intrinsic:str):
    assert(intrinsic in intrin_list+["scalar"])
    if(intrinsic == "scalar"): return simdbench_scalar
    elif (intrinsic in intrin_list): return SIMD_BENCH
    return None

# task range: [start_id, 135]
def samples(intrinsic:str, model:str, start_id:int = 0):
    assert(intrinsic in intrin_list+["scalar"])
    tasks = read_problems(get_data(intrinsic), intrinsic)
    with open(f"{model}-inline-ass-{intrinsic}.jsonl", "a", encoding="utf-8") as output:
        for task_id, task in tqdm.tqdm(tasks.items()):
            if( int(task_id.split('_')[1]) < start_id ): continue
            for _ in range(iteration):
                for retry in range(3):
                    try:
                        # completion = request_llm_once_local(task)
                        completion = request_llm_once(task, model)
                        # print(sys_prompt(task['intrinsic'], task['task']))
                        # breakpoint()
                        if completion is None:
                            continue
                        if inline_pattern.search(completion) is None:
                            
                            continue
                        data = {"task_id": task_id, "completion": completion}
                        json_line = json.dumps(data)
                        output.write( json_line + '\n')
                        output.flush()
                        break
                    except Exception as e:
                        print(f"[{task_id}] An error occurred: ", e)
                        pass

if __name__ == "__main__":
    # samples("scalar") # one by one

    # for item in ["SSE", "AVX", "SVE", "Neon", "RVV", "scalar"]:
    #     samples(item)
    for model in models:
        for item in ["SVE", "Neon", "scalar"]:
            samples(item, model)
