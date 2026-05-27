import os
import json
import re
from typing import Dict
from simdbench.global_var import intrin2fullname, get_header
from simdbench.data import SIMD_BENCH, simdbench_scalar
ROOT = os.path.dirname(os.path.abspath(__file__))

def remove_blank_lines(text):
    return '\n'.join(line for line in text.splitlines() if line.strip())

def prompt_scalar(task:Dict):
    '''
    generate scalar prompt
    '''
    prompt:str = task['prompt']
    prompt = prompt.replace(get_header(task['intrinsic']), "")
    prompt = prompt.replace(task['entrypoint_simd'], task['entrypoint_scalar'])

    simd_words = r"The requirement.*?intrinsics.*?parallelism."
    match = re.search(simd_words, prompt)
    assert(task['entrypoint_scalar'] != task['entrypoint_simd'])
    if(match == None):
        raise ValueError(f"Prompt in {task['task_id']}")
    else:
        prompt = re.sub(simd_words, "", prompt)
    return remove_blank_lines(prompt)


# generate scalar tasks from simd tasks
tasks = []
with open(SIMD_BENCH, "r", encoding="utf-8") as file:
    for line in file:
        data = json.loads(line)
        if(data['intrinsic'] != "AVX"): continue
        else: data['task_id'] = data['task_id'].replace("_AVX", "")
        data['prompt'] = prompt_scalar(data)
        data['intrinsic'] = "scalar"
        tasks.append(data)

with open(simdbench_scalar, "w", encoding="utf-8") as output:
    for data in tasks:
        json_line = json.dumps(data)
        output.write( json_line + '\n')