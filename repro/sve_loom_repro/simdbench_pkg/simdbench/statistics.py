import os
import argparse
import json
import re
from collections import Counter

skipped_id = []
data_type_candidate = \
['int', 'int64_t', 'int32_t', 'int16_t', 'int8_t'] + \
['uint64_t', 'uint32_t', 'uint16_t', 'uint8_t'] + \
['bool', 'float', 'double', 'size_t', 'string']

task_type_candidate = ['1', '2'] # 1: manual, 2: HumanEval

def parse_signature(signature: str, entry_point: str) -> dict:
    """
    Extract and count data types used in a given function signature.

    Args:
        signature: The full code block or source string containing the function.
        entry_point: The name of the function to extract and analyze.

    Returns:
        A dictionary with data type names as keys and their counts as values.
    """
    # Match function declaration
    def remove_comments(code):
        # Remove multi-line comments
        code = re.sub(r'/\*.*?\*/', lambda m: ' ' * len(m.group()), code, flags=re.DOTALL)
        # Remove single-line comments
        code = re.sub(r'//.*', lambda m: ' ' * len(m.group()), code)
        return code
    func_pattern = rf"([a-zA-Z_][\w\s\*]*?)\b{re.escape(entry_point)}\s*\((.*?)\)\s*\{{"
    match = re.search(func_pattern, remove_comments(signature), re.DOTALL)
    if not match:
        return {}

    # Extract return type and parameter string
    return_type_str = match.group(1).strip()
    params_str = match.group(2).strip()

    # std::vector<std::vector<T>> -> T
    def strip_std_vector(s: str) -> str:
        def parse_vector_type(s, i=0):
            result = ''
            while i < len(s):
                if s.startswith('std::vector<', i):
                    i += len('std::vector<')
                    inner, i = parse_vector_type(s, i)
                    result += inner
                elif s[i] == '<':
                    nested, i = parse_vector_type(s, i + 1)
                    result += nested
                elif s[i] == '>':
                    return result.strip(), i + 1
                else:
                    result += s[i]
                    i += 1
            return result.strip(), i
        result, _ = parse_vector_type(s.replace(' ', ''))
        return result

    # Normalize return type (remove const, *, etc.)
    def normalize(typename: str) -> str:
        typename = re.sub(r'\bconst\b', '', typename)  # remove 'const'
        typename = typename.replace('*', '')           # remove '*'
        typename = typename.replace('&', '')           # remove '&'
        typename = strip_std_vector(typename) # remove std::vector
        typename = typename.replace('std::', '')           # remove 'std::'
        return typename.strip()

    # Count return type
    types = []
    ret_type = normalize(return_type_str.split()[-1])
    if ret_type:
        types.append(ret_type)

    # Count parameter types
    if params_str and params_str.lower() != "void":
        param_parts = [p.strip() for p in params_str.split(',')]
        for part in param_parts:
            tokens = part.split()
            if len(tokens) >= 2:
                type_token = normalize(' '.join(tokens[:-1]))  # everything except var name
                types.append(type_token)

    # Return a frequency dictionary
    return dict(Counter(types))
    

def count(data_path:str):
    info_dict = {}
    sig_dict = {}
    entry_points_simd = {}

    # init
    res_task, res_data_type = {}, {}
    for candidate in task_type_candidate:
        res_task[candidate] = {}
        res_task[candidate]["count"] = 0
        res_task[candidate]["subtype"] = {}
        if(candidate == '1'):
            for idx in range(1, 7):
                res_task[candidate]["subtype"][f'1-{idx}'] = 0
    
    res_data_type["1"] = {} # manual tasks
    res_data_type["2"] = {} # humaneval tasks
    for dt in data_type_candidate:
        res_data_type["1"][dt] = 0
        res_data_type["2"][dt] = 0

    if(os.path.isdir(data_path)):
        for dir in os.listdir(data_path):
            match = re.match(r"SimdBench_([0-9]+)", dir)
            if(match == None):
                #id_lst.append(match.groups()[0])
                print(f"Skipped: {os.path.join(data_path, dir)}")
                continue
            id = match.groups()[0]
            if int(id) in skipped_id:
                print(f"Skipped: {os.path.join(data_path, dir)}")
                continue
            with open(os.path.join(data_path, dir, "info.json"), "r", encoding="utf-8") as file:
                info = json.load(file)
                info_dict[dir] = info
            with open(os.path.join(data_path, dir, "signature.cpp"), "r", encoding="utf-8") as file:
                signature = file.read()
                sig_dict[dir] = signature
    elif (data_path.endswith(".jsonl")):
        pass # TBD

    for key, info in info_dict.items():
        cur_type = info["type"]
        cur_subtype = info["subtype"]
        res_task[cur_type]["count"] += 1
        if(cur_type=="1"): res_task[cur_type]["subtype"][cur_subtype] += 1
        entry_points_simd[key] = info["entrypoint_simd"]
    
    skipped_type = set()
    for key, sig in sig_dict.items():
        types = parse_signature(signature=sig, entry_point=entry_points_simd[key])
        for t, count in types.items():
            if t in data_type_candidate:
                res_data_type[info_dict[key]["type"]][t] += count
            else:
                skipped_type.add(t)
    print(json.dumps(res_task, indent=4))
    print(json.dumps(res_data_type, indent=4))
    for t in data_type_candidate:
        print(f"{t}: {res_data_type["1"][t] + res_data_type["2"][t]}")
    print(f"Skipped types: {skipped_type}\n")
    return res_task, res_data_type


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help='path of data dir or data in .jsonl')
    args = parser.parse_args()
    data_path = args.data
    assert( os.path.isdir(data_path) or data_path.endswith(".jsonl") )

    count(data_path)