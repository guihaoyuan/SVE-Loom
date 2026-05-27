import re
import json
inline_pattern = re.compile(r'(?<![A-Za-z0-9_])_{0,2}asm_{0,2}\b', re.MULTILINE)
count = 0
total = 0
with open("/home/zhangyf/llms/paper_codellama/simdbench/deepseek-reasoner-inline-ass-SVE.jsonl", "r") as f:
    for line in f:
        total += 1
        data = json.loads(line)
        if not (inline_pattern.search(data["completion"]) is None):
            count += 1

print("Inline ass count: ", count, "in ", total)

    