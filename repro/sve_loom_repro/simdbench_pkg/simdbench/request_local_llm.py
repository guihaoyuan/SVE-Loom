from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from task import sys_prompt
from pathlib import Path

model_name_or_path = str(Path("~/hf_models/DeepSeek-R1-Distill-Qwen-32B").expanduser())
# 建议加个断言，避免又掉回去走 hub 校验
assert Path(model_name_or_path).is_dir(), f"Local model dir not found: {model_name_or_path}"
tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
    trust_remote_code=True,
    use_fast=True,
    padding_side="left",
    local_files_only=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    local_files_only=True
)
model.config.max_length = 8192
model.eval()
import re
def clean_code_block(text: str) -> str:
    text = re.sub(r"```(?:cpp|c\+\+)?", "", text)
    text = text.replace("```", "")
    text = re.sub(r"\[/?cpp\]", "", text)
    return text.strip()

@torch.no_grad()
def request_llm_once_local(task: dict):
    # print("Model config max_length:", getattr(model.config, 'max_length', 'N/A'))
    # print("Model max_position_embeddings:", model.config.max_position_embeddings)
    system_prompt = sys_prompt(task["intrinsic"], task["task"])
    user_prompt = task["prompt"]
    full_prompt = f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt} [/INST]"

    inputs = tokenizer(
        full_prompt,
        return_tensors="pt",
        # truncation=True,
        # max_length=4096
    ).to(model.device)
    tokenizer.pad_token = tokenizer.eos_token
    # print("Input token length:", inputs['input_ids'].shape[1])
    outputs = model.generate(
        **inputs,
        max_new_tokens=2048,
        temperature=0.8,
        do_sample=True,
        top_p=0.95,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated_tokens = outputs[0][inputs['input_ids'].shape[-1]:]
    completion = tokenizer.decode(generated_tokens).strip()
    if completion.endswith("</s>"):
        completion = completion[:-len("</s>")]
    # completion = completion[len(full_prompt):].strip()
    # print(completion)
    return completion