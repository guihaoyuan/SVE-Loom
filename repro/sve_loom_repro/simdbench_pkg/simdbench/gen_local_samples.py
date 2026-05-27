#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 你仓库里有 task.py + sys_prompt，就沿用它

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

def sys_prompt(intrinsic: str, task_text: str) -> str:
    intrin_desc = {
        "SVE": "SVE (Arm ACLE for Scalable Vector Extension)",
        "Neon": "Neon (Advanced SIMD)",
        "AVX": "AVX/AVX2",
        "SSE": "SSE/SSE2",
        "RVV": "RISC-V Vector Extension",
    }
    include_hint = {
        "SVE": "#include <arm_sve.h>",
        "Neon": "#include <arm_neon.h>",
        "AVX": "#include <immintrin.h>",
        "SSE": "#include <immintrin.h>",
        "RVV": "#include <riscv_vector.h>",
    }
    return f"""You are an expert SIMD programmer.
Target intrinsic: {intrinsic} - {intrin_desc.get(intrinsic, intrinsic)}.

Requirements:
- Write C/C++ code using {intrinsic} intrinsics (not scalar fallback).
- Include the correct header: {include_hint.get(intrinsic, "")}.
- Follow the task specification below and produce a compilable implementation.
- Output ONLY code. No markdown. No explanations.

Task:
{task_text}
"""


def clean_code_block(text: str) -> str:
    text = re.sub(r"```(?:cpp|c\+\+|c)?", "", text)
    text = text.replace("```", "")
    text = re.sub(r"\[/?cpp\]", "", text)
    return text.strip()


def load_problems(problem_file: str):
    problems = []
    with open(problem_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            problems.append(json.loads(line))
    # 兼容一些数据集可能是 {"task_id":..., "prompt":...} 或 {"task_id":..., "task":..., "prompt":...}
    # 必须至少有 task_id + prompt
    for p in problems:
        if "task_id" not in p:
            raise ValueError("problem missing task_id")
        if "prompt" not in p:
            raise ValueError(f"problem {p['task_id']} missing prompt")
        # task 字段若不存在，就用空串
        if "task" not in p:
            p["task"] = ""
    return problems


def build_full_prompt(intrinsic_arg: str, task_obj: dict) -> str:
    intrinsic = task_obj.get("intrinsic", intrinsic_arg)  # ★优先用 task 自己的
    system_prompt = sys_prompt(intrinsic, task_obj.get("task", ""))
    user_prompt = task_obj["prompt"]

    # ★把末尾空函数体 "{\n}" 改成 HumanEval 风格 "{\n"
    user_prompt = re.sub(r"\{\s*\}\s*$", "{\n", user_prompt)

    return f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt} [/INST]"


def get_dist_info():
    """Detect torchrun/torch.distributed environment."""
    ws = int(os.environ.get("WORLD_SIZE", "1"))
    if ws > 1:
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        return rank, ws, local_rank, True
    return 0, 1, 0, False


def shard_contiguous(items, num_shards: int, shard_id: int):
    """Split list into contiguous shards to keep global order after concatenation."""
    if num_shards <= 1:
        return items
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError(f"Invalid shard_id={shard_id} for num_shards={num_shards}")
    n = len(items)
    per = (n + num_shards - 1) // num_shards  # ceil
    start = shard_id * per
    end = min(start + per, n)
    return items[start:end]


def shard_output_path(base_out: Path, shard_id: int, num_shards: int) -> Path:
    if num_shards <= 1:
        return base_out
    return base_out.with_name(f"{base_out.stem}.shard{shard_id}of{num_shards}{base_out.suffix}")


def merge_shard_outputs(base_out: Path, num_shards: int):
    # 按 shard_id 顺序拼接，保证任务顺序一致（因为我们用 contiguous shard 划分）
    with base_out.open("w", encoding="utf-8") as wf:
        for sid in range(num_shards):
            part = shard_output_path(base_out, sid, num_shards)
            if not part.exists():
                continue
            with part.open("r", encoding="utf-8") as rf:
                for line in rf:
                    wf.write(line)


@torch.no_grad()
def generate_one(model, tokenizer, full_prompt: str, max_new_tokens: int, temperature: float, top_p: float, repetition_penalty: float):
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=(temperature > 0),
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen = outputs[0][inputs["input_ids"].shape[-1]:]
    text = tokenizer.decode(gen, skip_special_tokens=False).strip()
    if text.endswith("</s>"):
        text = text[:-len("</s>")]
    return clean_code_block(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True, help="Local HF model directory")
    ap.add_argument("--adapter_path", default="", help="Optional local LoRA/PEFT adapter dir")
    ap.add_argument("--problem_file", required=True, help="Problem jsonl (task_id + prompt required)")
    ap.add_argument("--intrinsic", default="SVE", choices=["SSE", "AVX", "SVE", "Neon", "RVV"])
    ap.add_argument("--output", required=True, help="Output samples jsonl (base path)")
    ap.add_argument("--n_samples", type=int, default=1, help="Samples per task (for pass@k)")
    ap.add_argument("--max_new_tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--repetition_penalty", type=float, default=1.1)
    ap.add_argument("--seed", type=int, default=1234)

    # 手动多进程切分（不使用 torchrun 时）
    ap.add_argument("--num_shards", type=int, default=1, help="Manual data parallel: total shards (ignored under torchrun)")
    ap.add_argument("--shard_id", type=int, default=0, help="Manual data parallel: this shard id (ignored under torchrun)")

    # torchrun 模式下，rank0 会自动 merge 成 --output；默认会清理 shard 文件
    ap.add_argument("--keep_shard_files", action="store_true", help="Keep per-shard output files (torchrun merge mode)")

    args = ap.parse_args()

    rank, world_size, local_rank, distributed = get_dist_info()

    # 统一“数据切分”参数：torchrun -> 用 rank/world_size；普通 python -> 用 args.*_shards
    if distributed:
        import torch.distributed as dist
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        shard_id = rank
        num_shards = world_size
    else:
        shard_id = args.shard_id
        num_shards = args.num_shards

    # 每个 shard/rank 用不同 seed，避免采样时随机数完全一致
    torch.manual_seed(args.seed + shard_id)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + shard_id)

    model_path = str(Path(args.model_path).expanduser())
    if not Path(model_path).is_dir():
        raise SystemExit(f"Model dir not found: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=True,
        padding_side="left",
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 注意：
    # - distributed(torchrun) 模式：每个进程把“整模型”放到自己的 local_rank GPU 上（数据并行）
    # - 普通模式：device_map=auto 会在当前进程可见的 GPU 上做模型切分（更适合模型放不下单卡的情况）
    if distributed:
        device_map = {"": f"cuda:{local_rank}"}
    else:
        device_map = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
        local_files_only=True,
    )
    model.eval()

    # 可选：加载 LoRA/PEFT adapter
    if args.adapter_path:
        adapter_path = str(Path(args.adapter_path).expanduser())
        if not Path(adapter_path).is_dir():
            raise SystemExit(f"Adapter dir not found: {adapter_path}")
        try:
            from peft import PeftModel
        except Exception as e:
            raise SystemExit(f"peft not installed. pip install peft. ({e})")
        model = PeftModel.from_pretrained(model, adapter_path, local_files_only=True)
        # 推荐 merge，提高推理稳定性/速度；如果你是量化 LoRA merge 失败就注释掉
        try:
            model = model.merge_and_unload()
        except Exception as e:
            if rank == 0:
                print("[WARN] merge_and_unload failed, continue without merge:", e)

    problems = load_problems(args.problem_file)
    my_problems = shard_contiguous(problems, num_shards=num_shards, shard_id=shard_id)

    base_out = Path(args.output)
    out_path = shard_output_path(base_out, shard_id=shard_id, num_shards=num_shards)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_local = len(my_problems) * args.n_samples

    pbar = None
    if tqdm is not None:
        # torchrun: local_rank 是当前机器上的 GPU 序号；非 torchrun 用 shard_id
        pos = local_rank if distributed else shard_id
        name = f"rank{rank}" if distributed else f"shard{shard_id}"
        pbar = tqdm(
            total=total_local,
            desc=f"{name}/{num_shards} -> {out_path.name}",
            position=pos,
            dynamic_ncols=True,
            leave=True,
        )

    # 写 jsonl：每条是 {"task_id":..., "completion":...}
    with out_path.open("w", encoding="utf-8") as wf:
        for p in my_problems:
            tid = p["task_id"]
            full_prompt = build_full_prompt(args.intrinsic, p)

            for s in range(args.n_samples):
                comp = generate_one(
                    model, tokenizer, full_prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                )
                wf.write(json.dumps({"task_id": tid, "completion": comp}, ensure_ascii=False) + "\n")
                if pbar is not None:
                    pbar.update(1)
            if pbar is not None:
                pbar.close()

    if distributed:
        import torch.distributed as dist
        dist.barrier()
        if rank == 0:
            # merge 到用户给的 --output
            merge_shard_outputs(base_out, num_shards=num_shards)
            if not args.keep_shard_files:
                for sid in range(num_shards):
                    part = shard_output_path(base_out, shard_id=sid, num_shards=num_shards)
                    try:
                        part.unlink()
                    except FileNotFoundError:
                        pass
            print(f"[OK] wrote merged: {base_out} (tasks={len(problems)} shards={num_shards} samples_per_task={args.n_samples})")
        dist.destroy_process_group()
    else:
        if num_shards == 1:
            print(f"[OK] wrote: {out_path} (tasks={len(problems)} samples_per_task={args.n_samples})")
        else:
            print(f"[OK] wrote shard: {out_path} (shard {shard_id+1}/{num_shards}, shard_tasks={len(my_problems)}, total_tasks={len(problems)})")


if __name__ == "__main__":
    main()