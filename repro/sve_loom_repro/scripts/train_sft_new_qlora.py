import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

try:
    from transformers import BitsAndBytesConfig
    BNB_AVAILABLE = True
except Exception:
    BNB_AVAILABLE = False

try:
    from peft import LoraConfig, PeftModel, get_peft_model
    try:
        from peft import prepare_model_for_kbit_training
    except Exception:
        prepare_model_for_kbit_training = None
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False
    prepare_model_for_kbit_training = None


def _try_bool(x: str) -> bool:
    if isinstance(x, bool):
        return x
    x = x.lower().strip()
    if x in ("1", "true", "yes", "y"):
        return True
    if x in ("0", "false", "no", "n"):
        return False
    raise ValueError(f"Cannot parse bool from: {x}")


def _has_chat_template(tokenizer) -> bool:
    tmpl = getattr(tokenizer, "chat_template", None)
    return bool(tmpl and isinstance(tmpl, str) and tmpl.strip())


def _apply_chat_template_to_ids(tokenizer, messages, add_generation_prompt: bool) -> List[int]:
    try:
        ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
        )
        if isinstance(ids, dict):
            ids = ids["input_ids"]
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return list(ids)
    except TypeError:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        return tokenizer(text, add_special_tokens=False).input_ids


def build_features_fn(tokenizer, max_length: int, use_chat_template: bool, plain_response_prefix: str):
    eos_id = tokenizer.eos_token_id

    def _build(ex: Dict[str, Any]) -> Dict[str, Any]:
        prompt = ex.get("prompt", "")
        response = ex.get("response", "")

        if use_chat_template and "messages" in ex and ex["messages"]:
            msgs = ex["messages"]
            if msgs[-1]["role"] != "assistant":
                msgs = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]

            msgs_prompt = msgs[:-1]
            prompt_ids = _apply_chat_template_to_ids(tokenizer, msgs_prompt, add_generation_prompt=True)

            response_text = msgs[-1]["content"]
            response_ids = tokenizer(response_text, add_special_tokens=False).input_ids
            if eos_id is not None:
                response_ids = response_ids + [eos_id]

            input_ids = prompt_ids + response_ids
            labels = [-100] * len(prompt_ids) + response_ids
        else:
            prompt_text = (prompt.rstrip() + "\n\n" + plain_response_prefix.rstrip() + "\n")
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids

            response_ids = tokenizer(response, add_special_tokens=False).input_ids
            if eos_id is not None:
                response_ids = response_ids + [eos_id]

            input_ids = prompt_ids + response_ids
            labels = [-100] * len(prompt_ids) + response_ids

        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
        attention_mask = [1] * len(input_ids)

        has_supervised = any(l != -100 for l in labels)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "has_supervised": has_supervised,
        }

    return _build


@dataclass
class DataCollatorForCausalLM:
    tokenizer: Any
    pad_to_multiple_of: Optional[int] = 8

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        for f in features:
            f.pop("has_supervised", None)

        max_len = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of:
            m = self.pad_to_multiple_of
            max_len = int(math.ceil(max_len / m) * m)

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        input_ids, attention_mask, labels = [], [], []
        for f in features:
            ids = f["input_ids"]
            mask = f["attention_mask"]
            lab = f["labels"]

            pad_n = max_len - len(ids)
            input_ids.append(ids + [pad_id] * pad_n)
            attention_mask.append(mask + [0] * pad_n)
            labels.append(lab + [-100] * pad_n)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def print_trainable_parameters(model) -> None:
    trainable, total = 0, 0
    for _, p in model.named_parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    if int(os.environ.get("RANK", "0")) == 0:
        pct = (100 * trainable / total) if total else 0.0
        print(f"Trainable params: {trainable:,} / {total:,} ({pct:.2f}%)")


def _load_tokenizer(model_path: str, adapter_path: Optional[str], trust_remote_code: bool):
    if adapter_path:
        try:
            tok = AutoTokenizer.from_pretrained(
                adapter_path,
                trust_remote_code=trust_remote_code,
                use_fast=False,
            )
            return tok
        except Exception:
            pass

    tok = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        use_fast=False,
    )
    return tok


def _resolve_qlora_compute_dtype(x: str):
    x = (x or "").lower().strip()
    if x == "bf16":
        return torch.bfloat16
    if x == "fp16":
        return torch.float16
    return torch.float32


def _resolve_device_map_for_qlora() -> Any:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return {"": int(os.environ.get("LOCAL_RANK", "0"))}
    return "auto"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True, help="基线模型目录（或HF模型名），例如 Qwen2.5-7B-Instruct")
    ap.add_argument("--adapter_path", type=str, default=None, help="可选：已有 adaptor(PEFT adapter) 目录；提供后将在其基础上继续训练")
    ap.add_argument("--train_file", type=str, required=True)
    ap.add_argument("--eval_file", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)

    ap.add_argument("--max_length", type=int, default=4096)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=16)
    ap.add_argument("--learning_rate", type=float, default=2e-4)
    ap.add_argument("--num_train_epochs", type=float, default=3)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--lr_scheduler_type", type=str, default="cosine")
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--eval_steps", type=int, default=200)
    ap.add_argument("--save_total_limit", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--gradient_checkpointing", action="store_true")

    ap.add_argument("--use_lora", type=_try_bool, default=True)
    ap.add_argument("--lora_r", type=int, default=64)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,up_proj,gate_proj,down_proj",
        help="逗号分隔；Qwen常用: q_proj,k_proj,v_proj,o_proj,up_proj,gate_proj,down_proj",
    )

    ap.add_argument("--use_qlora", action="store_true")
    ap.add_argument("--qlora_bits", type=int, default=4, choices=[4, 8])
    ap.add_argument("--qlora_double_quant", action="store_true")
    ap.add_argument("--qlora_quant_type", type=str, default="nf4", choices=["nf4", "fp4"])
    ap.add_argument("--qlora_compute_dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])

    ap.add_argument("--trust_remote_code", action="store_true", help="老Qwen/Qwen-7B(-Chat)可能需要")
    ap.add_argument("--deepspeed", type=str, default=None, help="可选：deepspeed json配置路径")
    ap.add_argument("--use_chat_template", type=str, default="auto", choices=["auto", "true", "false"])
    ap.add_argument("--plain_response_prefix", type=str, default="### UIR:", help="不用chat_template时拼到prompt后的响应前缀")
    ap.add_argument("--attn_implementation", type=str, default=None, help="可选: flash_attention_2 / sdpa / eager 等")
    ap.add_argument("--optim", type=str, default=None, help="默认: qlora用paged_adamw_8bit，否则adamw_torch")

    ap.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="可选：从 Trainer checkpoint 继续（含optimizer等状态），例如 output_dir/checkpoint-400",
    )

    args = ap.parse_args()

    if args.fp16 and args.bf16:
        raise ValueError("bf16 和 fp16 只能选一个；A100建议 bf16")

    if (args.adapter_path or args.use_lora or args.use_qlora) and not PEFT_AVAILABLE:
        raise RuntimeError("需要 peft，但未安装：pip install peft")

    if args.use_qlora:
        if not BNB_AVAILABLE:
            raise RuntimeError("需要 bitsandbytes，但未安装：pip install bitsandbytes")
        if prepare_model_for_kbit_training is None:
            raise RuntimeError("peft 版本过低，缺少 prepare_model_for_kbit_training")
        args.use_lora = True

    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    tokenizer = _load_tokenizer(
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.use_chat_template == "auto":
        use_chat_template = _has_chat_template(tokenizer)
    else:
        use_chat_template = (args.use_chat_template == "true")

    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else "auto")
    model_kwargs = dict(trust_remote_code=args.trust_remote_code, torch_dtype=dtype)
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    if args.use_qlora:
        qlora_compute_dtype = _resolve_qlora_compute_dtype(args.qlora_compute_dtype)
        if args.qlora_bits == 4:
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=args.qlora_quant_type,
                bnb_4bit_use_double_quant=args.qlora_double_quant,
                bnb_4bit_compute_dtype=qlora_compute_dtype,
            )
        else:
            bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
        model_kwargs["quantization_config"] = bnb_cfg
        model_kwargs["device_map"] = _resolve_device_map_for_qlora()
        model_kwargs["torch_dtype"] = qlora_compute_dtype

    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)

    if args.use_qlora:
        try:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
        except TypeError:
            model = prepare_model_for_kbit_training(model)
            if args.gradient_checkpointing:
                model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    else:
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            if hasattr(model.config, "use_cache"):
                model.config.use_cache = False

    if args.adapter_path:
        if int(os.environ.get("RANK", "0")) == 0:
            print(f"[INFO] Loading adapter from: {args.adapter_path}")
            print("[INFO] Will continue training on top of this adapter.")
        model = PeftModel.from_pretrained(model, args.adapter_path, is_trainable=True)
        print_trainable_parameters(model)
    else:
        if args.use_lora:
            target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
            lora_cfg = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=target_modules,
                task_type="CAUSAL_LM",
                bias="none",
            )
            model = get_peft_model(model, lora_cfg)
            print_trainable_parameters(model)
        else:
            print_trainable_parameters(model)

    ds = load_dataset("json", data_files={"train": args.train_file, "validation": args.eval_file})

    build_fn = build_features_fn(
        tokenizer=tokenizer,
        max_length=args.max_length,
        use_chat_template=use_chat_template,
        plain_response_prefix=args.plain_response_prefix,
    )

    def map_and_filter(split_name: str):
        mapped = ds[split_name].map(build_fn, remove_columns=ds[split_name].column_names)
        mapped = mapped.filter(lambda x: bool(x["has_supervised"]))
        mapped = mapped.remove_columns(["has_supervised"])
        return mapped

    train_ds = map_and_filter("train")
    eval_ds = map_and_filter("validation")

    optim_name = args.optim if args.optim else ("paged_adamw_8bit" if args.use_qlora else "adamw_torch")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        deepspeed=args.deepspeed,
        report_to=[],
        ddp_find_unused_parameters=False,
        optim=optim_name,
    )

    data_collator = DataCollatorForCausalLM(tokenizer=tokenizer, pad_to_multiple_of=8)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    if args.resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        trainer.train()

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if int(os.environ.get("RANK", "0")) == 0:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        (Path(args.output_dir) / "train_args.json").write_text(
            json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("Done. Saved to:", args.output_dir)
        if args.adapter_path or args.use_lora:
            print("NOTE: This output is a PEFT adapter directory; to load: base model + this adapter.")


if __name__ == "__main__":
    main()
