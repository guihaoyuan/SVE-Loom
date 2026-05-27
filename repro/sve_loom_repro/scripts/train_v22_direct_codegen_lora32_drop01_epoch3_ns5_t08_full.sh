#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../selective_repo/latest_pairs_DeesSeekR1-32B" && pwd)"
cd "$ROOT"

CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"
NPROC="${NPROC:-4}"
MODEL_PATH="${MODEL_PATH:-/home/user/hf_models/DeepSeek-R1-Distill-Qwen-32B}"
DATA_DIR="${DATA_DIR:-/home/user/simdbench_full/results/generated/train_dev_resplit_phase4i_atoms100_full_active_pool_9_1}"
TRAIN_FILE="${TRAIN_FILE:-$DATA_DIR/direct_codegen_train.resplit9_1.jsonl}"
EVAL_FILE="${EVAL_FILE:-$DATA_DIR/direct_codegen_dev.resplit9_1.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/simdbench_train_outputs/deepseek32b_v22_phase4i_atoms100_lora32_drop01_epoch3}"

MAX_LENGTH="${MAX_LENGTH:-4096}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
SAVE_STEPS="${SAVE_STEPS:-30}"
EVAL_STEPS="${EVAL_STEPS:-30}"
LORA_R="${LORA_R:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.1}"
LORA_ALPHA="${LORA_ALPHA:-16}"
DRY_RUN="${DRY_RUN:-0}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"

if [[ ! -f "$TRAIN_FILE" ]]; then
  echo "[v22_direct_codegen_train_lora32_drop01_epoch3] train file not found: $TRAIN_FILE" >&2
  exit 1
fi

if [[ ! -f "$EVAL_FILE" ]]; then
  echo "[v22_direct_codegen_train_lora32_drop01_epoch3] eval file not found: $EVAL_FILE" >&2
  exit 1
fi

CMD=(torchrun --nproc_per_node="$NPROC" scripts/train_sft.py
  --model_path "$MODEL_PATH"
  --train_file "$TRAIN_FILE"
  --eval_file "$EVAL_FILE"
  --output_dir "$OUTPUT_DIR"
  --bf16
  --max_length "$MAX_LENGTH"
  --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE"
  --per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH_SIZE"
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
  --learning_rate "$LEARNING_RATE"
  --num_train_epochs "$NUM_TRAIN_EPOCHS"
  --logging_steps "$LOGGING_STEPS"
  --save_steps "$SAVE_STEPS"
  --eval_steps "$EVAL_STEPS"
  --lora_r "$LORA_R"
  --lora_alpha "$LORA_ALPHA"
  --lora_dropout "$LORA_DROPOUT"
  --gradient_checkpointing)

if [[ "$TRUST_REMOTE_CODE" == "1" ]]; then
  CMD+=(--trust_remote_code)
fi

echo "[v22_direct_codegen_train_lora32_drop01_epoch3] data_dir=$DATA_DIR"
echo "[v22_direct_codegen_train_lora32_drop01_epoch3] train_file=$TRAIN_FILE"
echo "[v22_direct_codegen_train_lora32_drop01_epoch3] eval_file=$EVAL_FILE"
echo "[v22_direct_codegen_train_lora32_drop01_epoch3] output_dir=$OUTPUT_DIR"
echo "[v22_direct_codegen_train_lora32_drop01_epoch3] num_train_epochs=$NUM_TRAIN_EPOCHS"
echo "[v22_direct_codegen_train_lora32_drop01_epoch3] lora_r=$LORA_R"
echo "[v22_direct_codegen_train_lora32_drop01_epoch3] lora_alpha=$LORA_ALPHA"
echo "[v22_direct_codegen_train_lora32_drop01_epoch3] lora_dropout=$LORA_DROPOUT"
echo "[v22_direct_codegen_train_lora32_drop01_epoch3] trust_remote_code=$TRUST_REMOTE_CODE"
echo "CUDA_VISIBLE_DEVICES=$CUDA_DEVICES ${CMD[*]}"
if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
fi

mkdir -p "$OUTPUT_DIR"
CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" "${CMD[@]}"
