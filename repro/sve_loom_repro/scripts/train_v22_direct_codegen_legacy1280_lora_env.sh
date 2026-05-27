#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../selective_repo/latest_pairs_DeesSeekR1-32B" && pwd)"
cd "$ROOT"

CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"
NPROC="${NPROC:-4}"
MODEL_PATH="${MODEL_PATH:-/home/user/hf_models/DeepSeek-R1-Distill-Qwen-32B}"
DATA_DIR="${DATA_DIR:-/home/user/selective_repo/latest_pairs_DeesSeekR1-32B/semantic_pipeline_v1/v22_direct_codegen_legacy1280}"
TRAIN_FILE="${TRAIN_FILE:-$DATA_DIR/direct_codegen_train.jsonl}"
EVAL_FILE="${EVAL_FILE:-$DATA_DIR/direct_codegen_dev.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/user/selective_repo/latest_pairs_DeesSeekR1-32B/outputs/qwen32b_v22_direct_codegen_legacy1280}"

MAX_LENGTH="${MAX_LENGTH:-4096}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-4}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
SAVE_STEPS="${SAVE_STEPS:-100}"
EVAL_STEPS="${EVAL_STEPS:-100}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
SAVE_BEST_LIMIT="${SAVE_BEST_LIMIT:-0}"
METRIC_FOR_BEST_MODEL="${METRIC_FOR_BEST_MODEL:-eval_loss}"
GREATER_IS_BETTER="${GREATER_IS_BETTER:-false}"
LOAD_BEST_MODEL_AT_END="${LOAD_BEST_MODEL_AT_END:-false}"
if [[ "$SAVE_BEST_LIMIT" != "0" ]]; then
  LOAD_BEST_MODEL_AT_END="false"
fi
DRY_RUN="${DRY_RUN:-0}"
SEED="${SEED:-42}"
LORA_R="${LORA_R:-64}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.1}"

if [[ ! -f "$TRAIN_FILE" ]]; then
  echo "[v22_direct_codegen_train_lora_env] train file not found: $TRAIN_FILE" >&2
  exit 1
fi

if [[ ! -f "$EVAL_FILE" ]]; then
  echo "[v22_direct_codegen_train_lora_env] eval file not found: $EVAL_FILE" >&2
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
  --save_total_limit "$SAVE_TOTAL_LIMIT"
  --save_best_limit "$SAVE_BEST_LIMIT"
  --metric_for_best_model "$METRIC_FOR_BEST_MODEL"
  --greater_is_better "$GREATER_IS_BETTER"
  --load_best_model_at_end "$LOAD_BEST_MODEL_AT_END"
  --seed "$SEED"
  --gradient_checkpointing
  --lora_r "$LORA_R"
  --lora_alpha "$LORA_ALPHA"
  --lora_dropout "$LORA_DROPOUT")

echo "[v22_direct_codegen_train_lora_env] data_dir=$DATA_DIR"
echo "[v22_direct_codegen_train_lora_env] train_file=$TRAIN_FILE"
echo "[v22_direct_codegen_train_lora_env] eval_file=$EVAL_FILE"
echo "[v22_direct_codegen_train_lora_env] output_dir=$OUTPUT_DIR"
echo "[v22_direct_codegen_train_lora_env] seed=$SEED"
echo "[v22_direct_codegen_train_lora_env] lora_r=$LORA_R"
echo "[v22_direct_codegen_train_lora_env] lora_alpha=$LORA_ALPHA"
echo "[v22_direct_codegen_train_lora_env] lora_dropout=$LORA_DROPOUT"
echo "[v22_direct_codegen_train_lora_env] save_total_limit=$SAVE_TOTAL_LIMIT"
echo "[v22_direct_codegen_train_lora_env] save_best_limit=$SAVE_BEST_LIMIT"
echo "[v22_direct_codegen_train_lora_env] metric_for_best_model=$METRIC_FOR_BEST_MODEL"
echo "[v22_direct_codegen_train_lora_env] greater_is_better=$GREATER_IS_BETTER"
echo "[v22_direct_codegen_train_lora_env] load_best_model_at_end=$LOAD_BEST_MODEL_AT_END"
echo "CUDA_VISIBLE_DEVICES=$CUDA_DEVICES ${CMD[*]}"
if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
fi

mkdir -p "$OUTPUT_DIR"
CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" "${CMD[@]}"
