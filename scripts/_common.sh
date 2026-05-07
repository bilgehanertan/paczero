#!/bin/bash
# Common helpers sourced by every cell script. Sets paths and exposes
# run_pacmi / run_paczpl / run_kvar / run_dpaggzo / run_nonpriv. Set the
# SMOKE=1 environment variable to override --max_steps / --eval_steps with
# tiny values for a sanity check.
set -euo pipefail

PACZO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PACZO_ROOT"
export PYTHONPATH="${PACZO_ROOT}:${PYTHONPATH:-}"

OUT_BASE="${OUT_BASE:-${PACZO_ROOT}/result/runs}"
mkdir -p "$OUT_BASE"

if [[ -n "${SMOKE:-}" ]]; then
  STEPS_OVERRIDE="${SMOKE_STEPS:-1}"
  EVAL_OVERRIDE="${SMOKE_EVAL:-999999}"
  NTRAIN_OVERRIDE="${SMOKE_NTRAIN:-128}"
  NDEV_OVERRIDE="${SMOKE_NDEV:-10}"
  NEVAL_OVERRIDE="${SMOKE_NEVAL:-20}"
  BATCH_OVERRIDE="${SMOKE_BATCH:-128}"
  PAC_LOAD_BEST_DEV_FLAG=""
else
  NTRAIN_OVERRIDE=1000; NDEV_OVERRIDE=500; NEVAL_OVERRIDE=1000; BATCH_OVERRIDE=1000
  PAC_LOAD_BEST_DEV_FLAG="--pac_load_best_dev True"
fi

run_pacmi() {
  local run_id="$1" model="$2" task="$3" mode="$4" lr="$5" clip="$6" steps="$7" mi="$8" pac_m="${9:-128}" seed="${10:-0}"
  local out_dir="${OUT_BASE}/${run_id}"
  mkdir -p "$out_dir/checkpoints"
  local extra=()
  [[ "$mode" == "lora" ]] && extra+=( --lora )
  if [[ "$task" == "SQuAD" ]]; then extra+=( --non_diff ); else extra+=( --train_as_classification ); fi
  [[ -n "${STEPS_OVERRIDE:-}" ]] && steps="$STEPS_OVERRIDE"
  local eval_steps="${EVAL_OVERRIDE:-100}"
  python paczo/run_paczo.py \
    --model_name "$model" --task_name "$task" \
    --output_dir "$out_dir/checkpoints" --result_file "$out_dir/metrics.json" \
    --tag "$run_id" --train_set_seed "$seed" \
    --num_train "$NTRAIN_OVERRIDE" --num_dev "$NDEV_OVERRIDE" --num_eval "$NEVAL_OVERRIDE" \
    --max_steps "$steps" --logging_steps 20 --eval_steps "$eval_steps" \
    --trainer zo --load_float16 \
    --learning_rate "$lr" --zo_eps 1e-3 \
    --lr_scheduler_type polynomial --evaluation_strategy steps \
    --per_device_train_batch_size "$BATCH_OVERRIDE" \
    --pac_m "$pac_m" --pac_mi "$mi" --pac_clip "$clip" \
    --pac_adaptive_mi True $PAC_LOAD_BEST_DEV_FLAG \
    --overwrite_output_dir \
    "${extra[@]}"
}

run_paczpl() {
  local run_id="$1" model="$2" task="$3" mode="$4" lr="$5" clip="$6" steps="$7" pac_m="${8:-126}" seed="${9:-0}"
  local out_dir="${OUT_BASE}/${run_id}"
  mkdir -p "$out_dir/checkpoints"
  local extra=()
  [[ "$mode" == "lora" ]] && extra+=( --lora )
  if [[ "$task" == "SQuAD" ]]; then extra+=( --non_diff ); else extra+=( --train_as_classification ); fi
  [[ -n "${STEPS_OVERRIDE:-}" ]] && steps="$STEPS_OVERRIDE"
  local eval_steps="${EVAL_OVERRIDE:-100}"
  python paczo/run_paczo.py \
    --model_name "$model" --task_name "$task" \
    --output_dir "$out_dir/checkpoints" --result_file "$out_dir/metrics.json" \
    --tag "$run_id" --train_set_seed "$seed" \
    --num_train "$NTRAIN_OVERRIDE" --num_dev "$NDEV_OVERRIDE" --num_eval "$NEVAL_OVERRIDE" \
    --max_steps "$steps" --logging_steps 20 --eval_steps "$eval_steps" \
    --trainer zo --load_float16 \
    --learning_rate "$lr" --zo_eps 1e-3 \
    --lr_scheduler_type polynomial --evaluation_strategy steps \
    --per_device_train_batch_size "$BATCH_OVERRIDE" \
    --pac_m "$pac_m" --pac_clip "$clip" \
    --pac_zpl True $PAC_LOAD_BEST_DEV_FLAG \
    --overwrite_output_dir \
    "${extra[@]}"
}

run_kvar() {
  local run_id="$1" model="$2" task="$3" mode="$4" lr="$5" clip="$6" steps="$7" mi="$8" K="$9" pac_m="${10:-128}" seed="${11:-0}"
  local out_dir="${OUT_BASE}/${run_id}"
  mkdir -p "$out_dir/checkpoints"
  local extra=()
  [[ "$mode" == "lora" ]] && extra+=( --lora )
  if [[ "$task" == "SQuAD" ]]; then extra+=( --non_diff ); else extra+=( --train_as_classification ); fi
  [[ -n "${STEPS_OVERRIDE:-}" ]] && steps="$STEPS_OVERRIDE"
  local eval_steps="${EVAL_OVERRIDE:-200}"
  python paczo/run_paczo_kvar.py \
    --model_name "$model" --task_name "$task" \
    --output_dir "$out_dir/checkpoints" --result_file "$out_dir/metrics.json" \
    --tag "$run_id" --train_set_seed "$seed" \
    --num_train "$NTRAIN_OVERRIDE" --num_dev "$NDEV_OVERRIDE" --num_eval "$NEVAL_OVERRIDE" \
    --max_steps "$steps" --logging_steps 20 --eval_steps "$eval_steps" \
    --trainer zo --load_float16 \
    --learning_rate "$lr" --zo_eps 1e-3 \
    --lr_scheduler_type polynomial --evaluation_strategy steps \
    --per_device_train_batch_size "$BATCH_OVERRIDE" \
    --pac_m "$pac_m" --pac_mi "$mi" --pac_clip "$clip" --pac_k "$K" \
    --pac_adaptive_mi True $PAC_LOAD_BEST_DEV_FLAG \
    --overwrite_output_dir \
    "${extra[@]}"
}

run_dpaggzo() {
  local run_id="$1" model="$2" task="$3" mode="$4" lr="$5" steps="$6" dp_eps="$7" dp_clip="$8" N="$9" sample_rate="${10:-0.064}" seed="${11:-0}"
  local out_dir="${OUT_BASE}/${run_id}"
  mkdir -p "$out_dir"
  pushd baselines/dp-aggzo/opt > /dev/null
  mkdir -p result
  local short
  short="$(basename "$model")"
  local tag="dpzero-${mode}-${steps}-${sample_rate}-${lr}-1e-3-${seed}-${dp_eps}-${dp_clip}"
  ln -sfn "$out_dir" "result/${task}-${short}-${tag}"
  [[ -n "${STEPS_OVERRIDE:-}" ]] && steps="$STEPS_OVERRIDE"
  PYTHONPATH="$PWD" \
  MODEL="$model" TASK="$task" MODE="$mode" \
    LR="$lr" EPS=1e-3 SEED="$seed" \
    TRAIN="$NTRAIN_OVERRIDE" DEV="$NDEV_OVERRIDE" EVAL="$NEVAL_OVERRIDE" STEPS="$steps" EVAL_STEPS="${EVAL_OVERRIDE:-200}" \
    DP_EPS="$dp_eps" DP_CLIP="$dp_clip" DP_SAMPLE_RATE="$sample_rate" N="$N" \
    bash examples/dpaggzo.sh
  popd > /dev/null
}

run_nonpriv() {
  local run_id="$1" model="$2" task="$3" mode="$4" lr="$5" clip="$6" steps="$7" pac_m="${8:-128}" seed="${9:-0}" ablation="${10:-none}"
  local out_dir="${OUT_BASE}/${run_id}"
  mkdir -p "$out_dir/checkpoints"
  local extra=()
  [[ "$mode" == "lora" ]] && extra+=( --lora )
  if [[ "$task" == "SQuAD" ]]; then extra+=( --non_diff ); else extra+=( --train_as_classification ); fi
  [[ -n "${STEPS_OVERRIDE:-}" ]] && steps="$STEPS_OVERRIDE"
  local eval_steps="${EVAL_OVERRIDE:-100}"
  python paczo/run_paczo.py \
    --model_name "$model" --task_name "$task" \
    --output_dir "$out_dir/checkpoints" --result_file "$out_dir/metrics.json" \
    --tag "$run_id" --train_set_seed "$seed" \
    --num_train "$NTRAIN_OVERRIDE" --num_dev "$NDEV_OVERRIDE" --num_eval "$NEVAL_OVERRIDE" \
    --max_steps "$steps" --logging_steps 20 --eval_steps "$eval_steps" \
    --trainer zo --load_float16 \
    --learning_rate "$lr" --zo_eps 1e-3 \
    --lr_scheduler_type polynomial --evaluation_strategy steps \
    --per_device_train_batch_size "$BATCH_OVERRIDE" \
    --pac_m "$pac_m" --pac_clip "$clip" \
    --no_privacy True --ablation "$ablation" \
    $PAC_LOAD_BEST_DEV_FLAG \
    --overwrite_output_dir \
    "${extra[@]}"
}
