#!/bin/bash
# 6.7B LoRA rank ablation r=4
# Paper reference: appendix.tex Table rank-ablation
# Recipe: lr=1e-3 c=25 T=1000 MI=0.33 r=4 M=128 seed=0
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
OUT_DIR="$OUT_BASE/rankabl_sst2_6p7b_lora_mi0p33_r4_s0"; mkdir -p "$OUT_DIR/checkpoints"
python paczo/run_paczo.py \
  --model_name facebook/opt-6.7b --task_name SST2 \
  --output_dir "$OUT_DIR/checkpoints" --result_file "$OUT_DIR/metrics.json" \
  --tag rankabl_sst2_6p7b_lora_mi0p33_r4_s0 --train_set_seed 0 \
  --num_train 1000 --num_dev 500 --num_eval 1000 \
  --max_steps "${STEPS_OVERRIDE:-1000}" --logging_steps 20 --eval_steps "${EVAL_OVERRIDE:-100}" \
  --trainer zo --load_float16 --lora --lora_r 4 \
  --learning_rate 1e-3 --zo_eps 1e-3 \
  --lr_scheduler_type polynomial --evaluation_strategy steps \
  --per_device_train_batch_size 1000 \
  --pac_m 128 --pac_mi 0.33 --pac_clip 25 \
  --pac_adaptive_mi True --pac_load_best_dev True \
  --overwrite_output_dir --train_as_classification

