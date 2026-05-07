#!/bin/bash
# 6.7B LoRA clip ablation c=50 MI=0.33
# Paper reference: appendix.tex Table clip-ablation-67b
# Recipe: lr=1e-3 c=50 T=1000 MI=0.33 M=128 seed=0
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
run_pacmi clipabl_sst2_6p7b_lora_c50_mi0p33_s0 facebook/opt-6.7b SST2 lora 1e-3 50 1000 0.33 128 0
