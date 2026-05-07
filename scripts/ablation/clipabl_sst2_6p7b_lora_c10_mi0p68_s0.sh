#!/bin/bash
# 6.7B LoRA clip ablation c=10 MI=0.68
# Paper reference: appendix.tex Table clip-ablation-67b
# Recipe: lr=1e-3 c=10 T=1000 MI=0.68 M=128 seed=0
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
run_pacmi clipabl_sst2_6p7b_lora_c10_mi0p68_s0 facebook/opt-6.7b SST2 lora 1e-3 10 1000 0.68 128 0
