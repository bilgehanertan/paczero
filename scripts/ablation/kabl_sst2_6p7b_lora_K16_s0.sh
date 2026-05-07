#!/bin/bash
# K-aggregation ablation K=16
# Paper reference: appendix.tex Table k-ablation-67b (6.7B LoRA)
# Recipe: lora lr=1e-3 c=10 T=1000 MI=0.33 K=16 M=128 seed=0
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
run_kvar kabl_sst2_6p7b_lora_K16_s0 facebook/opt-6.7b SST2 lora 1e-3 10 1000 0.33 16 128 0
