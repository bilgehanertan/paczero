#!/bin/bash
# K-aggregation ablation K=4
# Paper reference: appendix.tex Table k-ablation-67b (6.7B LoRA)
# Recipe: lora lr=1e-3 c=10 T=1000 MI=0.33 K=4 M=128 seed=0
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
run_kvar kabl_sst2_6p7b_lora_K4_s0 facebook/opt-6.7b SST2 lora 1e-3 10 1000 0.33 4 128 0
