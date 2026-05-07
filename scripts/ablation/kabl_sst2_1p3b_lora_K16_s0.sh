#!/bin/bash
# K-aggregation ablation K=16
# Paper reference: appendix.tex Table k-ablation (1.3B LoRA)
# Recipe: lora lr=5e-4 c=25 T=2000 MI=0.33 K=16 M=128 seed=0
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
run_kvar kabl_sst2_1p3b_lora_K16_s0 facebook/opt-1.3b SST2 lora 5e-4 25 2000 0.33 16 128 0
