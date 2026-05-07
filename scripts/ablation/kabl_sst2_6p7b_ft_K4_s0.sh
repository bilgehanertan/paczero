#!/bin/bash
# K-aggregation ablation K=4
# Paper reference: appendix.tex Table k-ablation-67b (6.7B FT)
# Recipe: ft lr=1e-4 c=1000 T=1000 MI=0.33 K=4 M=128 seed=0
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
run_kvar kabl_sst2_6p7b_ft_K4_s0 facebook/opt-6.7b SST2 ft 1e-4 1000 1000 0.33 4 128 0
