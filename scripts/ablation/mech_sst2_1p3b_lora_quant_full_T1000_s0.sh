#!/bin/bash
# Mechanism decomposition quant_full, T=1000
# Paper reference: appendix.tex Table ablation
# Recipe: lora lr=5e-4 c=25 T=1000 M=128 ablation=quant_full no_privacy=True
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
run_nonpriv mech_sst2_1p3b_lora_quant_full_T1000_s0 facebook/opt-1.3b SST2 lora 5e-4 25 1000 128 0 quant_full
