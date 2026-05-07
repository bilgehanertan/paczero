#!/bin/bash
# Mechanism decomposition raw_full, T=2000
# Paper reference: appendix.tex Table ablation
# Recipe: lora lr=5e-4 c=25 T=2000 M=128 ablation=none no_privacy=True
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
run_nonpriv mech_sst2_1p3b_lora_raw_full_T2000_s0 facebook/opt-1.3b SST2 lora 5e-4 25 2000 128 0 none
