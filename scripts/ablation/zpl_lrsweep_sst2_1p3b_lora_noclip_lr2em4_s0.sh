#!/bin/bash
# PAC-ZPL LR sweep (lora, clip=1e9, lr=2e-4)
# Paper reference: appendix.tex Table zpl-lr-sweep
# Recipe: lora lr=2e-4 c=1e9 T=1000 M=128 seed=0
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
run_paczpl zpl_lrsweep_sst2_1p3b_lora_noclip_lr2em4_s0 facebook/opt-1.3b SST2 lora 2e-4 1e9 1000 128 0
