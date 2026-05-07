#!/bin/bash
# PAC-ZPL LR sweep (lora, clip=25, lr=1e-3)
# Paper reference: appendix.tex Table zpl-lr-sweep
# Recipe: lora lr=1e-3 c=25 T=1000 M=128 seed=0
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
run_paczpl zpl_lrsweep_sst2_1p3b_lora_c25_lr1em3_s0 facebook/opt-1.3b SST2 lora 1e-3 25 1000 128 0
