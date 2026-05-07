#!/bin/bash
# PAC-ZPL LR sweep (ft, clip=1000, lr=5e-5)
# Paper reference: appendix.tex Table zpl-lr-sweep
# Recipe: ft lr=5e-5 c=1000 T=1000 M=128 seed=0
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
run_paczpl zpl_lrsweep_sst2_1p3b_ft_c1000_lr5em5_s0 facebook/opt-1.3b SST2 ft 5e-5 1000 1000 128 0
