#!/bin/bash
# FT plateau, MI=1e-3
# Paper reference: appendix.tex Table FT-plateau-adaptive
# Recipe: lr=1e-4 c=1000 T=1000 MI=1e-3 M=128 seed=0
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
run_pacmi plateau_sst2_1p3b_ft_mi1em3_s0 facebook/opt-1.3b SST2 ft 1e-4 1000 1000 1e-3 128 0
