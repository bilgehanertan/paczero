#!/bin/bash
# DP-AggZO K=64 FT, eps=0.2, T=2000 (cliff confirmation)
# Paper reference: experiments.tex §dp-cliff
# Recipe: ft lr=5e-6 c=25 T=2000 N=64 eps=0.2
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
run_dpaggzo dpaggzo_sst2_1p3b_ft_eps0p2_K64_T2000 facebook/opt-1.3b SST2 ft 5e-6 2000 0.2 25 64 0.064 0
