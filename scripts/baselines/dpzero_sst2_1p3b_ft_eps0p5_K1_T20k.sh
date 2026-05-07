#!/bin/bash
# DPZero K=1 FT, eps=0.5
# Paper reference: experiments.tex Table dp-cliff
# Recipe: ft lr=5e-6 c=25 sample_rate=0.064 T=20000 N=1 eps=0.5
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
run_dpaggzo dpzero_sst2_1p3b_ft_eps0p5_K1_T20k facebook/opt-1.3b SST2 ft 5e-6 20000 0.5 25 1 0.064 0
