#!/bin/bash
# DPZero K=1 FT, eps=2.0
# Paper reference: experiments.tex Table dp-cliff
# Recipe: ft lr=5e-6 c=25 sample_rate=0.064 T=20000 N=1 eps=2.0
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
run_dpaggzo dpzero_sst2_1p3b_ft_eps2p0_K1_T20k facebook/opt-1.3b SST2 ft 5e-6 20000 2.0 25 1 0.064 0
